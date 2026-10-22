"""CPU-only ML/DL experiments invoked by both a notebook and a command line.

Completion records execution only. Every generated Cards sample remains in its
denominator; nothing here can promote a FrameGenX generator or verifier.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import platform
import time
import uuid

import numpy as np
from PIL import Image
import psutil
import torch
from safetensors.torch import save_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.research.artifacts import ROOT, atomic_json, experiment_lock, json_digest, sha256, workspace_path
from src.research.cards import load_data, prepare_data
from src.research.config import load_config
from src.research.diffusion import CardsDenoiser, LearnedVarianceDiffusion
from src.research.ihalla import inspect_ihalla
from src.training.tracking import ExperimentTracker


def source_identity():
    paths = sorted((ROOT / "src/research").glob("*.py"))
    # Training tracking also participates in an experiment's reproducibility.
    paths.append(ROOT / "src/training/tracking.py")
    paths.append(ROOT / "src/core/files.py")
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def runtime_identity():
    packages = ("torch", "numpy", "scipy", "Pillow", "scikit-learn", "safetensors",
                "mlflow-skinny", "SQLAlchemy", "alembic", "psutil", "threadpoolctl")
    return {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
            "packages": {name: version(name) for name in packages},
            "torch_threads": torch.get_num_threads(),
            "strict_determinism": torch.are_deterministic_algorithms_enabled()}


def snapshot_source(run_root, sources):
    for name, expected in sources.items():
        source = (ROOT / name).resolve()
        destination = (run_root / "source" / name).resolve()
        if not source.is_relative_to(ROOT.resolve()) or not destination.is_relative_to((run_root / "source").resolve()):
            raise ValueError("Research source snapshot escaped its roots")
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Research source changed before fitting")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)


def corrupt_image(pixels, kind, seed):
    rng = np.random.default_rng(seed)
    result = pixels.copy()
    if kind == "noise":
        result = rng.integers(0, 256, size=pixels.shape, dtype=np.uint8)
    elif kind == "missing_tile":
        result[:64, :64] = 255
    elif kind == "pixel_shuffle":
        result[:64, :64] = pixels[:64, :64].reshape(-1, 3)[rng.permutation(4096)].reshape(64, 64, 3)
    elif kind == "color_change":
        tile = result[:64, :64]
        foreground = tile.min(axis=2) < 180
        tile[foreground] = [30, 100, 230]  # blue symbols are outside these released templates
    else:
        raise ValueError("Unknown frozen corruption")
    return result


class Budget:
    def __init__(self, seconds):
        self.start = time.monotonic()
        self.seconds = seconds
        self.peak_rss = 0
        self.peak_system_used = 0
        self.minimum_available = None

    def check(self):
        if time.monotonic()-self.start > self.seconds:
            raise TimeoutError("Research wall-clock budget exhausted")
        process = psutil.Process()
        rss = process.memory_info().rss + sum(child.memory_info().rss for child in process.children() if child.is_running())
        memory = psutil.virtual_memory()
        self.peak_rss = max(self.peak_rss, rss)
        self.peak_system_used = max(self.peak_system_used, memory.used)
        self.minimum_available = memory.available if self.minimum_available is None else min(self.minimum_available, memory.available)
        if rss > 8_000_000_000 or memory.available < 750_000_000:
            raise MemoryError("CPU research memory budget exceeded")

    def summary(self):
        return {"elapsed_seconds": time.monotonic()-self.start,
                "observed_peak_process_tree_rss_bytes": self.peak_rss,
                "observed_peak_system_used_bytes": self.peak_system_used,
                "observed_minimum_available_bytes": self.minimum_available,
                "limits": {"process_tree_rss_bytes": 8_000_000_000, "minimum_available_bytes": 750_000_000,
                           "wall_seconds": self.seconds},
                "measurement": "sampled CPU RSS/system used; not committed memory or GPU peak",
                "device": "cpu", "cuda_initialized_by_experiment": False}


def pixels_for(record, config):
    path = workspace_path(config.data_root) / "subset" / record["file"]
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def train_classical(config, manifest, matcher, root, budget):
    features, labels, splits, kinds = [], [], [], []
    for index, record in enumerate(manifest["records"]):
        if record["source_status"] != "valid":
            continue
        budget.check()
        pixels = pixels_for(record, config)
        cases = [("source_valid", pixels)] + [(kind, corrupt_image(pixels, kind, config.seed+index))
                                             for kind in config.corruptions]
        for kind, variant in cases:
            features.append(matcher.inspect(variant, config.ncc_threshold, config.ambiguity_margin)["features"])
            labels.append(int(kind == "source_valid"))
            splits.append(record["split"])
            kinds.append(kind)
    values, target, splits = np.asarray(features), np.asarray(labels), np.asarray(splits)
    scaler = StandardScaler().fit(values[splits == "train"])
    model = LogisticRegression(C=1, max_iter=1000, random_state=config.seed).fit(
        scaler.transform(values[splits == "train"]), target[splits == "train"])
    # Fixed .5 decision; validation/test never choose this threshold or C.
    metrics = {}
    for split in ("validation", "test"):
        selection = splits == split
        probability = model.predict_proba(scaler.transform(values[selection]))[:, 1]
        prediction = probability >= .5
        metrics[split] = {"n": int(selection.sum()),
            "balanced_accuracy": float(balanced_accuracy_score(target[selection], prediction)),
            "roc_auc": float(roc_auc_score(target[selection], probability)),
            "confusion_matrix_true_rows_predicted_columns": confusion_matrix(target[selection], prediction, labels=[0, 1]).tolist(),
            "by_corruption": {kind: {"n": int(sum(np.asarray(kinds)[selection] == kind)),
                "correct": int(sum((prediction == target[selection])[np.asarray(kinds)[selection] == kind]))}
                for kind in ("source_valid", *config.corruptions)}}
    artifact = {"schema": "framegenx-cards-logistic-v1", "feature_order": "quadrants TL,TR,BL,BR each NCC,margin,RGB_RMSE",
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coefficient": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
        "threshold": .5, "labels": "released template-valid sources vs specified synthetic corruptions",
        "quality_status": "unverified", "calibrated": False}
    atomic_json(root / "ml_model.json", artifact)
    tracker = ExperimentTracker("runs/mlflow_research.db", "framegenx-cards-research", "cards-classical-ml")
    try:
        tracker.params({"dataset_revision": config.dataset_revision, "seed": config.seed,
                        "scope": "synthetic_corruption_diagnostic", "run_id": root.name})
        tracker.metrics({"test_balanced_accuracy": metrics["test"]["balanced_accuracy"],
                         "test_roc_auc": metrics["test"]["roc_auc"]})
        tracker.artifact(root / "ml_model.json")
        tracker.finish("FINISHED")
    except BaseException:
        tracker.finish("FAILED")
        raise
    return {"method": "logistic_regression", "status": "trained", "metrics": metrics,
            "mlflow_run_id": tracker.run_id, "quality_status": "unverified",
            "scope": "Cards template features on named synthetic corruptions; not independent generic visual verification"}


def train_arm(config, records, root, arm, rho, budget):
    arm_budget = Budget(config.max_seconds_per_arm)
    torch.manual_seed(config.seed)
    model = CardsDenoiser(config.width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    initial_hash = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    selection = [record for record in records if record["split"] == "train" and record["source_status"] == "valid"]
    # uint8 storage is bounded (~200MB for 4096 sources); convert minibatches only.
    data = np.stack([pixels_for(record, config) for record in selection])
    order = torch.Generator(device="cpu").manual_seed(config.seed+1)
    noise_stream = torch.Generator(device="cpu").manual_seed(config.seed+2)
    diffusion = LearnedVarianceDiffusion(config.diffusion_steps)
    tracker = ExperimentTracker("runs/mlflow_research.db", "framegenx-cards-research", "cards-"+arm)
    loss_rows = []
    arm_root = root / arm
    arm_root.mkdir()
    try:
        tracker.params({"run_id": root.name, "rho": rho, "device": "cpu", "steps": config.steps,
            "dataset_revision": config.dataset_revision, "initial_sha256": initial_hash,
            "schedule": config.vsm_schedule, "quality_scope": "cards_feasibility_not_sdxl"})
        model.train()
        for step in range(config.steps):
            budget.check()
            arm_budget.check()
            index = torch.randint(len(data), (config.batch_size,), generator=order)
            clean = torch.from_numpy(data[index.numpy()].copy()).permute(0, 3, 1, 2).float()/127.5-1
            timestep = torch.randint(diffusion.steps, (len(clean),), generator=noise_stream)
            noise = torch.randn(clean.shape, generator=noise_stream)
            noisy = diffusion.noisy(clean, timestep, noise)
            optimizer.zero_grad(set_to_none=True)
            terms = diffusion.loss(model(noisy, timestep), clean, noisy, noise, timestep, rho)
            terms["loss"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise ValueError("Optimizer produced nonfinite research weights")
            if step == 0 or (step+1) % 50 == 0 or step+1 == config.steps:
                metrics = {name: float(value.detach()) for name, value in terms.items()}
                metrics["gradient_norm"] = float(gradient_norm)
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise ValueError("Nonfinite training metrics")
                loss_rows.append({"step": step+1, **metrics})
                tracker.metrics(metrics, step=step+1)
                atomic_json(arm_root / "progress.json", {"status": "training", "step": step+1,
                    "target_steps": config.steps, "quality_status": "unverified", "metrics": loss_rows})
                print(f"{arm}: {step+1}/{config.steps} loss={metrics['loss']:.5f}", flush=True)
            if (step+1) % 100 == 0 or step+1 == config.steps:
                save_file(model.state_dict(), str(arm_root / "checkpoint.safetensors"),
                    metadata={"schema": "framegenx-cards-denoiser-v1", "architecture": "rgb_cpu_6channel",
                              "steps_completed": str(step+1), "quality_status": "unverified"})
        tracker.artifact(arm_root / "checkpoint.safetensors")
        tracker.finish("FINISHED")
    except BaseException as error:
        tracker.finish("KILLED" if isinstance(error, KeyboardInterrupt) else "FAILED")
        failure = {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "error_type": type(error).__name__, "quality_status": "unverified",
            "last_logged_step": loss_rows[-1]["step"] if loss_rows else 0,
            "resources": arm_budget.summary()}
        atomic_json(arm_root / "failure.json", failure)
        atomic_json(arm_root / "progress.json", {**failure, "metrics": loss_rows})
        raise
    result = {"status": "trained", "steps": config.steps, "rho": rho,
        "initial_sha256": initial_hash, "checkpoint_sha256": sha256(arm_root / "checkpoint.safetensors"),
        "metrics": loss_rows, "mlflow_run_id": tracker.run_id, "resources": arm_budget.summary(),
        "quality_status": "unverified"}
    atomic_json(arm_root / "training.json", result)
    atomic_json(arm_root / "progress.json", {"status": "trained", "step": config.steps,
                "target_steps": config.steps, "quality_status": "unverified", "metrics": loss_rows})
    return model, result


def sample_arm(config, model, root, arm, matcher, budget, training_thumbnail):
    diffusion = LearnedVarianceDiffusion(config.diffusion_steps)
    sample_root = root / arm / "samples"
    sample_root.mkdir()
    results = []
    for offset in range(0, len(config.sample_seeds), 4):
        budget.check()
        seeds = config.sample_seeds[offset:offset+4]
        pixels = diffusion.sample(model, seeds, check=budget.check)
        for seed, image in zip(seeds, pixels, strict=True):
            path = sample_root / f"seed_{seed}.png"
            Image.fromarray(image).save(path)
            observation = matcher.inspect(image, config.ncc_threshold, config.ambiguity_margin)
            small = np.asarray(Image.fromarray(image).resize((16, 16)), dtype=np.float32)/255
            distance = float(np.sqrt(((training_thumbnail-small.reshape(1, -1))**2).mean(1)).min())
            results.append({"seed": seed, "sha256": sha256(path), "status": observation["status"],
                "quadrants": observation["quadrants"], "nearest_train_16px_rmse": distance})
        atomic_json(root / arm / "sampling.json", {"status": "sampling", "samples": results,
                    "expected": len(config.sample_seeds), "quality_status": "unverified"})
        print(f"{arm}: sampled {len(results)}/{len(config.sample_seeds)}", flush=True)
    atomic_json(root / arm / "sampling.json", {"status": "completed", "samples": results,
                "expected": len(config.sample_seeds), "quality_status": "unverified"})
    return results


def wilson(successes, count):
    if count == 0:
        return None
    z = 1.95996398454
    center = (successes/count+z*z/(2*count))/(1+z*z/count)
    half = z*math.sqrt(successes/count*(1-successes/count)/count+z*z/(4*count*count))/(1+z*z/count)
    return [max(0, center-half), min(1, center+half)]


def paired_summary(arms, config):
    expected = config.sample_seeds
    indicators, metrics = {}, {}
    for arm in ("baseline", "candidate"):
        samples = arms[arm]
        if [item["seed"] for item in samples] != expected:
            raise ValueError("Missing, duplicate or out-of-order planned samples")
        counts = {status: sum(item["status"] == status for item in samples)
                  for status in ("valid", "invalid", "unknown", "error")}
        if sum(counts.values()) != len(expected):
            raise ValueError("Unknown sample outcome")
        indicators[arm] = np.asarray([item["status"] == "valid" for item in samples], dtype=float)
        metrics[arm] = {"n": len(expected), "counts": counts, "valid_rate": counts["valid"]/len(expected),
            "valid_rate_wilson_95": wilson(counts["valid"], len(expected)),
            "unique_image_hashes": len({item["sha256"] for item in samples}),
            "nearest_train_16px_rmse_mean": float(np.mean([item["nearest_train_16px_rmse"] for item in samples]))}
    delta = indicators["candidate"]-indicators["baseline"]
    rng = np.random.default_rng(config.bootstrap_seed)
    bootstrap = delta[rng.integers(0, len(delta), size=(config.bootstrap_samples, len(delta)))].mean(1)
    return {"scope": "color_aware_cards_template_validity_only", "arms": metrics,
        "paired_valid_rate_delta": float(delta.mean()),
        "paired_seed_bootstrap_95": np.quantile(bootstrap, [.025, .975]).tolist(),
        "bootstrap_limit": "Empirical interval conditional on one fitted pair; identical outcomes can give [0,0] without establishing population equality",
        "training_seed_replication": 1, "conclusion": "feasibility_only_no_qualified_reduction_claim",
        "full_contract_CSR": None, "quality_status": "unverified", "reduction_claim": False}


def run_experiment(config_file="configs/research_cards.json", *, reuse_completed=True):
    config = load_config(config_file)
    # Explicit CPU construction throughout. Do not probe or initialize CUDA here.
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    root = workspace_path(config.output_root)
    with experiment_lock(root), threadpool_limits(limits=config.threads):
        budget = Budget(config.max_total_seconds)
        manifest_file = prepare_data(config)
        manifest, matcher = load_data(manifest_file, config)
        signature = {"config": config.model_dump(by_alias=True), "data_sha256": sha256(manifest_file),
                     "source": source_identity(), "runtime": runtime_identity(),
                     "torch": torch.__version__, "numpy": np.__version__}
        digest = json_digest(signature)
        if reuse_completed:
            for report_file in sorted(root.glob("*/report.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    existing = json.loads(report_file.read_text(encoding="utf-8"))
                    if existing.get("signature") != digest or existing.get("execution_status") != "completed":
                        continue
                    inventory = json.loads((report_file.parent / "manifest.json").read_text(encoding="utf-8"))
                    if inventory["report_sha256"] != sha256(report_file):
                        raise ValueError("Completed report integrity changed")
                    for item in inventory["artifacts"]:
                        path = (report_file.parent / item["file"]).resolve()
                        if not path.is_relative_to(report_file.parent.resolve()) or sha256(path) != item["sha256"]:
                            raise ValueError("Completed research artifact changed")
                    print("Reusing intact completed feasibility run " + existing["id"], flush=True)
                    return report_file
                except (OSError, ValueError, KeyError):
                    raise ValueError("Existing research evidence is corrupt; inspect before rerunning")
        run_root = root / str(uuid.uuid4())
        run_root.mkdir()
        atomic_json(run_root / "protocol.json", signature)
        journal = {"id": run_root.name, "execution_status": "running", "quality_status": "unverified",
                   "stage": "snapshot_source", "signature": digest}
        atomic_json(run_root / "progress.json", journal)
        try:
            snapshot_source(run_root, signature["source"])
            journal["stage"] = "classical_ml"
            atomic_json(run_root / "progress.json", journal)
            classical = train_classical(config, manifest, matcher, run_root, budget)
            trained, arms = {}, {}
            for arm, rho in (("baseline", 0.), ("candidate", config.candidate_rho)):
                journal["stage"] = "train_"+arm
                atomic_json(run_root / "progress.json", journal)
                trained[arm], arms[arm] = train_arm(config, manifest["records"], run_root, arm, rho, budget)
            if arms["baseline"]["initial_sha256"] != arms["candidate"]["initial_sha256"]:
                raise ValueError("Paired model initializations differ")
            train_records = [r for r in manifest["records"] if r["split"] == "train" and r["source_status"] == "valid"]
            thumbnails = np.stack([np.asarray(Image.fromarray(pixels_for(r, config)).resize((16,16)), dtype=np.float32).reshape(-1)/255
                                   for r in train_records])
            samples = {}
            for arm in ("baseline", "candidate"):
                journal["stage"] = "sample_"+arm
                atomic_json(run_root / "progress.json", journal)
                samples[arm] = sample_arm(config, trained[arm], run_root, arm, matcher, budget, thumbnails)
            evaluation = paired_summary(samples, config)
            budget.check()
            report = {"schema": "framegenx-research-report-v1", "id": run_root.name,
                "created_at": datetime.now(timezone.utc).isoformat(), "signature": digest,
                "execution_status": "completed", "quality_status": "unverified",
                "study_scope": config.study_scope, "dataset": {"name": "VSM Cards", "revision": config.dataset_revision,
                    "license": "CC-BY-4.0", "manifest_sha256": sha256(manifest_file),
                    "records": len(manifest["records"]), "split_counts": manifest["verified_split_counts"]},
                "classical_ml": classical, "deep_learning": arms, "evaluation": evaluation,
                "ihalla": inspect_ihalla(ROOT / "reports/research_sources/ihalla_vsm_2026-10-16/ihalla"),
                "sdxl_integration": {"activation": "blocked_incompatible_checkpoint_and_unqualified_study",
                    "reason": "RGB six-channel Cards DDPM is not a four-latent-channel SDXL LoRA/variance-head adapter",
                    "automatic_loading": False}, "resources": budget.summary(),
                "limitations": ["single training seed and bounded archive subset", "synthetic corruption diagnostic is not calibrated image verification",
                    "template validity is not generic contract satisfaction", "paper-scale training not reproduced", "SDXL/QLoRA unchanged and inactive"]}
            atomic_json(run_root / "report.json", report)
            inventory = [{"file": path.relative_to(run_root).as_posix(), "sha256": sha256(path)}
                         for path in sorted(run_root.rglob("*")) if path.is_file() and path.name != "manifest.json"]
            atomic_json(run_root / "manifest.json", {"schema": "framegenx-research-manifest-v1",
                "report_sha256": sha256(run_root / "report.json"), "artifacts": inventory})
            journal.update(execution_status="completed", stage="finished")
            # Keep the final journal in the inventory consistent too.
            atomic_json(run_root / "progress.json", journal)
            for item in inventory:
                if item["file"] == "progress.json":
                    item["sha256"] = sha256(run_root / "progress.json")
            atomic_json(run_root / "manifest.json", {"schema": "framegenx-research-manifest-v1",
                "report_sha256": sha256(run_root / "report.json"), "artifacts": inventory})
            return run_root / "report.json"
        except BaseException as error:
            journal.update(execution_status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                           error_type=type(error).__name__, resources=budget.summary())
            atomic_json(run_root / "progress.json", journal)
            raise


def main():
    parser = argparse.ArgumentParser(description="CPU Cards ML/DL feasibility; never promotes SDXL")
    parser.add_argument("--config", default="configs/research_cards.json")
    parser.add_argument("--new-run", action="store_true", help="Deliberately rerun both arms instead of reusing intact completed evidence")
    args = parser.parse_args()
    print(run_experiment(args.config, reuse_completed=not args.new_run).relative_to(ROOT))


if __name__ == "__main__":
    main()
