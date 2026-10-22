"""Bounded local SDXL LoRA/QLoRA training with honest, durable run outcomes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback
import uuid

import numpy as np
import torch

from src.training.cache import load_cached_item, prepare_cache
from src.training.common import atomic_json, file_sha256, load_training_config, project_path
from src.training.data import validate_records
from src.training.export import complete_lora_metadata
from src.training.model import load_trainable_unet, training_model_source
from src.training.objective import denoising_loss, make_target
from src.training.tracking import ExperimentTracker
from src.core.device import exclusive_device
from src.training.budget import TrainingBudget


class ImageSampler:
    def __init__(self, size: int, seed: int):
        if size < 1:
            raise ValueError("Training split is empty")
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator)
        self.cursor = 0
        self.epoch = 0

    def next(self) -> int:
        if self.cursor == self.size:
            self.order = torch.randperm(self.size, generator=self.generator)
            self.cursor = 0
            self.epoch += 1
        index = int(self.order[self.cursor])
        self.cursor += 1
        return index

    def state_dict(self):
        return {"order": self.order, "cursor": self.cursor, "epoch": self.epoch,
                "generator_state": self.generator.get_state(), "size": self.size}

    def load_state_dict(self, state):
        if state["size"] != self.size:
            raise ValueError("Resume sampler size changed")
        self.order = state["order"].cpu()
        if sorted(self.order.tolist()) != list(range(self.size)):
            raise ValueError("Invalid checkpoint sampler permutation")
        if not 0 <= state["cursor"] <= self.size:
            raise ValueError("Invalid checkpoint sampler cursor")
        self.cursor, self.epoch = state["cursor"], state["epoch"]
        self.generator.set_state(state["generator_state"].cpu())


def source_files() -> list[tuple[Path, str]]:
    training = [(path, path.name) for path in sorted(Path(__file__).parent.glob("*.py"))]
    core = Path(__file__).resolve().parents[1] / "core"
    return training + [(core / name, "core/" + name) for name in ("device.py", "files.py")]


def code_fingerprint() -> str:
    digest = hashlib.sha256()
    for path, name in source_files():
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def save_source_snapshot(root: Path, dataset: Path, expected_fingerprint: str):
    """Preserve the executed training source and record manifest, without .env."""
    folder = root / "code_snapshot"
    folder.mkdir(parents=True, exist_ok=True)
    paths = source_files()
    for path, name in paths:
        destination = folder / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    shutil.copy2(dataset, root / "dataset_records.jsonl")
    digest = hashlib.sha256()
    for _, name in paths:
        digest.update(name.encode())
        digest.update((folder / name).read_bytes())
    if digest.hexdigest() != expected_fingerprint:
        raise RuntimeError("Training source changed while preserving its provenance snapshot")


def training_signature(config, dataset_sha: str, model_sha: str):
    profile = config.model_dump(exclude={"max_steps", "output_root", "tracking_db", "experiment",
                                         "max_runtime_seconds", "checkpoint_every", "validation_every"})
    return {"profile": profile, "dataset_sha256": dataset_sha, "model_sha256": model_sha,
            "training_code_sha256": code_fingerprint()}


def predict_loss(unet, scheduler, data, dtype, generator, snr_gamma: float):
    latent = data["latent"].to("cuda", dtype=dtype)
    noise = torch.randn(latent.shape, generator=generator, device="cuda", dtype=dtype)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latent.shape[0],),
                              generator=generator, device="cuda")
    noisy = scheduler.add_noise(latent, noise, timesteps)
    with torch.autocast("cuda", dtype=dtype):
        prediction = unet(noisy, timesteps, data["prompt_embeds"].to("cuda", dtype=dtype),
            added_cond_kwargs={"text_embeds": data["pooled_prompt_embeds"].to("cuda", dtype=dtype),
                               "time_ids": data["time_ids"].to("cuda", dtype=dtype)}).sample
    return denoising_loss(prediction, make_target(scheduler, latent, noise, timesteps), timesteps,
                           scheduler, snr_gamma)


@torch.no_grad()
def validation_losses(unet, scheduler, root, items, config, dtype):
    was_training = unet.training
    unet.eval()
    results = []
    try:
        for item in items:
            seed = (config.seed + int(hashlib.sha256(item["id"].encode()).hexdigest()[:8], 16)) % 2**63
            generator = torch.Generator(device="cuda").manual_seed(seed)
            weighted, raw = predict_loss(unet, scheduler, load_cached_item(root, item), dtype,
                                         generator, config.snr_gamma)
            if not torch.isfinite(weighted) or not torch.isfinite(raw):
                raise RuntimeError("Nonfinite held-out denoising loss")
            results.append({"id": item["id"], "source": item["source"], "weighted_loss": float(weighted),
                            "noise_mse": float(raw), "noise_seed": seed})
    finally:
        unet.train(was_training)
    return results


def choose_validation(items, limit: int):
    sources = sorted({i["source"] for i in items})
    groups = {s: sorted([i for i in items if i["source"] == s], key=lambda i: i["id"]) for s in sources}
    chosen = []
    for index in range(max(map(len, groups.values()), default=0)):
        for source in sources:
            if index < len(groups[source]):
                chosen.append(groups[source][index])
                if len(chosen) == limit:
                    return chosen
    return chosen


def save_checkpoint(unet, optimizer, sampler, noise_generator, root: Path, step: int, signature: dict,
                    trained_ids: Counter, run_id: str, scaler):
    from diffusers import StableDiffusionXLPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    folder = root / f"checkpoint-{step:06d}"
    folder.mkdir(parents=True, exist_ok=True)
    peft_state = {k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(unet).items()}
    if not peft_state or not all("lora_" in key for key in peft_state):
        raise RuntimeError("Export must contain only adapter parameters")
    metadata = complete_lora_metadata(unet.peft_config["default"])
    StableDiffusionXLPipeline.save_lora_weights(str(folder),
        unet_lora_layers=convert_state_dict_to_diffusers(peft_state),
        unet_lora_adapter_metadata=metadata, weight_name="adapter.safetensors")
    adapter = folder / "adapter.safetensors"
    state_path = folder / "training_state.pt"
    torch.save({"optimizer": optimizer.state_dict(), "sampler": sampler.state_dict(),
                "noise_generator": noise_generator.get_state(), "step": step,
                "adapter_state": peft_state, "trained_ids": dict(trained_ids),
                "grad_scaler": scaler.state_dict(), "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state()}, state_path)
    manifest = {"schema": "framegenx-adapter-v1", "run_id": run_id, "step": step,
                "base_model": signature["profile"]["model"], "signature": signature,
                "adapter_sha256": file_sha256(adapter), "state_sha256": file_sha256(state_path),
                "promotion_status": "unqualified", "quality_status": "unverified",
                "metric_scope": "training_objective_only"}
    atomic_json(adapter.with_suffix(".manifest.json"), manifest)
    return folder


@exclusive_device
def run_training(config, resume: str | None = None) -> dict:
    run_id = str(uuid.uuid4())
    root = project_path(config.output_root) / run_id
    root.mkdir(parents=True, exist_ok=True)
    result = {"run_id": run_id, "status": "failed", "quality_status": "unverified",
              "promotion_status": "unqualified", "metric_scope": "training_objective_only",
              "steps_completed": 0, "started_at": datetime.now(timezone.utc).isoformat(),
              "config": config.model_dump()}
    tracker = None
    started = time.monotonic()
    train_started = None
    unet = optimizer = params = initial_adapter = scaler = weighted = raw = grad_norm = None
    budget = None
    try:
        budget = TrainingBudget(config)
        budget.check()
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(config.cpu_threads)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(config.deterministic, warn_only=False)
        records, audit = validate_records(project_path(config.dataset))
        tracker = ExperimentTracker(config.tracking_db, config.experiment, run_id)
        result["mlflow_run_id"] = tracker.run_id
        result["dataset_audit"] = audit
        result["packages"] = {name: version(name) for name in
                ("torch", "numpy", "Pillow", "diffusers", "peft", "transformers", "bitsandbytes",
                 "safetensors", "psutil", "mlflow-skinny")}
        tracker.params({**config.model_dump(), "dataset_sha256": audit["dataset_sha256"],
                        "training_code_sha256": code_fingerprint(), **result["packages"]})
        atomic_json(root / "run_manifest.json", result)
        cache_root, manifest = prepare_cache(config, records, audit, check_budget=budget.check)
        training_items = [i for i in manifest["items"] if i["split"] == "train"]
        validation_items = choose_validation([i for i in manifest["items"] if i["split"] == "validation"],
                                              config.validation_limit)
        if not validation_items:
            raise ValueError("No independent validation image groups")
        source = Path(training_model_source(config.model))
        model_file = source / "unet/diffusion_pytorch_model.fp16.safetensors"
        model_hash = file_sha256(model_file)
        signature = training_signature(config, audit["dataset_sha256"], model_hash)
        result["signature"] = signature
        save_source_snapshot(root, project_path(config.dataset), signature["training_code_sha256"])
        result["model_snapshot"] = source.name
        result["cache_signature"] = manifest["signature"]
        result["train_images"] = len(training_items)
        result["validation_images_used"] = len(validation_items)
        result["test_images_used_for_training_or_selection"] = 0
        torch.cuda.reset_peak_memory_stats()
        unet, dtype = load_trainable_unet(config)
        budget.check()
        from diffusers import DDPMScheduler
        scheduler = DDPMScheduler.from_pretrained(source, subfolder="scheduler", local_files_only=True)
        params = [p for p in unet.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=config.learning_rate)
        scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
        sampler = ImageSampler(len(training_items), config.seed)
        noise_generator = torch.Generator(device="cuda").manual_seed(config.seed)
        baseline = validation_losses(unet, scheduler, cache_root, validation_items, config, dtype)
        atomic_json(root / "validation_baseline.json", {"items": baseline, "scope": "denoising_loss_only"})
        tracker.metrics({"validation_noise_mse": float(np.mean([i["noise_mse"] for i in baseline]))}, step=0)
        trained_ids = Counter()
        first_step = 0
        if resume:
            checkpoint = project_path(resume)
            saved = json.loads((checkpoint / "adapter.manifest.json").read_text())
            if saved["signature"] != signature:
                raise ValueError("Resume code, model, data or hyperparameters changed")
            state_path = checkpoint / "training_state.pt"
            if file_sha256(state_path) != saved["state_sha256"]:
                raise ValueError("Resume checkpoint state hash changed")
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            from peft.utils import set_peft_model_state_dict
            set_peft_model_state_dict(unet, state["adapter_state"])
            optimizer.load_state_dict(state["optimizer"])
            sampler.load_state_dict(state["sampler"])
            noise_generator.set_state(state["noise_generator"].cpu())
            scaler.load_state_dict(state["grad_scaler"])
            torch.set_rng_state(state["cpu_rng"].cpu())
            torch.cuda.set_rng_state(state["cuda_rng"].cpu())
            first_step = state["step"]
            if first_step >= config.max_steps:
                raise ValueError("Resume checkpoint already reached the requested step count")
            trained_ids.update(state["trained_ids"])
            result["resumed_from_run"] = saved["run_id"]
        result["trainable_parameters"] = sum(p.numel() for p in params)
        import bitsandbytes as bnb
        result["nf4_linear_modules"] = sum(isinstance(m, bnb.nn.Linear4bit) for m in unet.modules())
        result["dtype"] = str(dtype)
        train_started = time.monotonic()
        initial_adapter = [p.detach().cpu().clone() for p in params]
        for step in range(first_step + 1, config.max_steps + 1):
            budget.check()
            if time.monotonic() - train_started > config.max_runtime_seconds:
                raise TimeoutError("Training exceeded its configured wall-clock budget")
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(config.gradient_accumulation):
                item = training_items[sampler.next()]
                weighted, raw = predict_loss(unet, scheduler, load_cached_item(cache_root, item), dtype,
                                             noise_generator, config.snr_gamma)
                if not torch.isfinite(weighted) or not torch.isfinite(raw):
                    raise RuntimeError("Nonfinite training loss")
                scaler.scale(weighted / config.gradient_accumulation).backward()
                losses.append(float(raw.detach()))
                trained_ids[item["id"]] += 1
            scaler.unscale_(optimizer)
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in params):
                raise RuntimeError("Missing or nonfinite adapter gradient")
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            allocated = torch.cuda.max_memory_allocated() / 1e9
            reserved = torch.cuda.max_memory_reserved() / 1e9
            result.update(steps_completed=step, peak_allocated_gb=allocated, peak_reserved_gb=reserved,
                          training_seconds=time.monotonic() - train_started)
            if max(allocated, reserved) > config.max_vram_gb:
                raise RuntimeError("Measured training VRAM exceeded the configured budget")
            tracker.metrics({"train_noise_mse": float(np.mean(losses)), "gradient_norm": float(grad_norm),
                              "peak_allocated_gb": allocated, "peak_reserved_gb": reserved}, step)
            print(f"step {step}/{config.max_steps} mse={np.mean(losses):.6f} vram={reserved:.3f}GB "
                  f"elapsed={time.monotonic()-train_started:.1f}s", flush=True)
            if step % config.validation_every == 0 or step == config.max_steps:
                candidate = validation_losses(unet, scheduler, cache_root, validation_items, config, dtype)
                atomic_json(root / f"validation_{step:06d}.json", {"items": candidate, "scope": "denoising_loss_only"})
                tracker.metrics({"validation_noise_mse": float(np.mean([i["noise_mse"] for i in candidate]))}, step)
            if step % config.checkpoint_every == 0 or step == config.max_steps:
                folder = save_checkpoint(unet, optimizer, sampler, noise_generator, root, step, signature,
                                          trained_ids, run_id, scaler)
                result["latest_checkpoint"] = str(folder.relative_to(project_path(".")))
                atomic_json(root / "result.json", result)
        changed = sum(not torch.equal(old, p.detach().cpu()) for old, p in zip(initial_adapter, params))
        if not changed:
            raise RuntimeError("No adapter weights changed during training")
        result["changed_adapter_tensors"] = changed
        before = {i["id"]: i["noise_mse"] for i in baseline}
        after = {i["id"]: i["noise_mse"] for i in candidate}
        deltas = np.array([after[key] - before[key] for key in before])
        bootstrap = np.random.default_rng(config.seed).choice(deltas, size=(10000, len(deltas)), replace=True).mean(1)
        result["validation_comparison"] = {
            "paired_images": len(deltas), "baseline_mean_noise_mse": float(np.mean(list(before.values()))),
            "candidate_mean_noise_mse": float(np.mean(list(after.values()))),
            "mean_delta": float(deltas.mean()), "median_delta": float(np.median(deltas)),
            "bootstrap_mean_delta_95_ci": np.quantile(bootstrap, [.025, .975]).tolist(),
            "scope": "denoising_objective_only_not_CSR"}
        result["distinct_train_images_seen"] = len(trained_ids)
        result["train_presentations_by_source"] = dict(Counter(
            {source_name: sum(trained_ids[i["id"]] for i in training_items if i["source"] == source_name)
             for source_name in sorted({i["source"] for i in training_items})}))
        result["status"] = "trained_pending_evaluation"
        tracker.metrics({"validation_mean_delta_noise_mse": float(deltas.mean()),
                          "distinct_train_images_seen": len(trained_ids)}, config.max_steps)
        tracker.finish("FINISHED", "unverified")
        del unet, optimizer, params, initial_adapter
        gc.collect()
        torch.cuda.empty_cache()
    except KeyboardInterrupt:
        result["status"] = "interrupted"
        result["error"] = "Training interrupted; latest completed checkpoint remains available"
        if tracker:
            tracker.finish("KILLED")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        if tracker:
            tracker.finish("FAILED")
    finally:
        unet = optimizer = params = initial_adapter = scaler = weighted = raw = grad_norm = None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        result["elapsed_seconds"] = time.monotonic() - started
        result["host_resource_samples"] = budget.report() if budget is not None else None
        atomic_json(root / "result.json", result)
        if tracker:
            tracker.artifact(root / "result.json")
        print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training_pilot.yaml")
    parser.add_argument("--resume", help="A completed local checkpoint directory")
    args = parser.parse_args()
    result = run_training(load_training_config(args.config), args.resume)
    return 0 if result["status"] == "trained_pending_evaluation" else 1


if __name__ == "__main__":
    raise SystemExit(main())
