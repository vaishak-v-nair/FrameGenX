"""Post hoc validation diagnostics; never fits, repairs or qualifies a model."""
import argparse
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import numpy as np
import torch
from safetensors.torch import load_file
from threadpoolctl import threadpool_limits
from src.research.artifacts import atomic_json, sha256, workspace_path
from src.research.cards import load_data
from src.research.config import CardsConfig
from src.research.diffusion import CardsDenoiser, LearnedVarianceDiffusion
from src.research.experiment import pixels_for
from src.research.reports import read_json

TIMESTEPS = [0, 10, 25, 50, 75, 99]
NOISE_SEED = 271828


@torch.no_grad()
def diagnose(run_id, output):
    if str(uuid.UUID(run_id)) != run_id:
        raise ValueError("Use a canonical research UUID")
    folder = ROOT / "runs/research_cards" / run_id
    report = read_json(folder / "report.json")
    protocol = read_json(folder / "protocol.json")
    if report["execution_status"] != "completed":
        raise ValueError("Finish the experiment before exploring quality")
    config = CardsConfig.model_validate(protocol["config"])
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(config.threads):
        candidates = workspace_path(config.data_root).glob("prepared_*.json")
        manifest_file = next(path for path in candidates if sha256(path) == report["dataset"]["manifest_sha256"])
        manifest, _ = load_data(manifest_file, config)
        selected = [row for row in manifest["records"] if row["split"] == "validation" and row["source_status"] == "valid"][:8]
        if len(selected) != 8:
            raise ValueError("Exploration requires eight validation grids")
        clean = torch.from_numpy(np.stack([pixels_for(row, config) for row in selected])).permute(0, 3, 1, 2).float()/127.5-1
        diffusion = LearnedVarianceDiffusion(config.diffusion_steps)
        arms = {}
        for arm in ("baseline", "candidate"):
            checkpoint = folder / arm / "checkpoint.safetensors"
            if sha256(checkpoint) != report["deep_learning"][arm]["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed before exploratory diagnostics")
            model = CardsDenoiser(config.width)
            model.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True)
            model.eval()
            rows = []
            for index in TIMESTEPS:
                timestep = torch.full((len(clean),), index, dtype=torch.long)
                generator = torch.Generator(device="cpu").manual_seed(NOISE_SEED+index)
                noise = torch.randn(clean.shape, generator=generator)
                noisy = diffusion.noisy(clean, timestep, noise)
                output_tensor = model(noisy, timestep)
                epsilon = output_tensor[:, :3]
                bar = diffusion.take(diffusion.bar, timestep, clean)
                predicted_x0 = (noisy-(1-bar).sqrt()*epsilon)/bar.sqrt()
                _, log_variance = diffusion.distribution(output_tensor, noisy, timestep)
                rows.append({"timestep": index, "alpha_bar": float(diffusion.bar[index]),
                    "epsilon_mse": float((epsilon-noise).square().mean()),
                    "identity_epsilon_mse": float((noisy-noise).square().mean()),
                    "unclipped_x0_rmse": float((predicted_x0-clean).square().mean().sqrt()),
                    "clipped_x0_rmse": float((predicted_x0.clamp(-1,1)-clean).square().mean().sqrt()),
                    "predicted_x0_outside_unit_range_fraction": float((predicted_x0.abs()>1).float().mean()),
                    "mean_learned_variance": float(log_variance.exp().mean())})
            arms[arm] = rows
        result = {"schema": "framegenx-cards-exploratory-denoising-v1", "run_id": run_id,
            "study_type": "post_hoc_exploratory_validation_diagnostic", "quality_status": "unverified",
            "gate_effect": "none", "training_or_sampler_changes": "none",
            "selection": "first_eight_verified_validation_grids_in_frozen_manifest", "noise_seed": NOISE_SEED,
            "inputs": [{"file": row["file"], "sha256": row["sha256"]} for row in selected], "arms": arms,
            "cuda_initialized": torch.cuda.is_initialized(),
            "limits": ["Eight grids, no population confidence interval or causal attribution", "Identity epsilon is a terminal-noise diagnostic, not an image generator", "This does not qualify hallucination reduction"]}
        atomic_json(Path(output), result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", default="reports/research_execution_2026-10-16/exploratory_denoising.json")
    args = parser.parse_args()
    diagnose(args.run_id, args.output)
