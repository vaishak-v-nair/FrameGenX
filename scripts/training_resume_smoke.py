"""Real CUDA check: resumed optimizer state must match uninterrupted training.

Small technical runs exercise the current strict deterministic profile and
complete export metadata. Their training losses are not product quality gains.
"""
import gc
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.training.common import TrainingConfig, atomic_json, project_path
from src.training.train import run_training


from src.core.device import exclusive_device


@exclusive_device
def main():
    started = time.monotonic()
    config = TrainingConfig(max_steps=2, checkpoint_every=1, validation_every=2, validation_limit=3,
        output_root="runs/training/resume_smoke", experiment="framegenx-training-control-smoke",
        max_runtime_seconds=300, deterministic=True)
    result = {"status": "failed", "scope": "real_CUDA_optimizer_resume_only_not_quality_validation", "runs": []}
    try:
        partial = run_training(config)
        result["runs"].append(partial["run_id"])
        if partial["status"] != "trained_pending_evaluation":
            raise RuntimeError("Initial technical training run failed")
        gc.collect()
        torch.cuda.empty_cache()
        extended = config.model_copy(update={"max_steps": 3})
        resumed = run_training(extended, partial["latest_checkpoint"])
        result["runs"].append(resumed["run_id"])
        if resumed["status"] != "trained_pending_evaluation" or resumed.get("resumed_from_run") != partial["run_id"]:
            raise RuntimeError("Optimizer checkpoint continuation failed")
        gc.collect()
        torch.cuda.empty_cache()
        uninterrupted = run_training(extended)
        result["runs"].append(uninterrupted["run_id"])
        if uninterrupted["status"] != "trained_pending_evaluation":
            raise RuntimeError("Uninterrupted comparison run failed")
        state_a = torch.load(project_path(resumed["latest_checkpoint"]) / "training_state.pt",
                             map_location="cpu", weights_only=True)["adapter_state"]
        state_b = torch.load(project_path(uninterrupted["latest_checkpoint"]) / "training_state.pt",
                             map_location="cpu", weights_only=True)["adapter_state"]
        if state_a.keys() != state_b.keys():
            raise RuntimeError("Resumed adapter tensor keys differ")
        different = [k for k in state_a if not torch.equal(state_a[k], state_b[k])]
        result.update(compared_adapter_tensors=len(state_a), differing_tensors=len(different),
            max_abs_difference=max(float((state_a[k]-state_b[k]).abs().max()) for k in state_a),
            peak_reserved_gb=max(r["peak_reserved_gb"] for r in (partial, resumed, uninterrupted)))
        if different:
            raise RuntimeError("Resumed weights do not exactly match uninterrupted weights")
        import psutil
        memory = psutil.Process().memory_info()
        result["process_peak_working_set_gb"] = memory.peak_wset / 1e9
        result["status"] = "passed"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        atomic_json(PROJECT_ROOT / "reports/ml_resume_smoke_2026-10-14.json", result)
        print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
