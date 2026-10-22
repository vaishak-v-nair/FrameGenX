"""Exercise the real diffusion adapter backward pass before downloading images."""
import argparse
import time
import traceback

import torch

from src.training.common import atomic_json, load_training_config, project_path
from src.training.model import load_trainable_unet
from src.core.device import exclusive_device


@exclusive_device
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_pilot.yaml")
    parser.add_argument("--output", default="reports/ml_preflight_2026-10-14.json")
    args = parser.parse_args()
    config = load_training_config(args.config)
    started = time.monotonic()
    result = {"status": "failed", "method": config.method, "model": config.model,
              "resolution": config.resolution, "deterministic": config.deterministic}
    try:
        torch.manual_seed(config.seed)
        torch.cuda.reset_peak_memory_stats()
        unet, dtype = load_trainable_unet(config)
        print("Loaded quantized UNet; starting real 512px backward probe", flush=True)
        params = [p for p in unet.parameters() if p.requires_grad]
        result["trainable_parameters"] = sum(p.numel() for p in params)
        import bitsandbytes as bnb
        result["nf4_linear_modules"] = sum(isinstance(m, bnb.nn.Linear4bit) for m in unet.modules())
        before = [p.detach().clone() for p in params]
        optimizer = torch.optim.AdamW(params, lr=config.learning_rate)
        latent = torch.randn(1, 4, config.resolution // 8, config.resolution // 8, device="cuda", dtype=dtype)
        embedding = torch.randn(1, 77, 2048, device="cuda", dtype=dtype)
        conditioning = {"text_embeds": torch.randn(1, 1280, device="cuda", dtype=dtype),
                        "time_ids": torch.tensor([[512, 512, 0, 0, 512, 512]], device="cuda", dtype=dtype)}
        with torch.autocast("cuda", dtype=dtype):
            prediction = unet(latent, torch.tensor([999], device="cuda"), embedding,
                              added_cond_kwargs=conditioning).sample
            loss = prediction.float().square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss in preflight")
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in params):
            raise RuntimeError("Missing or nonfinite adapter gradient")
        optimizer.step()
        changed = sum(not torch.equal(old, p.detach()) for old, p in zip(before, params))
        if changed == 0:
            raise RuntimeError("Optimizer did not change any adapter weights")
        torch.cuda.synchronize()
        allocated = torch.cuda.max_memory_allocated() / 1e9
        reserved = torch.cuda.max_memory_reserved() / 1e9
        result.update(loss=float(loss.detach()), changed_adapter_tensors=changed,
                      peak_allocated_gb=allocated, peak_reserved_gb=reserved,
                      dtype=str(dtype))
        if max(allocated, reserved) > config.max_vram_gb:
            raise RuntimeError("Measured GPU allocation exceeded the configured budget")
        result["status"] = "passed"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        atomic_json(project_path(args.output), result)
        print(result, flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
