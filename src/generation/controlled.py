"""Explicit 6 GB NF4 SDXL-base backend for contract-driven instance fusion."""
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import secrets
import time

import numpy as np
from PIL import Image
import torch
from safetensors.torch import save_file

from src.generation.conditioning import encode_conditioning
from src.generation.fusion import encode_bootstrap_backgrounds, sample_fused
from src.training.common import TrainingConfig, file_sha256
from src.training.model import load_trainable_unet, training_model_source


def validate_control_profile(model: str, generation: dict):
    if model != "stabilityai/stable-diffusion-xl-base-1.0":
        raise ValueError("Instance fusion is validated with SDXL base 1.0; use its explicit profile")
    integer_options = {"width": 512, "height": 512, "steps": 25, "bootstrap_steps": 12, "regional_steps": 12}
    if any(isinstance(generation.get(key, default), bool) or
           not isinstance(generation.get(key, default), int) for key, default in integer_options.items()):
        raise ValueError("Fusion sampler dimensions and step counts must be integers")
    seed = generation.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63):
        raise ValueError("Fusion seed must be an integer in [0, 2**63)")
    guidance = generation.get("cfg_scale", 7)
    if isinstance(guidance, bool) or not isinstance(guidance, (int, float)):
        raise ValueError("Fusion guidance must be numeric")
    if (generation.get("width", 512) != 512 or generation.get("height", 512) != 512 or
            not 25 <= generation.get("steps", 25) <= 50 or
            not 1 < generation.get("cfg_scale", 7) <= 10 or
            not 0 <= generation.get("bootstrap_steps", 12) <= generation.get("regional_steps", 12) <= generation.get("steps", 25)):
        raise ValueError("Invalid 512px SDXL-base fusion sampler settings")


def generate_controlled(model: str, prompt: str, generation: dict, plan: dict,
                        output_path: Path, adapter_hook=None) -> dict:
    from src.core.device import require_device_lease
    require_device_lease()
    validate_control_profile(model, generation)
    if not plan.get("enabled"):
        raise ValueError("Controlled generation requires an eligible instance plan")
    from src.utils import release_clip
    release_clip()
    seed = generation.get("seed")
    seed = secrets.randbits(31) if seed is None else int(seed)
    config = TrainingConfig(model=model, resolution=512, seed=seed)
    source = Path(training_model_source(model))
    steps, guidance = generation.get("steps", 25), generation.get("cfg_scale", 7)
    started = time.monotonic()
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    pipe = unet = vae = None
    try:
        encoded = encode_conditioning(source, [prompt, plan["background_prompt"], *[r["prompt"] for r in plan["regions"]]])
        backgrounds = encode_bootstrap_backgrounds(source, 512)
        unet, dtype = load_trainable_unet(config)
        unet.eval().requires_grad_(False)
        from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler, AutoencoderKL
        pipe = StableDiffusionXLPipeline(vae=None, text_encoder=None, text_encoder_2=None,
            tokenizer=None, tokenizer_2=None, unet=unet,
            scheduler=EulerDiscreteScheduler.from_pretrained(source, subfolder="scheduler", local_files_only=True),
            force_zeros_for_empty_prompt=True, add_watermarker=False)
        pipe.set_progress_bar_config(disable=True)
        if adapter_hook is not None:
            adapter_hook(pipe)
        else:
            pipe.disable_lora()
        latent = sample_fused(pipe, encoded[1:], seed, 512, steps, guidance, dtype, plan,
            backgrounds, generation.get("bootstrap_steps", 12), global_condition=encoded[0],
            regional_steps=generation.get("regional_steps", 12))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        latent_file = output_path.with_suffix(".safetensors")
        save_file({"latent": latent}, str(latent_file))
        del pipe, unet, encoded
        pipe = unet = None
        gc.collect(); torch.cuda.empty_cache()
        vae = AutoencoderKL.from_pretrained(source, subfolder="vae", torch_dtype=torch.float32,
            variant="fp16", local_files_only=True).eval().to("cuda")
        vae.enable_slicing()
        with torch.inference_mode():
            decoded = vae.decode(latent.to("cuda") / vae.config.scaling_factor).sample
            if not torch.isfinite(decoded).all():
                raise RuntimeError("Nonfinite controlled image")
            pixels = (decoded / 2 + .5).clamp(0, 1).cpu().permute(0, 2, 3, 1)[0].numpy()
        if float(pixels.std()) < .01:
            raise RuntimeError("Blank controlled image")
        Image.fromarray(np.rint(pixels*255).astype(np.uint8)).save(output_path)
        attempt = output_path.stem.split("_")[-1]
        Image.open(output_path).save(output_path.parent / f"step_{attempt}_{steps-1}.png")
        reserved = torch.cuda.max_memory_reserved()/1e9
        if reserved > config.max_vram_gb:
            raise RuntimeError("Controlled backend exceeded the 5.5 GB VRAM budget")
        return {"prompt": prompt, "width": 512, "height": 512, "steps": steps, "cfg_scale": guidance,
            "seed": seed, "base_model": model, "resolved_base_model": str(source),
            "base_unet_sha256": file_sha256(source / "unet/diffusion_pytorch_model.fp16.safetensors"),
            "generation_backend": "nf4_sdxl_instance_fusion", "regional_plan": plan,
            "sampler": "EulerDiscreteScheduler", "negative_conditioning": "zeros",
            "compute_dtype": str(dtype), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "regions_per_batch": 2,
            "packages": {name: version(name) for name in
                ("torch", "diffusers", "transformers", "peft", "bitsandbytes", "safetensors")},
            "code_sha256": {str(p.relative_to(Path(__file__).resolve().parents[2])).replace("\\", "/"): file_sha256(p)
                for p in (Path(__file__).resolve(), Path(__file__).with_name("conditioning.py"),
                    Path(__file__).with_name("fusion.py"), Path(__file__).with_name("regions.py"),
                    Path(__file__).resolve().parents[1] / "training/model.py",
                    Path(__file__).resolve().parents[1] / "training/cache.py",
                    Path(__file__).resolve().parents[1] / "training/common.py")},
            "controller_schedule": {"bootstrap_steps": generation.get("bootstrap_steps", 12),
                "regional_steps": generation.get("regional_steps", 12),
                "global_refinement_steps": steps-generation.get("regional_steps", 12)},
            "regional_plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
            "latent_sha256": file_sha256(latent_file), "image_sha256": file_sha256(output_path),
            "peak_reserved_gb": reserved, "peak_allocated_gb": torch.cuda.max_memory_allocated()/1e9,
            "elapsed_seconds": time.monotonic()-started, "preview_source": "final_pipeline_decode",
            "control_image_mode": "none", "quality_status": "unverified",
            "created_at": datetime.now(timezone.utc).isoformat()}
    finally:
        pipe = unet = vae = None
        gc.collect(); torch.cuda.empty_cache()
