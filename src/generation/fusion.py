"""SDXL region-path fusion with early masked-background bootstrapping.

Inspired by MultiDiffusion (Bar-Tal et al., ICML 2023). Euler advances exactly
once per timestep after mask-weighted noise fusion; region batches never
advance shared scheduler state independently. This implementation is evaluated
separately from the original SD2/DDIM paper implementation.
Mask-centering during bootstrapping follows the stabilization idea in
SemanticDraw (Lee et al., CVPR 2025), implemented here for NF4 SDXL/Euler.
"""
from contextlib import nullcontext
import torch

from src.generation.regions import region_weights


def center_shifts(plan: dict, height: int, width: int) -> list[tuple[int, int]]:
    return [(0, 0)] + [(round(height/2-(r["box"][1]+r["box"][3])*height/2),
                       round(width/2-(r["box"][0]+r["box"][2])*width/2)) for r in plan["regions"]]


@torch.inference_mode()
def encode_bootstrap_backgrounds(source, resolution: int) -> torch.Tensor:
    from src.core.device import require_device_lease
    require_device_lease()
    from diffusers import AutoencoderKL
    import gc
    vae = AutoencoderKL.from_pretrained(source, subfolder="vae", torch_dtype=torch.float32,
                                       variant="fp16", local_files_only=True).eval().to("cuda")
    vae.enable_slicing()
    backgrounds = []
    for intensity in (.5, .85):
        pixels = torch.full((1, 3, resolution, resolution), intensity * 2 - 1, device="cuda")
        latent = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
        if not torch.isfinite(latent).all():
            raise RuntimeError("Nonfinite bootstrap background")
        backgrounds.append(latent.cpu().float())
    del vae, latent, pixels
    gc.collect(); torch.cuda.empty_cache()
    return torch.cat(backgrounds)


@torch.inference_mode()
def sample_fused(pipe, conditions: list[dict], seed: int, resolution: int, steps: int,
                 guidance: float, dtype, plan: dict | None = None,
                 backgrounds: torch.Tensor | None = None, bootstrap_steps: int = 0,
                 regions_per_batch: int = 2, global_condition: dict | None = None,
                 regional_steps: int | None = None) -> torch.Tensor:
    regional_steps = steps if regional_steps is None else regional_steps
    if not 1 <= regions_per_batch <= 2 or not 0 <= bootstrap_steps <= steps:
        raise ValueError("Invalid fusion work budget")
    if not 0 <= bootstrap_steps <= regional_steps <= steps or (regional_steps < steps and global_condition is None):
        raise ValueError("Global refinement requires global conditioning and a valid regional schedule")
    if plan is None and (len(conditions) != 1 or bootstrap_steps):
        raise ValueError("Uncontrolled parity mode requires one condition and no bootstrap")
    if plan is not None and len(conditions) != len(plan["regions"]) + 1:
        raise ValueError("Missing region/background conditioning")
    device = pipe.unet.device
    generator = torch.Generator(device=device).manual_seed(seed)
    scheduler = pipe.scheduler
    scheduler.set_timesteps(steps, device=device)
    latent = pipe.prepare_latents(1, pipe.unet.config.in_channels, resolution, resolution,
                                 dtype, device, generator, None)
    noise = latent / scheduler.init_noise_sigma
    if plan is None:
        weights = torch.ones(1, resolution//8, resolution//8, device=device, dtype=dtype)
    else:
        fractional = region_weights(plan, resolution//8, resolution//8, device, torch.float32)
        objects = (fractional[1:] > .5).to(dtype)
        if any(mask.sum() == 0 for mask in objects):
            raise ValueError("A region vanished at latent resolution")
        background = (objects.sum(0) == 0).to(dtype)
        weights = torch.cat([background[None], objects])
        weights = weights / weights.sum(0).clamp_min(1)
    if bootstrap_steps:
        if backgrounds is None or backgrounds.shape[1:] != latent.shape[1:]:
            raise ValueError("Bootstrap backgrounds missing or wrong shape")
        backgrounds = backgrounds.to(device, dtype=dtype)
    time_ids = torch.tensor([[resolution, resolution, 0, 0, resolution, resolution]], device=device, dtype=dtype)
    shifts = center_shifts(plan, resolution//8, resolution//8) if plan is not None else [(0, 0)]
    for index, timestep in enumerate(scheduler.timesteps):
        prediction = torch.zeros_like(latent)
        active_conditions = [global_condition] if index >= regional_steps else conditions
        active_weights = torch.ones_like(weights[:1]) if index >= regional_steps else weights
        for start in range(0, len(active_conditions), regions_per_batch):
            selected = active_conditions[start:start+regions_per_batch]
            views = latent.repeat(len(selected), 1, 1, 1)
            if index < bootstrap_steps:
                for offset in range(len(selected)):
                    region_index = start + offset
                    if region_index == 0:
                        continue
                    # Alternate fixed, encoded neutral backgrounds reproducibly.
                    clean = backgrounds[(seed + index + region_index) % len(backgrounds):][:1]
                    noised = scheduler.add_noise(clean, noise, timestep.reshape(1))
                    mask = (weights[region_index] > 0).to(dtype)[None, None]
                    views[offset:offset+1] = views[offset:offset+1] * mask + noised * (1-mask)
                    views[offset:offset+1] = torch.roll(views[offset:offset+1], shifts[region_index], dims=(-2, -1))
            embeddings = torch.cat([c["prompt"] for c in selected]).to(device, dtype=dtype)
            pooled = torch.cat([c["pooled"] for c in selected]).to(device, dtype=dtype)
            latent_input = torch.cat([views, views]) if guidance > 1 else views
            if guidance > 1:
                embeddings = torch.cat([torch.zeros_like(embeddings), embeddings])
                pooled = torch.cat([torch.zeros_like(pooled), pooled])
            latent_input = scheduler.scale_model_input(latent_input, timestep)
            with torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else nullcontext():
                output = pipe.unet(latent_input, timestep, encoder_hidden_states=embeddings,
                    added_cond_kwargs={"text_embeds": pooled,
                        "time_ids": time_ids.expand(len(latent_input), -1)}, return_dict=False)[0]
            if guidance > 1:
                negative, positive = output.chunk(2)
                output = negative + guidance * (positive-negative)
            if index < bootstrap_steps:
                for offset in range(len(selected)):
                    dy, dx = shifts[start+offset]
                    output[offset:offset+1] = torch.roll(output[offset:offset+1], (-dy, -dx), dims=(-2, -1))
            prediction += (output * active_weights[start:start+len(selected), None]).sum(0, keepdim=True)
        latent = scheduler.step(prediction, timestep, latent, return_dict=False)[0]
        if not torch.isfinite(latent).all():
            raise RuntimeError("Nonfinite fused latent")
        if device.type == "cuda" and torch.cuda.max_memory_reserved()/1e9 > 5.5:
            raise RuntimeError("Fusion exceeded the 5.5 GB VRAM budget")
    return latent.cpu().float()
