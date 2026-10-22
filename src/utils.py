import gc
import json
import torch
from pathlib import Path


def free_memory():
    """Aggressively free system RAM and GPU VRAM."""
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Shared CLIP singleton (CPU-resident).
#
# Retrieval (query embedding) and verification (CLIP score) both need the
# same CLIP model. Loading it from disk twice per run cost ~60-90s each
# through the WSL2 bind mount (measured 2026-08-24). One process-lifetime
# CPU instance (~600MB RAM) is cheap next to the 8-10GB pipe and keeps both
# consumers at ~0s load cost. VRAM discipline unchanged: consumers move it
# to GPU transiently and return it via _to_cpu().
# ---------------------------------------------------------------------------
_CLIP_CACHE: dict = {}


def get_clip(model_id: str = "openai/clip-vit-base-patch32"):
    """Process-lifetime CLIP model+processor singleton (CPU residence)."""
    if model_id not in _CLIP_CACHE:
        from transformers import CLIPModel, CLIPProcessor
        _CLIP_CACHE[model_id] = (
            CLIPModel.from_pretrained(model_id),
            CLIPProcessor.from_pretrained(model_id),
        )
    return _CLIP_CACHE[model_id]


def release_clip():
    """Evict the CLIP singleton (~0.6-1.2GB RAM). Called before SDXL pipe
    construction: under model-offload the pipe needs every GB of VM RAM,
    and CLIP reloads in seconds from the OS page cache when retrieval or
    verification next needs it. RAM budgeting is life or death on a 16GB
    host with a 10GB VM (host free RAM hit 0.2GB and nearly tore the VM
    down again, 2026-08-24)."""
    if _CLIP_CACHE:
        _CLIP_CACHE.clear()
        gc.collect()


def check_vram():
    """Print current VRAM usage."""
    from src.core.device import require_device_lease
    require_device_lease()
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"GPU: {device_name}")
        print(f"Total VRAM: {total_memory:.2f} GB")
        print(f"Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")


def cast_inputs(inputs, device, dtype):
    """Move processor outputs to device, casting FLOAT tensors to the model
    dtype (fp16 models + fp32 processor pixels = 'Input type (float) and
    bias type (Half)' crash). Integer tensors (input_ids) keep their dtype."""
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            out[k] = v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
        else:
            out[k] = v
    return out


def clip_text_embed(model, inputs):
    """Version-proof CLIP text embedding: transformers 4.x returned a tensor
    from get_text_features; 5.x returns an output object (observed:
    BaseModelOutputWithPooling — projection NOT applied). Handle all shapes."""
    out = model.get_text_features(**inputs)
    if torch.is_tensor(out):
        return out
    embeds = getattr(out, "text_embeds", None)
    if embeds is not None:
        return embeds
    # transformers 5.x: pooler_output IS the projected 512-d embedding
    # (verified empirically — re-projecting it crashes with a shape error)
    return out.pooler_output


def clip_image_embed(model, pixel_values):
    """Version-proof CLIP image embedding (see clip_text_embed)."""
    out = model.get_image_features(pixel_values=pixel_values)
    if torch.is_tensor(out):
        return out
    embeds = getattr(out, "image_embeds", None)
    if embeds is not None:
        return embeds
    # transformers 5.x: pooler_output IS the projected 512-d embedding
    return out.pooler_output


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
