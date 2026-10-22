"""Compile counted instances into separate regional prompts and route attention.

Each instance owns its noun and attributes. Repeated nouns do not share one
union of token boxes. This is an inference controller, not a trained adapter
or a guarantee that the generated pixels satisfy the contract.
"""
from contextlib import contextmanager
from copy import deepcopy
import math
import re

import torch


SURFACES = {"table", "road", "floor", "ground", "wall", "countertop", "background", "sky"}


def compile_regions(contract: dict, resolution: int = 512) -> dict:
    """Conservative disjoint layout for 2–6 objects with optional left/right.

    Contact, depth, unknown-count foregrounds, and ambiguous same-class
    relations are not approximated by disjoint boxes.
    """
    entities = deepcopy(contract.get("entities") or [])
    foreground = [e for e in entities if str(e.get("class_name", "")).lower() not in SURFACES]
    if not 2 <= len(foreground) <= 6 or any(e.get("uncounted") for e in foreground):
        return {"enabled": False, "reason": "requires 2-6 explicitly counted foreground instances"}
    names = [str(e.get("class_name", "")).strip().lower() for e in foreground]
    if not all(names):
        return {"enabled": False, "reason": "foreground class missing"}
    if any(e.get("bounding_box") is not None or e.get("relative_position") is not None for e in foreground):
        return {"enabled": False, "reason": "explicit foreground placement requires a different layout controller"}
    ids = [e["id"] for e in foreground if e.get("id")]
    if len(ids) != len(set(ids)):
        return {"enabled": False, "reason": "foreground instance IDs must be unique"}
    edges = []
    for relation in contract.get("spatial_relationships") or []:
        subject, obj = relation.get("subject", "").lower(), relation.get("object", "").lower()
        if subject not in names or obj not in names:
            continue  # background/support relationships stay in scene context
        predicate = relation.get("predicate", "").lower()
        if predicate not in {"left_of", "right_of"} or names.count(subject) != 1 or names.count(obj) != 1:
            return {"enabled": False, "reason": "foreground relation requires a different layout controller"}
        before, after = names.index(subject), names.index(obj)
        edges.append((before, after) if predicate == "left_of" else (after, before))
    # Stable topological order preserves input order unless a relation requires otherwise.
    remaining, ordered = list(range(len(foreground))), []
    while remaining:
        eligible = [i for i in remaining if not any(b == i and a in remaining for a, b in edges)]
        if not eligible:
            return {"enabled": False, "reason": "contradictory left/right relations"}
        chosen = eligible[0]
        ordered.append(chosen)
        remaining.remove(chosen)
    foreground = [foreground[i] for i in ordered]
    surface_descriptions = [" ".join([*map(str, e.get("attributes") or []), str(e["class_name"])])
                            for e in entities if str(e.get("class_name", "")).lower() in SURFACES]
    background = ", ".join(dict.fromkeys(surface_descriptions)) or "plain neutral background"
    style = re.search(r"\b(photorealistic|photograph|watercolor|illustration|drawing|painting|pixel art|cartoon)\b",
                      str(contract.get("prompt", "")), re.I)
    prefix = (style.group(0) + " of ") if style else ""
    # Small, explicit compositions use a row. Avoid a large grid of tiny objects.
    columns = len(foreground) if len(foreground) <= 3 or "row" in str(contract.get("prompt", "")).lower() else 3
    rows = math.ceil(len(foreground) / columns)
    regions = []
    for index, entity in enumerate(foreground):
        row, col = divmod(index, columns)
        x1, x2 = (col + .12) / columns, (col + .88) / columns
        y1, y2 = (.20, .88) if rows == 1 else ((row + .10) / rows, (row + .90) / rows)
        rounded = str(entity["class_name"]).lower() in {"apple", "orange", "ball", "sphere"}
        if rounded and rows == 1:
            y1, y2 = .45, .78
        if str(entity["class_name"]).lower() == "bottle" and rows == 1:
            center = (x1+x2)/2
            x1, x2 = center-.09, center+.09
        description = " ".join([*map(str, entity.get("attributes") or []), entity["class_name"]])
        regions.append({"entity_id": entity.get("id", f"instance-{index}"),
            "class_name": entity["class_name"], "attributes": entity.get("attributes") or [],
            "prompt": f"{prefix}one single {description}, centered whole object.",
            "shape": "box", "box": [x1, y1, x2, y2]})
    return {"enabled": True, "schema": "framegenx-instance-regions-v1", "resolution": resolution,
            "background_prompt": f"{prefix}{background}.", "regions": regions,
            "method": "spatially_routed_cross_attention", "quality_status": "unverified"}


def region_weights(plan: dict, height: int, width: int, device, dtype) -> torch.Tensor:
    masks = []
    for region in plan["regions"]:
        box = region["box"]
        if (len(box) != 4 or not all(math.isfinite(v) for v in box) or
                not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1):
            raise ValueError("Invalid normalized region box")
        # Fractional pixel coverage keeps small regions at coarse attention scales.
        x = torch.arange(width, device=device, dtype=torch.float32)
        y = torch.arange(height, device=device, dtype=torch.float32)
        wx = (torch.minimum(x + 1, torch.tensor(box[2] * width, device=device)) -
              torch.maximum(x, torch.tensor(box[0] * width, device=device))).clamp(0, 1)
        wy = (torch.minimum(y + 1, torch.tensor(box[3] * height, device=device)) -
              torch.maximum(y, torch.tensor(box[1] * height, device=device))).clamp(0, 1)
        mask = wy[:, None] * wx[None, :]
        if region.get("shape", "box") == "ellipse":
            cx, cy = (box[0]+box[2])*width/2, (box[1]+box[3])*height/2
            rx, ry = (box[2]-box[0])*width/2, (box[3]-box[1])*height/2
            ellipse = ((x[None, :]+.5-cx)/rx).square() + ((y[:, None]+.5-cy)/ry).square() <= 1
            mask = mask * ellipse
        masks.append(mask)
    objects = torch.stack(masks)
    background = (1 - objects.sum(0)).clamp_min(0)
    weights = torch.cat([background[None], objects])
    return (weights / weights.sum(0).clamp_min(1e-8)).to(dtype=dtype)


class InstanceAttentionProcessor:
    def __init__(self, original, plan: dict, embeddings: list[torch.Tensor], cfg: bool):
        if len(embeddings) != len(plan["regions"]) + 1:
            raise ValueError("Every region and the background need separate conditioning")
        self.original, self.plan, self.embeddings, self.cfg = original, plan, embeddings, cfg
        self._weights = {}

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        if encoder_hidden_states is None:
            return self.original(attn, hidden_states, None, attention_mask, temb, *args, **kwargs)
        batch = hidden_states.shape[0]
        if batch != (2 if self.cfg else 1):
            raise ValueError("Instance conditioning supports one image per call")
        if hidden_states.ndim == 4:
            height, width = hidden_states.shape[-2:]
        else:
            height = width = math.isqrt(hidden_states.shape[1])
            if height * width != hidden_states.shape[1]:
                raise ValueError("Instance controller currently requires square attention grids")
        key = (height, width, hidden_states.device, hidden_states.dtype)
        if key not in self._weights:
            self._weights[key] = region_weights(self.plan, height, width, hidden_states.device, hidden_states.dtype)
        weights = self._weights[key]
        total, unconditional = None, None
        for index, embedding in enumerate(self.embeddings):
            positive = embedding.to(device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
            if positive.shape != encoder_hidden_states[-1:].shape:
                raise ValueError("Regional conditioning shape differs from pipeline conditioning")
            context = torch.cat([encoder_hidden_states[:1], positive]) if self.cfg else positive
            value = self.original(attn, hidden_states, context, attention_mask, temb, *args, **kwargs)
            if self.cfg:
                unconditional = value[:1] if unconditional is None else unconditional
                value = value[1:]
            weight = weights[index]
            weight = weight[None, None] if value.ndim == 4 else weight.reshape(1, height * width, 1)
            weighted = value * weight
            total = weighted if total is None else total + weighted
        return torch.cat([unconditional, total]) if self.cfg else total


@contextmanager
def instance_attention(unet, plan: dict, embeddings: list[torch.Tensor], cfg: bool):
    """Install only cross-attention routes and restore exact processors on exit."""
    original = dict(unet.attn_processors)
    replacement = {name: InstanceAttentionProcessor(proc, plan, embeddings, cfg)
                   if name.endswith("attn2.processor") else proc for name, proc in original.items()}
    if not any(isinstance(p, InstanceAttentionProcessor) for p in replacement.values()):
        raise ValueError("UNet has no supported cross-attention processors")
    try:
        unet.set_attn_processor(replacement.copy())
        yield
    finally:
        unet.set_attn_processor(original.copy())
