"""Validate durable paired-generation progress before reusing local outputs."""
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch

from src.training.common import file_sha256


def expected_images(protocol: dict) -> list[dict]:
    return [{"case_id": case["id"], "seed": seed, "arm": arm,
             "file": f"{case['id']}_{seed}_{arm}.png", "prompt": case["prompt"],
             "requirements": case["requirements"]}
            for case in protocol["cases"] for seed in protocol["seeds"]
            for arm in ("baseline", "candidate")]


def validate_progress(root: Path, saved: dict, signature: dict,
                      expected: list[dict], resolution: int) -> dict[str, dict]:
    """A partial journal is evidence of work only after identity and hash checks.

    Never reuse outputs across changed models, settings, code or packages.
    Missing image hashes mean decoding has not durably completed yet.
    """
    if saved.get("generation_signature") != signature:
        raise ValueError("Evaluation resume model, adapter, protocol, code or settings changed")
    if (saved.get("quality_status") != "unverified" or
            saved.get("promotion_status") != "unqualified" or
            saved.get("full_contract_CSR") is not None):
        raise ValueError("Generation progress cannot contain qualification or a CSR claim")
    plan = {item["file"]: item for item in expected}
    found = {}
    for item in saved["images"]:
        name = item["file"]
        if name in found or name not in plan or any(item.get(k) != v for k, v in plan[name].items()):
            raise ValueError("Evaluation progress has duplicate or unexpected image identities")
        latent_file = root / Path(name).with_suffix(".safetensors")
        if not latent_file.is_file() or file_sha256(latent_file) != item.get("latent_sha256"):
            raise ValueError("Evaluation resume latent is missing or changed")
        tensors = load_file(str(latent_file), device="cpu")
        latent = tensors.get("latent")
        if (set(tensors) != {"latent"} or latent.shape != (1, 4, resolution // 8, resolution // 8)
                or not torch.isfinite(latent).all()):
            raise ValueError("Evaluation resume latent has invalid shape or values")
        if item.get("image_sha256"):
            image_file = root / name
            if not image_file.is_file() or file_sha256(image_file) != item["image_sha256"]:
                raise ValueError("Evaluation resume image is missing or changed")
            with Image.open(image_file) as image:
                if image.mode != "RGB" or image.size != (resolution, resolution):
                    raise ValueError("Evaluation resume image has invalid dimensions or mode")
                if float(np.asarray(image).astype(np.float32).std()) / 255 < .01:
                    raise ValueError("Evaluation resume image is blank")
        found[name] = item
    return found
