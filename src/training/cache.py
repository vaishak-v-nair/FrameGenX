"""Cache frozen SDXL conditioning and full-image VAE latents, one model at a time."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from safetensors.torch import load_file, save_file

from src.training.common import atomic_json, file_sha256, load_training_config, project_path
from src.training.data import validate_records
from src.training.model import training_model_source
from src.core.device import exclusive_device, require_device_lease


PREPROCESSING = "rgb_full_image_letterbox_gray_no_crop_no_flip_latent_mode_v1"


def full_image_tensor(path: Path, resolution: int) -> tuple[torch.Tensor, tuple[int, int]]:
    with Image.open(path) as source:
        if source.getexif().get(274, 1) != 1:
            raise ValueError("Rotated EXIF image needs annotation orientation validation before training")
        image = source.convert("RGB")
        original = (image.height, image.width)
        contained = ImageOps.contain(image, (resolution, resolution), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (resolution, resolution), (127, 127, 127))
        canvas.paste(contained, ((resolution - contained.width) // 2,
                                 (resolution - contained.height) // 2))
        array = np.asarray(canvas, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0), original


def checked_tokens(tokenizer, caption: str, record_id: str) -> torch.Tensor:
    ids = tokenizer(caption, truncation=False, add_special_tokens=True)["input_ids"]
    if len(ids) > tokenizer.model_max_length:
        raise ValueError(f"Caption {record_id} has {len(ids)} tokens; silent CLIP truncation is prohibited")
    return tokenizer(caption, padding="max_length", max_length=tokenizer.model_max_length,
                     truncation=False, return_tensors="pt").input_ids


def prepare_cache(config, records: list[dict], audit: dict, check_budget=None) -> tuple[Path, dict]:
    require_device_lease()
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Latent caching requires CUDA for this bounded SDXL profile")
    source = Path(training_model_source(config.model))
    signature = {"dataset_sha256": audit["dataset_sha256"], "model": config.model,
                 "model_snapshot": source.name, "resolution": config.resolution,
                 "preprocessing": PREPROCESSING, "cache_code_sha256": file_sha256(Path(__file__)),
                 "scope": "positive_train_and_validation_only_no_test_images"}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
    root = project_path(config.dataset).parent / "cache" / key
    root.mkdir(parents=True, exist_ok=True)
    manifest_file = root / "manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        if manifest["signature"] != signature:
            raise ValueError("Cache signature changed")
        expected = {r["id"]: r for r in records if r["positive"] and r["split"] != "test"}
        if (len(manifest["items"]) != len(expected) or
                {i["id"] for i in manifest["items"]} != expected.keys()):
            raise ValueError("Cache contains missing, duplicate or nonpositive records")
        for item in manifest["items"]:
            if any(item[field] != expected[item["id"]][field]
                   for field in ("source", "split", "image_sha256")):
                raise ValueError("Cache record differs from validated dataset")
            path = (root / item["file"]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Cache tensor file is outside its root")
            if not path.is_file() or file_sha256(path) != item["sha256"]:
                raise ValueError("Cached training tensors are missing or changed")
        return root, manifest

    positives = [r for r in records if r["positive"] and r["split"] != "test"]
    # Negative statements never enter the generative training cache.
    embeddings = {}
    for slot, (folder, encoder_class) in enumerate([
            ("text_encoder", CLIPTextModel), ("text_encoder_2", CLIPTextModelWithProjection)]):
        tokenizer_folder = "tokenizer" if slot == 0 else "tokenizer_2"
        tokenizer = CLIPTokenizer.from_pretrained(source / tokenizer_folder, local_files_only=True)
        ids = {r["id"]: checked_tokens(tokenizer, r["caption"], r["id"]) for r in positives}
        encoder = encoder_class.from_pretrained(source / folder, torch_dtype=torch.float16,
                                                 variant="fp16", local_files_only=True).eval().to("cuda")
        encoder.requires_grad_(False)
        with torch.inference_mode():
            for record in positives:
                if check_budget is not None:
                    check_budget()
                output = encoder(ids[record["id"]].to("cuda"), output_hidden_states=True)
                entry = embeddings.setdefault(record["id"], {})
                entry[f"hidden_{slot}"] = output.hidden_states[-2].cpu().float()
                if slot == 1:
                    entry["pooled"] = output.text_embeds.cpu().float()
        del output, encoder, ids
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Cached {folder} for {len(positives)} positive captions", flush=True)

    # fp32 VAE avoids the known SDXL fp16 overflow problem.
    vae = AutoencoderKL.from_pretrained(source, subfolder="vae", torch_dtype=torch.float32,
                                       variant="fp16", local_files_only=True).eval().to("cuda")
    vae.requires_grad_(False)
    vae.enable_slicing()
    items = []
    dataset_root = project_path(config.dataset).parent
    with torch.inference_mode():
        for index, record in enumerate(positives):
            if check_budget is not None:
                check_budget()
            pixels, original = full_image_tensor(dataset_root / record["image"], config.resolution)
            latent = vae.encode(pixels.to("cuda")).latent_dist.mode() * vae.config.scaling_factor
            entry = embeddings[record["id"]]
            tensors = {"latent": latent.cpu().float(),
                       "prompt_embeds": torch.cat([entry["hidden_0"], entry["hidden_1"]], dim=-1),
                       "pooled_prompt_embeds": entry["pooled"],
                       "time_ids": torch.tensor([[*original, 0, 0, config.resolution, config.resolution]],
                                                dtype=torch.float32)}
            if not all(torch.isfinite(t).all() for t in tensors.values()):
                raise RuntimeError(f"Nonfinite frozen-model output for {record['id']}")
            filename = hashlib.sha256(record["id"].encode()).hexdigest()[:24] + ".safetensors"
            save_file(tensors, str(root / filename))
            items.append({"id": record["id"], "source": record["source"], "split": record["split"],
                          "image_sha256": record["image_sha256"], "file": filename,
                          "sha256": file_sha256(root / filename)})
            if (index + 1) % 12 == 0:
                print(f"Cached full-image latents: {index + 1}/{len(positives)}", flush=True)
    del vae, latent, pixels, embeddings
    gc.collect()
    torch.cuda.empty_cache()
    manifest = {"signature": signature, "items": items, "negatives_excluded": True,
                "test_images_excluded": True}
    atomic_json(manifest_file, manifest)
    return root, manifest


def load_cached_item(root: Path, item: dict) -> dict[str, torch.Tensor]:
    path = (root / item["file"]).resolve()
    if not path.is_relative_to(root.resolve()) or file_sha256(path) != item["sha256"]:
        raise ValueError("Training tensor cache changed or escaped its root")
    return load_file(str(path), device="cpu")


@exclusive_device
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training_pilot.yaml")
    args = parser.parse_args()
    config = load_training_config(args.config)
    records, audit = validate_records(project_path(config.dataset))
    root, manifest = prepare_cache(config, records, audit)
    print(json.dumps({"cache": str(root), "items": len(manifest["items"]), "signature": manifest["signature"]}), flush=True)


if __name__ == "__main__":
    main()
