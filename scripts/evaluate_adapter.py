"""Generate sealed paired pilot cases with the actual exported Diffusers adapter.

The generator uses NF4 SDXL with identical seeds and settings for both arms.
This is not an evaluation of the production fp16 Turbo backend. Images and
latents are saved; no uncalibrated score qualifies or promotes an adapter.
"""
import argparse
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file, save_file

from src.training.cache import checked_tokens
from src.training.common import atomic_json, file_sha256, load_training_config, project_path
from src.training.model import load_trainable_unet, training_model_source
from src.training.registry import adapter_manifest
from src.training.evaluation import expected_images, validate_progress
from src.generation.lora import LoraManager


def encode_prompts(source: Path, prompts: list[str]) -> list[dict]:
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    result = [{} for _ in prompts]
    for index, encoder_class in enumerate((CLIPTextModel, CLIPTextModelWithProjection)):
        folder = "text_encoder" if index == 0 else "text_encoder_2"
        tokenizer_folder = "tokenizer" if index == 0 else "tokenizer_2"
        tokenizer = CLIPTokenizer.from_pretrained(source / tokenizer_folder, local_files_only=True)
        encoder = encoder_class.from_pretrained(source / folder, torch_dtype=torch.float16,
                                                 variant="fp16", local_files_only=True).eval().to("cuda")
        with torch.inference_mode():
            for i, prompt in enumerate(prompts):
                ids = checked_tokens(tokenizer, prompt, f"evaluation-prompt-{i}")
                output = encoder(ids.to("cuda"), output_hidden_states=True)
                result[i][f"hidden_{index}"] = output.hidden_states[-2].cpu().float()
                if index == 1:
                    result[i]["pooled"] = output.text_embeds.cpu().float()
        del encoder, output
        gc.collect()
        torch.cuda.empty_cache()
    return [{"prompt": torch.cat([r["hidden_0"], r["hidden_1"]], dim=-1), "pooled": r["pooled"]}
            for r in result]


@torch.inference_mode()
def sample(pipe, positive, seed, config, protocol, dtype):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.autocast("cuda", dtype=dtype):
        latent = pipe(prompt_embeds=positive["prompt"].to("cuda", dtype=dtype),
            pooled_prompt_embeds=positive["pooled"].to("cuda", dtype=dtype),
            negative_prompt=None, num_inference_steps=protocol["steps"],
            guidance_scale=protocol["guidance_scale"], width=config.resolution, height=config.resolution,
            output_type="latent", generator=generator).images
    if not torch.isfinite(latent).all():
        raise RuntimeError("Nonfinite evaluation latent")
    return latent.cpu().float()


from src.core.device import exclusive_device


@exclusive_device
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/training_pilot.yaml")
    parser.add_argument("--protocol", default="configs/evaluation_pilot.json")
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="Reuse only intact, identically fingerprinted progress")
    args = parser.parse_args()
    config = load_training_config(args.config)
    checkpoint = project_path(args.checkpoint)
    protocol_path = project_path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    protocol_sha = file_sha256(protocol_path)
    if protocol.get("schema") != "framegenx-paired-pilot-v1" or not 1 <= protocol["steps"] <= 50:
        raise ValueError("Invalid paired evaluation protocol")
    if (protocol.get("sampler") != "diffusers_StableDiffusionXLPipeline" or
            protocol.get("negative_prompt_policy") != "default_zero_conditioning"):
        raise ValueError("Pilot must use the actual Diffusers sampling and empty-prompt policy")
    if len(protocol["cases"]) * len(protocol["seeds"]) > args.max_pairs:
        raise ValueError("Evaluation exceeds the explicit image-pair budget")
    if not all(isinstance(s, int) and 0 <= s < 2**63 for s in protocol["seeds"]):
        raise ValueError("Invalid evaluation seed")
    if (len(set(protocol["seeds"])) != len(protocol["seeds"]) or
            len({c["id"] for c in protocol["cases"]}) != len(protocol["cases"])):
        raise ValueError("Duplicate seeds or cases would bias the paired evaluation")
    if not all(c["id"].replace("_", "").isalnum() and c["requirements"] for c in protocol["cases"]):
        raise ValueError("Invalid evaluation case")
    adapter = checkpoint / "adapter.safetensors"
    provenance = adapter_manifest(str(adapter), config.model, allow_unqualified=True, require_manifest=True)
    source = Path(training_model_source(config.model))
    saved = json.loads(adapter.with_suffix(".manifest.json").read_text())
    base_sha = file_sha256(source / "unet/diffusion_pytorch_model.fp16.safetensors")
    if saved["signature"]["model_sha256"] != base_sha:
        raise ValueError("Evaluation base checkpoint hash differs from the trained backbone")
    code_paths = [Path(__file__).resolve(), PROJECT_ROOT / "src/generation/lora.py",
                  *sorted((PROJECT_ROOT / "src/training").glob("*.py"))]
    signature = {"model_sha256": base_sha, "model_snapshot": source.name,
                 "adapter_sha256": provenance["adapter_sha256"], "protocol_sha256": protocol_sha,
                 "profile": config.model_dump(include={"model", "method", "resolution", "rank", "lora_alpha",
                    "seed", "deterministic", "target_modules"}),
                 "code_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): file_sha256(p) for p in code_paths},
                 "packages": {name: version(name) for name in
                    ("torch", "diffusers", "transformers", "peft", "bitsandbytes", "safetensors")},
                 "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    root = checkpoint.parent / ("evaluation_" + protocol_sha[:12] + "_" + fingerprint[:8])
    root.mkdir(parents=True, exist_ok=True)
    result = {"status": "running", "metric_scope": "paired_NF4_generation_pilot_only",
              "full_contract_CSR": None, "quality_status": "unverified", "promotion_status": "unqualified",
              "adapter": provenance, "protocol_sha256": protocol_sha, "protocol": protocol,
              "evaluator_code_sha256": file_sha256(Path(__file__)), "images": [],
              "generation_signature": signature, "generation_fingerprint": fingerprint,
              "model_snapshot": source.name, "model_sha256": base_sha}
    expected = expected_images(protocol)
    if (root / "result.json").exists():
        if not args.resume:
            raise FileExistsError(f"Progress already exists; use --resume: {root}")
        result = json.loads((root / "result.json").read_text())
        existing = validate_progress(root, result, signature, expected, config.resolution)
        if (result["status"] == "generated_pending_independent_review" and len(existing) == len(expected)
                and all(i.get("image_sha256") for i in existing.values())):
            print(f"Validated completed evaluation: {root}", flush=True)
            return 0
    else:
        existing = {}
        for path in code_paths:
            target = root / "code_snapshot" / path.relative_to(PROJECT_ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            if file_sha256(target) != signature["code_sha256"][path.relative_to(PROJECT_ROOT).as_posix()]:
                raise RuntimeError("Evaluation source changed while preserving its snapshot")
    result.update(status="running", reused_samples=len(existing))
    result.pop("error", None)
    previous_elapsed = result.get("elapsed_seconds", 0)
    atomic_json(root / "protocol.json", protocol)
    atomic_json(root / "result.json", result)
    print(f"Evaluation directory: {root}", flush=True)
    started = time.monotonic()
    try:
        torch.manual_seed(config.seed)
        torch.cuda.reset_peak_memory_stats()
        prompts = [c["prompt"] for c in protocol["cases"]]
        encoded = encode_prompts(source, prompts)
        unet, dtype = load_trainable_unet(config)
        unet.eval()
        unet.requires_grad_(False)
        from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
        pipe = StableDiffusionXLPipeline(vae=None, text_encoder=None, text_encoder_2=None,
            tokenizer=None, tokenizer_2=None, unet=unet,
            scheduler=EulerDiscreteScheduler.from_pretrained(source, subfolder="scheduler", local_files_only=True),
            force_zeros_for_empty_prompt=True, add_watermarker=False)
        pipe.set_progress_bar_config(disable=True)
        manager = LoraManager({"models": {"generator": config.model}, "lora": {"adapters": [
            {"name": "evaluated", "source": str(adapter), "kind": "trained", "allow_unqualified": True}]}})
        manager.hook(pipe)
        if not manager.load_report or not manager.load_report[0].get("activated"):
            raise RuntimeError("Exported candidate did not activate through the real application LoRA hook")
        result["runtime_adapter_load_report"] = manager.load_report
        result["exported_diffusers_adapter_loaded"] = True
        for case_index, case in enumerate(protocol["cases"]):
            for seed in protocol["seeds"]:
                for arm in ("baseline", "candidate"):
                    stem = f"{case['id']}_{seed}_{arm}"
                    latent_file = root / (stem + ".safetensors")
                    if stem + ".png" in existing:
                        print(f"Reusing verified sample {stem}", flush=True)
                        continue
                    if arm == "baseline":
                        pipe.disable_lora()
                    else:
                        pipe.enable_lora()
                    latent = sample(pipe, encoded[case_index], seed, config, protocol, dtype)
                    partial = latent_file.with_suffix(".safetensors.partial")
                    save_file({"latent": latent}, str(partial))
                    partial.replace(latent_file)
                    result["images"].append({"case_id": case["id"], "seed": seed, "arm": arm,
                        "file": stem + ".png", "latent_sha256": file_sha256(latent_file),
                        "prompt": case["prompt"], "requirements": case["requirements"]})
                    result["peak_reserved_gb"] = max(result.get("peak_reserved_gb", 0),
                                                       torch.cuda.max_memory_reserved() / 1e9)
                    if result["peak_reserved_gb"] > config.max_vram_gb:
                        raise RuntimeError("Evaluation exceeded its VRAM budget")
                    result["elapsed_seconds"] = previous_elapsed + time.monotonic() - started
                    atomic_json(root / "result.json", result)
                    print(f"Sampled {stem}; {time.monotonic() - started:.1f}s", flush=True)
        del pipe, unet, encoded
        gc.collect()
        torch.cuda.empty_cache()
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(source, subfolder="vae", torch_dtype=torch.float32,
                                           variant="fp16", local_files_only=True).eval().to("cuda")
        vae.enable_slicing()
        with torch.inference_mode():
            for item in result["images"]:
                if item.get("image_sha256"):
                    continue
                latent = load_file(str(root / Path(item["file"]).with_suffix(".safetensors")))["latent"]
                decoded = vae.decode(latent.to("cuda") / vae.config.scaling_factor).sample
                if not torch.isfinite(decoded).all():
                    raise RuntimeError("Nonfinite VAE evaluation output")
                pixels = (decoded / 2 + .5).clamp(0, 1).cpu().permute(0, 2, 3, 1)[0].numpy()
                if float(pixels.std()) < .01:
                    raise RuntimeError("Blank evaluation image")
                Image.fromarray(np.rint(pixels * 255).astype(np.uint8)).save(root / item["file"])
                item.update(image_sha256=file_sha256(root / item["file"]), pixel_std=float(pixels.std()))
                result["elapsed_seconds"] = previous_elapsed + time.monotonic() - started
                atomic_json(root / "result.json", result)
                print(f"Decoded {item['file']}", flush=True)
        result["peak_allocated_gb"] = max(result.get("peak_allocated_gb", 0), torch.cuda.max_memory_allocated() / 1e9)
        result["peak_reserved_gb"] = max(result.get("peak_reserved_gb", 0), torch.cuda.max_memory_reserved() / 1e9)
        if result["peak_reserved_gb"] > config.max_vram_gb:
            raise RuntimeError("Evaluation decode exceeded its VRAM budget")
        result["status"] = "generated_pending_independent_review"
    except KeyboardInterrupt:
        result["status"] = "interrupted"
        result["error"] = "Interrupted; intact samples may be resumed with --resume"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        result["elapsed_seconds"] = previous_elapsed + time.monotonic() - started
        atomic_json(root / "result.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "images"}, indent=2), flush=True)
    return 0 if result["status"] == "generated_pending_independent_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
