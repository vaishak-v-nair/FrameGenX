"""Paired regional-controller experiment against intact, saved NF4 baselines."""
import argparse
from copy import deepcopy
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

from scripts.evaluate_adapter import encode_prompts, sample
from src.generation.regions import compile_regions, instance_attention
from src.generation.fusion import encode_bootstrap_backgrounds, sample_fused
from src.training.common import atomic_json, file_sha256, load_training_config, project_path
from src.training.evaluation import expected_images, validate_progress
from src.training.model import load_trainable_unet, training_model_source


from src.core.device import exclusive_device


@exclusive_device
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--cases", nargs="+", default=["apple_count_color", "vehicle_binding", "animal_relation", "bottle_negation"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--method", choices=["attention", "fusion"], default="fusion")
    parser.add_argument("--bootstrap-steps", type=int, default=12)
    parser.add_argument("--regional-steps", type=int, default=12)
    args = parser.parse_args()
    config = load_training_config("configs/training_pilot.yaml")
    baseline_root = project_path(args.baseline)
    baseline = json.loads((baseline_root / "result.json").read_text())
    if baseline["status"] != "generated_pending_independent_review":
        raise ValueError("Baseline generation incomplete")
    saved_images = validate_progress(baseline_root, baseline, baseline["generation_signature"],
                                    expected_images(baseline["protocol"]), config.resolution)
    if len(set(args.cases)) != len(args.cases) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate cases/seeds")
    if len(args.cases) * len(args.seeds) > 12 or not args.cases or not args.seeds:
        raise ValueError("Budget is 1-12 pairs")
    if not set(args.seeds) <= set(baseline["protocol"]["seeds"]):
        raise ValueError("New seeds require a separately generated matched baseline")
    cases_file = PROJECT_ROOT / "configs/constraint_cases.json"
    cases = json.loads(cases_file.read_text())["cases"]
    protocol = deepcopy(baseline["protocol"])
    protocol["cases"] = [c for c in protocol["cases"] if c["id"] in args.cases]
    if len(protocol["cases"]) != len(args.cases):
        raise ValueError("Unknown evaluation case")
    protocol["seeds"] = args.seeds
    plans = {}
    for case in protocol["cases"]:
        case["contract"] = {**cases[case["id"]], "prompt": case["prompt"]}
        plans[case["id"]] = compile_regions(case["contract"], config.resolution)
        if not plans[case["id"]]["enabled"]:
            raise ValueError("Pilot case cannot use instance regions")
    source = Path(training_model_source(config.model))
    model_sha = file_sha256(source / "unet/diffusion_pytorch_model.fp16.safetensors")
    packages = {name: version(name) for name in baseline["generation_signature"]["packages"]}
    if model_sha != baseline["model_sha256"] or packages != baseline["generation_signature"]["packages"]:
        raise ValueError("Saved baseline model/packages differ from the current sampler")
    code_paths = [Path(__file__).resolve(), PROJECT_ROOT / "scripts/evaluate_adapter.py",
                  PROJECT_ROOT / "src/generation/fusion.py",
                  PROJECT_ROOT / "src/generation/regions.py", PROJECT_ROOT / "src/training/model.py",
                  PROJECT_ROOT / "src/training/cache.py", PROJECT_ROOT / "src/training/common.py",
                  PROJECT_ROOT / "src/training/evaluation.py"]
    signature = {"model_sha256": model_sha, "packages": packages, "config": config.model_dump(),
        "method": args.method, "bootstrap_steps": args.bootstrap_steps if args.method == "fusion" else 0,
        "regional_steps": args.regional_steps if args.method == "fusion" else 0,
        "protocol": protocol, "region_plans": plans, "case_contract_sha256": file_sha256(cases_file),
        "baseline_result_sha256": file_sha256(baseline_root / "result.json"),
        "code_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): file_sha256(p) for p in code_paths}}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    root = project_path("runs/constraint_control") / fingerprint[:16]
    root.mkdir(parents=True, exist_ok=True)
    result = {"status": "running", "method": args.method, "images": [],
        "full_contract_CSR": None, "quality_status": "unverified", "promotion_status": "unqualified",
        "generation_signature": signature, "generation_fingerprint": fingerprint, "protocol": protocol,
        "baseline_source": str(baseline_root), "adapters_enabled": False, "region_plans": plans}
    if (root / "result.json").exists():
        if not args.resume:
            raise FileExistsError(f"Use --resume for {root}")
        result = json.loads((root / "result.json").read_text())
        existing = validate_progress(root, result, signature, expected_images(protocol), config.resolution)
    else:
        existing = {}
        for path in code_paths:
            target = root / "code_snapshot" / path.relative_to(PROJECT_ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    started = time.monotonic()
    result.update(status="running", reused_images=len(existing))
    print(f"Regional evaluation directory: {root}", flush=True)
    atomic_json(root / "result.json", result)
    try:
        torch.manual_seed(config.seed)
        torch.cuda.reset_peak_memory_stats()
        prompts, prompt_indices = [], {}
        for case in protocol["cases"]:
            plan = plans[case["id"]]
            texts = [case["prompt"], plan["background_prompt"], *[r["prompt"] for r in plan["regions"]]]
            prompt_indices[case["id"]] = list(range(len(prompts), len(prompts) + len(texts)))
            prompts += texts
        encoded = encode_prompts(source, prompts)
        backgrounds = encode_bootstrap_backgrounds(source, config.resolution) if args.method == "fusion" else None
        if backgrounds is not None:
            save_file({"backgrounds": backgrounds}, str(root / "bootstrap_backgrounds.safetensors"))
            result["bootstrap_backgrounds_sha256"] = file_sha256(root / "bootstrap_backgrounds.safetensors")
        unet, dtype = load_trainable_unet(config)
        unet.eval().requires_grad_(False)
        from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
        pipe = StableDiffusionXLPipeline(vae=None, text_encoder=None, text_encoder_2=None,
            tokenizer=None, tokenizer_2=None, unet=unet,
            scheduler=EulerDiscreteScheduler.from_pretrained(source, subfolder="scheduler", local_files_only=True),
            force_zeros_for_empty_prompt=True, add_watermarker=False)
        pipe.set_progress_bar_config(disable=True)
        pipe.disable_lora()
        if args.method == "fusion":
            # Verify manual Euler/CFG/negative-prompt semantics against the
            # actual Diffusers baseline before interpreting controller results.
            case = protocol["cases"][0]
            seed = protocol["seeds"][0]
            index = prompt_indices[case["id"]][0]
            parity = sample_fused(pipe, [encoded[index]], seed, config.resolution,
                protocol["steps"], protocol["guidance_scale"], dtype)
            baseline_name = f"{case['id']}_{seed}_baseline.safetensors"
            reference = load_file(str(baseline_root / baseline_name))["latent"]
            result["native_pipeline_parity"] = {"exact": torch.equal(parity, reference),
                "max_abs_difference": float((parity-reference).abs().max())}
            atomic_json(root / "result.json", result)
            if not torch.equal(parity, reference):
                raise RuntimeError("Fusion's uncontrolled sampler does not exactly match the native baseline")
            print("Uncontrolled Euler sampler exactly matches native Diffusers baseline", flush=True)
        for case in protocol["cases"]:
            indices = prompt_indices[case["id"]]
            for seed in protocol["seeds"]:
                for arm in ("baseline", "candidate"):
                    name = f"{case['id']}_{seed}_{arm}.png"
                    if name in existing:
                        continue
                    latent_file = Path(name).with_suffix(".safetensors")
                    entry = {"case_id": case["id"], "seed": seed, "arm": arm, "file": name,
                             "prompt": case["prompt"], "requirements": case["requirements"]}
                    if arm == "baseline":
                        original = saved_images[name]
                        shutil.copy2(baseline_root / name, root / name)
                        shutil.copy2(baseline_root / latent_file, root / latent_file)
                        entry.update(image_sha256=original["image_sha256"], latent_sha256=original["latent_sha256"])
                    else:
                        embeddings = [encoded[i]["prompt"] for i in indices[1:]]
                        if args.method == "attention":
                            with instance_attention(unet, plans[case["id"]], embeddings, protocol["guidance_scale"] > 1):
                                latent = sample(pipe, encoded[indices[0]], seed, config, protocol, dtype)
                        else:
                            latent = sample_fused(pipe, [encoded[i] for i in indices[1:]], seed,
                                config.resolution, protocol["steps"], protocol["guidance_scale"], dtype,
                                plans[case["id"]], backgrounds, args.bootstrap_steps,
                                global_condition=encoded[indices[0]], regional_steps=args.regional_steps)
                        save_file({"latent": latent}, str(root / latent_file))
                        entry["latent_sha256"] = file_sha256(root / latent_file)
                    result["images"].append(entry)
                    result["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
                    if result["peak_reserved_gb"] > config.max_vram_gb:
                        raise RuntimeError("Regional sampling exceeded its VRAM budget")
                    atomic_json(root / "result.json", result)
                    print(f"Saved {name}; {time.monotonic()-started:.1f}s", flush=True)
        del pipe, unet, encoded
        gc.collect(); torch.cuda.empty_cache()
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
                    raise RuntimeError("Nonfinite regional image")
                pixels = (decoded / 2 + .5).clamp(0, 1).cpu().permute(0, 2, 3, 1)[0].numpy()
                if float(pixels.std()) < .01:
                    raise RuntimeError("Blank regional image")
                Image.fromarray(np.rint(pixels*255).astype(np.uint8)).save(root / item["file"])
                item.update(image_sha256=file_sha256(root / item["file"]), pixel_std=float(pixels.std()))
                atomic_json(root / "result.json", result)
        result.update(status="generated_pending_independent_review",
            peak_reserved_gb=torch.cuda.max_memory_reserved()/1e9,
            peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9)
        if result["peak_reserved_gb"] > config.max_vram_gb:
            raise RuntimeError("Regional decode exceeded its VRAM budget")
    except KeyboardInterrupt:
        result["status"] = "interrupted"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        result["elapsed_seconds"] = time.monotonic()-started
        atomic_json(root / "result.json", result)
        print(json.dumps({k:v for k,v in result.items() if k in
            ("status","error","elapsed_seconds","peak_reserved_gb","generation_fingerprint")}), flush=True)
    return 0 if result["status"] == "generated_pending_independent_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
