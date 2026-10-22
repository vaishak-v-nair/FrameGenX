"""
Ablation benchmark runner.

  python benchmark.py --config full
  python benchmark.py --config baseline          # maps to --disable retrieval,adapt,reflect
  python benchmark.py --disable retrieval,adapt  # custom

Configs (design doc ablation semantics — repair loop present in ALL configs;
scoring verifiers always run; reflect toggles only gating + edits):

  baseline   = disable retrieval, adapt, reflect
  rag        = disable adapt, reflect
  rag_lora   = disable reflect
  full       = retrieval + adaptation + reflection (refinement excluded)

Resume: completed rows are keyed on (experiment fingerprint, config_name,
prompt, seed) regardless of acceptance. Infrastructure errors remain retryable.
"""
import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from src.core.context import load_config
from src.core.stage import current_cache_key
from src.pipeline import FrameGenXPipeline

PROJECT_ROOT = Path(__file__).resolve().parent

ABLATION_CONFIGS = {
    "baseline": ["retrieval", "adapt", "reflect"],
    "rag": ["adapt", "reflect"],
    "rag_lora": ["reflect"],
    "full": [],
}

ALL_STAGES = ["intent", "retrieval", "plan", "adapt", "generate", "verify", "reflect"]


def benchmark_overrides(config_name: str, stages: list, seed: int) -> dict:
    """Keep an ablation's active components explicit and reproducible."""
    overrides = {
        "pipeline": {"stages": stages},
        "generation": {"seed": seed},
    }
    if config_name in ("rag_lora", "full"):
        config = load_config(str(Path(__file__).resolve().parent / "config.yaml"))
        adapters = (config.get("lora") or {}).get("adapters") or []
        if len(adapters) != 1 or not adapters[0].get("source"):
            raise ValueError("LoRA ablations require exactly one configured adapter source")
        overrides["lora"] = {"adapters": [{**adapters[0], "enabled": True}]}
    return overrides


def load_subset(dataset_path: str = "benchmark_subset.json"):
    p = Path(dataset_path)
    if not p.exists():
        # Fallback: derive from the raw dataset (curated subset preferred)
        with open("benchmark_dataset.json", "r") as f:
            dataset = json.load(f)
        return [d for d in dataset if d.get("category") == "object_count"][:20]
    with open(p, "r") as f:
        return json.load(f)


def build_benchmark_manifest(test_cases: list, config_name: str,
                             stages: list) -> dict:
    """Fingerprint the code, settings, selected cases and installed packages.

    Model IDs are recorded, but floating remote model revisions are not
    checkpoint hashes. The manifest says so explicitly.
    """
    base_config = load_config(str(PROJECT_ROOT / "config.yaml"))
    packages = {}
    for name in ("torch", "diffusers", "transformers", "accelerate", "Pillow"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    selected = json.dumps(test_cases, sort_keys=True, separators=(",", ":"))
    payload = {
        "format": "framegenx-benchmark-v2",
        "base_code_config_sha256": current_cache_key(base_config),
        "runner_sha256": hashlib.sha256((PROJECT_ROOT / "benchmark.py").read_bytes()).hexdigest(),
        "selected_cases_sha256": hashlib.sha256(selected.encode("utf-8")).hexdigest(),
        "selected_cases": len(test_cases),
        "package_versions": packages,
        "model_refs": base_config.get("models", {}),
        "adapter_refs": (base_config.get("lora") or {}).get("adapters", []),
        "checkpoint_revisions_verified": False,
        "provenance_limit": "Configured model IDs are recorded; resolved weight commit hashes are not captured.",
    }
    if config_name not in ABLATION_CONFIGS:
        payload["custom_stages"] = stages
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return {"experiment_key": "benchmark-v2:" + hashlib.sha256(encoded).hexdigest(),
            **payload}


def save_benchmark_manifest(config: dict, manifest: dict) -> Path:
    output_dir = Path(config.get("paths", {}).get("output_dir", "runs"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    manifest_dir = output_dir / "benchmark_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"{manifest['experiment_key'].split(':', 1)[1]}.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


from src.core.device import exclusive_device


@exclusive_device
def run_benchmark(config_name: str = "full", disable: list = None,
                  dataset_path: str = "benchmark_subset.json", limit: int = 0,
                  seed: int = 42):
    disable = disable if disable is not None else ABLATION_CONFIGS.get(config_name, [])
    stages = [s for s in ALL_STAGES if s not in disable]
    print(f"--- FrameGenX Ablation Benchmark [{config_name}] stages={stages} seed={seed} ---")

    test_cases = load_subset(dataset_path)
    if limit and limit < len(test_cases):
        # Balanced cut: keep the category mix (a naive first-N would drop
        # color_constraints entirely since the subset lists counts first).
        by_cat = {}
        for c in test_cases:
            by_cat.setdefault(c.get("category", "other"), []).append(c)
        cut = []
        share = limit / len(test_cases)
        for cat, cases in by_cat.items():
            cut.extend(cases[: max(1, round(len(cases) * share))])
        test_cases = cut[:limit]

    overrides = benchmark_overrides(config_name, stages, seed)
    pipeline = FrameGenXPipeline(config_overrides=overrides)
    manifest = build_benchmark_manifest(test_cases, config_name, stages)
    experiment_key = manifest["experiment_key"]
    manifest_path = save_benchmark_manifest(pipeline.config, manifest)
    print(f"[benchmark] suite={experiment_key} manifest={manifest_path}")

    total = len(test_cases)
    done = skipped = successful = errors = 0
    results_by_category = {}  # category -> [bool]
    start_time = time.time()

    for idx, case in enumerate(test_cases):
        prompt = case["prompt"]
        category = case.get("category", "unknown")
        if pipeline.db.has_run(prompt, config_name, seed=seed,
                               experiment_key=experiment_key):
            skipped += 1
            print(f"[{idx+1}/{total}] SKIP (already ran): '{prompt}'")
            continue
        print(f"\n[{idx+1}/{total}] [{category}] '{prompt}'")
        try:
            ctx = pipeline.run(prompt, config_name=config_name,
                               is_benchmark=True, benchmark_seed=seed,
                               benchmark_key=experiment_key)
            done += 1
            ok = ctx.is_successful
            successful += int(ok)
            results_by_category.setdefault(category, []).append(ok)
        except Exception as e:
            print(f"ERROR for '{prompt}': {e}")
            # Preserve the error for diagnosis without counting it as an
            # observed constraint failure or suppressing a later retry.
            from src.core.context import RunContext
            ctx = RunContext(prompt=prompt, config=pipeline.config)
            ctx.config_name = config_name
            ctx.is_benchmark = True
            ctx.benchmark_seed = seed
            ctx.cache_key = experiment_key
            ctx.is_successful = False
            ctx.verification_report = {"accepted": False, "error": str(e)[:300]}
            pipeline.db.log_run(ctx)
            errors += 1

    elapsed = time.time() - start_time
    print(f"\n--- [{config_name}] complete: {done} scored, {errors} errors, {skipped} skipped, "
          f"{successful} accepted, {elapsed/60:.1f} min ---")

    # Per-category hard-check pass breakdown
    if results_by_category:
        print(f"\n--- Per-Category Count/Spatial Pass Breakdown [{config_name}] ---")
        for cat in sorted(results_by_category.keys()):
            cat_results = results_by_category[cat]
            pass_rate = sum(cat_results) / len(cat_results) if cat_results else 0
            print(f"  {cat}: {pass_rate:.0%} ({sum(cat_results)}/{len(cat_results)})")

    return {"experiment_key": experiment_key,
            "done": done, "errors": errors, "skipped": skipped, "successful": successful,
            "elapsed": elapsed, "by_category": results_by_category}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="full", choices=list(ABLATION_CONFIGS))
    parser.add_argument("--disable", default=None,
                        help="comma-separated stage names (overrides --config mapping)")
    parser.add_argument("--dataset", default="benchmark_subset.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default=None,
                        help="comma-separated seeds for multi-seed runs (overrides --seed)")
    parser.add_argument("--json", action="store_true",
                        help="export metrics as JSON")
    args = parser.parse_args()

    disable = args.disable.split(",") if args.disable else None
    name = args.config if disable is None else "custom"

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    all_results = []

    for seed in seeds:
        result = run_benchmark(config_name=name, disable=disable,
                               dataset_path=args.dataset, limit=args.limit, seed=seed)
        all_results.append({"seed": seed, **result})

    if len(seeds) > 1:
        total_done = sum(r["done"] for r in all_results)
        total_success = sum(r["successful"] for r in all_results)
        total_elapsed = sum(r["elapsed"] for r in all_results)
        if total_done > 0:
            overall_pass_rate = total_success / total_done
            print(f"\n=== MULTI-SEED SUMMARY [{name}] ===")
            print(f"Seeds: {seeds}")
            print(f"Overall count/spatial pass: {overall_pass_rate:.0%} ({total_success}/{total_done})")
            print(f"Total time: {total_elapsed/60:.1f} min")

    if args.json:
        print(json.dumps(all_results, indent=2, default=str))
    if any(result["errors"] for result in all_results):
        raise SystemExit(1)
