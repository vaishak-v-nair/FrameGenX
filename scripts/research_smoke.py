"""Offline GPU preview and isolated Milvus memory smoke from a saved contract.

This checks execution and artifact integrity, not constraint satisfaction.
It writes a new run folder and a synthetic memory database in that folder.
No provider calls or production index writes are performed.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
import uuid

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from src.core.device import exclusive_device


@exclusive_device
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=uuid.UUID)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from PIL import Image, ImageChops, ImageStat
    import torch
    from src.core.context import RunContext, load_config
    from src.generation.engine import GenerationEngine
    from src.retrieval.memory import FailureMemoryDB
    from src.retrieval.retriever import LayoutBlueprintRetriever
    from src.scene.planner import PlanStage
    from src.utils import free_memory

    config = load_config()
    config.setdefault("generation", {})["seed"] = 42
    source_path = PROJECT_ROOT / "runs" / str(args.source_run) / "contract_v2.json"
    source_contract = json.loads(source_path.read_text(encoding="utf-8"))
    ctx = RunContext(prompt=source_contract["prompt"], config=config)
    ctx.contract = deepcopy(source_contract)
    result = {
        "run_id": ctx.run_id, "source_run": str(args.source_run),
        "scope": "offline_generation_preview_and_synthetic_memory_only",
        "status": "incomplete", "checks": {},
    }
    engine = GenerationEngine(config)
    memory = FailureMemoryDB(index_dir=str(ctx.run_dir / "synthetic_memory"))
    memory_prompt = f"FrameGenX synthetic QA memory {ctx.run_id}"
    synthetic_edits = ["synthetic correction for local QA"]
    try:
        memory.log_failure(memory_prompt, [{"type": "synthetic_check"}], synthetic_edits)
        memory.unload()
        free_memory()
        result["checks"]["memory_write"] = True

        entities = [e for e in ctx.contract.get("entities", []) if not e.get("uncounted")]
        index_dir = PROJECT_ROOT / config.get("retrieval", {}).get("index_dir", "data/index")
        ctx.contract["retrieved_layout"] = LayoutBlueprintRetriever(index_dir).retrieve(
            len(entities), ctx.contract.get("spatial_relationships", []), entities=entities,
        ) if len(entities) >= 2 else None
        original_requirements = deepcopy(ctx.contract)
        PlanStage(config).run(ctx)
        if ctx.contract != original_requirements:
            raise RuntimeError("Planning mutated the extracted contract")
        result["checks"]["contract_preserved"] = True
        ctx.save_artifact("contract_v2.json", source_contract)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        meta = engine.generate(
            ctx.final_prompt, ctx.gen_config, ctx.control_hints, ctx.masks,
            ctx.run_dir / "generated_0.png", entities=ctx.scene_plan["entities"],
        )
        result["generation_seconds"] = round(time.monotonic() - started, 2)
        ctx.save_artifact("generation_meta_0.json", meta)
        preview = ctx.run_dir / f"step_0_{ctx.gen_config['steps'] - 1}.png"
        with Image.open(ctx.run_dir / "generated_0.png") as draft, Image.open(preview) as decoded:
            equal = ImageChops.difference(draft, decoded).getbbox() is None
            nonblank = sum(ImageStat.Stat(decoded).stddev) > 1.0
        if not equal or not nonblank:
            raise RuntimeError("Decoded preview differs from the draft or is blank")
        result["checks"].update(preview_matches_draft=equal, preview_nonblank=nonblank)
        if torch.cuda.is_available():
            result["process_peak_allocated_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
            result["process_peak_reserved_vram_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 3)
        engine.teardown()

        warnings = memory.retrieve_warnings(memory_prompt, top_k=1)
        if not warnings or warnings[0]["edits"] != synthetic_edits:
            raise RuntimeError("Synthetic repair memory did not survive connection reopen")
        result["checks"]["memory_reopen_roundtrip"] = True
        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        memory.unload()
        engine.teardown()
        result["total_seconds"] = round(ctx.get_duration(), 2)
        ctx.save_artifact("smoke_result.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
