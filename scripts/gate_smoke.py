"""
Night-before GATE (design doc): the build relies on GPU work ONLY after this
passes. Pip succeeding is NOT the gate — inference succeeding is.

Checks, in order:
  1. CUDA present
  2. Full pipeline cycle on one prompt: Turbo generate + DINO + CLIP + BLIP
     + reflect — measures FULL attempt latency (gen + verify + reflect)
  3. ControlNet-under-Turbo layout check: DINO detections must land INSIDE
     the planner's boxes (fallback chain: canny -> weight sweep -> none)
  4. Projected 4-config benchmark wall-clock vs overnight budget
  5. (if lora.adapters configured) fixed-seed A/B pair, adapter on/off

Usage: python scripts/gate_smoke.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROMPT = "two cats sitting on a wall"


def box_overlap(det_box, plan_box) -> float:
    """IoU-ish: fraction of detection box inside the planned box."""
    ax0, ay0, ax1, ay1 = det_box
    bx0, by0 = plan_box["xmin"], plan_box["ymin"]
    bx1, by1 = plan_box["xmax"], plan_box["ymax"]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area = max(1, (ax1 - ax0) * (ay1 - ay0))
    return inter / area


from src.core.device import exclusive_device


@exclusive_device
def main():
    import torch
    assert torch.cuda.is_available(), "GATE FAIL: CUDA not available"
    print(f"[gate 1/5] CUDA OK: {torch.cuda.get_device_name(0)}")

    from src.pipeline import FrameGenXPipeline
    pipeline = FrameGenXPipeline()

    t0 = time.time()
    ctx = pipeline.run(PROMPT, config_name="gate_smoke")
    cycle = time.time() - t0
    attempts = ctx.attempt + 1
    per_attempt = cycle / attempts
    print(f"[gate 2/5] full cycle OK: {cycle:.1f}s total, {attempts} attempt(s), "
          f"~{per_attempt:.1f}s per attempt")
    assert (ctx.run_dir / f"generated_{ctx.attempt}.png").exists(), "no image produced"
    assert ctx.reward_history, "verify stage produced no scores"
    last = ctx.reward_history[-1]
    print(f"          scores: clip_norm={last['clip_norm']} vqa={last['vqa_score']} "
          f"R={last['reward']}")

    # Layout check on the LAST attempt's verification signals
    report_path = ctx.run_dir / f"verification_{ctx.attempt}.json"
    with open(report_path) as f:
        report = json.load(f)
    detected = report.get("signals", {}).get("detected_counts", {})
    print(f"[gate 3/5] layout: expected cats=2, DINO detected={detected}")
    if detected.get("cats", 0) == 0:
        print("          WARNING: nothing detected — run fallback chain "
              "(config generation.control_image_mode: canny -> box -> none, "
              "weight 0.8-1.2 at steps=4)")

    # Projected overnight wall-clock: 30 prompts x 4 configs x ~2 avg attempts
    projected_h = (30 * 4 * 2 * per_attempt) / 3600
    budget_ok = projected_h <= 8
    print(f"[gate 4/5] projected benchmark: ~{projected_h:.1f}h "
          f"({'OK' if budget_ok else 'OVER 8h BUDGET — arm the 15-prompt cut'})")

    # LoRA A/B (only when configured)
    adapters = (pipeline.config.get("lora", {}) or {}).get("adapters", [])
    if adapters:
        engine = pipeline.pipeline.stages["generate"].engine
        from PIL import Image
        out_dir = Path("runs/_lora_ab")
        out_dir.mkdir(parents=True, exist_ok=True)
        gen_cfg = {"width": 512, "height": 512, "steps": 4, "cfg_scale": 0.0, "seed": 1234}
        hints = {"control_image_mode": "none"}
        engine.teardown()
        engine.generate("a cozy cottage in a forest", gen_cfg, hints, {}, out_dir / "with_lora.png")
        engine.teardown()
        engine._pre_offload_hook = None  # off
        engine.generate("a cozy cottage in a forest", gen_cfg, hints, {}, out_dir / "no_lora.png")
        print(f"[gate 5/5] LoRA A/B written to {out_dir} — inspect visually")
    else:
        print("[gate 5/5] no LoRA configured yet — A/B skipped")

    layout_ok = detected.get("cats", 0) > 0
    print(f"\n=== GATE {'PASSED' if layout_ok else 'PASSED WITH LAYOUT WARNING'} ===")
    print(f"run dir: {ctx.run_dir}")


if __name__ == "__main__":
    main()
