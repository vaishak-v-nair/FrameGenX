"""Plugin smoke validation — runs the pipeline natively with both
diagnostic plugins enabled, prints measured timings and the soft signals.
Not part of the test suite: live LLM + GPU."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.pipeline import FrameGenXPipeline

overrides = {
    "verification": {"plugins": {"scene_comprehension": True, "easyocr": True}},
}

pipeline = FrameGenXPipeline(config_overrides=overrides)

prompts = [
    "a rustic wooden sign that says OPEN",   # text_rendering-ish
    "three red apples on a wooden table",    # object_count classic
]

for prompt in prompts:
    t0 = time.time()
    ctx = pipeline.run(prompt)
    dt = time.time() - t0
    rep = ctx.verification_report or {}
    signals = rep.get("signals", {})
    print("\n" + "=" * 70)
    print(f"PROMPT: {prompt}")
    print(f"accepted={ctx.is_successful} attempts={ctx.attempt + 1} latency={dt:.1f}s")
    print(f"hard_failures={rep.get('hard_failures')}")
    print(f"clip_norm={rep.get('clip_norm')} vqa={rep.get('vqa_score')} reward={rep.get('reward')}")
    ocr = signals.get("ocr")
    if ocr:
        print(f"OCR signal: found={ocr.get('texts_found')} requests_text={ocr.get('prompt_requests_text')} note={ocr.get('note')}")
    scene = signals.get("scene_comprehension")
    if scene:
        print(f"SCENE signal: described={scene.get('described')} missing={scene.get('missing')}")
        print(f"  excerpt: {scene.get('caption_excerpt', '')[:120]}")
    if not ocr and not scene:
        print("(no plugin signals — check attempt count vs max_repair_loops)")
    print("=" * 70)
