"""
T8: Tau pilot — calibrate the reflect threshold from real images instead of
trusting the slides' 0.575 (BLIP's yes-bias saturates VQA near 1.0, which
would make a fixed tau inert; eng review OV#7).

Scores existing runs/ images with the new verifiers, prints the observed
reward distribution, and recommends tau at the 60th percentile.

Usage: python scripts/tau_pilot.py [--apply]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src.core.context import load_config  # noqa: E402
from src.intent.extractor import IntentExtractor  # noqa: E402
from src.verification.verify_stage import CLIPScoreVerifier, VQAVerifier, compute_reward  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def collect_samples(max_n: int = 12):
    """(image_path, prompt) pairs from existing runs with saved contracts."""
    samples = []
    runs_dir = PROJECT_ROOT / "runs"
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        contract_path = run_dir / "contract_v2.json"
        images = sorted(run_dir.glob("generated_*.png"))
        if not contract_path.exists() or not images:
            continue
        with open(contract_path, "r", encoding="utf-8") as f:
            contract = json.load(f)
        samples.append((images[-1], contract.get("prompt", ""), contract))
        if len(samples) >= max_n:
            break
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="write recommended tau into config.yaml")
    parser.add_argument("--percentile", type=float, default=60.0)
    args = parser.parse_args()

    config = load_config()
    samples = collect_samples()
    if len(samples) < 4:
        print(f"Only {len(samples)} scored samples available — need >=4. "
              "Generate a few pipeline runs first; keeping paper tau.")
        return

    clip_v = CLIPScoreVerifier(config)
    vqa_v = VQAVerifier(config)
    ex = IntentExtractor()

    rewards = []
    for img_path, prompt, contract in samples:
        if not contract.get("entities"):
            # Old runs may predate entity extraction; re-extract for VQA questions
            contract = ex.extract(prompt) if prompt else contract
        image = Image.open(img_path).convert("RGB")
        c = clip_v.score(image, prompt)
        v = vqa_v.score(image, contract)
        r = compute_reward(c["clip_norm"], v["vqa_score"])
        rewards.append(r)
        print(f"{img_path.parent.name[:8]} clip_norm={c['clip_norm']:.3f} "
              f"vqa={v['vqa_score']:.3f} R={r:.3f}  '{prompt[:50]}'")

    arr = np.array(rewards)
    tau = float(np.percentile(arr, args.percentile))
    print(f"\nR distribution: min={arr.min():.3f} p25={np.percentile(arr,25):.3f} "
          f"median={np.median(arr):.3f} p60={np.percentile(arr,60):.3f} "
          f"max={arr.max():.3f}")
    print(f"Recommended tau (p{args.percentile:.0f}): {tau:.3f}  (paper value: 0.575)")

    if args.apply:
        cfg_path = PROJECT_ROOT / "config.yaml"
        text = cfg_path.read_text(encoding="utf-8")
        text = text.replace("tau: 0.575", f"tau: {tau:.3f}")
        text = text.replace("tau_source: paper", "tau_source: pilot")
        cfg_path.write_text(text, encoding="utf-8")
        print(f"Applied to config.yaml: tau={tau:.3f}, tau_source=pilot")


if __name__ == "__main__":
    main()
