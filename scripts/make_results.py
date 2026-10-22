"""
T10: Render results.md — the 4-row ablation table — from the run DB.

Reads runs.config_name + metrics_json written by benchmark.py.
Honest numbers: measured on THIS machine at THIS scale; paper-reported
values are labeled as such and never mixed into measured rows.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.db import RunTrackerDB  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ORDER = ["baseline", "rag", "rag_lora", "full"]
CONFIG_LABELS = {
    "baseline": "Baseline (SDXL-Turbo + count-repair)",
    "rag": "+ Retrieval (RAG)",
    "rag_lora": "+ RAG + LoRA",
    "full": "Full (RAG + LoRA + Reflect)",
}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, digits=3):
    return f"{x:.{digits}f}" if x is not None else "—"


def main():
    db = RunTrackerDB(str(PROJECT_ROOT / "runs" / "framegenx.db"))
    experiment_key = db.latest_benchmark_experiment()
    rows = db.all_benchmark_rows()

    by_config = defaultdict(list)
    for config_name, prompt, ok, loops, metrics_json in rows:
        if config_name not in CONFIG_ORDER:
            continue
        m = json.loads(metrics_json) if metrics_json else {}
        by_config[config_name].append({"ok": bool(ok), "loops": loops, **m})

    lines = [
        "# FrameGenX — Measured Ablation Results",
        "",
        "Recorded local benchmark rows for one fingerprinted suite.",
        f"Suite ID: `{experiment_key or 'none'}`. The table aggregates its recorded seeds.",
        "No fingerprinted rows means there is no current comparable result.",
        "Scoring verifiers (CLIP, VQA) run in every config; `reflect` toggles only",
        "reward gating + corrective prompts. CLIP scored against the ORIGINAL user",
        "prompt. These are honest small-scale numbers — NOT the values reported in",
        "the published paper (see README). +LoRA rows run the pixel-art style",
        "adapter (nerijs/pixel-art-xl) — a visible style shift is expected and",
        "intended; it demonstrates plug-and-play adaptation, not quality gain.",
        "The pass rate below covers hard object-count and spatial checks, not all",
        "contract constraints. Attribute VQA is diagnostic and may be unavailable.",
        "Configured model references are recorded, but weight revisions are not pinned.",
        "",
        "| Config | N | Count/spatial pass | CLIP_norm | VQA | Reward | Avg attempts | Avg latency (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cfg in CONFIG_ORDER:
        runs = by_config.get(cfg, [])
        if not runs:
            lines.append(f"| {CONFIG_LABELS[cfg]} | 0 | — | — | — | — | — | — |")
            continue
        n = len(runs)
        csr = sum(r["ok"] for r in runs) / n
        lines.append(
            f"| {CONFIG_LABELS[cfg]} | {n} "
            f"| {csr:.0%} "
            f"| {fmt(mean([r.get('clip_norm') for r in runs]))} "
            f"| {fmt(mean([r.get('vqa_score') for r in runs]))} "
            f"| {fmt(mean([r.get('reward') for r in runs]))} "
            f"| {fmt(mean([r.get('attempts') for r in runs]), 2)} "
            f"| {fmt(mean([r.get('latency_s') for r in runs]), 1)} |"
        )

    lines += [
        "",
        "Paper-reported values (different system scale, different eval protocol —",
        "reported for transparency, not comparison): FID 7.2, CLIP 0.63, VQA 0.68,",
        "81% human preference.",
        "",
    ]

    out = PROJECT_ROOT / "results.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
