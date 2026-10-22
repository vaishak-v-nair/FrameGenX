"""
Benchmark Report Generator — Produces a detailed markdown report from benchmark
runs with per-category hard-check pass breakdown, reward distributions, and latency analysis.

Usage:
    python scripts/benchmark_report.py
    python scripts/benchmark_report.py --output reports/benchmark_report.md
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_ORDER = ["baseline", "rag", "rag_lora", "full"]
CONFIG_LABELS = {
    "baseline": "Baseline",
    "rag": "+ RAG",
    "rag_lora": "+ RAG + LoRA",
    "full": "Full (RAG+LoRA+Reflect)",
}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def fmt(x, digits=3):
    return f"{x:.{digits}f}" if x is not None else "—"


def load_data():
    from src.core.db import RunTrackerDB

    db = RunTrackerDB(str(PROJECT_ROOT / "runs" / "framegenx.db"))
    rows = db.all_benchmark_rows()

    # Also try to match prompts to categories from benchmark_dataset.json
    categories = {}
    for ds_file in ["benchmark_subset.json", "benchmark_dataset.json"]:
        ds_path = PROJECT_ROOT / ds_file
        if ds_path.exists():
            with open(ds_path, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    categories[item["prompt"]] = item.get("category", "unknown")

    by_config = defaultdict(list)
    for config_name, prompt, ok, loops, metrics_json in rows:
        if config_name not in CONFIG_ORDER:
            continue
        m = json.loads(metrics_json) if metrics_json else {}
        m["ok"] = bool(ok)
        m["loops"] = loops
        m["prompt"] = prompt
        m["category"] = categories.get(prompt, "unknown")
        by_config[config_name].append(m)

    db.close()
    return by_config


def generate_report(by_config: dict) -> str:
    lines = [
        "# FrameGenX — Detailed Benchmark Report",
        "",
        f"Generated from `runs/framegenx.db`",
        "",
    ]

    # 1. Summary Table
    lines.append("## 1. Ablation Summary")
    lines.append("")
    lines.append("Count/spatial pass is the recorded hard gate; color and other attributes remain diagnostic.")
    lines.append("")
    lines.append("| Config | N | Count/spatial pass | CLIP_norm | VQA | Reward | Avg Attempts | Avg Latency (s) |")
    lines.append("|--------|---|-----|-----------|-----|--------|--------------|-----------------|")

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
    lines.append("")

    # 2. Per-Category Breakdown
    lines.append("## 2. Per-Category Count/Spatial Pass Breakdown")
    lines.append("")

    all_categories = set()
    for runs in by_config.values():
        for r in runs:
            all_categories.add(r["category"])
    all_categories = sorted(all_categories)

    if len(all_categories) > 1:
        header = "| Category |"
        separator = "|----------|"
        for cfg in CONFIG_ORDER:
            if cfg in by_config:
                header += f" {CONFIG_LABELS[cfg]} |"
                separator += "------|"
        lines.append(header)
        lines.append(separator)

        for cat in all_categories:
            row = f"| {cat} |"
            for cfg in CONFIG_ORDER:
                if cfg not in by_config:
                    continue
                cat_runs = [r for r in by_config[cfg] if r["category"] == cat]
                if cat_runs:
                    csr = sum(r["ok"] for r in cat_runs) / len(cat_runs)
                    row += f" {csr:.0%} ({len(cat_runs)}) |"
                else:
                    row += " — |"
            lines.append(row)
        lines.append("")
    else:
        lines.append("*Only one category detected. Expand benchmark_dataset.json for breakdown.*")
        lines.append("")

    # 3. Reward Distribution
    lines.append("## 3. Reward Distribution")
    lines.append("")

    for cfg in CONFIG_ORDER:
        runs = by_config.get(cfg, [])
        if not runs:
            continue
        rewards = [r.get("reward") for r in runs if r.get("reward") is not None]
        if not rewards:
            continue
        lines.append(f"### {CONFIG_LABELS[cfg]}")
        lines.append(f"- N: {len(rewards)}")
        lines.append(f"- Mean: {mean(rewards):.4f}")
        lines.append(f"- Std: {std(rewards):.4f}")
        lines.append(f"- Min: {min(rewards):.4f}")
        lines.append(f"- Max: {max(rewards):.4f}")

        # Histogram (text-based)
        buckets = defaultdict(int)
        for r in rewards:
            bucket = round(r * 10) / 10  # bucket to nearest 0.1
            buckets[bucket] += 1
        lines.append(f"- Distribution:")
        for b in sorted(buckets.keys()):
            bar = "█" * buckets[b]
            lines.append(f"  {b:.1f} | {bar} ({buckets[b]})")
        lines.append("")

    # 4. Attempt Distribution
    lines.append("## 4. Attempt Distribution")
    lines.append("")
    lines.append("| Config | 1 attempt | 2 attempts | 3 attempts | 4 attempts |")
    lines.append("|--------|-----------|------------|------------|------------|")

    for cfg in CONFIG_ORDER:
        runs = by_config.get(cfg, [])
        if not runs:
            continue
        attempt_counts = defaultdict(int)
        for r in runs:
            a = r.get("attempts", 1)
            attempt_counts[int(a)] += 1
        row = f"| {CONFIG_LABELS[cfg]} |"
        for a in [1, 2, 3, 4]:
            count = attempt_counts.get(a, 0)
            row += f" {count} |"
        lines.append(row)
    lines.append("")

    # 5. Failed Prompts
    lines.append("## 5. Failed Prompts (Full Config)")
    lines.append("")
    full_runs = by_config.get("full", [])
    failed = [r for r in full_runs if not r["ok"]]
    if failed:
        lines.append(f"**{len(failed)} of {len(full_runs)} prompts failed:**")
        lines.append("")
        for r in failed:
            reward = r.get("reward") or 0.0
            lines.append(f"- `{r['prompt']}` (R={reward:.3f}, category={r['category']})")
    else:
        lines.append("All prompts passed! 🎉")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate detailed benchmark report")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path (default: stdout + reports/benchmark_report.md)")
    args = parser.parse_args()

    by_config = load_data()
    if not by_config:
        print("No benchmark data found. Run benchmark.py first.")
        return

    report = generate_report(by_config)

    output_path = Path(args.output) if args.output else PROJECT_ROOT / "reports" / "benchmark_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Report written to: {output_path}")
    # Windows cp1252 can't print emoji; replace gracefully
    import sys
    sys.stdout.buffer.write(report.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
