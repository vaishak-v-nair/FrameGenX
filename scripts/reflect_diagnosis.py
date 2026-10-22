"""
Reflect Diagnosis Tool — Traces whether corrective edits improved or degraded
subsequent attempts across historical benchmark runs.

Core analysis:
1. For each run with multiple attempts, compare reward at attempt_k vs attempt_k+1.
2. Flag cases where reflect accepted an image that had hard failures (τ too low)
   vs rejected an image that had no hard failures (τ too high).
3. Identify the dominant failure mode: "τ too aggressive" vs "edits poisoning prompt".

Usage:
    python scripts/reflect_diagnosis.py
    python scripts/reflect_diagnosis.py --config full --verbose
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"


def load_attempt_artifacts(run_dir: Path, max_attempts: int = 8):
    """Load per-attempt verification and decision artifacts for a run."""
    attempts = []
    for n in range(max_attempts):
        verification_path = run_dir / f"verification_{n}.json"
        decision_path = run_dir / f"decision_{n}.json"
        prompt_path = run_dir / f"final_prompt_{n}.txt"

        if not verification_path.exists():
            break

        verification = {}
        decision = {}
        final_prompt = ""

        try:
            with open(verification_path, "r", encoding="utf-8") as f:
                verification = json.load(f)
        except Exception:
            pass
        try:
            with open(decision_path, "r", encoding="utf-8") as f:
                decision = json.load(f)
        except Exception:
            pass
        try:
            if prompt_path.exists():
                final_prompt = prompt_path.read_text(encoding="utf-8").strip()
        except Exception:
            pass

        attempts.append({
            "attempt": n,
            "verification": verification,
            "decision": decision,
            "final_prompt": final_prompt,
            "reward": verification.get("reward", 0.0),
            "clip_norm": verification.get("clip_norm", 0.0),
            "vqa_score": verification.get("vqa_score", 0.0),
            "hard_failures": verification.get("hard_failures", []),
            "accepted": decision.get("accept", False),
            "prompt_edits": decision.get("prompt_edits", []),
            "tau": decision.get("tau"),
        })
    return attempts


def analyze_run(run_dir: Path, prompt: str, config_name: str, verbose: bool = False):
    """Analyze a single run for reflect effectiveness."""
    attempts = load_attempt_artifacts(run_dir)
    if len(attempts) < 1:
        return None

    result = {
        "prompt": prompt,
        "config_name": config_name,
        "run_dir": str(run_dir.name),
        "num_attempts": len(attempts),
        "issues": [],
        "transitions": [],
    }

    for i in range(len(attempts)):
        a = attempts[i]
        hard_ok = len(a["hard_failures"]) == 0
        reward = a["reward"]
        accepted = a["accepted"]
        tau = a["tau"]

        # Issue: rejected despite no hard failures (τ too aggressive)
        if not accepted and hard_ok and tau is not None and reward is not None and reward < tau:
            result["issues"].append({
                "type": "tau_too_aggressive",
                "attempt": i,
                "reward": reward,
                "tau": tau,
                "gap": round(tau - reward, 4),
                "detail": f"Attempt {i}: R={reward:.3f} < τ={tau:.3f} but no hard failures — "
                          f"image was count-correct but rejected for low reward"
            })

        # Issue: accepted despite hard failures (shouldn't happen but check)
        if accepted and not hard_ok:
            result["issues"].append({
                "type": "accepted_with_hard_failures",
                "attempt": i,
                "hard_failures": a["hard_failures"],
                "detail": f"Attempt {i}: accepted despite {len(a['hard_failures'])} hard failure(s)"
            })

        # Track reward transitions between attempts
        if i > 0:
            prev = attempts[i - 1]
            delta_reward = (reward - prev["reward"]
                            if reward is not None and prev["reward"] is not None
                            else None)
            prev_edits = prev["prompt_edits"]
            prev_hard_ok = len(prev["hard_failures"]) == 0

            transition = {
                "from_attempt": i - 1,
                "to_attempt": i,
                "reward_delta": round(delta_reward, 4) if delta_reward is not None else None,
                "improved": delta_reward > 0 if delta_reward is not None else None,
                "edits_applied": prev_edits,
                "num_edits": len(prev_edits),
                "from_hard_ok": prev_hard_ok,
                "to_hard_ok": hard_ok,
                "prompt_len_before": len(prev["final_prompt"]),
                "prompt_len_after": len(a["final_prompt"]),
            }
            result["transitions"].append(transition)

            # Issue: edits made things worse
            if delta_reward is not None and delta_reward < -0.05 and prev_edits:
                result["issues"].append({
                    "type": "edits_degraded_reward",
                    "attempt": i,
                    "reward_delta": round(delta_reward, 4),
                    "edits": prev_edits,
                    "detail": f"Attempt {i-1}→{i}: edits [{', '.join(prev_edits[:2])}...] "
                              f"degraded reward by {abs(delta_reward):.3f}"
                })

            # Issue: was count-correct, edits broke count
            if prev_hard_ok and not hard_ok:
                result["issues"].append({
                    "type": "edits_broke_count",
                    "attempt": i,
                    "hard_failures": a["hard_failures"],
                    "edits": prev_edits,
                    "detail": f"Attempt {i-1}→{i}: was count-correct, edits caused "
                              f"{len(a['hard_failures'])} count failure(s)"
                })

            # Issue: prompt length exceeded reasonable bounds (77-token proxy)
            if len(a["final_prompt"]) > 300:  # ~77 CLIP tokens ≈ ~55 words ≈ ~300 chars
                result["issues"].append({
                    "type": "prompt_overflow",
                    "attempt": i,
                    "prompt_len": len(a["final_prompt"]),
                    "detail": f"Attempt {i}: final prompt is {len(a['final_prompt'])} chars "
                              f"(likely exceeds 77 CLIP tokens)"
                })

    return result


def scan_runs(config_filter: str = None):
    """Scan all runs directories for analysis."""
    from src.core.db import RunTrackerDB

    db = RunTrackerDB(str(PROJECT_ROOT / "runs" / "framegenx.db"))
    experiment_key = db.latest_benchmark_experiment()
    rows = db.all_benchmark_rows()

    run_analyses = []
    for config_name, prompt, is_successful, repair_loops, metrics_json in rows:
        if config_filter and config_name != config_filter:
            continue

        # Find the run directory by matching prompt
        cursor = db.conn.cursor()
        cursor.execute(
            "SELECT run_id FROM runs WHERE prompt = ? AND config_name = ? "
            "AND cache_key = ? ORDER BY start_time DESC LIMIT 1",
            (prompt, config_name, experiment_key),
        )
        row = cursor.fetchone()
        if not row:
            continue

        run_dir = RUNS_DIR / row[0]
        if not run_dir.exists():
            continue

        analysis = analyze_run(run_dir, prompt, config_name)
        if analysis:
            analysis["is_successful"] = bool(is_successful)
            run_analyses.append(analysis)

    db.close()
    return run_analyses


def print_report(analyses, verbose: bool = False):
    """Print a summary diagnosis report."""
    if not analyses:
        print("No runs found for analysis.")
        return

    # Aggregate issue counts
    issue_counts = defaultdict(int)
    all_transitions = []
    total_runs = len(analyses)
    successful_runs = sum(1 for a in analyses if a.get("is_successful"))

    for a in analyses:
        for issue in a["issues"]:
            issue_counts[issue["type"]] += 1
        all_transitions.extend(a["transitions"])

    print("=" * 72)
    print("REFLECT STAGE DIAGNOSIS REPORT")
    print("=" * 72)
    print(f"\nRuns analyzed: {total_runs}")
    print(f"Successful runs: {successful_runs} ({successful_runs/total_runs:.0%})")
    print(f"Config filter: {analyses[0]['config_name'] if analyses else 'all'}")

    print(f"\n--- Issue Summary ---")
    for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        labels = {
            "tau_too_aggressive": "τ rejected count-correct image (τ too high)",
            "edits_degraded_reward": "Corrective edits degraded reward",
            "edits_broke_count": "Corrective edits broke object count",
            "prompt_overflow": "Final prompt likely exceeded 77 CLIP tokens",
            "accepted_with_hard_failures": "Accepted image with hard failures (bug)",
        }
        print(f"  {labels.get(issue_type, issue_type)}: {count} occurrences")

    if all_transitions:
        observed = [t for t in all_transitions if t["reward_delta"] is not None]
        improved = sum(1 for t in observed if t["improved"])
        degraded = len(observed) - improved
        print(f"\n--- Repair Transition Analysis ---")
        print(f"  Total transitions: {len(all_transitions)}")
        if observed:
            print(f"  Reward improved: {improved} ({improved/len(observed):.0%})")
            print(f"  Reward degraded: {degraded} ({degraded/len(observed):.0%})")
        else:
            print("  Reward transitions unavailable (vision QA incomplete).")

        # Mean delta by edit count
        with_edits = [t for t in observed if t["num_edits"] > 0]
        without_edits = [t for t in observed if t["num_edits"] == 0]
        if with_edits:
            mean_delta = sum(t["reward_delta"] for t in with_edits) / len(with_edits)
            print(f"  Mean reward delta (with edits): {mean_delta:+.4f}")
        if without_edits:
            mean_delta = sum(t["reward_delta"] for t in without_edits) / len(without_edits)
            print(f"  Mean reward delta (no edits): {mean_delta:+.4f}")

        # Count transitions lost
        broke_count = sum(1 for t in all_transitions
                         if t["from_hard_ok"] and not t["to_hard_ok"])
        fixed_count = sum(1 for t in all_transitions
                         if not t["from_hard_ok"] and t["to_hard_ok"])
        print(f"  Transitions that broke count: {broke_count}")
        print(f"  Transitions that fixed count: {fixed_count}")

    # Root cause diagnosis
    print(f"\n--- ROOT CAUSE DIAGNOSIS ---")
    tau_issues = issue_counts.get("tau_too_aggressive", 0)
    edit_degrade = issue_counts.get("edits_degraded_reward", 0)
    edit_broke = issue_counts.get("edits_broke_count", 0)
    overflow = issue_counts.get("prompt_overflow", 0)

    if tau_issues > edit_degrade and tau_issues > edit_broke:
        print("  PRIMARY: τ threshold is too aggressive.")
        print(f"  {tau_issues} cases where count-correct images were rejected")
        print("  because R < τ, causing unnecessary re-generation into worse states.")
        print("  RECOMMENDATION: Lower τ to the 40th percentile of observed rewards")
        print("  or implement a hard-failure-only acceptance mode.")
    elif edit_broke > tau_issues:
        print("  PRIMARY: Corrective edits are destructive.")
        print(f"  {edit_broke} cases where edits broke object count.")
        print("  RECOMMENDATION: Limit corrective edits to count-related prompts")
        print("  and cap total edit tokens.")
    elif overflow > 0:
        print("  PRIMARY: Prompt token budget overflow.")
        print(f"  {overflow} cases where final prompt likely exceeded 77 CLIP tokens.")
        print("  RECOMMENDATION: Enforce strict character limit before generation.")
    else:
        print("  No dominant single root cause identified.")
        print("  Consider multi-factor analysis with --verbose flag.")

    if verbose:
        print(f"\n--- Detailed Per-Run Issues ---")
        for a in analyses:
            if not a["issues"]:
                continue
            print(f"\n  [{a['config_name']}] '{a['prompt'][:60]}...'")
            print(f"  Successful: {a.get('is_successful')}, Attempts: {a['num_attempts']}")
            for issue in a["issues"]:
                print(f"    [{issue['type']}] {issue['detail']}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose reflect stage effectiveness")
    parser.add_argument("--config", default=None,
                        help="Filter to specific config (baseline/rag/rag_lora/full)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-run detailed issues")
    parser.add_argument("--json", action="store_true",
                        help="Output raw analysis as JSON")
    args = parser.parse_args()

    analyses = scan_runs(config_filter=args.config)

    if args.json:
        print(json.dumps(analyses, indent=2))
    else:
        print_report(analyses, verbose=args.verbose)


if __name__ == "__main__":
    main()
