"""
Tau Sweep — Simulates different τ thresholds against existing benchmark data
to find the empirically optimal acceptance threshold.

Instead of running full generation (expensive), this script replays the
reward_history from past runs and computes what the accept/reject decision
WOULD HAVE BEEN at each τ value. This reveals the CSR-optimal τ in minutes.

Usage:
    python scripts/tau_sweep.py
    python scripts/tau_sweep.py --config full --tau-min 0.40 --tau-max 0.85 --step 0.025
    python scripts/tau_sweep.py --apply  # writes optimal τ to config.yaml
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"


def collect_run_data(config_filter: str = None):
    """Collect reward histories and hard-failure data from past runs."""
    from src.core.db import RunTrackerDB

    db = RunTrackerDB(str(PROJECT_ROOT / "runs" / "framegenx.db"))
    experiment_key = db.latest_benchmark_experiment()
    rows = db.all_benchmark_rows()

    run_data = []
    for config_name, prompt, is_successful, repair_loops, metrics_json in rows:
        if config_filter and config_name != config_filter:
            continue

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

        # Load per-attempt data
        attempts = []
        for n in range(8):
            v_path = run_dir / f"verification_{n}.json"
            if not v_path.exists():
                break
            try:
                with open(v_path, "r", encoding="utf-8") as f:
                    v = json.load(f)
                attempts.append({
                    "attempt": n,
                    "reward": v.get("reward", 0.0),
                    "hard_failures": v.get("hard_failures", []),
                    "clip_norm": v.get("clip_norm", 0.0),
                    "vqa_score": v.get("vqa_score", 0.0),
                })
            except Exception:
                break

        if attempts:
            run_data.append({
                "prompt": prompt,
                "config_name": config_name,
                "attempts": attempts,
                "run_id": row[0],
            })

    db.close()
    return run_data


def simulate_tau(run_data, tau: float, max_repair_loops: int = 3):
    """Simulate what would happen at a given τ threshold.

    For each run, walks through attempts in order. An attempt is accepted if:
    (1) no hard failures AND (2) reward >= τ.
    If not accepted and more attempts remain (up to max_repair_loops), continue.
    The final state (accepted or not) determines CSR contribution.
    """
    total = len(run_data)
    accepted = 0
    total_attempts = 0
    rewards = []

    for run in run_data:
        run_accepted = False
        last_reward = None
        used_attempts = 0

        for a in run["attempts"]:
            used_attempts += 1
            hard_ok = len(a["hard_failures"]) == 0
            reward = a["reward"]
            last_reward = reward

            if hard_ok and reward is not None and reward >= tau:
                run_accepted = True
                break

            # If this was the last allowed attempt, check if we'd accept on hard failures alone
            if used_attempts > max_repair_loops:
                break

        if run_accepted:
            accepted += 1
        if last_reward is not None:
            rewards.append(last_reward)
        total_attempts += used_attempts

    csr = accepted / total if total > 0 else 0.0
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    mean_attempts = total_attempts / total if total > 0 else 0.0

    return {
        "tau": tau,
        "csr": csr,
        "accepted": accepted,
        "total": total,
        "mean_reward": mean_reward,
        "mean_attempts": mean_attempts,
    }


def simulate_hard_only(run_data, max_repair_loops: int = 3):
    """Simulate acceptance based ONLY on hard failures (no τ gate).
    This is the "reflect disabled" baseline."""
    total = len(run_data)
    accepted = 0
    total_attempts = 0

    for run in run_data:
        run_accepted = False
        used_attempts = 0

        for a in run["attempts"]:
            used_attempts += 1
            hard_ok = len(a["hard_failures"]) == 0

            if hard_ok:
                run_accepted = True
                break

            if used_attempts > max_repair_loops:
                break

        if run_accepted:
            accepted += 1
        total_attempts += used_attempts

    return {
        "tau": "hard_only",
        "csr": accepted / total if total > 0 else 0.0,
        "accepted": accepted,
        "total": total,
        "mean_attempts": total_attempts / total if total > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep τ threshold to find optimal CSR")
    parser.add_argument("--config", default=None,
                        help="Filter to specific config name")
    parser.add_argument("--tau-min", type=float, default=0.35,
                        help="Minimum τ to test (default: 0.35)")
    parser.add_argument("--tau-max", type=float, default=0.85,
                        help="Maximum τ to test (default: 0.85)")
    parser.add_argument("--step", type=float, default=0.025,
                        help="Step size for τ sweep (default: 0.025)")
    parser.add_argument("--apply", action="store_true",
                        help="Write the optimal τ to config.yaml")
    parser.add_argument("--csv", action="store_true",
                        help="Output as CSV for plotting")
    args = parser.parse_args()

    run_data = collect_run_data(config_filter=args.config)
    if not run_data:
        print("No benchmark runs found. Run benchmark.py first.")
        return

    configs_found = set(r["config_name"] for r in run_data)
    print(f"Loaded {len(run_data)} runs across configs: {sorted(configs_found)}")

    # Sweep τ values
    results = []
    tau = args.tau_min
    while tau <= args.tau_max + 1e-6:
        r = simulate_tau(run_data, tau)
        results.append(r)
        tau += args.step

    # Also compute hard-only baseline
    hard_only = simulate_hard_only(run_data)

    if args.csv:
        print("tau,csr,accepted,total,mean_reward,mean_attempts")
        for r in results:
            print(f"{r['tau']:.3f},{r['csr']:.3f},{r['accepted']},{r['total']},"
                  f"{r['mean_reward']:.4f},{r['mean_attempts']:.2f}")
    else:
        print(f"\n{'τ':>6}  {'CSR':>6}  {'Accept':>6}/{run_data[0]['config_name'] if run_data else 'total':>6}"
              f"  {'AvgR':>6}  {'AvgAttempts':>11}")
        print("-" * 55)

        # Hard-only baseline
        print(f"{'hard':>6}  {hard_only['csr']:>5.0%}  "
              f"{hard_only['accepted']:>6}/{hard_only['total']:<6}  "
              f"{'—':>6}  {hard_only['mean_attempts']:>11.2f}  ◀ no τ gate")

        best = max(results, key=lambda r: r["csr"])
        for r in results:
            marker = " ◀ OPTIMAL" if r["tau"] == best["tau"] else ""
            current = " ◀ CURRENT" if abs(r["tau"] - 0.730) < 0.01 else ""
            print(f"{r['tau']:>6.3f}  {r['csr']:>5.0%}  "
                  f"{r['accepted']:>6}/{r['total']:<6}  "
                  f"{r['mean_reward']:>6.4f}  {r['mean_attempts']:>11.2f}{marker}{current}")

        print(f"\n--- Summary ---")
        print(f"Hard-only baseline CSR: {hard_only['csr']:.0%}")
        print(f"Current τ=0.730 CSR:    {simulate_tau(run_data, 0.730)['csr']:.0%}")
        print(f"Optimal τ={best['tau']:.3f} CSR:   {best['csr']:.0%}")

        if best["csr"] < hard_only["csr"]:
            print(f"\n⚠ WARNING: Even the optimal τ ({best['csr']:.0%}) is worse than "
                  f"hard-only ({hard_only['csr']:.0%}).")
            print("  This means the τ reward gate is actively hurting CSR.")
            print("  Consider: (a) disabling τ gate, (b) lower τ, or (c) fixing edit quality.")

        if args.apply and best["csr"] >= hard_only["csr"]:
            cfg_path = PROJECT_ROOT / "config.yaml"
            text = cfg_path.read_text(encoding="utf-8")
            import re
            text = re.sub(r"tau:\s+[\d.]+", f"tau: {best['tau']:.3f}", text)
            text = re.sub(r"tau_source:\s+\w+", "tau_source: sweep", text)
            cfg_path.write_text(text, encoding="utf-8")
            print(f"\n✓ Applied to config.yaml: tau={best['tau']:.3f}, tau_source=sweep")
        elif args.apply:
            print(f"\n✗ NOT applied: optimal τ CSR ({best['csr']:.0%}) < hard-only CSR ({hard_only['csr']:.0%}).")
            print("  Fix the edit quality first before enabling τ gating.")


if __name__ == "__main__":
    main()
