"""Reflexion-style baseline runner for Interphyre puzzles using the Claude CLI.

Wraps the existing run_react_claude pipeline with a per-seed K-trial loop that
uses verbal self-reflection between trials (Shinn et al. 2023,
https://arxiv.org/pdf/2303.11366).

Usage:
    python -m react_agent.run_react_claude_reflexion \\
        --level catapult --seeds 10 --k-trials 2 --verbose

Both the actor and the reflection step use `claude -p` (no Anthropic SDK,
no API key — same auth as run_react_claude.py).
"""

import argparse
import glob
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "interphyre"))

from react_agent.run_react_claude import load_claude_cli_model
from react_agent.reflexion.runner import run_seed_with_reflexion


def save_aggregate_summary(results: list, log_dir: str, args) -> None:
    os.makedirs(log_dir, exist_ok=True)

    n = len(results)
    successes = sum(1 for r in results if r.get("success"))

    trial_of_success_dist: dict = {}
    for r in results:
        rs = r.get("reflexion_summary", {})
        t = rs.get("trial_of_success")
        if t is not None:
            trial_of_success_dist[t] = trial_of_success_dist.get(t, 0) + 1

    summary = {
        "model": args.model,
        "level": args.level,
        "k_trials": args.k_trials,
        "reflection_model": args.reflection_model,
        "max_iterations": args.max_iterations,
        "temperature": args.temperature,
        "seeds": sorted([r["seed"] for r in results]),
        "total_seeds": n,
        "successes": successes,
        "success_rate": (successes / n) if n else 0.0,
        "avg_total_iterations": (sum(r["iterations"] for r in results) / n) if n else 0,
        "trial_of_success_distribution": trial_of_success_dist,
        "results": [],
    }
    for r in sorted(results, key=lambda x: x["seed"]):
        rs = r.get("reflexion_summary", {})
        summary["results"].append({
            "seed": r["seed"],
            "success": r["success"],
            "trial_of_success": rs.get("trial_of_success"),
            "total_iterations": r["iterations"],
            "total_elapsed_time": r.get("elapsed_time"),
            "successful_action": rs.get("successful_action"),
        })

    path = os.path.join(log_dir, f"summary_{args.model}_reflexion.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAggregate Reflexion summary saved → {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Reflexion-style ReAct baseline for Interphyre with Claude CLI",
    )
    parser.add_argument(
        "--level", type=str, default="catapult",
        choices=[
            "down_to_earth", "two_body_problem", "catapult", "falling_into_place",
            "basket_case", "pass_the_parcel", "tipping_point", "cliffhanger",
        ],
    )
    parser.add_argument(
        "--model", type=str, default="claude",
        help="Logging label for the actor model (Claude CLI selects the actual model).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--max-iterations", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--eval-dir", type=str, default=None,
        help="Where to save trajectory + summary files. "
             "Defaults to eval_results_<level>_claude_reflexion.",
    )
    parser.add_argument(
        "--k-trials", type=int, default=2,
        help="Number of Reflexion trials per seed (default 2 = 1 reflection between trials).",
    )
    parser.add_argument(
        "--reflection-model", type=str, default="claude-sonnet-4-6",
        help="Model used for the self-reflection step (passed to claude -p --model).",
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="(Default ON) Skip fully-completed seeds and reload completed trials "
             "from disk. Mid-seed resume (e.g. trial 1 done but trial 2 killed) is "
             "always on regardless of this flag — the trial-level resume inside "
             "the per-seed runner looks at the trajectory file directly.",
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Force a clean re-run: do not skip seeds with existing summaries. "
             "(Existing trial trajectory files will still be loaded by the "
             "per-seed runner — delete them manually to force per-trial re-runs.)",
    )

    args = parser.parse_args()

    if args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [42]
    args.seeds = seeds

    if not args.eval_dir:
        args.eval_dir = f"eval_results_{args.level}_claude_reflexion"

    print(
        f"[Reflexion Runner] level={args.level} k_trials={args.k_trials} "
        f"reflection_model={args.reflection_model} seeds={seeds} eval_dir={args.eval_dir}"
    )

    model_fn = load_claude_cli_model()

    # Resume: load any seed*_reflexion_summary.json already on disk and skip those seeds
    completed_seeds: set = set()
    results: list = []
    if args.resume and args.eval_dir:
        pattern = os.path.join(args.eval_dir, "seed*_reflexion_summary.json")
        for fp in sorted(glob.glob(pattern)):
            with open(fp) as f:
                summary = json.load(f)
            sd = summary.get("seed")
            if sd is None:
                continue
            completed_seeds.add(sd)
            results.append({
                "seed": sd,
                "success": summary.get("final_success", False),
                "action": summary.get("successful_action"),
                "iterations": summary.get("total_iterations", 0),
                "elapsed_time": summary.get("total_elapsed_time"),
                "trajectory": [],
                "final_observation": "",
                "reflexion_summary": summary,
            })
        if completed_seeds:
            print(
                f"[Resume] Found completed summaries in: {os.path.abspath(args.eval_dir)}\n"
                f"[Resume] Skipping {len(completed_seeds)} seed(s): {sorted(completed_seeds)}"
            )

    for seed in seeds:
        if seed in completed_seeds:
            print(f"\n[Resume] Seed {seed} already completed — skipping.")
            continue

        result = run_seed_with_reflexion(
            seed=seed,
            model_fn=model_fn,
            args=args,
            k_trials=args.k_trials,
            reflection_model=args.reflection_model,
        )
        results.append(result)

        save_aggregate_summary(results, args.eval_dir, args)

    # Final overview
    print(f"\n{'='*60}")
    print(f"  REFLEXION RUN SUMMARY")
    print(f"{'='*60}")
    print(f"  Level         : {args.level}")
    print(f"  K trials      : {args.k_trials}")
    print(f"  Reflect model : {args.reflection_model}")
    print(f"  Total seeds   : {len(results)}")
    succ = sum(1 for r in results if r['success'])
    print(f"  Success rate  : {succ}/{len(results)}"
          + (f" ({100*succ/len(results):.1f}%)" if results else ""))
    print()
    for r in sorted(results, key=lambda x: x["seed"]):
        rs = r.get("reflexion_summary", {})
        ts = rs.get("trial_of_success")
        status = f"✓ trial {ts}" if r["success"] else "✗"
        print(f"    {status:>10}  seed {r['seed']:<4} total_iters={r['iterations']}")


if __name__ == "__main__":
    main()
