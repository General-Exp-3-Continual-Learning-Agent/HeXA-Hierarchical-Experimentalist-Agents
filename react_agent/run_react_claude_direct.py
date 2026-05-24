"""Direct-answer baseline: ReAct with only 2 iterations (minimal reasoning).

This is a pure zero-shot baseline using Claude Sonnet 4.6 via CLI.
Strategy: Peek at scene once, then immediately commit an answer.

Usage:
    python -m react_agent.run_react_claude_direct \\
        --level pass_the_parcel \\
        --seeds 0 1 2 3 4 5 ... 49 \\
        --max-iterations 2 \\
        --verbose
"""

import argparse
import glob
import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "interphyre"))

from react_agent.react_agent import ReactAgent
from react_agent.tools import InterphyreToolkit
from react_agent.level_prompts import build_system_prompt, build_initial_user_message
from react_agent.run_react_claude import load_claude_cli_model, save_trajectory
import cv2


def run_single_seed(model_fn, seed: int, args) -> dict:
    """Run the ReAct agent on a single seed."""
    print(f"\n{'#'*60}")
    print(f"  Seed: {seed}")
    print(f"{'#'*60}")

    toolkit = InterphyreToolkit(level_name=args.level, seed=seed, is_oss=False)

    system_prompt = build_system_prompt(args.level)
    initial_message = build_initial_user_message(args.level)

    agent = ReactAgent(
        model_fn=model_fn,
        toolkit=toolkit,
        level_name=args.level,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    if args.eval_dir:
        os.makedirs(args.eval_dir, exist_ok=True)
        img_initial = toolkit.get_image()
        cv2.imwrite(
            os.path.join(
                args.eval_dir,
                f"seed{seed}_rawclaude_initial.png",
            ),
            img_initial,
        )

    start_time = time.time()
    result = agent.solve()
    elapsed = time.time() - start_time

    if args.eval_dir:
        img_final = toolkit.get_image()
        cv2.imwrite(
            os.path.join(
                args.eval_dir,
                f"seed{seed}_rawclaude_final.png",
            ),
            img_final,
        )

    result["seed"] = seed
    result["elapsed_time"] = elapsed

    toolkit.close()

    print(f"  Success: {result['success']}")
    print(f"  Action: {result['action']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Time: {elapsed:.1f}s")
    if result.get("final_observation"):
        print(f"  Final: {result['final_observation'][:200]}")

    return result


def save_results(results: list, log_dir: str, args):
    """Save summary JSON to the log directory."""
    os.makedirs(log_dir, exist_ok=True)

    all_seeds = sorted(r["seed"] for r in results)
    summary = {
        "model": args.model,
        "level": args.level,
        "strategy": "direct (2 iterations, minimal reasoning)",
        "seeds": all_seeds,
        "max_iterations": args.max_iterations,
        "temperature": args.temperature,
        "total_seeds": len(results),
        "successes": sum(1 for r in results if r["success"]),
        "success_rate": sum(1 for r in results if r["success"]) / len(results) if results else 0,
        "avg_iterations": sum(r["iterations"] for r in results) / len(results) if results else 0,
        "results": [],
    }

    for r in sorted(results, key=lambda x: x["seed"]):
        summary["results"].append({
            "seed": r["seed"],
            "success": r["success"],
            "action": r["action"],
            "iterations": r["iterations"],
            "elapsed_time": r.get("elapsed_time"),
        })

    summary_path = os.path.join(log_dir, f"summary_{args.model}_direct.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Direct-answer baseline: ReAct with 2 iterations (zero-shot ability)"
    )
    parser.add_argument(
        "--level", type=str, default="pass_the_parcel",
        choices=[
            "down_to_earth", "two_body_problem", "catapult", "falling_into_place",
            "basket_case", "pass_the_parcel", "tipping_point", "cliffhanger",
        ],
    )
    parser.add_argument(
        "--model", type=str, default="claude",
        help="Logging label for the model (Claude CLI selects the actual model)."
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--max-iterations", type=int, default=2,
        help="Max ReAct iterations (default 2 for direct baseline)."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=700,
        help="Max tokens per generation"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print full ReAct trace"
    )
    parser.add_argument(
        "--eval-dir", type=str, default=None,
        help="Directory to save results. Defaults to eval_results_<level>_claude_direct."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from a previous interrupted run."
    )

    args = parser.parse_args()

    # Determine seeds
    if args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [42]
    args.seeds = seeds

    # Set eval directory default
    if not args.eval_dir:
        args.eval_dir = f"eval_results_{args.level}_claude_direct"

    print(
        f"[Direct Baseline] level={args.level} max_iterations={args.max_iterations} "
        f"seeds={seeds} eval_dir={args.eval_dir}"
    )

    model_fn = load_claude_cli_model()

    # Resume logic
    results = []
    completed_seeds = set()

    if args.resume and args.eval_dir:
        pattern = os.path.join(args.eval_dir, f"trajectory_seed*_rawclaude.json")
        for fpath in sorted(glob.glob(pattern)):
            with open(fpath) as f:
                prior = json.load(f)
            seed_done = prior.get("seed")
            if seed_done is not None:
                completed_seeds.add(seed_done)
                results.append({
                    "seed": seed_done,
                    "success": prior["success"],
                    "action": prior["action"],
                    "iterations": prior["iterations"],
                    "trajectory": [(t["thought"], t["action"], t["observation"]) for t in prior.get("trajectory", [])],
                    "elapsed_time": prior.get("elapsed_time"),
                    "final_observation": "",
                })
        if completed_seeds:
            print(
                f"[Resume] Found trajectory files in: {os.path.abspath(args.eval_dir)}\n"
                f"[Resume] Skipping {len(completed_seeds)} seed(s): {sorted(completed_seeds)}"
            )

    for seed in seeds:
        if seed in completed_seeds:
            print(f"\n[Resume] Seed {seed} already completed — skipping.")
            continue

        result = run_single_seed(model_fn, seed, args)
        results.append(result)

        if args.eval_dir:
            save_trajectory(result, args.eval_dir, args.model)

        save_results(results, args.eval_dir, args)

    # Final overview
    print(f"\n{'='*60}")
    print(f"  DIRECT BASELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  Level         : {args.level}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Total seeds   : {len(results)}")
    succ = sum(1 for r in results if r["success"])
    print(f"  Success rate  : {succ}/{len(results)}"
          + (f" ({100*succ/len(results):.1f}%)" if results else ""))
    print()
    for r in sorted(results, key=lambda x: x["seed"]):
        status = "✓" if r["success"] else "✗"
        print(f"    {status}  seed {r['seed']:<4} iters={r['iterations']} time={r.get('elapsed_time', 0):.1f}s")


if __name__ == "__main__":
    main()
