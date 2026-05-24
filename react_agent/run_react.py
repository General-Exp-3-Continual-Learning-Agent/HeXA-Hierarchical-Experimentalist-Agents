"""
CLI entry point for the ReAct agent.

Usage:
    python -m react_agent.run_react --model "Qwen2.5-7B-Instruct" --seed 42 --verbose
    python -m react_agent.run_react --model mock --seed 42 --verbose  # dry run
"""

import argparse
import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "interphyre"))

from react_agent.react_agent import (
    ReactAgent,
    MockModel,
    load_qwen_model,
    load_qwen3_thinking_model,
    is_thinking_model,
    load_gpt_oss_model,
    load_openai_compatible_model,
)
from react_agent.tools import InterphyreToolkit
import cv2


def run_single_seed(model_fn, seed: int, args) -> dict:
    """Run the ReAct agent on a single seed."""
    print(f"\n{'#'*60}")
    print(f"  Seed: {seed}")
    print(f"{'#'*60}")

    is_oss = "gpt-oss" in getattr(args, "model", "").lower()
    toolkit = InterphyreToolkit(level_name=args.level, seed=seed, is_oss=is_oss)

    agent = ReactAgent(
        model_fn=model_fn,
        toolkit=toolkit,
        level_name=args.level,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        is_oss=is_oss,
    )

    # Save initial state image if evaluation directory is provided
    if args.eval_dir:
        os.makedirs(args.eval_dir, exist_ok=True)
        img_initial = toolkit.get_image()
        cv2.imwrite(os.path.join(args.eval_dir, f"seed{seed}_initial.png"), img_initial)

    start_time = time.time()
    result = agent.solve()
    elapsed = time.time() - start_time

    # Save final state image if evaluation directory is provided
    if args.eval_dir:
        img_final = toolkit.get_image()
        cv2.imwrite(os.path.join(args.eval_dir, f"seed{seed}_final.png"), img_final)

    result["seed"] = seed
    result["elapsed_time"] = elapsed

    print(f"\n{'='*60}")
    print(f"  Result for seed {seed}")
    print(f"{'='*60}")
    print(f"  Success: {result['success']}")
    print(f"  Action:  {result['action']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Time: {elapsed:.1f}s")
    if result.get("final_observation"):
        print(f"  Final: {result['final_observation'][:200]}")

    toolkit.close()

    return result


def save_results(results: list, log_dir: str, args):
    """Save results and trajectories to the log directory."""
    os.makedirs(log_dir, exist_ok=True)

    # Summary
    summary = {
        "model": args.model,
        "level": args.level,
        "seeds": args.seeds,
        "max_iterations": args.max_iterations,
        "temperature": args.temperature,
        "total_seeds": len(results),
        "successes": sum(1 for r in results if r["success"]),
        "success_rate": sum(1 for r in results if r["success"]) / len(results) if results else 0,
        "avg_iterations": sum(r["iterations"] for r in results) / len(results) if results else 0,
        "results": [],
    }

    for r in results:
        summary["results"].append({
            "seed": r["seed"],
            "success": r["success"],
            "action": r["action"],
            "iterations": r["iterations"],
            "elapsed_time": r.get("elapsed_time"),
        })

    summary_path = os.path.join(log_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    # Save trajectories
    for r in results:
        traj_path = os.path.join(log_dir, f"trajectory_seed{r['seed']}.json")
        traj_data = {
            "seed": r["seed"],
            "success": r["success"],
            "action": r["action"],
            "iterations": r["iterations"],
            "elapsed_time": r.get("elapsed_time"),
            "avg_time_per_iteration": (
                r.get("elapsed_time") / r["iterations"]
                if r.get("elapsed_time") and r.get("iterations")
                else None
            ),
            "trajectory": [
                {"thought": t, "action": a, "observation": o}
                for t, a, o in r.get("trajectory", [])
            ],
        }
        with open(traj_path, "w") as f:
            json.dump(traj_data, f, indent=2)

    print(f"Trajectories saved to: {log_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="ReAct agent for solving Interphyre physics puzzles"
    )
    parser.add_argument(
        "--level", type=str, default="down_to_earth",
        choices=["down_to_earth", "two_body_problem", "falling_into_place", "catapult", "basket_case", "pass_the_parcel", "tipping_point", "cliffhanger"],
        help="Level to solve (default: down_to_earth)"
    )
    parser.add_argument(
        "--model", type=str, default="mock",
        help="Qwen model name (e.g. 'Qwen2.5-7B-Instruct') or 'mock' for testing"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Single level seed"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Multiple level seeds"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=10,
        help="Max ReAct iterations (safety guardrail)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=800,
        help="Max tokens per generation"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print full ReAct trace"
    )
    parser.add_argument(
        "--eval-dir", type=str, default=None,
        help="Directory to save evaluation results including summary JSON and images"
    )
    parser.add_argument(
        "--vllm-url", type=str, default=None,
        help="If set, call model via OpenAI-compatible API at this URL (e.g. http://localhost:8000/v1)"
    )

    args = parser.parse_args()

    # Determine seeds
    if args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [42]  # default seed
    args.seeds = seeds

    # Load model
    if args.model == "mock":
        print("Using mock model for testing...")
        mock = MockModel(level_name=args.level)
        model_fn = mock
    elif args.vllm_url:
        model_fn = load_openai_compatible_model(
            args.model,
            base_url=args.vllm_url,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
    elif is_thinking_model(args.model):
        model_fn = load_qwen3_thinking_model(
            args.model,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        model_fn = load_qwen_model(
            args.model,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )

    # Run on each seed
    results = []
    for seed in seeds:
        result = run_single_seed(model_fn, seed, args)
        results.append(result)
        
        # Save results incrementally if eval-dir specified
        if args.eval_dir:
            save_results(results, args.eval_dir, args)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"  Model: {args.model}")
    print(f"  Seeds: {seeds}")
    print(f"  Success rate: {sum(1 for r in results if r['success'])}/{len(results)}")
    avg_iter = sum(r["iterations"] for r in results) / len(results)
    print(f"  Avg iterations: {avg_iter:.1f}")

    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"    {status} Seed {r['seed']}: action={r['action']}, iters={r['iterations']}")


if __name__ == "__main__":
    main()
