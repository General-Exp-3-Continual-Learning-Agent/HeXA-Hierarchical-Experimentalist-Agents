"""
CLI entry point for the ReAct agent using Claude Code CLI.

Usage:
    python -m react_agent.run_react_claude --level down_to_earth --seed 42 --verbose
"""

import argparse
import json
import os
import sys
import time
import subprocess

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "interphyre"))

from react_agent.react_agent import ReactAgent
from react_agent.tools import InterphyreToolkit
from react_agent.level_prompts import build_system_prompt, build_initial_user_message
import cv2


def load_claude_cli_model():
    """
    Returns a callable `model_fn` that is compatible with `ReactAgent`
    and calls the `claude -p` CLI command.
    """
    def generate(messages: list, temp: float = 0.3, max_tokens: int = 800) -> str:
        # Convert messages to a single flat prompt
        prompt = ""
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            prompt += f"{role}:\n{content}\n\n"
        prompt += "USER:\nIMPORTANT: Output ONLY your next single Thought and Action (and Action Input if needed). Do NOT write any Observation or additional steps. Stop after the Action Input.\n\nASSISTANT:\n"

        print(f"\n[Claude CLI] Prompt length: {len(prompt)} — waiting for response...", flush=True)
        call_start = time.time()
        try:
            result = subprocess.run(
    [
                 "claude", "-p", prompt,
                    "--model", "claude-sonnet-4-6",
                    "--max-turns", "1",
                    "--tools", "",
                    "--no-session-persistence",
                    "--output-format", "text",

    ],
    capture_output=True,
    text=True,
    env=os.environ.copy(),
    timeout=2000,
)
            elapsed = time.time() - call_start

            if result.stderr:
                print(f"[Claude stderr]: {result.stderr.strip()}", flush=True)
            if result.returncode != 0:
                err_out = (result.stderr or "").strip()
                std_out = (result.stdout or "").strip()
                print(
                    f"[Claude Error] Exit code {result.returncode} after {elapsed:.1f}s. "
                    f"stderr: {err_out[:1000] if err_out else 'none'} | "
                    f"stdout (first 1500): {std_out[:1500] if std_out else 'none'}",
                    flush=True,
                )
                return "Thought: I encountered an error communicating with Claude CLI.\nAction: finish\nAction Input: {}"

            print(f"[Claude CLI] Response received in {elapsed:.1f}s", flush=True)
            return result.stdout.strip()

        except FileNotFoundError:
             print("[Claude Error] The 'claude' CLI command was not found. Install: https://claude.ai/code", flush=True)
             return "Thought: I encountered an error communicating with Claude CLI.\nAction: finish\nAction Input: {}"
        except subprocess.TimeoutExpired:
             print(f"[Claude Error] Claude CLI timed out after {time.time() - call_start:.1f}s.", flush=True)
             return "Thought: I encountered an error communicating with Claude CLI.\nAction: finish\nAction Input: {}"
        except Exception as e:
             print(f"[Claude Error] Unexpected after {time.time() - call_start:.1f}s: {type(e).__name__}: {e}", flush=True)
             return "Thought: I encountered an error communicating with Claude CLI.\nAction: finish\nAction Input: {}"

    return generate


def run_single_seed(model_fn, seed: int, args) -> dict:
    """Run the ReAct agent on a single seed."""
    print(f"\n{'#'*60}")
    print(f"  Seed: {seed}")
    print(f"{'#'*60}")

    # Claude models are never OSS
    toolkit = InterphyreToolkit(level_name=args.level, seed=seed, is_oss=False)

    # Print full prompt before model is loaded
    system_prompt = build_system_prompt(args.level)
    initial_message = build_initial_user_message(args.level)
    # print(f"\n{'='*60}")
    # print("  FULL PROMPT")
    # print(f"{'='*60}")
    # print("\n--- SYSTEM PROMPT ---")
    # print(system_prompt)
    # print("\n--- INITIAL USER MESSAGE ---")
    # print(initial_message)
    # print(f"\n{'='*60}\n")

    agent = ReactAgent(
        model_fn=model_fn,
        toolkit=toolkit,
        level_name=args.level,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    # Save initial state image if evaluation directory is provided
    if args.eval_dir:
        os.makedirs(args.eval_dir, exist_ok=True)
        img_initial = toolkit.get_image()
        cv2.imwrite(os.path.join(args.eval_dir, f"seed{seed}_raw{args.model}_initial.png"), img_initial)

    start_time = time.time()
    result = agent.solve()
    elapsed = time.time() - start_time

    # Save final state image if evaluation directory is provided
    if args.eval_dir:
        img_final = toolkit.get_image()
        cv2.imwrite(os.path.join(args.eval_dir, f"seed{seed}_raw{args.model}_final.png"), img_final)

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


def save_trajectory(result: dict, log_dir: str, model: str):
    """Save a single trajectory file for one result."""
    os.makedirs(log_dir, exist_ok=True)
    traj_path = os.path.join(log_dir, f"trajectory_seed{result['seed']}_raw{model}.json")
    traj_data = {
        "seed": result["seed"],
        "success": result["success"],
        "action": result["action"],
        "iterations": result["iterations"],
        "elapsed_time": result.get("elapsed_time"),
        "trajectory": [
            {"thought": t, "action": a, "observation": o}
            for t, a, o in result.get("trajectory", [])
        ],
    }
    with open(traj_path, "w") as f:
        json.dump(traj_data, f, indent=2)
    print(f"Trajectory saved to: {os.path.abspath(traj_path)}")


def save_results(results: list, log_dir: str, args):
    """Save summary JSON to the log directory."""
    os.makedirs(log_dir, exist_ok=True)

    all_seeds = sorted(r["seed"] for r in results)
    summary = {
        "model": args.model,
        "level": args.level,
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

    summary_path = os.path.join(log_dir, f"summary_{args.model}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="ReAct agent for solving Interphyre physics puzzles with Claude Code CLI"
    )
    parser.add_argument(
        "--level", type=str, default="down_to_earth",
        choices=["down_to_earth", "two_body_problem", "catapult", "falling_into_place", "basket_case", "pass_the_parcel"],
    )
    parser.add_argument(
        "--model", type=str, default="claude",
        help="Model name (purely for logging since Claude Code chooses the model automatically)"
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
        "--max-iterations", type=int, default=25,
        help="Max ReAct iterations (safety guardrail)"
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
        help="Directory to save evaluation results including summary JSON and images"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from a previous interrupted run. Seeds with existing trajectory files will be skipped."
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

    # Load model wrapper
    model_fn = load_claude_cli_model()

    # --- Resume Logic ---
    # On --resume, load any previously completed trajectory files and skip those seeds.
    results = []
    completed_seeds = set()

    if args.resume and args.eval_dir:
        import glob
        pattern = os.path.join(args.eval_dir, f"trajectory_seed*_raw{args.model}.json")
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
                    "elapsed_time": None,
                    "final_observation": "",
                })
        if completed_seeds:
            print(f"\n[Resume] Found trajectory files in: {os.path.abspath(args.eval_dir)}")
            print(f"[Resume] Skipping {len(completed_seeds)} already completed seed(s): {sorted(completed_seeds)}")

    # Run on each seed
    for seed in seeds:
        if seed in completed_seeds:
            print(f"\n[Resume] Seed {seed} already completed — skipping.")
            continue

        result = run_single_seed(model_fn, seed, args)
        results.append(result)

        # Save incrementally: only write this seed's trajectory + update summary
        if args.eval_dir:
            save_trajectory(result, args.eval_dir, args.model)
            save_results(results, args.eval_dir, args)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"  Model: {args.model}")
    print(f"  Seeds: {seeds}")
    print(f"  Success rate: {sum(1 for r in results if r['success'])}/{len(results)}")
    if results:
        avg_iter = sum(r["iterations"] for r in results) / len(results)
        print(f"  Avg iterations: {avg_iter:.1f}")

    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"    {status} Seed {r['seed']}: action={r['action']}, iters={r['iterations']}")


if __name__ == "__main__":
    main()
