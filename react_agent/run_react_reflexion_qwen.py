"""Reflexion baseline for Interphyre with an OpenAI-compatible actor (local vLLM).

The Claude entry point (`run_react_claude_reflexion.py`) drives both the actor and
the reflection through the `claude -p` CLI. This module is the same K-trial loop
against a served open-weight model instead, so Reflexion can be measured on the
identical actor that produces the HeXA and ReAct numbers on this level.

Everything method-side is shared: `reflexion/runner.run_seed_with_reflexion` and
`reflexion/reflect.reflect_on_trajectory` are the same code the Claude arm uses.
Only the two model handles differ, and BOTH point at the same served model --
actor and critic are one frozen model, which is what makes this self-improvement
rather than distillation from a stronger critic.

Budget note, to be carried into any writeup: K=2 gives Reflexion two attempts per
seed where HeXA and ReAct get one. That is Reflexion as published, and it is the
convention the existing Claude arm already uses, but it is NOT budget-matched.

Usage:
    python -m react_agent.run_react_reflexion_qwen \
        --level down_to_earth --seeds $(seq 57 106) \
        --model openai:Qwen/Qwen2.5-7B-Instruct \
        --eval-dir <dir> --k-trials 2 --verbose
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "interphyre"))

from react_agent.react_agent import load_openai_compatible_model
from react_agent.reflexion.runner import run_seed_with_reflexion
from react_agent.run_react_claude_reflexion import save_aggregate_summary


def main():
    parser = argparse.ArgumentParser(
        description="Reflexion baseline for Interphyre with a served OpenAI-compatible model",
    )
    parser.add_argument(
        "--level", type=str, default="down_to_earth",
        choices=[
            "down_to_earth", "two_body_problem", "catapult", "falling_into_place",
            "basket_case", "pass_the_parcel", "tipping_point", "cliffhanger",
        ],
    )
    parser.add_argument(
        "--model", type=str, default="openai:Qwen/Qwen2.5-7B-Instruct",
        help="Served model, `openai:<served-name>`. Also used for the reflection step.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    # Default matches the Claude catapult Reflexion convention: 12 iterations
    # PER TRIAL, so K=2 totals 24, inside the 25-total per-seed cap every other
    # arm (HeXA, ReAct) uses for its single attempt. Confirmed against
    # skillrl/RESULTS/catapult_claude/Reflexion/*/seed10_reflexion_summary.json.
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-new-tokens", type=int, default=800)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--eval-dir", type=str, default=None)
    parser.add_argument("--k-trials", type=int, default=2)
    parser.add_argument(
        "--reflection-model", type=str, default=None,
        help="Defaults to --model, i.e. the actor reflects on itself.",
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--max", action="store_true", help="Accepted for parity; unused.")

    args = parser.parse_args()

    if args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [42]
    args.seeds = seeds

    # Actor and critic are the same served model unless explicitly split.
    if not args.reflection_model:
        args.reflection_model = args.model

    # The runner names trajectory files `..._raw{args.model}.json`, so a raw
    # `openai:Qwen/Qwen2.5-7B-Instruct` would put a slash in the filename and the
    # open() would fail on a directory that does not exist. Keep the served name
    # for the model call and a filesystem-safe label for paths.
    served = args.model
    args.model = served.split(":", 1)[1].split("/")[-1] if ":" in served else served

    if not args.eval_dir:
        args.eval_dir = f"eval_results_{args.level}_{args.model}_reflexion"

    print(
        f"[Reflexion/vLLM] level={args.level} actor={served} "
        f"critic={args.reflection_model} k_trials={args.k_trials} "
        f"seeds={seeds[0]}-{seeds[-1]} ({len(seeds)}) eval_dir={args.eval_dir}"
    )

    model_fn = load_openai_compatible_model(
        served.split(":", 1)[1] if ":" in served else served,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

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
            print(f"[Resume] Skipping {len(completed_seeds)} completed seed(s).")

    for seed in seeds:
        if seed in completed_seeds:
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

    n = len(results)
    succ = sum(1 for r in results if r["success"])
    print(f"\n{'='*60}\n  REFLEXION SUMMARY — {args.level} / {args.model}\n{'='*60}")
    print(f"  seeds {n}   success {succ}/{n}" + (f" ({100*succ/n:.1f}%)" if n else ""))


if __name__ == "__main__":
    main()
