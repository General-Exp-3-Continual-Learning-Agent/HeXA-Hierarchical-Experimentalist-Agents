"""Direct-answer baseline for Interphyre with an OpenAI-compatible actor
(hosted API or local vLLM).

Same protocol as run_direct_answer_claude.py -- one scripted get_level_state
peek, one model call, parse JSON, simulate -- but through
react_agent.react_agent.load_openai_compatible_model instead of the `claude -p`
CLI. Every prompt template, level blurb, and file-format helper is imported
from the Claude module rather than duplicated, so a fix there (e.g. a new
level's LEVEL_BLURBS entry) applies here too without a second edit.

Usage (hosted API, e.g. gpt-6-astra on api.openai.com):
    export OPENAI_API_KEY=...   # or source an env file that sets it
    python -m react_agent.run_direct_answer_openai \
        --level catapult --variant cot --model gpt-6-astra \
        --base-url https://api.openai.com/v1 --seeds $(seq 57 106)

Usage (local vLLM, no key needed):
    python -m react_agent.run_direct_answer_openai \
        --level catapult --model Qwen/Qwen2.5-7B-Instruct \
        --base-url http://localhost:8000/v1 --seeds $(seq 57 106)
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "interphyre"))

from react_agent.react_agent import load_openai_compatible_model
from react_agent.run_direct_answer_claude import (
    LEVEL_BLURBS,
    run_single_seed_direct,
    save_trajectory,
    save_aggregate,
)


def main():
    p = argparse.ArgumentParser(
        description="Direct-answer baseline (no ReAct) for Interphyre with an "
                     "OpenAI-compatible actor",
    )
    p.add_argument("--level", type=str, default="catapult",
                   choices=list(LEVEL_BLURBS.keys()),
                   help="Add a level by extending LEVEL_BLURBS in run_direct_answer_claude.py.")
    p.add_argument(
        "--variant", type=str, default="cot", choices=["no_cot", "cot"],
        help="no_cot = JSON-only answer, no reasoning. cot = 2-3 sentences then JSON.",
    )
    p.add_argument(
        "--model", type=str, required=True,
        help="Model id passed to the API, e.g. gpt-6-astra or Qwen/Qwen2.5-7B-Instruct. "
             "Also used as the filesystem-safe label in output filenames.",
    )
    p.add_argument(
        "--base-url", type=str,
        default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible endpoint. Defaults to $OPENAI_BASE_URL, else local vLLM.",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-new-tokens", type=int, default=700)
    p.add_argument(
        "--reasoning-effort", type=str, default=os.environ.get("OPENAI_REASONING_EFFORT"),
        choices=["low", "medium", "high"],
        help="Reasoning-family models only (o1/o3/o4, gpt-5.x, gpt-6.x); ignored "
             "otherwise. Defaults to $OPENAI_REASONING_EFFORT; if that's also "
             "unset, uses the API's own implicit default -- confirmed non-zero "
             "on a real prompt against gpt-6-astra, not 'off'. An explicit CLI "
             "value here always wins over the env var.",
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--eval-dir", type=str, default=None,
        help="Where to save trajectories + summary. Defaults to "
             "eval_results_<level>_<model>_direct_<variant>[_<reasoning_effort>].",
    )
    p.add_argument(
        "--resume", action="store_true", default=True,
        help="(Default ON) Skip seeds whose trajectory file already exists in --eval-dir.",
    )
    p.add_argument("--no-resume", dest="resume", action="store_false")
    args = p.parse_args()

    if args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [42]
    args.seeds = seeds

    # Trajectory/summary filenames embed args.model directly (see
    # run_direct_answer_claude.save_trajectory) -- keep it filesystem-safe.
    raw_model = args.model
    args.model = raw_model.replace("/", "_")

    if not args.eval_dir:
        suffix = f"_{args.reasoning_effort}" if args.reasoning_effort else ""
        args.eval_dir = f"eval_results_{args.level}_{args.model}_direct_{args.variant}{suffix}"

    print(
        f"[Direct-Answer/OpenAI] level={args.level} variant={args.variant} "
        f"model={raw_model} base_url={args.base_url} "
        f"reasoning_effort={args.reasoning_effort or '<api default>'} "
        f"seeds={seeds[0]}-{seeds[-1]} ({len(seeds)}) eval_dir={args.eval_dir}"
    )

    model_fn = load_openai_compatible_model(
        raw_model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        reasoning_effort=args.reasoning_effort,
    )

    completed = set()
    results = []
    if args.resume and args.eval_dir:
        for fp in sorted(glob.glob(os.path.join(args.eval_dir, f"trajectory_seed*_raw{args.model}.json"))):
            with open(fp) as f:
                prior = json.load(f)
            sd = prior.get("seed")
            if sd is not None:
                completed.add(sd)
                results.append({
                    "seed": sd,
                    "success": prior["success"],
                    "action": tuple(prior["action"]) if prior["action"] else None,
                    "iterations": prior["iterations"],
                    "elapsed_time": prior.get("elapsed_time"),
                    "trajectory": [
                        (t["thought"], t["action"], t["observation"])
                        for t in prior.get("trajectory", [])
                    ],
                    "final_observation": "",
                    "usage": prior.get("usage"),
                })
        if completed:
            print(f"[Resume] Skipping {len(completed)} completed seeds: {sorted(completed)}")

    for seed in seeds:
        if seed in completed:
            print(f"[Resume] seed {seed} already done — skipping.")
            continue
        r = run_single_seed_direct(model_fn, seed, args)
        results.append(r)
        save_trajectory(r, args.eval_dir, args.model)
        save_aggregate(results, args.eval_dir, args)

    n = len(results)
    succ = sum(1 for r in results if r["success"])
    print(f"\n{'='*60}\n  DIRECT-ANSWER SUMMARY — {args.level} / {raw_model}\n{'='*60}")
    print(f"  Total seeds   : {n}")
    print(f"  Success rate  : {succ}/{n}" + (f" ({100*succ/n:.1f}%)" if n else ""))
    for r in sorted(results, key=lambda x: x["seed"]):
        status = "✓" if r["success"] else "✗"
        print(f"    {status} seed {r['seed']:<4} action={r['action']}")


if __name__ == "__main__":
    main()
