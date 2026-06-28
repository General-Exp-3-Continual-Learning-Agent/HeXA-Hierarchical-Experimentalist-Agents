"""
Fast trajectory collection using vLLM batched inference.
Drop-in replacement for interphyre_eval.py for the evolving skill bank pipeline.
All seeds are batched together per turn — much faster than HF sequential inference.

Usage:
    python SKILL-RL/tools/vllm_eval.py \
        --checkpoint <path/to/huggingface> \
        --level down_to_earth \
        --seed_start 6 --num_seeds 3 \
        --max_turns 25 \
        --skill_bank SKILL-RL/Skill_banks/skill_bank_round_1.json \
        --skillrl_output_dir SKILL-RL/data/round_1/down_to_earth \
        --output logs/run/round_1/eval_seeds_6-8.jsonl
"""

import argparse
import json
import os
import re
import sys

from vllm import LLM, SamplingParams

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from verl_tool.servers.tools.interphyre import _PhysicsToolkit
from examples.data_preprocess.interphyre_data import (
    build_system_prompt, build_initial_user_message, load_skill_bank
)

STOP_STRING = "\nObservation:"
ACTION_RE = re.compile(r"Action:\s*([A-Za-z_][A-Za-z0-9_]*)", re.DOTALL)
INPUT_RE  = re.compile(r"Action Input:\s*(\{.*?\})", re.DOTALL | re.IGNORECASE)


def call_tool(toolkit: _PhysicsToolkit, action_text: str):
    """Parse and execute one tool call. Returns (observation, done)."""
    action_match = ACTION_RE.search(action_text)
    if not action_match:
        return "ERROR: Could not parse Action.", False

    tool_name = action_match.group(1).strip().lower()
    input_match = INPUT_RE.search(action_text)
    args = {}
    if input_match:
        try:
            raw = input_match.group(1)
            raw = re.sub(r'(\d+)\.\s*([,}\]])', r'\1.0\2', raw)
            raw = re.sub(r':\s*(\d+)\.\s*$', r': \1.0', raw)
            args = json.loads(raw)
        except json.JSONDecodeError as e:
            return f"ERROR: Invalid JSON: {e}", False

    if tool_name == "finish":
        x, y, r = float(args.get("x", 0)), float(args.get("y", 0)), float(args.get("radius", 0.5))
        obs = "[FINAL_RESULT]\n" + toolkit.simulate_action(x, y, r)
        return obs, True
    elif tool_name == "get_level_state":
        obs = toolkit.get_level_state()
    elif tool_name == "simulate_action":
        obs = toolkit.simulate_action(float(args.get("x", 0)), float(args.get("y", 0)), float(args.get("radius", 0.5)))
    elif tool_name == "simulate_partial":
        obs = toolkit.simulate_partial(float(args.get("x", 0)), float(args.get("y", 0)), float(args.get("radius", 0.5)), int(args.get("stop_step", 50)))
    elif tool_name == "get_contact_log":
        obs = toolkit.get_contact_log()
    elif tool_name == "compute_gap_analysis":
        obs = toolkit.compute_gap_analysis()
    elif tool_name == "compute_relative_positions":
        obs = toolkit.compute_relative_positions()
    elif tool_name == "compute_tipping_analysis":
        obs = toolkit.compute_tipping_analysis()
    elif tool_name == "compute_wall_distance_analysis":
        obs = toolkit.compute_wall_distance_analysis()
    elif tool_name == "compute_catapult_analysis":
        obs = toolkit.compute_catapult_analysis()
    else:
        obs = f"ERROR: Unknown tool '{tool_name}'."
    return obs, False


def run_episodes(llm, seeds, level, max_turns, skill_bank=None, temperature=0.0):
    """Run ReAct episodes for all seeds, batched per turn via vLLM."""
    tokenizer = llm.get_tokenizer()
    system_prompt = build_system_prompt(level, skill_bank=skill_bank)
    user_msg = build_initial_user_message(level)

    initial_prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user",   "content": user_msg}],
        add_generation_prompt=True, tokenize=False,
    )

    running_texts  = {s: initial_prompt for s in seeds}
    full_responses = {s: "" for s in seeds}
    toolkits       = {s: _PhysicsToolkit(level, seed=s) for s in seeds}
    done           = {s: False for s in seeds}
    success        = {s: False for s in seeds}
    n_turns        = {s: 0 for s in seeds}

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=512,
        stop=[STOP_STRING],
        include_stop_str_in_output=False,
    )

    for _ in range(max_turns):
        active = [s for s in seeds if not done[s]]
        if not active:
            break

        outputs = llm.generate([running_texts[s] for s in active], sampling_params)

        for seed, out in zip(active, outputs):
            generated = out.outputs[0].text.rstrip()
            full_responses[seed] += generated
            n_turns[seed] += 1

            obs, is_done = call_tool(toolkits[seed], generated)
            full_responses[seed] += f"\nObservation: {obs}\n"

            if is_done:
                success[seed] = "[FINAL_RESULT]" in obs and "SUCCESS" in obs
                done[seed] = True
            else:
                running_texts[seed] += generated + f"\nObservation: {obs}\n"

    return [
        {"seed": s, "success": success[s], "n_turns": n_turns[s],
         "response": full_responses[s][-2000:]}
        for s in seeds
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",            type=str,   required=True)
    parser.add_argument("--level",                 type=str,   default="down_to_earth")
    parser.add_argument("--seed_start",            type=int,   default=51)
    parser.add_argument("--num_seeds",             type=int,   default=50)
    parser.add_argument("--max_turns",             type=int,   default=25)
    parser.add_argument("--output",                type=str,   default="eval_results.jsonl")
    parser.add_argument("--skill_bank",            type=str,   default=None)
    parser.add_argument("--skillrl_output_dir",    type=str,   default=None)
    parser.add_argument("--temperature",           type=float, default=0.0)
    parser.add_argument("--gpu_memory_utilization",type=float, default=0.8)
    args = parser.parse_args()

    skill_bank = load_skill_bank(args.skill_bank) if args.skill_bank else None
    if skill_bank:
        print(f"Loaded skill bank: {args.skill_bank}")
    if args.skillrl_output_dir:
        os.makedirs(args.skillrl_output_dir, exist_ok=True)
        print(f"SKILL-RL output dir: {args.skillrl_output_dir}")

    print(f"Loading model via vLLM from {args.checkpoint}...")
    llm = LLM(
        model=args.checkpoint,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=8192,
    )

    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    print(f"Running {len(seeds)} seeds batched per turn...")
    results = run_episodes(llm, seeds, args.level, args.max_turns, skill_bank=skill_bank, temperature=args.temperature)

    successes = sum(r["success"] for r in results)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, "w") as f:
        for r in results:
            print(f"  seed={r['seed']} -> {'SUCCESS' if r['success'] else 'FAIL'} (turns={r['n_turns']})")
            f.write(json.dumps(r) + "\n")

            if args.skillrl_output_dir:
                path = os.path.join(args.skillrl_output_dir, f"trajectory_seed{r['seed']}.json")
                with open(path, "w") as sf:
                    json.dump({
                        "success":      r["success"],
                        "seed":         r["seed"],
                        "iterations":   r["n_turns"],
                        "level":        args.level,
                        "trajectory":   [],
                        "raw_response": r["response"],
                    }, sf, indent=2)

    print(f"\n=== Results ===")
    print(f"Level:  {args.level}")
    print(f"Seeds:  {args.seed_start} – {args.seed_start + args.num_seeds - 1}")
    print(f"Score:  {successes}/{len(seeds)} = {successes/len(seeds)*100:.1f}%")
    print(f"Saved:  {args.output}")


if __name__ == "__main__":
    main()
