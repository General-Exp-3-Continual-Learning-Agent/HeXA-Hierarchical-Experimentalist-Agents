"""
Standalone inference script for Interphyre.
Loads an HF checkpoint, runs the ReAct agent on specified seeds, and reports scores.

Usage:
    python examples/eval/interphyre_eval.py \
        --checkpoint <path/to/huggingface> \
        --level down_to_earth \
        --seed_start 51 --num_seeds 50 \
        --max_turns 25 \
        --output results.jsonl
"""

import argparse
import json
import os
import re
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from verl_tool.servers.tools.interphyre import _PhysicsToolkit
from examples.data_preprocess.interphyre_data import build_system_prompt, build_initial_user_message, load_skill_bank

STOP_STRING = "\nObservation:"
ACTION_RE = re.compile(r"Action:\s*([A-Za-z_][A-Za-z0-9_]*)", re.DOTALL)
INPUT_RE  = re.compile(r"Action Input:\s*(\{.*?\})", re.DOTALL | re.IGNORECASE)


class StopOnTokenSequence(StoppingCriteria):
    """Stop generation when a specific token sequence appears at the end of output."""
    def __init__(self, stop_ids: list, prompt_length: int):
        self.stop_ids = stop_ids
        self.n = len(stop_ids)
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[0][self.prompt_length:]
        if len(generated) >= self.n and generated[-self.n:].tolist() == self.stop_ids:
            return True
        return False


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

    done = False
    if tool_name == "finish":
        x, y, r = float(args.get("x", 0)), float(args.get("y", 0)), float(args.get("radius", 0.5))
        obs = "[FINAL_RESULT]\n" + toolkit.simulate_action(x, y, r)
        done = True
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
    return obs, done


def run_episode(model, tokenizer, stop_ids: list, level: str, seed: int, max_turns: int, device: str, skill_bank: dict = None) -> dict:
    system_prompt = build_system_prompt(level, skill_bank=skill_bank)
    user_msg = build_initial_user_message(level)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_msg},
    ]

    # Build the initial prompt as a flat string ending with <|im_start|>assistant\n
    # All subsequent turns are appended to this string directly — no re-wrapping.
    running_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    toolkit = _PhysicsToolkit(level, seed=seed)
    full_response = ""
    success = False
    n_turns = 0

    for _ in range(max_turns):
        input_ids = tokenizer(running_text, return_tensors="pt").input_ids.to(device)
        prompt_length = input_ids.shape[1]

        stopping_criteria = StoppingCriteriaList([
            StopOnTokenSequence(stop_ids, prompt_length)
        ])

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=512,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )

        new_tokens = output[0][prompt_length:]
        generated = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Truncate at the first \nObservation: — discard anything the model
        # hallucinated beyond that point within this single generation.
        if STOP_STRING in generated:
            generated = generated[:generated.index(STOP_STRING)]
        generated = generated.rstrip()

        full_response += generated
        n_turns += 1

        obs, done = call_tool(toolkit, generated)

        if done:
            success = "[FINAL_RESULT]" in obs and "SUCCESS" in obs
            full_response += f"\nObservation: {obs}\n"
            break

        # Append generated text + observation inline to the running flat string
        running_text += generated + f"\nObservation: {obs}\n"
        full_response += f"\nObservation: {obs}\n"

    return {"seed": seed, "success": success, "n_turns": n_turns, "response": full_response[-2000:]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--level", type=str, default="down_to_earth")
    parser.add_argument("--seed_start", type=int, default=51)
    parser.add_argument("--num_seeds", type=int, default=50)
    parser.add_argument("--max_turns", type=int, default=25)
    parser.add_argument("--output", type=str, default="eval_results.jsonl")
    parser.add_argument("--skill_bank", type=str, default=None,
                        help="Optional path to skill bank JSON; injects learned physics skills into the system prompt (matches what the skilled checkpoints saw at training time).")
    parser.add_argument("--skillrl_output_dir", type=str, default=None,
                        help="If set, write a SKILL-RL-compatible trajectory_seed{N}.json per seed to this directory (for use by the evolving skill bank pipeline).")
    args = parser.parse_args()

    skill_bank = load_skill_bank(args.skill_bank) if args.skill_bank else None
    if skill_bank:
        print(f"Loaded skill bank: {args.skill_bank}")

    if args.skillrl_output_dir:
        os.makedirs(args.skillrl_output_dir, exist_ok=True)
        print(f"SKILL-RL output dir: {args.skillrl_output_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {args.checkpoint} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()

    # Pre-encode the stop string once
    stop_ids = tokenizer.encode(STOP_STRING, add_special_tokens=False)
    print(f"Stop string '{STOP_STRING}' encodes to token ids: {stop_ids}")

    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    successes = 0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, "w") as f:
        for i, seed in enumerate(seeds):
            print(f"[{i+1}/{len(seeds)}] seed={seed}", end=" ", flush=True)
            result = run_episode(model, tokenizer, stop_ids, args.level, seed, args.max_turns, device, skill_bank=skill_bank)
            if result["success"]:
                successes += 1
            print(f"-> {'SUCCESS' if result['success'] else 'FAIL'} (turns={result['n_turns']})")
            f.write(json.dumps(result) + "\n")
            f.flush()

            if args.skillrl_output_dir:
                skillrl_record = {
                    "success":      result["success"],
                    "seed":         seed,
                    "iterations":   result["n_turns"],
                    "level":        args.level,
                    "trajectory":   [],
                    "raw_response": result["response"],
                }
                skillrl_path = os.path.join(args.skillrl_output_dir, f"trajectory_seed{seed}.json")
                with open(skillrl_path, "w") as sf:
                    json.dump(skillrl_record, sf, indent=2)

    print(f"\n=== Results ===")
    print(f"Level:   {args.level}")
    print(f"Seeds:   {args.seed_start} - {args.seed_start + args.num_seeds - 1}")
    print(f"Score:   {successes}/{len(seeds)} = {successes/len(seeds)*100:.1f}%")
    print(f"Saved:   {args.output}")


if __name__ == "__main__":
    main()
