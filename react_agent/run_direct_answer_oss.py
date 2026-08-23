"""Direct-answer baseline for Interphyre puzzles using OSS models (Qwen 7B/14B, GPT-OSS-20B).

Protocol per seed — no ReAct loop:
  1. Call `get_level_state` via toolkit (scripted, not model-driven).
  2. ONE model call.
  3. Parse JSON, run `simulate_action`, record success/failure.

Model-specific prompt strategy:
  Qwen 7B / 14B:
    Uses DIRECT_SYSTEM_TEMPLATE + USER_TEMPLATE (same as Claude direct baseline).
    apply_chat_template without tools — plain text answer expected.

  GPT-OSS-20B:
    Uses OSS system prompt (build_system_prompt is_oss=True) + native function-calling.
    Conversation has the get_level_state tool exchange pre-filled so the model
    only needs to emit one `finish` call with the coordinates.
    apply_chat_template with tools=get_oss_tools(level) — harmony format.

Usage:
    python -m react_agent.run_direct_answer_oss \\
        --model /datasets/ai/qwen2/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/... \\
        --model-tag qwen7b \\
        --level catapult --variant cot --seeds $(seq 0 49) --eval-dir eval_7b_catapult_direct
"""

import argparse
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "interphyre"))

import cv2

from react_agent.tools import InterphyreToolkit
from react_agent.level_prompts import (
    build_system_prompt,
    build_initial_user_message,
    get_oss_tools,
    OSS_TOOLS_SHARED,
)
from react_agent.react_agent import _parse_oss_native

# Direct-answer: only the `finish` tool is exposed so the model cannot re-call
# get_level_state / analysis / simulate_action. It MUST commit a placement.
_OSS_FINISH_ONLY_TOOLS = [
    t for t in OSS_TOOLS_SHARED if t["function"]["name"] == "finish"
]

# Forcing message appended after the scripted get_level_state result. This
# overrides the ReAct "start by calling get_level_state, then simulate..."
# framing in the OSS initial message — there is no loop in the direct baseline.
_OSS_FORCE_FINISH = (
    "You have now inspected the complete level state shown above. "
    "Do NOT call get_level_state, any analysis tool, or simulate_action — none "
    "are available. Reason briefly about the physics, then submit your FINAL "
    "red ball placement NOW by calling the finish function with concrete "
    "floating-point values for x, y, and radius, respecting every placement "
    "constraint."
)

# Shared prompts, parser, and save helpers with the Claude direct-answer baseline.
from react_agent.run_direct_answer_claude import (
    LEVEL_BLURBS,
    DIRECT_SYSTEM_TEMPLATE,
    USER_TEMPLATE,
    _ANSWER_BLOCKS,
    extract_placement,
    save_trajectory,
    save_aggregate,
)


_OSS_CONTROL_TOKEN_RE = re.compile(r"<\|[a-z_]+\|>")


def _wrap_forced_finish(raw: str) -> str:
    """The forced-channel generation emits only the finish() JSON args.

    Strip any trailing control tokens (<|call|>, <|end|>, ...) and return the
    args in ReAct format so extract_placement can parse them.
    """
    cleaned = _OSS_CONTROL_TOKEN_RE.sub(" ", raw).strip()
    # Grab the first balanced {...} JSON object.
    m = re.search(r"\{[^{}]*\}", cleaned)
    args = m.group(0) if m else cleaned
    return f"Thought: (direct finish)\nAction: finish\nAction Input: {args}"


# ─── Model loader ─────────────────────────────────────────────────────────────

def load_oss_model_direct(model_path: str, temperature: float = 0.3, max_new_tokens: int = 700):
    """Load an OSS model for direct-answer inference.

    Returns generate(messages, temp, max_tokens, tools=None).
    When tools is passed (GPT-OSS-20B), apply_chat_template injects tool schemas
    so the model can emit native function calls.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    is_gpt_oss = "gpt-oss" in model_path.lower()
    print(f"[load_oss_model_direct] Loading {'GPT-OSS' if is_gpt_oss else 'Qwen'}: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if is_gpt_oss:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto",
            attn_implementation="eager",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto",
        )
    print("[load_oss_model_direct] Model loaded.")

    _CALL_TOKEN_ID = 200012  # <|call|> — marks end of OSS tool call

    def generate(messages: list, temp: float = temperature,
                 max_tokens: int = max_new_tokens, tools=None,
                 dump_prompt: bool = False, force_finish: bool = False) -> str:
        if tools is not None:
            # GPT-OSS-20B native function-calling path.
            text = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
                model_identity="You are a physics puzzle solving assistant.",
                reasoning_effort="low",
            )
            if force_finish:
                # DIRECT baseline: prefill the assistant turn straight into the
                # finish() commentary channel. This skips the analysis channel
                # entirely — the model emits ONLY the JSON args (no reasoning
                # tokens, no chance to run out of budget mid-thought).
                # apply_chat_template ends the prompt at "<|start|>assistant".
                text += " to=functions.finish<|channel|>commentary json<|message|>"
            eos_ids = [tokenizer.eos_token_id, _CALL_TOKEN_ID]
        else:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            eos_ids = tokenizer.eos_token_id

        if dump_prompt:
            print("\n" + "=" * 72)
            print("  FULL PROMPT (verbatim apply_chat_template output)")
            print("=" * 72)
            print(text)
            print("=" * 72 + "\n")

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temp,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=eos_ids,
            )
        generated_ids = output[0][inputs["input_ids"].shape[1]:]

        if tools is not None:
            # Decode without stripping special tokens — control tokens are NOT
            # registered as special, so skip_special_tokens=True would drop them.
            raw = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
            if force_finish:
                # Generated text is just the finish() JSON args (+ <|call|>).
                return _wrap_forced_finish(raw)
            return _parse_oss_native(raw)
        else:
            return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    generate._is_gpt_oss = is_gpt_oss
    return generate


# ─── Seed runners ─────────────────────────────────────────────────────────────

def run_single_seed_qwen(model_fn, seed: int, args) -> dict:
    """Direct-answer for Qwen 7B/14B — same DIRECT_SYSTEM_TEMPLATE as Claude."""
    print(f"\n{'#'*60}")
    print(f"  Direct-Answer Qwen Seed: {seed}  (level={args.level})")
    print(f"{'#'*60}")

    blurb = LEVEL_BLURBS[args.level]
    toolkit = InterphyreToolkit(level_name=args.level, seed=seed, is_oss=False)

    if args.eval_dir:
        os.makedirs(args.eval_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(args.eval_dir, f"seed{seed}_raw{args.model}_initial.png"),
            toolkit.get_image(),
        )

    scene_obs = toolkit.dispatch_tool("get_level_state", {})
    if args.verbose:
        print(f"\n--- get_level_state ---\n{scene_obs}\n")

    system_prompt = DIRECT_SYSTEM_TEMPLATE.format(
        puzzle_name=blurb["name"],
        elements=blurb["elements"],
        goal=blurb["goal"],
    )
    answer_block = _ANSWER_BLOCKS[args.variant]
    user_msg = USER_TEMPLATE.format(scene_obs=scene_obs, answer_block=answer_block)
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    start = time.time()
    resp = model_fn(msgs, args.temperature, args.max_new_tokens)
    if args.verbose:
        print(f"--- ASSISTANT ---\n{resp}\n")

    return _finish_seed(toolkit, seed, args, scene_obs, resp, start)


def run_single_seed_gpt_oss(model_fn, seed: int, args) -> dict:
    """Direct-answer for GPT-OSS-20B — pre-fills get_level_state tool exchange,
    then one model call with native function-calling to emit finish({x,y,radius}).
    """
    print(f"\n{'#'*60}")
    print(f"  Direct-Answer GPT-OSS Seed: {seed}  (level={args.level})")
    print(f"{'#'*60}")

    toolkit = InterphyreToolkit(level_name=args.level, seed=seed, is_oss=True)

    if args.eval_dir:
        os.makedirs(args.eval_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(args.eval_dir, f"seed{seed}_raw{args.model}_initial.png"),
            toolkit.get_image(),
        )

    scene_obs = toolkit.dispatch_tool("get_level_state", {})
    if args.verbose:
        print(f"\n--- get_level_state ---\n{scene_obs}\n")

    # Pre-fill the get_level_state tool exchange so the scene is already in
    # context, then force a single finish call (no ReAct loop).
    msgs = [
        {"role": "system", "content": build_system_prompt(args.level, is_oss=True)},
        {"role": "user", "content": build_initial_user_message(args.level, is_oss=True)},
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_prefill_0",
                "type": "function",
                "function": {"name": "get_level_state", "arguments": "{}"},
            }],
        },
        {
            "role": "tool",
            "content": scene_obs,
            "tool_call_id": "call_prefill_0",
        },
        {"role": "user", "content": _OSS_FORCE_FINISH},
    ]

    # Expose ONLY the finish tool so the model cannot loop back to inspection.
    start = time.time()
    resp = model_fn(msgs, args.temperature, args.max_new_tokens,
                    tools=_OSS_FINISH_ONLY_TOOLS, dump_prompt=args.verbose,
                    force_finish=True)
    if args.verbose:
        print(f"--- ASSISTANT (parsed) ---\n{resp}\n")

    return _finish_seed(toolkit, seed, args, scene_obs, resp, start)


def _finish_seed(toolkit, seed: int, args, scene_obs: str, resp: str, start: float) -> dict:
    """Shared: parse placement from response, simulate, record result."""
    trajectory = [
        ("(direct-answer baseline: scripted peek)", "get_level_state({})", scene_obs),
    ]

    placement = extract_placement(resp)

    if placement is None:
        print("[Direct] Failed to parse JSON placement from response.")
        action = None
        sim_obs = "(parse failure — no JSON placement extracted from model output)"
        success = False
    else:
        x, y, r = placement
        action = (x, y, r)
        sim_obs = toolkit.dispatch_tool("simulate_action", {"x": x, "y": y, "radius": r})
        success = "SUCCESS!" in sim_obs

    final_action_str = (
        f'finish({{"x": {placement[0]}, "y": {placement[1]}, "radius": {placement[2]}}})'
        if placement else "finish({})"
    )
    trajectory.append((resp, final_action_str, sim_obs))
    elapsed = time.time() - start

    if args.eval_dir:
        cv2.imwrite(
            os.path.join(args.eval_dir, f"seed{seed}_raw{args.model}_final.png"),
            toolkit.get_image(),
        )

    print(f"  Success: {success}    Action: {action}    Elapsed: {elapsed:.1f}s")
    toolkit.close()

    return {
        "seed": seed,
        "success": success,
        "action": action,
        "iterations": 2,
        "elapsed_time": elapsed,
        "trajectory": trajectory,
        "final_observation": sim_obs,
    }


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Direct-answer baseline (no ReAct loop) for Interphyre with OSS models",
    )
    p.add_argument(
        "--level", type=str, default="catapult",
        choices=list(LEVEL_BLURBS.keys()),
    )
    p.add_argument(
        "--variant", type=str, default="cot", choices=["no_cot", "cot"],
        help="no_cot = JSON-only. cot = 2-3 sentence Thought then JSON. (Qwen only; OSS uses native format)",
    )
    p.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model name or local snapshot path.",
    )
    p.add_argument(
        "--model-tag", type=str, default=None,
        help="Short tag for filenames (e.g. qwen7b). Defaults to basename of --model.",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-new-tokens", type=int, default=700)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--eval-dir", type=str, default=None)
    p.add_argument(
        "--resume", action="store_true",
        help="Skip seeds whose trajectory file already exists.",
    )
    args = p.parse_args()

    if args.model_tag is None:
        args.model_tag = os.path.basename(args.model.rstrip("/")) or args.model

    model_path = args.model
    args.model = args.model_tag  # save helpers key on short tag

    if args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [42]
    args.seeds = seeds

    if not args.eval_dir:
        args.eval_dir = f"eval_results_{args.level}_{args.model}_direct_{args.variant}"

    is_gpt_oss = "gpt-oss" in model_path.lower()
    print(
        f"[Direct-Answer OSS Runner] level={args.level} "
        f"model={'GPT-OSS' if is_gpt_oss else 'Qwen'} ({args.model}) "
        f"seeds={seeds} eval_dir={args.eval_dir}"
    )

    model_fn = load_oss_model_direct(
        model_path, temperature=args.temperature, max_new_tokens=args.max_new_tokens,
    )

    run_seed = run_single_seed_gpt_oss if is_gpt_oss else run_single_seed_qwen

    # Resume: skip completed seeds.
    completed = set()
    results = []
    if args.resume and args.eval_dir:
        for fp in sorted(glob.glob(
            os.path.join(args.eval_dir, f"trajectory_seed*_raw{args.model}.json")
        )):
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
                })
        if completed:
            print(f"[Resume] Skipping {len(completed)} completed seeds: {sorted(completed)}")

    for seed in seeds:
        if seed in completed:
            print(f"[Resume] seed {seed} already done — skipping.")
            continue
        r = run_seed(model_fn, seed, args)
        results.append(r)
        save_trajectory(r, args.eval_dir, args.model)
        save_aggregate(results, args.eval_dir, args)

    print(f"\n{'='*60}")
    print(f"  DIRECT-ANSWER SUMMARY (OSS)")
    print(f"{'='*60}")
    succ = sum(1 for r in results if r["success"])
    print(f"  Level         : {args.level}")
    print(f"  Model tag     : {args.model}")
    print(f"  Total seeds   : {len(results)}")
    print(
        f"  Success rate  : {succ}/{len(results)}"
        + (f" ({100*succ/len(results):.1f}%)" if results else "")
    )
    parse_failures = sum(1 for r in results if r["action"] is None)
    if parse_failures:
        print(f"  Parse failures: {parse_failures}")
    print()
    for r in sorted(results, key=lambda x: x["seed"]):
        status = "✓" if r["success"] else "✗"
        print(f"    {status} seed {r['seed']:<4} action={r['action']}")


if __name__ == "__main__":
    main()
