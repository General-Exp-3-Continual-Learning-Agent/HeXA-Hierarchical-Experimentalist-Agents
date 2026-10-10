"""Direct-answer baseline for Interphyre puzzles using the Claude CLI.

No ReAct loop, no agentic tool dispatch. Protocol per seed:
  1. We call `get_level_state` directly via the Interphyre toolkit (real
     simulator-API call returning all object positions, sizes, properties).
  2. ONE Claude call: system prompt + user prompt that embeds the scene
     observation AND asks for the final placement as JSON.
  3. Parse JSON, run `simulate_action`, record success/failure.

Two answer-prompt variants control whether the model is allowed to write
short reasoning before the JSON:
  - --variant no_cot : JSON only, no prose.

Prompts here are self-contained — they intentionally do NOT reuse
`react_agent.level_prompts` so this baseline cannot accidentally inherit
ReAct framing. Add new levels by extending LEVEL_BLURBS.

Usage:
    python -m react_agent.run_direct_answer_claude --level catapult --variant cot --seeds $(seq 0 49)
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "interphyre"))

from react_agent.tools import InterphyreToolkit


def load_claude_cli_model(model_name: str = "claude-sonnet-4-6"):
    """Local model_fn that calls `claude -p` with a configurable --model.

    Default is `claude-sonnet-4-6` to match the actor model used by
    run_react_claude.py:45 — the direct-answer baseline must share the actor
    model with ReAct/Reflexion so any performance gap is attributable to the
    method, not to model strength.
    """
    def generate(messages: list, temp: float = 0.3, max_tokens: int = 800) -> str:
        prompt = ""
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            prompt += f"{role}:\n{content}\n\n"
        # SAME trailer as run_react_claude.py:37 — without this, Sonnet treats
        # the prompt as an open-ended decision and triggers extended thinking
        # (~200s/call). With it, Sonnet emits a single structured turn (~27s).
        prompt += (
            "USER:\nIMPORTANT: Output ONLY your single Action (and Action Input "
            "if needed). Do NOT write any Observation or additional steps. "
            "Stop after the Action Input.\n\n"
            "ASSISTANT:\n"
        )

        print(f"\n[Claude CLI / {model_name}] Prompt length: {len(prompt)} — waiting...", flush=True)
        call_start = time.time()
        try:
            result = subprocess.run(
                [
                    "claude", "-p", prompt,
                    "--model", model_name,
                    "--max-turns", "1",
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
                print(
                    f"[Claude Error] Exit {result.returncode} after {elapsed:.1f}s. "
                    f"stderr: {(result.stderr or '')[:500]} | "
                    f"stdout: {(result.stdout or '')[:500]}",
                    flush=True,
                )
                return '{"x": 0.0, "y": 0.0, "radius": 0.5}'
            print(f"[Claude CLI / {model_name}] Response in {elapsed:.1f}s", flush=True)
            return result.stdout.strip()
        except FileNotFoundError:
            print("[Claude Error] 'claude' CLI not found.", flush=True)
            return '{"x": 0.0, "y": 0.0, "radius": 0.5}'
        except subprocess.TimeoutExpired:
            print(f"[Claude Error] Timed out after {time.time() - call_start:.1f}s", flush=True)
            return '{"x": 0.0, "y": 0.0, "radius": 0.5}'
        except Exception as e:
            print(f"[Claude Error] {type(e).__name__}: {e}", flush=True)
            return '{"x": 0.0, "y": 0.0, "radius": 0.5}'

    return generate


# ─── Per-level prompt blurbs (rewritten, no ReAct framing) ───────────────────

LEVEL_BLURBS = {
    "catapult": {
        "name": "Catapult",
        "elements": (
            "- Green Ball (dynamic): the object you want to bring into contact with the blue ball.\n"
            "- Blue Ball (dynamic): rests in a basket on a ledge; this is the success target.\n"
            "- Gray Platform (dynamic): a long bar resting across a small pivot — behaves as a lever / catapult arm.\n"
            "- Pivot Ball (static gray ball): the lever's fulcrum.\n"
            "- Basket (static): holds the blue ball on a ledge above ground.\n"
            "- Black ceiling bar / blocker (static): obstacles that may also be used as ricochet surfaces.\n"
        ),
        "goal": (
            "Place ONE red ball so that, after the simulation runs, the green ball "
            "touches the blue ball for at least 3 seconds."
        ),
        # No explicit radius bound in react_agent/level_prompts/down_to_earth.py or
        # catapult.py -- the shared 0.1-2.0 default (below) matches those levels.
    },
    "down_to_earth": {
        "name": "Down to Earth",
        "elements": (
            "- Green Ball (dynamic, target): will fall under gravity.\n"
            "- Black High Platform (static): a horizontal platform below the green ball. "
            "Without intervention, the green ball lands on this platform and stays there.\n"
            "- Purple Ground (static): the floor at the very bottom of the box (y ~ -5).\n"
        ),
        "goal": (
            "Place ONE red ball so that the green ball is knocked off the platform and "
            "touches the purple ground for at least 3 seconds."
        ),
    },
    "two_body_problem": {
        "name": "Two Body Problem",
        "elements": (
            "- Green Ball (dynamic): falls under gravity from rest.\n"
            "- Blue Ball (dynamic): falls under gravity from rest, separated horizontally from the green ball.\n"
        ),
        "goal": (
            "Place ONE red ball at t=0 so that the green ball collides with the blue "
            "ball and stays in contact."
        ),
        # react_agent/level_prompts/two_body_problem.py:50 -- narrower than the
        # 0.1-2.0 default used by catapult/down_to_earth.
        "radius_range": "0.1 <= radius <= 1.5",
    },
    "pass_the_parcel": {
        "name": "Pass The Parcel",
        "elements": (
            "- Top Basket (dynamic, inverted): sits on the platform with its opening "
            "facing DOWNWARD, trapping the green ball underneath it.\n"
            "- Green Ball (dynamic): trapped beneath the inverted top basket on the platform.\n"
            "- Bottom Basket (dynamic, upright): sits below the platform with its opening "
            "facing UPWARD, holds the blue ball.\n"
            "- Blue Ball (dynamic): sits inside the bottom basket; the success target.\n"
            "- Black Platform (static): a horizontal bar the top basket rests on.\n"
            "- Ramp (static): an angled bar rising from the platform's left edge upward "
            "to the right, useful for rolling the red ball onto the top basket.\n"
        ),
        "goal": (
            "Place ONE red ball so that the green ball is released from under the "
            "inverted top basket and contacts the blue ball for at least 3 seconds."
        ),
        # react_agent/level_prompts/pass_the_parcel.py:50 -- same as the default.
    },
    "basket_case": {
        "name": "Basket Case",
        "elements": (
            "- Green Ball (dynamic): positioned high above the basket. Falls straight "
            "down under gravity.\n"
            "- Basket (Gray, dynamic): sits near the purple ground with its opening "
            "facing UPWARD. Without intervention, the green ball falls directly into "
            "it and gets trapped.\n"
            "- Purple Ground (static): the target surface at the very bottom of the "
            "box (y ~ -5).\n"
        ),
        "goal": (
            "Place ONE red ball so that the green ball is deflected sideways, misses "
            "the basket, and touches the purple ground for at least 3 seconds."
        ),
        # react_agent/level_prompts/basket_case.py -- same as the default.
    },
    "tipping_point": {
        "name": "Tipping Point",
        "elements": (
            "- Green Bar (dynamic, vertical, length 2.0-5.0): rests upright inside a "
            "small gray basket on the ground. Its center of mass sits above the "
            "basket floor; small horizontal pushes at the top can topple it.\n"
            "- Gray Basket (dynamic): a small basket sitting near the bottom of the "
            "box, holding the green bar. Can slide if struck hard.\n"
            "- Purple Wall (static, vertical, spans top to bottom): standing flush "
            "against either the LEFT side (x ~ -4.9) or the RIGHT side (x ~ 4.9) of "
            "the box.\n"
        ),
        "goal": (
            "Place ONE red ball so that the green bar tips over and contacts the "
            "purple wall for at least 3 seconds."
        ),
        # react_agent/level_prompts/tipping_point.py -- same as the default.
    },
}


DIRECT_SYSTEM_TEMPLATE = """You are a physics reasoning agent answering a 2D physics puzzle in a single shot.

The world is a 2D box with x, y in [-5, 5]. Gravity pulls objects downward.

Puzzle: {puzzle_name}

Key elements:
{elements}
Goal: {goal}

Placement constraints:
- Red ball must be inside the box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- Red ball must NOT overlap with any existing object at t=0.
- {radius_range}.

You will be given ONE inspection of the scene (output of get_level_state with object positions and sizes). There is no simulator access — no testing, no retries. One inspection, one final answer."""


# User prompt template — scene observation is embedded directly.
# {scene_obs} is filled in at call time with the real get_level_state output.
# {answer_block} is one of the variants below.
USER_TEMPLATE = """Here is the current scene state from `get_level_state` (this is the actual output of the simulator API — object positions, sizes, dynamic flags):

{scene_obs}

{answer_block}"""


# Variant 1: NO chain-of-thought. Single Action turn, no Thought.
# Format mirrors ReAct's per-iteration output shape so Sonnet treats this as a
# routine structured turn (~27s) rather than an open-ended one-shot decision
# (which triggers extended thinking, ~200s).
ANSWER_BLOCK_NO_COT = """Submit your FINAL red ball placement now using the finish action.

Output EXACTLY this format and NOTHING else (no Thought line, no preamble, no commentary):
Action: finish
Action Input: {"x": <X>, "y": <Y>, "radius": <R>}

Replace <X>, <Y>, <R> with your chosen floating-point numbers (subject to the placement constraints above)."""


# Variant 2: short chain-of-thought via a Thought line, then Action.
# Same per-iteration shape as ReAct — Sonnet is trained on this and emits it fast.
ANSWER_BLOCK_COT = """Submit your FINAL red ball placement now using the finish action.

Output EXACTLY this format and NOTHING else (no preamble, no commentary outside the Thought line):
Thought: <2–3 short sentences explaining your placement choice>
Action: finish
Action Input: {"x": <X>, "y": <Y>, "radius": <R>}

Replace <X>, <Y>, <R> with your chosen floating-point numbers (subject to the placement constraints above)."""


_ANSWER_BLOCKS = {
    "no_cot": ANSWER_BLOCK_NO_COT,
    "cot": ANSWER_BLOCK_COT,
}


# ─── JSON extraction ─────────────────────────────────────────────────────────

_JSON_FALLBACK_RE = re.compile(
    r'\{[^{}]*?"x"\s*:\s*(-?\d+\.?\d*)[^{}]*?"y"\s*:\s*(-?\d+\.?\d*)[^{}]*?"radius"\s*:\s*(-?\d+\.?\d*)[^{}]*?\}',
    re.DOTALL,
)
_ACTION_INPUT_RE = re.compile(r"Action\s+Input\s*:\s*(\{.*?\})", re.DOTALL | re.IGNORECASE)


def extract_placement(response: str):
    """Return (x, y, radius) tuple, or None if parsing fails.

    Handles three response shapes:
      1. ReAct-style: '... Action Input: {"x":..., "y":..., "radius":...}'
      2. Bare JSON on a line: '{"x":..., "y":..., "radius":...}'
      3. JSON embedded in prose (regex fallback).
    """
    if not response:
        return None

    # Pass 1: ReAct format — extract from `Action Input: {...}`.
    m = _ACTION_INPUT_RE.search(response)
    if m:
        try:
            obj = json.loads(m.group(1))
            if all(k in obj for k in ("x", "y", "radius")):
                return float(obj["x"]), float(obj["y"]), float(obj["radius"])
        except Exception:
            pass

    # Pass 2: try parsing each line, last-first, as bare JSON.
    for line in reversed(response.strip().splitlines()):
        s = line.strip().rstrip("`").strip()
        if not s.startswith("{"):
            continue
        try:
            obj = json.loads(s)
            if all(k in obj for k in ("x", "y", "radius")):
                return float(obj["x"]), float(obj["y"]), float(obj["radius"])
        except Exception:
            continue

    # Pass 3: regex over the whole response (handles JSON embedded in prose).
    m = _JSON_FALLBACK_RE.search(response)
    if m:
        try:
            return float(m.group(1)), float(m.group(2)), float(m.group(3))
        except ValueError:
            return None
    return None


# ─── Single-seed runner ──────────────────────────────────────────────────────

def run_single_seed_direct(model_fn, seed: int, args) -> dict:
    print(f"\n{'#'*60}")
    print(f"  Direct-Answer Seed: {seed}")
    print(f"{'#'*60}")

    blurb = LEVEL_BLURBS.get(args.level)
    if blurb is None:
        raise SystemExit(
            f"No direct-answer prompt for level '{args.level}'. "
            f"Add it to LEVEL_BLURBS in {__file__}."
        )

    toolkit = InterphyreToolkit(level_name=args.level, seed=seed, is_oss=False)

    if args.eval_dir:
        os.makedirs(args.eval_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(args.eval_dir, f"seed{seed}_raw{args.model}_initial.png"),
            toolkit.get_image(),
        )

    system_prompt = DIRECT_SYSTEM_TEMPLATE.format(
        puzzle_name=blurb["name"],
        elements=blurb["elements"],
        goal=blurb["goal"],
        radius_range=blurb.get("radius_range", "0.1 <= radius <= 2.0"),
    )

    trajectory = []
    start = time.time()

    # ── Step 1: real simulator API call — fetch the scene state ──
    scene_obs = toolkit.dispatch_tool("get_level_state", {})
    trajectory.append((
        "(direct-answer baseline: scripted peek)",
        "get_level_state({})",
        scene_obs,
    ))
    if args.verbose:
        print(f"\n--- get_level_state (Interphyre toolkit) returned {len(scene_obs)} chars ---")
        print(scene_obs)
        print()

    # ── Step 2: ONE Claude call — system + user (with embedded scene state) ──
    answer_block = _ANSWER_BLOCKS[args.variant]
    user_msg = USER_TEMPLATE.format(scene_obs=scene_obs, answer_block=answer_block)
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    if args.verbose:
        print(f"--- SYSTEM ({len(system_prompt)} chars) ---\n{system_prompt}\n")
        print(f"--- USER (variant={args.variant}, {len(user_msg)} chars) ---\n{user_msg}\n")

    resp = model_fn(msgs, args.temperature, args.max_new_tokens)
    # OpenAI-compatible model_fns attach usage after the call (see
    # load_openai_compatible_model in react_agent.py); the Claude CLI model_fn
    # has no such attribute, so this is None there rather than an error.
    usage = getattr(model_fn, "last_usage", None)
    if args.verbose:
        print(f"--- ASSISTANT ---\n{resp}\n")
        if usage:
            print(f"--- USAGE --- {usage}\n")

    placement = extract_placement(resp)

    # ── Simulate ────────────────────────────────────────────
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
        "iterations": 2,                  # peek + answer (for summary compat)
        "elapsed_time": elapsed,
        "trajectory": trajectory,
        "final_observation": sim_obs,
        "usage": usage,
    }


# ─── Save / main ─────────────────────────────────────────────────────────────

def save_trajectory(result: dict, log_dir: str, model: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"trajectory_seed{result['seed']}_raw{model}.json")
    data = {
        "seed": result["seed"],
        "success": result["success"],
        "action": list(result["action"]) if result["action"] else None,
        "iterations": result["iterations"],
        "elapsed_time": result.get("elapsed_time"),
        "trajectory": [
            {"thought": t, "action": a, "observation": o}
            for t, a, o in result["trajectory"]
        ],
        # Token usage from an OpenAI-compatible backend (None for Claude CLI,
        # which reports no usage). See load_openai_compatible_model in
        # react_agent.py for what this dict contains.
        "usage": result.get("usage"),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Trajectory saved → {os.path.abspath(path)}")


def save_aggregate(results: list, log_dir: str, args) -> None:
    os.makedirs(log_dir, exist_ok=True)
    n = len(results)
    succ = sum(1 for r in results if r["success"])
    parse_failures = sum(1 for r in results if r["action"] is None)
    usages = [r.get("usage") for r in results if r.get("usage")]
    usage_totals = None
    if usages:
        usage_totals = {
            "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usages),
            "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usages),
            "reasoning_tokens": sum(u.get("reasoning_tokens") or 0 for u in usages),
            "n_calls_with_usage": len(usages),
        }
    summary = {
        "model": args.model,
        "level": args.level,
        "baseline": "direct_answer",
        "variant": args.variant,
        "seeds": sorted(r["seed"] for r in results),
        "total_seeds": n,
        "successes": succ,
        "success_rate": succ / n if n else 0.0,
        "parse_failures": parse_failures,
        # Run-level token totals from an OpenAI-compatible backend, when
        # reported. None if the actor doesn't report usage (e.g. Claude CLI).
        "usage_totals": usage_totals,
        "results": [
            {
                "seed": r["seed"],
                "success": r["success"],
                "action": list(r["action"]) if r["action"] else None,
                "elapsed_time": r.get("elapsed_time"),
                "usage": r.get("usage"),
            }
            for r in sorted(results, key=lambda x: x["seed"])
        ],
    }
    path = os.path.join(log_dir, f"summary_{args.model}_direct.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Aggregate summary saved → {path}")


def main():
    p = argparse.ArgumentParser(
        description="Direct-answer baseline (no ReAct) for Interphyre with Claude CLI",
    )
    p.add_argument("--level", type=str, default="catapult", choices=list(LEVEL_BLURBS.keys()))
    p.add_argument(
        "--variant", type=str, default="cot", choices=["no_cot", "cot"],
        help="no_cot = JSON-only answer, no reasoning. cot = 2-3 sentences of reasoning then JSON.",
    )
    p.add_argument("--model", type=str, default="claude",
                   help="Logging label only.")
    p.add_argument(
        "--actor-model", type=str, default="claude-sonnet-4-6",
        help="Model passed to `claude -p --model`. Default matches "
             "run_react_claude.py for like-for-like comparison with the "
             "ReAct/Reflexion baselines.",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-new-tokens", type=int, default=700)
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--eval-dir", type=str, default=None,
        help="Where to save trajectories + summary. Defaults to eval_results_<level>_claude_direct.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip seeds whose trajectory file already exists in --eval-dir.",
    )
    args = p.parse_args()

    if args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [42]
    args.seeds = seeds

    if not args.eval_dir:
        args.eval_dir = f"eval_results_{args.level}_claude_direct_{args.variant}"

    print(
        f"[Direct-Answer Runner] level={args.level} variant={args.variant} "
        f"seeds={seeds} eval_dir={args.eval_dir}"
    )

    model_fn = load_claude_cli_model()

    # Resume
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

    # Final overview
    print(f"\n{'='*60}")
    print(f"  DIRECT-ANSWER SUMMARY")
    print(f"{'='*60}")
    succ = sum(1 for r in results if r["success"])
    print(f"  Level         : {args.level}")
    print(f"  Total seeds   : {len(results)}")
    print(
        f"  Success rate  : {succ}/{len(results)}"
        + (f" ({100*succ/len(results):.1f}%)" if results else "")
    )
    print()
    for r in sorted(results, key=lambda x: x["seed"]):
        status = "✓" if r["success"] else "✗"
        print(f"    {status} seed {r['seed']:<4} action={r['action']}")


if __name__ == "__main__":
    main()
