"""Direct-answer baseline for Interphyre puzzles using the Claude CLI.

No ReAct loop, no agentic tool dispatch. Protocol per seed:
  1. Call `get_level_state` directly via the Interphyre toolkit (real
     simulator-API call returning all object positions, sizes, properties).
  2. ONE Claude call: system prompt + user prompt that embeds the scene
     observation AND asks for the final placement as JSON.
  3. Parse JSON, run `simulate_action`, record success/failure.

Two answer-prompt variants control whether the model may reason before answering:
  --variant cot    : 2-3 sentence Thought line before the JSON (default)
  --variant no_cot : JSON only, no prose

Prompts here are self-contained — they intentionally do NOT reuse
`react_agent.level_prompts` so this baseline cannot accidentally inherit
ReAct framing.

Usage:
    python -m react_agent.run_direct_answer_claude \\
        --level catapult --variant cot --seeds $(seq 0 49)
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


# ─── Claude CLI model loader ─────────────────────────────────────────────────

def load_claude_cli_model(model_name: str = "claude-sonnet-4-6"):
    """Return a generate(messages, temp, max_tokens) callable backed by `claude -p`.

    Default model matches run_react_claude.py so any performance gap between
    this baseline and ReAct is attributable to method, not model strength.
    """
    def generate(messages: list, temp: float = 0.3, max_tokens: int = 800) -> str:
        prompt = ""
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            prompt += f"{role}:\n{content}\n\n"
        # Same trailer as run_react_claude.py — without it Sonnet enters
        # extended-thinking mode (~200 s/call); with it a single structured
        # turn is emitted (~27 s).
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


# ─── Per-level prompt blurbs ─────────────────────────────────────────────────
# Self-contained: no ReAct framing, no tool list. Add new levels here.

LEVEL_BLURBS = {
    "down_to_earth": {
        "name": "Down to Earth",
        "elements": (
            "- Green Ball (dynamic): falls under gravity. Without intervention it lands on the platform and stays there.\n"
            "- Black High Platform (static): a horizontal bar below the green ball that blocks its fall to the ground.\n"
            "- Purple Ground (static): the floor at the very bottom of the box (y ≈ -5). This is the success target.\n"
        ),
        "goal": (
            "Place ONE red ball so that the green ball is knocked off the platform "
            "and touches the purple ground for at least 3 seconds."
        ),
    },
    "two_body_problem": {
        "name": "Two Body Problem",
        "elements": (
            "- Green Ball (dynamic): falls under gravity from rest.\n"
            "- Blue Ball (dynamic): falls under gravity from rest, separated horizontally from the green ball.\n"
            "- Without intervention both balls fall straight down and NEVER meet.\n"
        ),
        "goal": (
            "Place ONE red ball so that the green ball collides with the blue ball "
            "and stays in contact. The red ball must be placed ABOVE the green ball "
            "with a slight horizontal offset AWAY from the blue ball so the collision "
            "deflects green toward blue."
        ),
    },
    "catapult": {
        "name": "Catapult",
        "elements": (
            "- Green Ball (dynamic): small ball sitting on the LEFT end of the gray catapult arm.\n"
            "- Gray Bar / Catapult Arm (dynamic): a long lever resting across the gray pivot ball. Green ball sits on its left end.\n"
            "- Gray Ball / Pivot (dynamic): fulcrum sitting on the left black platform.\n"
            "- Black Ball (static): ceiling blocker near the top of the scene.\n"
            "- Black Platform (left, static): horizontal platform on the left side.\n"
            "- Black Ledge (right, static): platform on the right side.\n"
            "- Basket (gray, dynamic): sits on the right ledge.\n"
            "- Blue Ball (dynamic): sits inside the basket on the right ledge. This is the success target.\n"
        ),
        "goal": (
            "Place ONE red ball so that the green ball contacts the blue ball for at least 3 seconds. "
            "Use more radius for better energy (r > 1). "
            "The most reliable approach: drop the red ball on the RIGHT end of the catapult arm to "
            "launch the green ball upward and to the right toward the blue ball."
        ),
    },
    "falling_into_place": {
        "name": "Falling Into Place",
        "elements": (
            "- Green Ball (dynamic): sits on one of the two black platforms (left or right side).\n"
            "- Left Platform / Right Platform (black, static): two horizontal bars with a gap between them in the center.\n"
            "- Bottom Ramp (black, static): a slightly angled bar near the bottom of the scene.\n"
            "- Blue Jar (dynamic basket): a basket positioned above the scene with its opening facing DOWNWARD. It falls due to gravity.\n"
        ),
        "goal": (
            "Place ONE red ball so that the green ball touches the blue jar for at least 3 seconds. "
            "Place the red ball HIGH ABOVE the green ball (at least 2-3 units higher in y) "
            "with a slight horizontal offset AWAY from the jar's side, so it falls onto the green ball "
            "and deflects it off the platform edge toward where the jar will land."
        ),
    },
    "basket_case": {
        "name": "Basket Case",
        "elements": (
            "- Green Ball (dynamic): positioned high above the basket. Falls straight down due to gravity.\n"
            "- Basket (gray, dynamic): a container sitting near the purple ground with its opening facing UPWARD. "
            "Without intervention the green ball falls directly into it and gets trapped.\n"
            "- Purple Ground (static): the floor at the very bottom of the box (y ≈ -5). This is the success target.\n"
        ),
        "goal": (
            "Place ONE red ball so that the green ball touches the purple ground for at least 3 seconds. "
            "You must deflect the green ball sideways so it misses the basket and lands on the ground. "
            "Place the red ball just to the side of the green ball's fall path to nudge it away from the basket."
        ),
    },
    "pass_the_parcel": {
        "name": "Pass The Parcel",
        "elements": (
            "- Top Basket (gray, inverted, dynamic): sits on the black platform with its opening facing DOWNWARD, trapping the green ball underneath.\n"
            "- Green Ball (dynamic): trapped beneath the inverted top basket on the platform.\n"
            "- Bottom Basket (gray, upright, dynamic): below the platform with its opening facing UPWARD, holds the blue ball.\n"
            "- Blue Ball (dynamic): inside the bottom basket. This is the success target.\n"
            "- Black Platform (static): horizontal bar. The top basket sits on it.\n"
            "- Ramp (black, static): an angled bar rising from the platform's left edge upward to the right.\n"
        ),
        "goal": (
            "Place ONE red ball so that the green ball contacts the blue ball for at least 3 seconds. "
            "You must knock the inverted top basket off the platform to release the green ball. "
            "Place the red ball on or above the ramp so it rolls/falls and strikes the top basket, "
            "toppling it and freeing the green ball to fall into the bottom basket where the blue ball is."
        ),
    },
    "tipping_point": {
        "name": "Tipping Point",
        "elements": (
            "- Green Bar (dynamic): a vertical bar (length 2.0–5.0) resting upright inside a small gray basket on the ground. "
            "Small horizontal pushes near its top can topple it.\n"
            "- Gray Basket (small, dynamic): holds the green bar near the bottom of the box.\n"
            "- Purple Wall (static): a vertical bar flush against either the LEFT (x ≈ -4.9) or RIGHT (x ≈ 4.9) side of the box, "
            "spanning nearly the full height. This is the success target.\n"
        ),
        "goal": (
            "Place ONE red ball so that the green bar contacts the purple wall and stays in contact "
            "for the success-time duration. "
            "Drop the red ball from ABOVE near the TOP of the green bar, slightly offset toward the purple wall, "
            "to topple the bar in the correct direction."
        ),
    },
    "cliffhanger": {
        "name": "Cliffhanger",
        "elements": (
            "- Green Bar (dynamic): a vertical bar (length 2.0–3.0) standing upright on the black platform near one of its edges.\n"
            "- Black Platform (static): a horizontal bar (length 4.0–6.0) at variable height y ∈ [-3, 0]. The green bar stands on top of it.\n"
            "- Ceiling (static): a horizontal bar above the platform.\n"
            "- Purple Ground (static): the floor at the bottom of the box (y ≈ -5). This is the success target.\n"
        ),
        "goal": (
            "Place ONE red ball so that the green bar contacts the purple ground for at least 3 seconds. "
            "You must tip the green bar off the platform edge so it falls to the ground. "
            "Place the red ball above the platform near the top of the green bar, slightly offset toward the closer edge, "
            "to push it off."
        ),
    },
}


# ─── Prompt templates ────────────────────────────────────────────────────────

DIRECT_SYSTEM_TEMPLATE = """You are a physics reasoning agent answering a 2D physics puzzle in a single shot.

The world is a 2D box with x, y in [-5, 5]. Gravity pulls objects downward.

Puzzle: {puzzle_name}

Key elements:
{elements}
Goal: {goal}

Placement constraints:
- Red ball must be inside the box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- Red ball must NOT overlap with any existing object at t=0.
- 0.1 <= radius <= 2.0.

You will be given ONE inspection of the scene (output of get_level_state with object positions and sizes). There is no simulator access — no testing, no retries. One inspection, one final answer."""


USER_TEMPLATE = """Here is the current scene state from `get_level_state` (actual simulator output — object positions, sizes, dynamic flags):

{scene_obs}

{answer_block}"""


# Variant 1: NO chain-of-thought. Single Action turn.
ANSWER_BLOCK_NO_COT = """Submit your FINAL red ball placement now using the finish action.

Output EXACTLY this format and NOTHING else (no Thought line, no preamble, no commentary):
Action: finish
Action Input: {"x": <X>, "y": <Y>, "radius": <R>}

Replace <X>, <Y>, <R> with your chosen floating-point numbers (subject to the placement constraints above)."""


# Variant 2: short chain-of-thought Thought line, then Action.
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

_ACTION_INPUT_RE = re.compile(r"Action\s+Input\s*:\s*(\{.*?\})", re.DOTALL | re.IGNORECASE)
_JSON_FALLBACK_RE = re.compile(
    r'\{[^{}]*?"x"\s*:\s*(-?\d+\.?\d*)[^{}]*?"y"\s*:\s*(-?\d+\.?\d*)[^{}]*?"radius"\s*:\s*(-?\d+\.?\d*)[^{}]*?\}',
    re.DOTALL,
)


def extract_placement(response: str):
    """Return (x, y, radius) tuple from model response, or None on parse failure.

    Handles three response shapes:
      1. ReAct-style:  'Action Input: {"x":..., "y":..., "radius":...}'
      2. Bare JSON on a line: '{"x":..., "y":..., "radius":...}'
      3. JSON embedded in prose (regex fallback).
    """
    if not response:
        return None

    # Pass 1: ReAct format.
    m = _ACTION_INPUT_RE.search(response)
    if m:
        try:
            obj = json.loads(m.group(1))
            if all(k in obj for k in ("x", "y", "radius")):
                return float(obj["x"]), float(obj["y"]), float(obj["radius"])
        except Exception:
            pass

    # Pass 2: last bare-JSON line.
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

    # Pass 3: regex fallback.
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
    print(f"  Direct-Answer Seed: {seed}  (level={args.level} variant={args.variant})")
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
    )

    trajectory = []
    start = time.time()

    # Step 1: real simulator call — fetch scene state.
    scene_obs = toolkit.dispatch_tool("get_level_state", {})
    trajectory.append((
        "(direct-answer baseline: scripted peek)",
        "get_level_state({})",
        scene_obs,
    ))
    if args.verbose:
        print(f"\n--- get_level_state ({len(scene_obs)} chars) ---\n{scene_obs}\n")

    # Step 2: ONE model call — system + user with embedded scene.
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
    if args.verbose:
        print(f"--- ASSISTANT ---\n{resp}\n")

    placement = extract_placement(resp)

    # Step 3: simulate and record.
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
        "iterations": 2,           # peek + answer (for summary compat)
        "elapsed_time": elapsed,
        "trajectory": trajectory,
        "final_observation": sim_obs,
    }


# ─── Persistence helpers ──────────────────────────────────────────────────────

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
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Trajectory saved → {os.path.abspath(path)}")


def save_aggregate(results: list, log_dir: str, args) -> None:
    os.makedirs(log_dir, exist_ok=True)
    n = len(results)
    succ = sum(1 for r in results if r["success"])
    parse_failures = sum(1 for r in results if r["action"] is None)
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
        "results": [
            {
                "seed": r["seed"],
                "success": r["success"],
                "action": list(r["action"]) if r["action"] else None,
                "elapsed_time": r.get("elapsed_time"),
            }
            for r in sorted(results, key=lambda x: x["seed"])
        ],
    }
    path = os.path.join(log_dir, f"summary_{args.model}_direct.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Aggregate summary saved → {path}")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Direct-answer baseline (no ReAct loop) for Interphyre with Claude CLI",
    )
    p.add_argument(
        "--level", type=str, default="catapult",
        choices=list(LEVEL_BLURBS.keys()),
    )
    p.add_argument(
        "--variant", type=str, default="cot", choices=["no_cot", "cot"],
        help="no_cot = JSON-only answer. cot = 2-3 sentence Thought then JSON.",
    )
    p.add_argument(
        "--model", type=str, default="claude",
        help="Logging label only (does not select the model).",
    )
    p.add_argument(
        "--actor-model", type=str, default="claude-sonnet-4-6",
        help="Model passed to `claude -p --model`. Default matches run_react_claude.py.",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-new-tokens", type=int, default=700)
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--eval-dir", type=str, default=None,
        help="Where to save trajectories + summary. "
             "Defaults to eval_results_<level>_claude_direct_<variant>.",
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
        f"actor_model={args.actor_model} seeds={seeds} eval_dir={args.eval_dir}"
    )

    model_fn = load_claude_cli_model(args.actor_model)

    # Resume: skip seeds with existing trajectory files.
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
        r = run_single_seed_direct(model_fn, seed, args)
        results.append(r)
        save_trajectory(r, args.eval_dir, args.model)
        save_aggregate(results, args.eval_dir, args)

    # Final summary.
    print(f"\n{'='*60}")
    print(f"  DIRECT-ANSWER SUMMARY")
    print(f"{'='*60}")
    succ = sum(1 for r in results if r["success"])
    print(f"  Level         : {args.level}")
    print(f"  Variant       : {args.variant}")
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
