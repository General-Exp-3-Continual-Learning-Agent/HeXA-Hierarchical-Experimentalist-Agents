"""Skills-in-context, single-shot: HExA's frozen bank with the tool interface removed.

Separates two readings of HExA's frozen-bank result:
  - skills encode the solution  -> this scores near the HExA frozen eval;
  - skills are procedure for experimenting -> this collapses toward zero-shot.

Protocol per seed is the zero-shot direct-answer baseline
(run_direct_answer_claude.py, --variant cot) unchanged -- same system prompt,
same scripted get_level_state peek, same answer block, same parser, one
simulate_action to score -- with ONE addition: the frozen skill bank block is
appended to the system prompt exactly as HExA's frozen eval injects it
(augmented_runner.py: retrieve_skills top-k by confidence, first k mistakes,
SkillBank.format_skills_as_prompt, appended after "\n\n").

Without --skill-bank this is the matched zero-shot control on the same seeds
and the same CLI settings.

Differences from run_direct_answer_claude.py's CLI call, applied to BOTH arms:
  - `--tools ""`: no tool can run inside the single turn, so the model cannot
    experiment even implicitly.
  - a CLI error/timeout is recorded as `cli_error` and scored as a failure,
    instead of silently becoming a (0, 0, 0.5) placement.

Arm B (--max-attempts 25): same prompt, but up to 25 placements, and after each
only SUCCESS / FAILURE / INVALID (static overlap reason) is shown -- no
trajectory, no trace or probe tools, no simulator output.

Arm C (--reasoning-turns 25): no feedback at all. 24 scratchpad turns
(`Action: think`), then ONE committed placement on turn 25; only that placement
is simulated. Extended CoT, zero environment signal.

Usage:
    python -m react_agent.run_direct_skills_claude --level catapult \
        --skill-bank .../skill_bank_evolving_17.json --seeds $(seq 57 106) \
        --eval-dir OUT --resume
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "interphyre"))

from react_agent.tools import InterphyreToolkit
from react_agent.run_direct_answer_claude import (
    LEVEL_BLURBS,
    DIRECT_SYSTEM_TEMPLATE,
    USER_TEMPLATE,
    _ANSWER_BLOCKS,
    extract_placement,
)
from skillrl.core.skill_bank import SkillBank
from skillrl.core.retriever import retrieve_skills

CLI_ERROR = "__CLI_ERROR__"


def load_claude_cli_model(model_name: str):
    def generate(messages: list) -> str:
        prompt = ""
        for msg in messages:
            prompt += f"{msg['role'].upper()}:\n{msg['content']}\n\n"
        # Same trailer as run_direct_answer_claude.py.
        prompt += (
            "USER:\nIMPORTANT: Output ONLY your single Action (and Action Input "
            "if needed). Do NOT write any Observation or additional steps. "
            "Stop after the Action Input.\n\n"
            "ASSISTANT:\n"
        )
        print(f"\n[Claude CLI / {model_name}] Prompt length: {len(prompt)} — waiting...", flush=True)
        t0 = time.time()
        try:
            result = subprocess.run(
                [
                    "claude", "-p", prompt,
                    "--model", model_name,
                    "--max-turns", "1",
                    "--tools", "",
                    "--no-session-persistence",
                    "--output-format", "text",
                ],
                capture_output=True, text=True, env=os.environ.copy(), timeout=2000,
            )
        except Exception as e:
            print(f"[Claude Error] {type(e).__name__}: {e}", flush=True)
            return CLI_ERROR
        if result.returncode != 0:
            print(f"[Claude Error] Exit {result.returncode} after {time.time() - t0:.1f}s. "
                  f"stderr: {(result.stderr or '')[:500]} | stdout: {(result.stdout or '')[:500]}",
                  flush=True)
            return CLI_ERROR
        print(f"[Claude CLI / {model_name}] Response in {time.time() - t0:.1f}s", flush=True)
        return result.stdout.strip()

    return generate


def build_skills_block(bank_path, level, max_general, max_specific, max_mistakes):
    """Identical selection and rendering to augmented_runner.run_skill_augmented."""
    bank = SkillBank.load(bank_path)
    skills = retrieve_skills(bank, level, max_general=max_general, max_specific=max_specific)
    mistakes = bank.get_mistakes_for_level(level)[:max_mistakes]
    block = bank.format_skills_as_prompt(level, skills=skills, mistakes=mistakes)
    return block, [s.title for s in skills], [m.description for m in mistakes]


SINGLE_SHOT_LINE = ("You will be given ONE inspection of the scene (output of get_level_state with "
                    "object positions and sizes). There is no simulator access — no testing, no "
                    "retries. One inspection, one final answer.")

RETRY_LINE = ("You will be given ONE inspection of the scene (output of get_level_state with object "
              "positions and sizes). You have no tools. You may submit up to {n} placements; after "
              "each one you are told ONLY whether it succeeded, failed, or was an invalid placement "
              "-- no trajectory, positions, or any other simulator output. The episode ends at the "
              "first success or after {n} placements.")

FEEDBACK_MSG = """Attempt {k}/{n}: {outcome}
{remaining} placement(s) remaining. Submit your next red ball placement in the same format:
Thought: <2–3 short sentences>
Action: finish
Action Input: {{"x": <X>, "y": <Y>, "radius": <R>}}"""


def outcome_only(sim_obs: str) -> str:
    """Strip simulate_action output down to the binary outcome (arm B)."""
    if "SUCCESS!" in sim_obs:
        return "SUCCESS."
    if sim_obs.startswith("INVALID"):
        # Static geometry only (overlap / out of box), derivable from the scene.
        return sim_obs.replace("\nThe simulation was not run. Fix the placement and try again.", "") \
            + "\n(Not simulated.)"
    if sim_obs.startswith("FAILURE"):
        return "FAILURE."
    return "FAILURE (simulation error)."


def run_seed(model_fn, seed, args, skills_block):
    blurb = LEVEL_BLURBS[args.level]
    system_prompt = DIRECT_SYSTEM_TEMPLATE.format(
        puzzle_name=blurb["name"],
        elements=blurb["elements"],
        goal=blurb["goal"],
        radius_range=blurb.get("radius_range", "0.1 <= radius <= 2.0"),
    )
    n = args.max_attempts
    if n > 1:
        assert SINGLE_SHOT_LINE in system_prompt
        system_prompt = system_prompt.replace(SINGLE_SHOT_LINE, RETRY_LINE.format(n=n))
    if skills_block:
        system_prompt += "\n\n" + skills_block

    toolkit = InterphyreToolkit(level_name=args.level, seed=seed, is_oss=False)
    start = time.time()
    scene_obs = toolkit.dispatch_tool("get_level_state", {})
    answer_block = _ANSWER_BLOCKS[args.variant]
    if n > 1:
        answer_block = answer_block.replace("Submit your FINAL red ball placement now",
                                            f"Submit your first red ball placement (of up to {n})")
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": USER_TEMPLATE.format(scene_obs=scene_obs, answer_block=answer_block)},
    ]
    if args.verbose:
        print(f"--- SYSTEM ({len(system_prompt)} chars) ---\n{system_prompt}\n")
        print(f"--- USER ---\n{msgs[1]['content']}\n")

    attempts = []
    success = cli_error = False
    for k in range(1, n + 1):
        resp = model_fn(msgs)
        if resp == CLI_ERROR:
            cli_error = True
            break
        placement = extract_placement(resp)
        if placement is None:
            action, sim_obs = None, "(parse failure — no placement extracted)"
            feedback = "INVALID: no placement could be parsed from your answer."
        else:
            x, y, r = placement
            action = [x, y, r]
            sim_obs = toolkit.dispatch_tool("simulate_action", {"x": x, "y": y, "radius": r})
            feedback = outcome_only(sim_obs)
        success = "SUCCESS!" in sim_obs
        attempts.append({"attempt": k, "response": resp, "action": action,
                         "feedback_shown": feedback, "sim_obs_hidden": sim_obs})
        if args.verbose:
            print(f"--- ASSISTANT (attempt {k}) ---\n{resp}\n--- FEEDBACK ---\n{feedback}\n")
        if success or k == n:
            break
        msgs.append({"role": "assistant", "content": resp})
        msgs.append({"role": "user", "content": FEEDBACK_MSG.format(
            k=k, n=n, outcome=feedback, remaining=n - k)})
    toolkit.close()

    last = attempts[-1] if attempts else {}
    elapsed = time.time() - start
    print(f"  seed {seed}: success={success} attempts={len(attempts)} action={last.get('action')} "
          f"cli_error={cli_error} ({elapsed:.1f}s)", flush=True)
    return {
        "seed": seed,
        "success": success,
        "action": last.get("action"),
        "cli_error": cli_error,
        "parse_failure": bool(attempts) and all(a["action"] is None for a in attempts),
        "iterations": len(attempts),
        "elapsed_time": elapsed,
        "attempts": attempts,
    }


REASONING_LINE = ("You will be given ONE inspection of the scene (output of get_level_state with "
                  "object positions and sizes). You do NOT have access to any simulation tools and "
                  "will receive NO feedback from the environment. Reason using the skills above and "
                  "your own physics knowledge. You have a scratchpad of up to {n} turns: each turn, "
                  "write one step of reasoning (hypotheses, geometry, working through the skills, "
                  "eliminating options). You commit your single final answer on turn {n}, and only "
                  "that one answer is evaluated.")

REASONING_TURN_FORMAT = """On turns before the last, output EXACTLY this format and NOTHING else:
Thought: <one step of reasoning>
Action: think

You will be told when it is your final turn; only then do you commit an answer."""

REASONING_CONTINUE = """Turn {k}/{n}. (No environment feedback -- this is your own scratchpad.) Continue reasoning with `Action: think`; you commit on turn {n}."""

REASONING_FINAL = """Turn {k}/{n}. This is your FINAL turn: you must commit now.
Output EXACTLY:
Thought: <brief justification>
Action: finish
Action Input: {{"x": <X>, "y": <Y>, "radius": <R>}}"""


def run_seed_reasoning(model_fn, seed, args, skills_block):
    """Arm C: up to N turns of scratchpad, zero environment signal, one committed answer."""
    blurb = LEVEL_BLURBS[args.level]
    n = args.reasoning_turns
    system_prompt = DIRECT_SYSTEM_TEMPLATE.format(
        puzzle_name=blurb["name"],
        elements=blurb["elements"],
        goal=blurb["goal"],
        radius_range=blurb.get("radius_range", "0.1 <= radius <= 2.0"),
    )
    assert SINGLE_SHOT_LINE in system_prompt
    system_prompt = system_prompt.replace(SINGLE_SHOT_LINE, REASONING_LINE.format(n=n))
    if skills_block:
        system_prompt += "\n\n" + skills_block

    toolkit = InterphyreToolkit(level_name=args.level, seed=seed, is_oss=False)
    start = time.time()
    scene_obs = toolkit.dispatch_tool("get_level_state", {})
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": USER_TEMPLATE.format(
            scene_obs=scene_obs,
            answer_block=f"Turn 1/{n}. Begin reasoning.\n\n" + REASONING_TURN_FORMAT)},
    ]
    if args.verbose:
        print(f"--- SYSTEM ({len(system_prompt)} chars) ---\n{system_prompt}\n")
        print(f"--- USER ---\n{msgs[1]['content']}\n")

    turns, placement, cli_error = [], None, False
    for k in range(1, n + 1):
        resp = model_fn(msgs)
        if resp == CLI_ERROR:
            cli_error = True
            break
        # The flat CLI prompt invites the model to script later turns itself
        # ("USER: Turn 4/25 ..."); keep only this turn's step.
        resp = resp.split("\nUSER:")[0].strip()
        turns.append(resp)
        if args.verbose:
            print(f"--- ASSISTANT (turn {k}) ---\n{resp}\n")
        # The answer is committed ONLY on the final turn; an early finish is
        # treated as a scratchpad step (it is never simulated).
        if k == n:
            placement = extract_placement(resp)
            break
        msgs.append({"role": "assistant", "content": resp})
        nxt = k + 1
        msgs.append({"role": "user", "content": REASONING_FINAL.format(k=nxt, n=n) if nxt == n
                     else REASONING_CONTINUE.format(k=nxt, n=n)})

    if placement is None:
        action = None
        sim_obs = "(CLI error)" if cli_error else "(parse failure — no committed placement)"
        success = False
    else:
        x, y, r = placement
        action = [x, y, r]
        sim_obs = toolkit.dispatch_tool("simulate_action", {"x": x, "y": y, "radius": r})
        success = "SUCCESS!" in sim_obs
    toolkit.close()

    elapsed = time.time() - start
    print(f"  seed {seed}: success={success} turns={len(turns)} action={action} "
          f"cli_error={cli_error} ({elapsed:.1f}s)", flush=True)
    return {
        "seed": seed,
        "success": success,
        "action": action,
        "cli_error": cli_error,
        "parse_failure": placement is None and not cli_error,
        "iterations": len(turns),
        "elapsed_time": elapsed,
        "turns": turns,
        "final_observation": sim_obs,
    }


def save_summary(results, args, skill_titles, mistake_descs, skills_block):
    valid = [r for r in results if not r["cli_error"]]
    succ = sum(r["success"] for r in results)
    summary = {
        "baseline": ("direct_skills" if args.skill_bank else "direct_zeroshot")
                    + ("" if args.max_attempts == 1 else f"_retry{args.max_attempts}")
                    + (f"_reason{args.reasoning_turns}" if args.reasoning_turns else ""),
        "level": args.level,
        "actor_model": args.actor_model,
        "variant": args.variant,
        "skill_bank": args.skill_bank,
        "retrieval": {"max_general": args.max_general_skills,
                      "max_specific": args.max_specific_skills,
                      "max_mistakes": args.max_mistakes},
        "skills_injected": skill_titles,
        "mistakes_injected": mistake_descs,
        "skills_block_chars": len(skills_block),
        "seeds": sorted(r["seed"] for r in results),
        "total_seeds": len(results),
        "successes": succ,
        "success_rate": succ / len(results) if results else 0.0,
        "cli_errors": len(results) - len(valid),
        "parse_failures": sum(r["parse_failure"] for r in results),
        "max_attempts": args.max_attempts,
        "reasoning_turns": args.reasoning_turns,
        "avg_iterations": (sum(r["iterations"] for r in results) / len(results)) if results else 0.0,
        "results": [{k: r[k] for k in ("seed", "success", "action", "iterations", "cli_error",
                                       "parse_failure", "elapsed_time")}
                    for r in sorted(results, key=lambda r: r["seed"])],
    }
    with open(os.path.join(args.eval_dir, "summary_direct_skills.json"), "w") as f:
        json.dump(summary, f, indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--level", required=True, choices=list(LEVEL_BLURBS.keys()))
    p.add_argument("--skill-bank", default=None,
                   help="Frozen bank JSON. Omit for the matched zero-shot control.")
    p.add_argument("--variant", default="cot", choices=list(_ANSWER_BLOCKS.keys()))
    p.add_argument("--actor-model", default="claude-sonnet-4-6")
    p.add_argument("--max-general-skills", type=int, default=0)
    p.add_argument("--max-specific-skills", type=int, default=6)
    p.add_argument("--max-mistakes", type=int, default=4)
    p.add_argument("--max-attempts", type=int, default=1,
                   help="1 = single shot (arm A). >1 = up to N placements with outcome-only "
                        "feedback (arm B).")
    p.add_argument("--reasoning-turns", type=int, default=0,
                   help=">0 = arm C: up to N scratchpad turns, no environment feedback, one "
                        "committed placement.")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if args.reasoning_turns and args.max_attempts > 1:
        p.error("--reasoning-turns and --max-attempts > 1 are separate arms")
    os.makedirs(args.eval_dir, exist_ok=True)

    skills_block, skill_titles, mistake_descs = ("", [], [])
    if args.skill_bank:
        skills_block, skill_titles, mistake_descs = build_skills_block(
            args.skill_bank, args.level, args.max_general_skills,
            args.max_specific_skills, args.max_mistakes)
    print(f"[direct-skills] level={args.level} bank={args.skill_bank} "
          f"skills={len(skill_titles)} mistakes={len(mistake_descs)} eval_dir={args.eval_dir}",
          flush=True)

    model_fn = load_claude_cli_model(args.actor_model)
    results = {}
    if args.resume:
        for fp in glob.glob(os.path.join(args.eval_dir, "trajectory_seed*_direct_skills.json")):
            with open(fp) as f:
                r = json.load(f)
            # CLI errors are retried on resume rather than kept as failures.
            if not r["cli_error"]:
                results[r["seed"]] = r

    for seed in args.seeds:
        if seed in results:
            continue
        r = (run_seed_reasoning if args.reasoning_turns else run_seed)(
            model_fn, seed, args, skills_block)
        results[seed] = r
        with open(os.path.join(args.eval_dir, f"trajectory_seed{seed}_direct_skills.json"), "w") as f:
            json.dump(r, f, indent=2)
        save_summary(list(results.values()), args, skill_titles, mistake_descs, skills_block)

    save_summary(list(results.values()), args, skill_titles, mistake_descs, skills_block)
    res = list(results.values())
    print(f"\n[direct-skills] {args.level}: {sum(r['success'] for r in res)}/{len(res)} solved, "
          f"{sum(r['cli_error'] for r in res)} CLI errors, "
          f"{sum(r['parse_failure'] for r in res)} parse failures")


if __name__ == "__main__":
    main()
