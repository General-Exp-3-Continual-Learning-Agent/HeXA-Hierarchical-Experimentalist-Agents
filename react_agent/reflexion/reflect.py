"""Self-reflection LLM call via the same `claude -p` CLI subprocess used by
the actor. No Anthropic SDK / API key required.

Mirrors the subprocess invocation in run_react_claude.py:42-55.
"""

import os
import subprocess
import time

REFLECTION_SYSTEM = """You are analyzing a failed attempt at a 2D physics puzzle.

You will receive the full Thought/Action/Observation trajectory from a single failed attempt by a ReAct-style agent, followed by the final outcome and any prior reflections accumulated on this same task.

Your job: produce a short reflection (≤5 sentences) that the agent will read before its next attempt on the SAME task. Cover:
(a) which strategy/approach the agent pursued in this attempt;
(b) the specific kinematic, geometric, or procedural reason it failed (cite concrete coordinates, distances, or contact events from the observations);
(c) one concrete different action or strategy to try next — be specific (object names, approximate (x, y, radius), expected mechanism).

Hard rules:
- Do NOT repeat lessons that already appear in the prior reflections list.
- Do NOT re-state the goal or the puzzle rules.
- Output ONLY the reflection text. No preamble, no headers, no markdown."""


def _format_trajectory_compact(trajectory, max_obs_chars: int = 600) -> str:
    """Format a trajectory (list of (thought, action, observation) tuples OR
    list of dicts with those keys) into a compact numbered text dump.
    Long observations are truncated head+tail to keep the prompt manageable.
    """
    lines = []
    for i, step in enumerate(trajectory, 1):
        if isinstance(step, dict):
            t = step.get("thought") or ""
            a = step.get("action") or ""
            o = step.get("observation") or ""
        else:
            # Tuple form: (thought, action, observation)
            t = step[0] if len(step) > 0 else ""
            a = step[1] if len(step) > 1 else ""
            o = step[2] if len(step) > 2 else ""
        if o and len(o) > max_obs_chars:
            head = max_obs_chars // 2
            tail = max_obs_chars - head
            o = o[:head] + " ...[truncated]... " + o[-tail:]
        lines.append(f"--- Step {i} ---")
        if t:
            lines.append(f"Thought: {t.strip()}")
        if a:
            lines.append(f"Action: {a.strip()}")
        if o:
            lines.append(f"Observation: {o.strip()}")
    return "\n".join(lines)


def reflect_on_trajectory(
    trajectory,
    final_observation: str,
    success: bool,
    past_reflections: list,
    model: str = "claude-sonnet-4-6",
    timeout: int = 300,
) -> str:
    """Generate a Reflexion-style reflection on a failed trajectory.

    Returns the reflection text, or an empty string on success / empty trajectory /
    CLI error. Empty return is treated as "no new lesson" by the caller.
    """
    if success:
        return ""
    if not trajectory:
        return ""

    traj_text = _format_trajectory_compact(trajectory)

    past_block = ""
    if past_reflections:
        past_lines = ["", "## Prior reflections (do not repeat these)"]
        for i, r in enumerate(past_reflections, 1):
            past_lines.append(f"{i}. {r.strip()}")
        past_block = "\n".join(past_lines)

    final_obs_block = ""
    if final_observation:
        final_obs_block = f"\nFinal observation:\n{final_observation.strip()[:1500]}\n"

    user_msg = (
        "Outcome: FAILURE.\n"
        + final_obs_block
        + f"\nTrajectory:\n{traj_text}\n"
        + past_block
        + "\n\nWrite the reflection now. Be specific and concise (≤5 sentences)."
    )

    # Flatten into a single text prompt — same convention as
    # run_react_claude.py:30-37.
    prompt = (
        f"SYSTEM:\n{REFLECTION_SYSTEM}\n\n"
        f"USER:\n{user_msg}\n\n"
        f"ASSISTANT:\n"
    )

    print(f"\n[Reflexion] Reflecting on failed trial — prompt length: {len(prompt)}", flush=True)
    call_start = time.time()
    try:
        result = subprocess.run(
            [
                "claude", "-p", prompt,
                "--model", model,
                "--max-turns", "1",
                "--tools", "",
                "--no-session-persistence",
                "--output-format", "text",
            ],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            timeout=timeout,
        )
        elapsed = time.time() - call_start

        if result.stderr:
            err_short = result.stderr.strip()[:300]
            print(f"[Reflexion stderr]: {err_short}", flush=True)
        if result.returncode != 0:
            err = (result.stderr or "")[:500]
            std = (result.stdout or "")[:500]
            print(
                f"[Reflexion Error] Exit code {result.returncode} after {elapsed:.1f}s. "
                f"stderr: {err} | stdout: {std}",
                flush=True,
            )
            return ""

        reflection = result.stdout.strip()
        print(f"[Reflexion] Got reflection in {elapsed:.1f}s ({len(reflection)} chars).", flush=True)
        return reflection

    except FileNotFoundError:
        print("[Reflexion Error] 'claude' CLI not found. Install: https://claude.ai/code", flush=True)
        return ""
    except subprocess.TimeoutExpired:
        print(f"[Reflexion Error] Timed out after {timeout}s.", flush=True)
        return ""
    except Exception as e:
        print(f"[Reflexion Error] Unexpected: {type(e).__name__}: {e}", flush=True)
        return ""
