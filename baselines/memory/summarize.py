"""MemRL's memory-building calls, via the `claude -p` CLI.

Two call types, matching the official release:

  "script"      proceduralization of a *successful* trajectory into 3-5
                high-level steps (`memrl/providers/llm.py:generate_script`,
                run at the config temperature, 0.0 for ALFWorld);
  "reflection"  post-mortem of a *failed* trajectory
                (`memrl/service/updater.py:_generate_reflection`, temperature 0.3).

Both prompts are ports of those two, with the wording adapted from
household/coding tasks to 2D physics placement puzzles.

Two backends:

  `make_summarizer`            `claude -p` subprocess, same pattern as the actor.
                               Note that `claude -p` has no temperature flag
                               (`react_agent/run_react_claude.py:41`), so the
                               paper's 0.0 / 0.3 split is NOT reproducible here.
  `make_model_fn_summarizer`   any local `model_fn` (e.g. Qwen2.5-7B-Instruct via
                               `load_qwen_model`), which does take a temperature
                               — so this backend runs the script call at 0.0 and
                               the reflection call at 0.3 exactly as specified.
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import Callable, Optional

SCRIPT_SYSTEM = """You are analyzing a SUCCESSFUL attempt at a 2D physics placement puzzle.

You will receive the task (level description and scene geometry) and the agent's Thought/Action/Observation trajectory.

Produce a concise, high-level script that captures the essential steps and decision points. The script must be:
1. Generic enough to apply to similar puzzle instances (different seeds, shifted geometry).
2. Specific enough to provide useful guidance (name objects, give approximate coordinates or offsets, state the mechanism).
3. At most 5 high-level steps.
4. Focused on strategy and key decisions, not on individual tool calls.

Output ONLY the numbered script. No preamble, no headers, no markdown fences."""

REFLECTION_SYSTEM = """You are analyzing a FAILED attempt at a 2D physics placement puzzle.

You will receive the task (level description and scene geometry) and the agent's Thought/Action/Observation trajectory.

This attempt failed. Analyze what went wrong and suggest improvements for future similar tasks. Cover:
1. Incorrect assumptions the agent made about the physics or geometry.
2. Steps to improve — cite concrete coordinates, distances, or contact events from the observations.
3. What to avoid next time.

Keep it under 8 sentences. Output ONLY the reflection text. No preamble, no headers, no markdown fences."""

_SYSTEMS = {"script": SCRIPT_SYSTEM, "reflection": REFLECTION_SYSTEM}


def claude_cli_call(prompt: str, *, model: str = "claude-sonnet-4-6", timeout: int = 900) -> str:
    """One-shot `claude -p` call — same subprocess pattern as the actor."""
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
    except FileNotFoundError:
        print("[MemRL summarizer] 'claude' CLI not found. Install: https://claude.ai/code", flush=True)
        return ""
    except subprocess.TimeoutExpired:
        print(f"[MemRL summarizer] timed out after {timeout}s.", flush=True)
        return ""

    if result.returncode != 0:
        print(
            f"[MemRL summarizer] exit {result.returncode}: "
            f"{(result.stderr or '').strip()[:500]}",
            flush=True,
        )
        return ""
    return (result.stdout or "").strip()


def make_summarizer(
    *,
    model: str = "claude-sonnet-4-6",
    verbose: bool = False,
    call_fn: Optional[Callable[[str], str]] = None,
) -> Callable[[str, str, str], str]:
    """Return `summarizer(kind, intent, trajectory_text) -> str`."""

    def summarize(kind: str, intent: str, trajectory_text: str) -> str:
        system = _SYSTEMS.get(kind)
        if system is None:
            raise ValueError(f"Unknown summarizer kind: {kind}")
        prompt = (
            f"SYSTEM:\n{system}\n\n"
            f"USER:\nTask:\n{intent}\n\nTrajectory:\n{trajectory_text}\n\n"
            f"ASSISTANT:\n"
        )
        start = time.time()
        text = (call_fn or (lambda p: claude_cli_call(p, model=model)))(prompt)
        if verbose:
            print(
                f"  [MemRL {kind}] {len(text)} chars in {time.time() - start:.1f}s",
                flush=True,
            )
        return text

    return summarize


#: MemRL's configured temperatures: memory-building 0.0, adjustment 0.3.
SCRIPT_TEMPERATURE = 0.0
REFLECTION_TEMPERATURE = 0.3


def make_model_fn_summarizer(
    model_fn,
    *,
    max_new_tokens: int = 1024,
    script_temperature: float = SCRIPT_TEMPERATURE,
    reflection_temperature: float = REFLECTION_TEMPERATURE,
    verbose: bool = False,
) -> Callable[[str, str, str], str]:
    """Summarize with a local `model_fn(messages, temp, max_tokens) -> str`.

    Reuses whatever model the actor loaded (e.g. one Qwen2.5-7B-Instruct on the
    GPU serves both roles), so a run needs no second model in memory and no CLI
    subprocess.  Unlike the `claude -p` backend this honours MemRL's per-call
    temperatures.
    """

    def summarize(kind: str, intent: str, trajectory_text: str) -> str:
        system = _SYSTEMS.get(kind)
        if system is None:
            raise ValueError(f"Unknown summarizer kind: {kind}")
        temperature = (
            script_temperature if kind == "script" else reflection_temperature
        )
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Task:\n{intent}\n\nTrajectory:\n{trajectory_text}",
            },
        ]
        start = time.time()
        try:
            out = model_fn(messages, temperature, max_new_tokens)
        except Exception as exc:
            print(f"[MemRL summarizer] {type(exc).__name__}: {exc}", flush=True)
            return ""
        # OSS-style model_fns return {"thinking": ..., "content": ...}.
        if isinstance(out, dict):
            out = out.get("content") or ""
        text = str(out).strip()
        if verbose:
            print(
                f"  [MemRL {kind} @ T={temperature}] {len(text)} chars in "
                f"{time.time() - start:.1f}s",
                flush=True,
            )
        return text

    return summarize


def mock_summarizer(kind: str, intent: str, trajectory_text: str) -> str:
    """Deterministic stand-in for smoke tests (no LLM call)."""
    del intent
    label = "SCRIPT" if kind == "script" else "REFLECTION"
    return f"[mock {label}] {len(trajectory_text)} chars of trajectory summarized."
