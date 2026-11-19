"""Route DiscoverPhysics LLM calls through the Claude Code CLI — no API key.

DiscoverPhysics makes every model call through `scienceagent.llm_client.complete`
(actor = DiscoveryAgent, judge = ExplanationJudge, supervisor = CriticAgent).
`patch_llm_client()` replaces that one function with a CLI-backed implementation
that shells out to `claude -p ...`, mirroring HeXA's `call_teacher` /
`load_claude_cli_model` pattern. The teacher (`skillrl...distill.call_teacher`)
already uses the CLI for `claude-*` model names, so it needs no change.

The model string passed to `complete()` is forwarded to the CLI's `--model`
(so `--model sonnet` / `--model opus` / a full `claude-*` id all work); pass a
`model_override` to force one model regardless of caller.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Optional

_SESSION_LIMIT_MARKERS = ("session limit", "usage limit", "rate limit")


class SessionLimitError(RuntimeError):
    """CLI usage window exhausted — propagate so a run stops cleanly."""


def _flatten(messages: list[dict], system: Optional[str]) -> str:
    """Flatten a system prompt + chat messages into one CLI prompt string.

    Unlike HeXA's ReAct flattener we append NO 'emit one Thought/Action'
    instruction — DiscoveryAgent expects raw XML tags (<run_experiment>,
    <final_law>, ...), which its own system prompt already specifies.
    """
    parts: list[str] = []
    if system:
        parts.append(f"SYSTEM:\n{system}\n")
    for m in messages:
        parts.append(f"{m['role'].upper()}:\n{m['content']}\n")
    parts.append("ASSISTANT:")
    return "\n".join(parts)


def cli_complete(
    model: str,
    messages: list[dict],
    system: Optional[str] = None,
    max_tokens: int = 4096,  # accepted for signature parity; CLI manages length
    *,
    model_override: Optional[str] = None,
    timeout: int = 2000,
) -> str:
    """Drop-in replacement for scienceagent.llm_client.complete via `claude -p`."""
    prompt = _flatten(messages, system)
    cli_model = model_override or model
    # The prompt goes on STDIN, not argv. Linux caps a single argument at
    # MAX_ARG_STRLEN = 128 KiB regardless of the much larger ARG_MAX, and the
    # actor's prompt is the whole conversation so far: system prompt + every
    # round's experiment output. DiscoverPhysics observations are raw numeric
    # trajectories, so a long episode on a many-particle world crosses that
    # limit and `subprocess.run` raises before the process starts:
    #     OSError: [Errno 7] Argument list too long: 'claude'
    # Measured: this killed the loo_fracfamily run mid-episode on
    # extra_dimensions. The same bug was fixed for the teacher in
    # `teacher_cli.py`; the actor path was missed because short episodes stay
    # under the cap, so it only bites on the longest conversations.
    cmd = [
        "claude", "-p",
        "--model", cli_model,
        "--max-turns", "1",
        "--tools", "",
        "--no-session-persistence",
        "--output-format", "json",
    ]
    start = time.time()
    try:
        result = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True,
            env=os.environ.copy(), timeout=timeout,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "'claude' CLI not found. Install Claude Code: https://claude.ai/code"
        )
    except subprocess.TimeoutExpired:
        print(f"[cli] timeout after {time.time() - start:.0f}s", flush=True)
        return ""

    blob = ((result.stdout or "") + " " + (result.stderr or "")).lower()
    if result.returncode != 0:
        if any(m in blob for m in _SESSION_LIMIT_MARKERS):
            raise SessionLimitError(
                (result.stdout or result.stderr or "")[:200]
            )
        print(f"[cli] exit {result.returncode}: {(result.stderr or '')[:300]}", flush=True)
        return ""

    try:
        return json.loads(result.stdout).get("result", "").strip()
    except (json.JSONDecodeError, AttributeError):
        # Fall back to raw stdout if --output-format json wasn't honored.
        return (result.stdout or "").strip()


_ORIG_COMPLETE = None


def patch_llm_client(model_override: Optional[str] = None) -> None:
    """Install the CLI backend into scienceagent.llm_client.complete (idempotent).

    Also mirrors it onto the already-imported module bindings the agent/judge/
    critic captured, so patching works regardless of import order.
    """
    global _ORIG_COMPLETE
    import scienceagent.llm_client as _lc

    if getattr(_lc.complete, "_is_cli_shim", False):
        return
    _ORIG_COMPLETE = _lc.complete

    def _shim(model, messages, system=None, max_tokens=4096):
        return cli_complete(
            model, messages, system, max_tokens, model_override=model_override
        )

    _shim._is_cli_shim = True  # type: ignore[attr-defined]
    _lc.complete = _shim
