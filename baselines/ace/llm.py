"""Text-in/text-out calls for ACE's Reflector and Curator.

ACE uses one model for all three roles (paper §4.2: "preventing knowledge
transfer from a stronger Reflector or Curator to a weaker Generator"), so the
Reflector and Curator here run on whatever backend the actor uses:

  HF path / openai:<name>   the actor's own `model_fn(messages, temp, max_tokens)`.
                            For a local Qwen that is the same weights already on
                            the GPU, so a run needs no second model.
  claude                    `claude -p` with the prompt on **stdin**.  The actor's
                            CLI `model_fn` cannot be reused: it appends a
                            "output ONLY your next Thought and Action" turn that
                            would corrupt a JSON reflection.  stdin rather than
                            argv because a full playbook + 25-step trajectory can
                            exceed the 128 KiB argv cap.

Temperature 0.0 is ACE's setting (`llm.py:timed_llm_call`).  Two failure modes
are handled rather than swallowed:

  * an empty answer (a thinking model spent its budget reasoning) is retried
    once at 2x the token budget, then returned empty and counted by the caller;
  * a backend that rejects temperature 0 for sampling (HF `generate` with
    `do_sample=True`) is retried at 1e-5, which is greedy in effect.
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import Callable, Optional

ACE_TEMPERATURE = 0.0
_GREEDY_FALLBACK = 1e-5


def claude_stdin_call(prompt: str, *, model: str, timeout: int = 900) -> str:
    try:
        result = subprocess.run(
            [
                "claude", "-p",
                "--model", model,
                "--max-turns", "1",
                "--tools", "",
                "--no-session-persistence",
                "--output-format", "text",
            ],
            input=prompt,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            timeout=timeout,
        )
    except FileNotFoundError:
        print("[ACE llm] 'claude' CLI not found.", flush=True)
        return ""
    except subprocess.TimeoutExpired:
        print(f"[ACE llm] claude timed out after {timeout}s.", flush=True)
        return ""
    if result.returncode != 0:
        print(
            f"[ACE llm] claude exit {result.returncode}: "
            f"{(result.stderr or result.stdout or '').strip()[:500]}",
            flush=True,
        )
        return ""
    return (result.stdout or "").strip()


def make_text_llm(
    model_name: str,
    model_fn: Optional[Callable] = None,
    *,
    claude_model: str = "claude-sonnet-4-6",
    temperature: float = ACE_TEMPERATURE,
) -> Callable[[str, int], str]:
    """Return `call(prompt, max_tokens) -> str` on the actor's backend.

    ACE sends every role prompt as a single user message
    (`llm.py:timed_llm_call`), so this does too.
    """
    if model_name == "claude":
        def call(prompt: str, max_tokens: int) -> str:
            del max_tokens  # `claude -p` exposes no output cap
            return claude_stdin_call(prompt, model=claude_model)
        return call

    if model_name == "mock":
        # Plumbing smoke tests only: MockModel emits random ReAct actions, so the
        # roles answer empty and every reflection/curation is counted as such.
        return lambda prompt, max_tokens: ""

    if model_fn is None:
        raise ValueError("model_fn is required for non-claude backends")

    state = {"temperature": temperature}

    def _once(prompt: str, max_tokens: int) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            out = model_fn(messages, state["temperature"], max_tokens)
        except ValueError as exc:
            if state["temperature"] > 0 or "temperature" not in str(exc).lower():
                raise
            print(
                f"[ACE llm] backend rejected temperature 0 ({exc}); using "
                f"{_GREEDY_FALLBACK} for the rest of the run",
                flush=True,
            )
            state["temperature"] = _GREEDY_FALLBACK
            out = model_fn(messages, state["temperature"], max_tokens)
        if isinstance(out, dict):  # Qwen3-thinking loaders return {thinking, content}
            out = out.get("content") or ""
        return str(out).strip()

    def call(prompt: str, max_tokens: int) -> str:
        started = time.time()
        text = _once(prompt, max_tokens)
        if not text:
            print(
                f"[ACE llm] empty answer at max_tokens={max_tokens}; retrying "
                f"at {2 * max_tokens}",
                flush=True,
            )
            text = _once(prompt, 2 * max_tokens)
        print(f"[ACE llm] {len(text)} chars in {time.time() - started:.1f}s", flush=True)
        return text

    return call
