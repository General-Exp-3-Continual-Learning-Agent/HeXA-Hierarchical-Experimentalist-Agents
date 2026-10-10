"""MemRL's two memory-building prompts, written for DiscoverPhysics.

MemRL calls an LLM once per finished episode:

  "script"      proceduralization of a *successful* trajectory into a few
                high-level steps (`memrl/providers/llm.py:generate_script`,
                temperature 0.0);
  "reflection"  post-mortem of a *failed* one
                (`memrl/service/updater.py:_generate_reflection`, temp 0.3).

`baselines/memory/summarize.py` carries the InterPhyre wording and
`baselines/orbit/summarize.py` the ORBIT wording. These are the same two calls
with the task description swapped -- the method is the call structure, not the
prose.

Three things are stated explicitly, because each is a failure mode this
benchmark actually produces:

* **Do not name the law.** A memory is read by an agent facing a world whose
  hidden law may differ, and on the within-world arms it is read by an agent
  facing *the same* law -- which would make the memory an answer key rather
  than a lesson. Trajectories reaching the summarizer are already numeric- and
  law-form-redacted (`traj_adapter.redact_text`), and the prompt asks for
  experiment design rather than conclusions so the summarizer does not
  reconstruct what redaction removed. `bank_audit.scan_for_law_forms` over the
  finished store is the check.
* **Name only controls that exist.** The trajectory shows one world's
  experiment schema; a summary that turns it into advice exports that schema to
  worlds that do not accept it. This is the measured Episodic RAG failure --
  14/15 hubble episodes emitted a control key hubble rejects -- and the one
  thing a summariser can fix that verbatim retrieval cannot.
* **Say what was measured, not what was concluded.** The transferable content
  of a DiscoverPhysics episode is which experiment isolated which quantity.

`claude -p` exposes no temperature flag, so the paper's 0.0 / 0.3 split is not
reproducible through the CLI backend; recorded as a deviation on every run.
"""

from __future__ import annotations

from typing import Callable, Optional

from baselines.memory.summarize import (
    REFLECTION_TEMPERATURE,
    SCRIPT_TEMPERATURE,
    claude_cli_call,
)
SCRIPT_SYSTEM = """You are analyzing a SUCCESSFUL attempt to discover the hidden law of motion in a simulated physics world.

You will receive the world's setup (what is observable and which controls the agent may set) and the agent's Thought/Action/Observation trajectory. Numeric literals are masked as <n>.

Produce a concise, high-level script capturing the experimental strategy. It must be:
1. Generic enough to apply to a different world with a different hidden law.
2. Specific about EXPERIMENT DESIGN: which quantity was varied, what was held fixed, which measurement times or ranges isolated the effect.
3. At most 5 high-level steps.
4. Restricted to controls named in the world setup you are given. Never suggest a control that is not listed there.
5. Free of any claim about what the law turned out to BE. No functional forms, no named laws, no exponents, no constants. Describe what was measured and how, not what was concluded.

Output ONLY the numbered script. No preamble, no headers, no markdown fences."""

REFLECTION_SYSTEM = """You are analyzing a FAILED attempt to discover the hidden law of motion in a simulated physics world.

You will receive the world's setup (what is observable and which controls the agent may set) and the agent's Thought/Action/Observation trajectory. Numeric literals are masked as <n>.

This attempt failed. Analyze what went wrong at the level of EXPERIMENT DESIGN and suggest improvements for future discovery tasks. Cover:
1. What the agent failed to measure, or measured in a way that could not separate competing explanations.
2. Concrete design fixes -- which variable to sweep, over what range, what to hold fixed, which regime to sample.
3. What to avoid next time.

Constraints:
- Restrict every suggestion to controls named in the world setup you are given. Never suggest a control that is not listed there.
- Do not state or guess what the hidden law is. No functional forms, no named laws, no exponents, no constants.

Keep it under 8 sentences. Output ONLY the reflection text. No preamble, no headers, no markdown fences."""

_SYSTEMS = {"script": SCRIPT_SYSTEM, "reflection": REFLECTION_SYSTEM}


class EmptySummary(RuntimeError):
    """The summarizer produced nothing, twice. See `make_summarizer`."""


def make_summarizer(
    *, model: str = "sonnet", verbose: bool = False,
    call_fn: Optional[Callable[[str], str]] = None,
) -> Callable[[str, str, str], str]:
    """`summarizer(kind, intent, trajectory_text) -> str`, DiscoverPhysics wording.

    Built directly on `claude_cli_call` rather than wrapping
    `baselines.memory.summarize.make_summarizer`: that helper hard-codes the
    InterPhyre system prompts and takes no override, so wrapping it would send
    the placement-puzzle instructions as SYSTEM and demote these to task text.
    ORBIT sidesteps this through `make_model_fn_summarizer(systems=...)`, which
    needs a local `model_fn`; the DiscoverPhysics actor is the Claude CLI.

    `model` defaults to the actor's CLI alias: MemRL builds memory with the same
    model that produced the episode, matching HeXA's rule that a stronger
    teacher makes it distillation rather than self-improvement.
    """
    import time

    def summarize(kind: str, intent: str, trajectory_text: str) -> str:
        system = _SYSTEMS.get(kind)
        if system is None:
            raise ValueError(f"Unknown summarizer kind: {kind}")
        prompt = (
            f"SYSTEM:\n{system}\n\n"
            f"USER:\nWorld setup:\n{intent}\n\n"
            f"Trajectory:\n{trajectory_text}\n\n"
            f"ASSISTANT:\n"
        )
        call = call_fn or (lambda p: claude_cli_call(p, model=model))
        start = time.time()
        text = call(prompt)
        if not text.strip():
            # Retry once: a dropped call is transient, a refused one is not.
            print(f"  [MemRL {kind}] empty summary, retrying once", flush=True)
            text = call(prompt)
        if not text.strip():
            # Storing it anyway writes a memory whose SCRIPT/reflection body is
            # blank while everything around it looks healthy -- the record is
            # created, the run reports success, and the arm silently becomes
            # "inject a raw trajectory with an empty header". That happened: a
            # usage limit mid-run left the oscillator store with 1 hollow record
            # and hubble's with 5, and only a byte-level check of `content`
            # revealed it. Fail the arm instead; both passes resume from disk.
            raise EmptySummary(
                f"MemRL {kind} summarizer returned empty text twice. The arm is "
                f"stopped rather than storing a memory with no content -- rerun "
                f"the same command to resume once the model is available."
            )
        if verbose:
            print(f"  [MemRL {kind}] {len(text)} chars in "
                  f"{time.time() - start:.1f}s", flush=True)
        return text

    return summarize


__all__ = ["make_summarizer", "EmptySummary", "SCRIPT_SYSTEM", "REFLECTION_SYSTEM",
           "SCRIPT_TEMPERATURE", "REFLECTION_TEMPERATURE"]
