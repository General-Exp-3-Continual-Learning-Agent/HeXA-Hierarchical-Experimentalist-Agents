"""MemRL's memory-building calls (script / reflection) for CORE's reasoning tasks.

Adaptation of `baselines/memory/summarize.py`'s wording from 2D physics placement
to CORE's four reasoning tasks (equation-stick puzzles, word problems, logic
grids, disk-move search) -- generic across all four rather than per-task, since
unlike ORBIT's game-specific wording (`baselines/orbit/summarize.py`) there is no
shared vocabulary across Matchstick/MathGAP/ZebraLogic/Hanoi worth hand-tuning
per task, only "a reasoning problem with a verified right answer".

Per HeXA's own actor/teacher-parity rule, the summarizer here calls the SAME
model serving the actor (`generate`), not a second model -- the port keeps
MemRL's official split of a 0.0-temperature script call and a 0.3-temperature
reflection call (`configs/rl_alf_config.yaml`), which `VLLMClient.generate`
exposes directly, unlike the `claude -p` backend `baselines/memory/summarize.py`
notes cannot honour it.
"""

from __future__ import annotations

from typing import Callable

SCRIPT_SYSTEM = """You are analyzing a SUCCESSFUL solution to a reasoning problem with a single verified correct answer (an equation-stick puzzle, a word problem, a logic grid, or a disk-move search).

You will receive the problem statement and the solver's full reasoning-plus-answer text.

Produce a concise, high-level script that captures the essential solution strategy. The script must be:
1. Generic enough to apply to similar problem instances (different numbers, different grid entries, different disk counts).
2. Specific enough to provide useful guidance (name the technique, the key constraint check, or the invariant that made this solvable).
3. At most 5 high-level steps.
4. Focused on strategy and the decisive insight, not a transcription of the arithmetic.

Output ONLY the numbered script. No preamble, no headers, no markdown fences."""

REFLECTION_SYSTEM = """You are analyzing a FAILED attempt at a reasoning problem with a single verified correct answer (an equation-stick puzzle, a word problem, a logic grid, or a disk-move search).

You will receive the problem statement and the solver's full reasoning-plus-answer text, which the verifier scored incorrect.

This attempt failed. Analyze what went wrong and suggest improvements for future similar problems. Cover:
1. The incorrect assumption or arithmetic/logic slip that produced the wrong answer.
2. What to check or verify before committing to a final answer next time.
3. What to avoid repeating.

Keep it under 8 sentences. Output ONLY the reflection text. No preamble, no headers, no markdown fences."""

_SYSTEMS = {"script": SCRIPT_SYSTEM, "reflection": REFLECTION_SYSTEM}

#: MemRL's configured temperatures (`baselines/memory/summarize.py`).
SCRIPT_TEMPERATURE = 0.0
REFLECTION_TEMPERATURE = 0.3


def make_core_summarizer(
    generate: Callable[..., str | None],
    *,
    max_new_tokens: int = 1024,
    verbose: bool = False,
) -> Callable[[str, str, str], str]:
    """Return `summarizer(kind, intent, trajectory_text) -> str`.

    `generate` is `VLLMClient.generate(prompt, temperature=..., max_new_tokens=...)`
    -- a plain prompt string in, scored text out, exactly the actor's own
    interface, deliberately unseeded (script/reflection calls are not part of
    the paired-eval seed accounting and MemRL's own release does not seed them).
    """

    def summarize(kind: str, intent: str, trajectory_text: str) -> str:
        system = _SYSTEMS.get(kind)
        if system is None:
            raise ValueError(f"Unknown summarizer kind: {kind}")
        temperature = SCRIPT_TEMPERATURE if kind == "script" else REFLECTION_TEMPERATURE
        prompt = (
            f"{system}\n\n"
            f"Problem:\n{intent}\n\n"
            f"Solver's reasoning and answer:\n{trajectory_text}\n"
        )
        text = (
            generate(prompt, temperature=temperature, max_new_tokens=max_new_tokens) or ""
        ).strip()
        if not text:
            print(f"  [MemRL {kind}] WARNING: summarizer returned empty text", flush=True)
        elif verbose:
            print(f"  [MemRL {kind} @ T={temperature}] {len(text)} chars", flush=True)
        return text

    return summarize
