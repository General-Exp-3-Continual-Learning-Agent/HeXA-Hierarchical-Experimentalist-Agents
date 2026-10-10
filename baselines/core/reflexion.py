"""Reflexion (Shinn et al., 2023) on CORE's four single-turn tasks.

CORE's tasks have no multi-episode structure the way ORBIT's Maze/Mastermind
do (`baselines/orbit/reflexion.py`), so the ORBIT mapping (reflect between an
instance's own episodes, reset the chat) does not transfer: a CORE problem is
graded once, on one completion. The analogue that *does* transfer is
InterPhyre's own Reflexion baseline (`docs/BASELINES.md`): K independent
trials on the SAME item, reflecting on a failure before the next trial, no
memory transfer across items, and no persistent bank -- so unlike Episodic
RAG and MemRL there is no build phase for this method at all (matching
ORBIT Reflexion's "no build pass, no calibration" property).

Because a trial stops as soon as it succeeds, the per-K accuracies reported
below are **cumulative** ("solved within K trials"), not per-trial rates the
way ORBIT's non-early-stopping ep1/ep2/ep3 are: an item solved on trial 1
counts toward trial2_acc and trial3_acc too, since it was never re-attempted.
trial1_acc alone is the true first-attempt (no-reflection) rate.

The reflection call uses `generate_structured(...).final_text`, not the raw
scored text: CORE's own verifier reads `reasoning_text + final_text`
concatenated (`hexa_core/adapter.py`), but injecting the WHOLE reasoning
channel into the next trial's prompt would dump potentially thousands of
characters of scratch reasoning as if it were a lesson. `final_text` is the
model's own conclusion, mirroring exactly what `baselines/orbit/reflexion.py`
does for the same reason.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

# Generic across all four tasks, for the same reason `baselines/core/summarize.py`
# is: there is no shared vocabulary across an equation-stick puzzle, a word
# problem, a logic grid and a disk-move search worth hand-tuning per task, only
# "a reasoning problem with a single verified correct answer" (unlike ORBIT's
# Maze/Mastermind, which get their own wording since the game mechanics differ).
REFLECTION_SYSTEM = """You are analysing a FAILED attempt at a reasoning problem with a single verified correct answer (an equation-stick puzzle, a word problem, a logic grid, or a disk-move search).

You will be shown the problem and the solver's own reasoning-and-answer text. A verifier has already checked the final answer and confirmed it is WRONG -- you are not told what the right answer is, and you must not guess or assert one.

Write a short reflection (at most 5 sentences) that a fresh attempt at this SAME problem will read, with no memory of the attempt you are analysing. Cover:
(a) the specific step where the reasoning went wrong -- an arithmetic slip, a misread constraint, an unchecked case;
(b) what to verify before committing to a final answer next time;
(c) one concrete different approach to try.

Hard rules:
- Do NOT state or imply what the correct answer is; you were not told it.
- Do NOT repeat lessons already listed as prior reflections.
- Output ONLY the reflection text. No preamble, no headings, no markdown."""

#: Mirrors `baselines/orbit/reflexion.py`'s DEFAULT_REFLECTION_MAX_TOKENS and its
#: reasoning: on a hybrid-thinking model the budget spans reasoning + answer, so
#: too small a cap silently returns an empty answer channel. CORE runs at
#: reasoning_effort=high, so the same generous budget is used here.
DEFAULT_REFLECTION_MAX_TOKENS = 8192


def format_trial_core(
    problem_text: str, raw_text: str, parsed_answer: str, *, max_chars: int = 2000
) -> str:
    """Render one failed CORE trial for the reflector.

    `raw_text` is CORE's scored string (reasoning + final, concatenated), which
    can run long under reasoning_effort=high; truncated head+tail like
    `baselines/orbit/reflexion.py.format_trial` truncates observations.
    """
    text = raw_text or ""
    if len(text) > max_chars:
        head = max_chars // 2
        text = text[:head] + " ...[truncated]... " + text[-(max_chars - head):]
    return (
        f"Problem:\n{problem_text}\n\n"
        f"Solver's reasoning and answer:\n{text}\n\n"
        f"Parsed final answer: {parsed_answer}\n"
        f"Verdict: INCORRECT"
    )


def reflect_core(
    generate_structured: Callable[..., Any],
    problem_text: str,
    raw_text: str,
    parsed_answer: str,
    past_reflections: Sequence[str] = (),
    *,
    max_new_tokens: int = DEFAULT_REFLECTION_MAX_TOKENS,
) -> str:
    """One reflection on one failed CORE trial. ``""`` means "no lesson".

    ``generate_structured`` is `VLLMClient.generate_structured`-shaped: called
    as `generate_structured(prompt=..., system_prompt=..., max_new_tokens=...)`
    and returning an object with `.final_text` / `.reasoning_text` /
    `.finish_reason`, or None on failure. Degrades to "" on any failure path
    rather than raising -- a dead reflection call must fall the trial back to
    the no-reflection condition, not kill the whole arm.
    """
    past_block = ""
    if past_reflections:
        past_block = "\n\nPrior reflections (do not repeat these):\n" + "\n".join(
            f"{i}. {r.strip()}" for i, r in enumerate(past_reflections, 1)
        )
    user = (
        format_trial_core(problem_text, raw_text, parsed_answer)
        + past_block
        + "\n\nWrite the reflection now. At most 5 sentences."
    )

    # Two attempts, second at double budget -- see DEFAULT_REFLECTION_MAX_TOKENS.
    for attempt, budget in enumerate((max_new_tokens, max_new_tokens * 2)):
        try:
            generation = generate_structured(
                prompt=user, system_prompt=REFLECTION_SYSTEM, max_new_tokens=budget,
            )
        except Exception as exc:  # noqa: BLE001 -- see docstring
            print(f"[reflexion] reflection call failed: {type(exc).__name__}: {exc}", flush=True)
            return ""
        if generation is None:
            print("[reflexion] reflection call returned no answer", flush=True)
            return ""
        text = (generation.final_text or "").strip()
        if text:
            return text
        reasoned = len(generation.reasoning_text or "")
        print(
            f"[reflexion] empty answer channel at budget {budget} "
            f"(finish={generation.finish_reason}, {reasoned} reasoning chars)"
            + ("; retrying" if attempt == 0 else "; giving up"),
            flush=True,
        )
    return ""


def reflexion_block(reflections: Sequence[str]) -> Optional[str]:
    """The prompt block for a retry, or ``None`` for an untouched prompt.

    Identical in shape to `baselines/orbit/reflexion.py`'s function of the same
    name (copied rather than imported, so this module has no dependency on
    `hexa_orbit`). ``None`` rather than ``""`` so trial 1 -- and any trial whose
    reflections all came back empty -- is byte-identical to the no-reflection
    prompt: `Problem.build_prompt(None)` reproduces the base prompt exactly,
    which is what makes trial 1 usable as a first-attempt control.
    """
    kept = [r.strip() for r in reflections if r and r.strip()]
    if not kept:
        return None
    lines = [
        "## Lessons from your previous attempts at THIS problem",
        "You have attempted this exact problem before and failed. You cannot see",
        "those attempts, but you wrote down what went wrong:",
        "",
    ]
    lines += [f"{i}. {r}" for i, r in enumerate(kept, 1)]
    lines += [
        "",
        "Use these lessons to try a different approach. Do not repeat the "
        "reasoning that already failed.",
    ]
    return "\n".join(lines)
