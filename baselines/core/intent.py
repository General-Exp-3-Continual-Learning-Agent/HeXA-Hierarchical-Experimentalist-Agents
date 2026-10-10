"""Retrieval-key text for CORE problems: strip boilerplate before embedding.

Mirrors `baselines/orbit/intent.py`'s "geometry" mode -- the same problem that
motivated it here: a sentence embedder (and even character n-gram hashing)
measures similarity dominated by whatever text is *shared* across every item,
and for Tower of Hanoi that is ~90% of the prompt (the rules and instructions
are identical on every item; only the peg configuration lines vary). Measured
directly: embedding the full `problem_text` with MiniLM gives Hanoi's 60 train
items a pairwise-cosine std of 0.0012 (matchstick and mathgap are 0.10-0.13 on
the same embedder, so this is Hanoi-specific, not a general embedder problem);
character-hashing over the full text only gets to 0.0073, still below the
"nearly degenerate" line -- the boilerplate dominates n-gram frequency too.

`retrieval_text` extracts just the state description for tasks where the
distinguishing content is a small fraction of the prompt, and is the identity
function for tasks where it is not (Matchstick, MathGAP, ZebraLogic: the whole
problem statement varies meaningfully item to item, confirmed non-degenerate
above, so stripping anything there would throw signal away rather than noise).
"""

from __future__ import annotations

#: Tower of Hanoi's own prompt template (hexa_core.tasks -> CORE's
#: `tasks.single_turn` factory) always contains both markers, in this order:
#: "...Initial configuration:\n\n- Peg 0: ...\n\nGoal configuration:\n\n- Peg
#: 0: ...\n\nRules:\n\n- Only one disk...". Slicing between them keeps both the
#: initial AND goal peg layouts (the only two things that vary) and drops the
#: fixed rules/instructions paragraph.
_HANOI_START = "Initial configuration:"
_HANOI_END = "Rules:"


def _hanoi_state(problem_text: str) -> str:
    start = problem_text.find(_HANOI_START)
    end = problem_text.find(_HANOI_END, start + 1) if start >= 0 else -1
    if start < 0 or end < 0:
        # Template changed underneath us -- fail open to the full text rather
        # than silently embedding an empty string for every item, which would
        # make every retrieval a tie.
        return problem_text
    return problem_text[start:end]


_EXTRACTORS = {"tower-of-hanoi": _hanoi_state}


def retrieval_text(task_name: str, problem_text: str) -> str:
    """The text actually embedded for retrieval on this task."""
    extractor = _EXTRACTORS.get(task_name)
    return extractor(problem_text) if extractor else problem_text
