"""Added-token metric checks: clamping, monotonicity, and an honest counter label.

Run with::

    PYTHONDONTWRITEBYTECODE=1 python -m hexa_core.tests.test_tokens

Imports only :mod:`hexa_core.tokens` and stdlib -- deliberately not ``skillrl``
or CORE. This suite backs CORE's Figure 3 number (added task-external context per
eval item: CORE 0.92k, GEPA 1.29k, Episodic RAG 33.6k, MemRL 32.7k), and a
token-counting test that fails because the skill bank moved would be worse than
no test.

Which counter is live is an environment fact, not an assumption: ``tiktoken`` was
absent when this suite was written (making the word/char heuristic the live path)
and is present in the vLLM env used for the pilots (making counts exact BPE).
Every assertion here holds either way, branching on :func:`hexa_core.tokens.is_exact`
where the two genuinely differ, and
:func:`test_counter_name_is_honest_about_the_live_path` asserts the label names
whichever one actually ran.
"""

from __future__ import annotations

import importlib.util
import math
import re
import time

from hexa_core.tests import check
from hexa_core.tokens import (
    TOKENIZER_NOTE,
    added_tokens,
    count_tokens,
    counter_name,
    is_exact,
)

_SKILL_TEMPLATE = (
    "{i}. [Strategy: {title}] {principle}\n"
    "   Apply when: {when}\n"
    "   Example: {example}\n"
)

_MISTAKE_TEMPLATE = (
    "{i}. **Emitting the move list at the wrong nesting depth (variant {i}).**\n"
    "   Why it happens: the required format is a list of [disk, from_peg, to_peg]\n"
    "   triples, and it is easy to flatten it into a single list of integers.\n"
    "   Instead: emit exactly one bracketed list of three-element lists, and\n"
    "   nothing after it.\n"
)


def _skills_block(n_skills: int, n_mistakes: int) -> str:
    """Text shaped like ``SkillBank.format_skills_as_prompt`` output.

    Heading and section labels follow the relabelled form
    (``hexa_core.hexa_bridge.SKILLS_BLOCK_HEADING``) rather than the InterPhyre
    original, and the entry layout follows ``skill_bank.py:149-175``.
    """
    lines = [
        "=== LEARNED PROBLEM-SOLVING STRATEGIES ===",
        "",
        "## Tower-Of-Hanoi-Specific Skills",
    ]
    for i in range(1, n_skills + 1):
        lines.append(
            _SKILL_TEMPLATE.format(
                i=i,
                title=f"Restate peg occupancy before committing to move {i}",
                principle=(
                    "Before emitting any move, restate the current tuple of top disks per "
                    "peg and confirm the disk being moved is strictly smaller than the "
                    "destination top disk; one illegal move invalidates the entire "
                    "remaining sequence."
                ),
                when=(
                    "Whenever the response runs longer than a dozen moves and the "
                    "intermediate state is no longer obvious from the goal description "
                    "alone."
                ),
                example=(
                    "For a 6-disk instance starting from a scrambled state, write the "
                    "state after every third move so a mistake is caught locally rather "
                    "than at the end of a 63-move list."
                ),
            )
        )
    lines.append("## Common Mistakes to Avoid")
    for i in range(1, n_mistakes + 1):
        lines.append(_MISTAKE_TEMPLATE.format(i=i))
    return "\n".join(lines)


#: 6 skills + 3 mistakes. Measured under the live heuristic counter:
#: chars=4713, words=775, count_tokens=1178. Sized past 0.92k on purpose --
#: hitting CORE's reported figure with a hand-written fixture would be arithmetic
#: luck, and nobody should be able to read this suite as validating that number.
SKILLS_BLOCK = _skills_block(6, 3)

#: A plausible CORE base prompt to inject against: long enough that the injected
#: block is not the whole string, so the subtraction is doing real work.
BASE_PROMPT = (
    "Solve the following Tower of Hanoi instance. Move all disks to the goal "
    "configuration using legal moves only.\n"
) * 12

#: One extra skill entry, 67 tokens, used as the monotonicity increment.
GROWTH = _SKILL_TEMPLATE.format(
    i=9,
    title="Check the parity of the disk count before choosing a first move",
    principle="P" * 80,
    when="W" * 40,
    example="E" * 40,
)

#: Many short words: 216 tokens on its own via the ``words * 1.35`` branch, while
#: :data:`BASE_PROMPT` is scored by the ``chars // 4`` branch. Appending it moves
#: the combined string across the branch boundary, which is why ``added_tokens``
#: is not additive and why the band assertion below is a band.
WORD_DENSE = "a bb cc dd ee ff gg hh " * 20


def test_added_tokens_zero_when_identical() -> None:
    prompts = ("", "one line of prompt", BASE_PROMPT, SKILLS_BLOCK, BASE_PROMPT + SKILLS_BLOCK)
    for prompt in prompts:
        assert added_tokens(prompt, prompt) == 0, repr(prompt[:40])
    check(f"added_tokens is 0 when augmented == base ({len(prompts)} prompts)", True)


def test_added_tokens_clamps_at_zero_when_shorter() -> None:
    """A shrinking augmentation reads as zero, never as a negative credit."""
    long_prompt = BASE_PROMPT + SKILLS_BLOCK
    for shorter in (long_prompt[:50], long_prompt[: len(long_prompt) // 2], ""):
        # Prove the clamp fired rather than the case passing vacuously.
        assert count_tokens(shorter) < count_tokens(long_prompt)
        assert added_tokens(long_prompt, shorter) == 0, repr(shorter[:30])
    check("added_tokens clamps at 0 when the augmented prompt is shorter", True)


def test_added_tokens_monotonic_in_added_content() -> None:
    increment = count_tokens(GROWTH)
    assert increment > 20, f"the growth fixture is too small to be a real increment: {increment}"

    previous = -1
    for k in range(8):
        current = added_tokens(BASE_PROMPT, BASE_PROMPT + GROWTH * k)
        assert current >= 0
        assert current > previous, f"k={k}: {current} did not exceed {previous}"
        previous = current
    check(f"added_tokens strictly increases over 8 repetitions of a {increment}-token entry", True)

    # count_tokens itself is non-decreasing over growing prefixes.
    block = BASE_PROMPT + SKILLS_BLOCK
    counts = [count_tokens(block[:cut]) for cut in range(0, len(block) + 1, 137)]
    check(
        f"count_tokens is monotone over {len(counts)} growing prefixes",
        counts == sorted(counts),
    )


def test_added_tokens_tracks_the_injected_block() -> None:
    """The delta must be within a band of the block's own size, not equal to it.

    ``count_tokens`` is ``max(words * 1.35, chars // 4)``, so it is not additive in
    general: the branch that wins can differ between ``base`` and
    ``base + block``. Asserting equality would be asserting an accident of this
    fixture -- both strings happen to land in the char-dense branch, so the delta
    *is* exactly additive for ``SKILLS_BLOCK`` today. :data:`WORD_DENSE` is the
    counterexample that keeps the band honest.
    """
    block_tokens = count_tokens(SKILLS_BLOCK)
    delta = added_tokens(BASE_PROMPT, BASE_PROMPT + SKILLS_BLOCK)
    check(
        f"a {block_tokens}-token skills block adds {delta} tokens (within band)",
        0.5 * block_tokens <= delta <= block_tokens + 2,
        f"delta={delta} outside band for a {block_tokens}-token block",
    )

    # Non-additivity is a property of the *heuristic* (`max(words*1.35, chars//4)`:
    # the winning branch can differ between `base` and `base + block`), not of the
    # metric. A real BPE tokenizer concatenates additively except where the join seam
    # merges, so the counterexample only applies when the heuristic is the live path.
    # Asserting it unconditionally made an exact tokenizer look like a regression.
    dense_tokens = count_tokens(WORD_DENSE)
    dense_delta = added_tokens(BASE_PROMPT, BASE_PROMPT + WORD_DENSE)
    if is_exact():
        check(
            f"exact counter is additive at the seam: {dense_tokens} -> {dense_delta}",
            abs(dense_delta - dense_tokens) <= 2,
            f"BPE delta={dense_delta} for a {dense_tokens}-token block",
        )
    else:
        check(
            f"heuristic word-dense block: {dense_tokens} tokens -> {dense_delta} added, "
            f"so not additive",
            dense_delta != dense_tokens,
            "the heuristic turned additive; the band above could be tightened",
        )
    assert 0.5 * dense_tokens <= dense_delta <= dense_tokens + 2, (
        f"dense delta={dense_delta} outside band for a {dense_tokens}-token block"
    )


def test_counter_name_is_honest_about_the_live_path() -> None:
    """The label must name the method that actually ran, and must not overclaim."""
    have_tiktoken = importlib.util.find_spec("tiktoken") is not None
    name = counter_name()
    if have_tiktoken:
        assert name == "tiktoken:o200k_base", name
    else:
        assert name.startswith("heuristic:"), name
        assert "words+chars" in name, name
        # A heuristic label naming a real tokenizer would make every added-token
        # figure in a run manifest look citable when it is not.
        assert "tiktoken" not in name, name

    # The lru_cached encoder must not make the label flap between calls.
    assert counter_name() == counter_name() == name
    assert "token_counter" in TOKENIZER_NOTE, TOKENIZER_NOTE
    check(f"counter_name() names the live path: {name} (tiktoken installed={have_tiktoken})", True)


def test_heuristic_is_within_a_sane_factor_of_a_hand_reference() -> None:
    """Bracket the ~1k-token block between two independently hand-derivable bounds.

    * lower = whitespace-delimited word count. Every word costs at least one
      token under any BPE vocabulary.
    * upper = ``sum(ceil(len(w) / 3) for w in words)`` plus one token per
      punctuation character. No BPE merge table for English prose is coarser
      than roughly one token per three characters, and punctuation is the
      worst case for splitting.

    Measured on ``SKILLS_BLOCK``: words=775, count=1178, upper=1686.
    """
    text = SKILLS_BLOCK
    words = text.split()
    lower = len(words)
    upper = sum(math.ceil(len(word) / 3) for word in words) + len(re.findall(r"[^\w\s]", text))
    count = count_tokens(text)
    detail = f"words={lower} count={count} upper={upper} chars={len(text)}"

    assert 900 <= count <= 1400, f"fixture drifted away from ~1k tokens: {detail}"
    check(
        f"one-token-per-word floor <= heuristic <= subword ceiling ({detail})",
        lower <= count <= upper,
        detail,
    )

    ratio = count / (len(text) / 4)
    check(
        f"heuristic is {ratio:.2f}x the chars/4 rule of thumb",
        1 / 1.6 <= ratio <= 1.6,
        detail,
    )


def test_count_tokens_edge_cases() -> None:
    assert count_tokens("") == 0
    assert count_tokens(" ") >= 0
    assert count_tokens("\n\n\t") >= 0
    assert count_tokens("a") >= 1

    # The metric is computed per eval item (100 items x 3 runs x 3 tasks), so a
    # counter that is accidentally quadratic in length matters.
    big = "the quick brown fox jumps over the lazy dog. " * 4500
    assert len(big) > 200_000
    started = time.perf_counter()
    tokens = count_tokens(big)
    elapsed = time.perf_counter() - started
    assert tokens > 0
    check(
        f"count_tokens handled {len(big)} chars in {elapsed * 1000:.1f}ms",
        elapsed < 1.0,
        f"took {elapsed:.3f}s",
    )


if __name__ == "__main__":
    import sys

    from hexa_core.tests import run_module

    sys.exit(run_module(sys.modules[__name__]))
