"""The length-matched placebo bank -- arm C of DESIGN_PHASE3_MAZE §13.

Arm B injects ~1000 tokens of confident, task-relevant prose into the system
prompt; arm A injects nothing. So if B beats A, two explanations fit the data
equally well: *the bank taught the agent something*, and *a longer, more focused
system prompt helps*. Nothing in the A/B contrast separates them, which is why
this arm is not optional.

**What is held fixed and what is destroyed.** A skill is useful because of a
mapping: a trigger (``when_to_apply``) paired with an action (``principle``).
:func:`scramble_bank` keeps every skill's ``title`` and ``when_to_apply`` where
they are and **deranges the ``principle`` and ``example`` across skills**, so the
injected block keeps its token count, register, task vocabulary, structure and
apparent specificity, and loses only the correspondence. A derangement (no fixed
points) is used rather than a shuffle, because a random permutation leaves some
skills intact and those would still carry real advice.

**The bias this introduces, stated up front.** A deranged block is internally
incoherent -- a trigger about corridor choice followed by advice about turn
budgets -- and incoherent instructions may *actively mislead* rather than merely
fail to help. So arm C is not a clean "same tokens, no information" floor; it can
land below arm A. Read the three arms together:

* ``A < C ≈ B`` -- the gain is prose, not content. The mechanism is not what is
  working.
* ``A ≈ C < B`` -- the trigger→action mapping is doing the work. This is the
  result the design predicts.
* ``C < A < B`` -- the placebo actively confused the agent, so C is uninformative
  as a floor. B vs A still stands, but C settles nothing and should be reported
  as inconclusive rather than as support.

:func:`foreign_bank` is the alternative control for when that ambiguity matters:
internally coherent skills that are simply about the wrong task. It trades the
coherence problem for a register mismatch, so neither control is strictly better
and running both is cheap.
"""

from __future__ import annotations

import copy
import random
from dataclasses import replace
from typing import Any, Optional, Sequence

from hexa_core.tokens import count_tokens

#: Fraction by which a placebo block's token count may differ from the real
#: bank's before :func:`scramble_bank` complains. Deranging text of unequal
#: lengths cannot change the multiset of principles, so the rendered totals
#: should match almost exactly; a mismatch means something other than a
#: permutation happened.
TOKEN_TOLERANCE = 0.02


def derange(n: int, rng: random.Random) -> list[int]:
    """A permutation of ``range(n)`` with no fixed points.

    Rejection-samples, which is fine at bank sizes (n ≤ 10): the probability a
    random permutation is a derangement tends to 1/e, so this succeeds in a
    handful of draws. ``n == 1`` has no derangement and raises -- a one-skill bank
    cannot be scrambled, and silently returning the identity would produce a
    "placebo" that is the real bank.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n == 0:
        return []
    if n == 1:
        raise ValueError(
            "a 1-skill bank has no derangement; scrambling it would return the "
            "real bank unchanged. Use foreign_bank() for this case."
        )
    order = list(range(n))
    for _ in range(10_000):
        rng.shuffle(order)
        if all(i != j for i, j in enumerate(order)):
            return list(order)
    raise RuntimeError("failed to sample a derangement")  # pragma: no cover


def scramble_bank(
    bank: Any,
    level_name: str = "maze",
    *,
    seed: int = 0,
    scramble_mistakes: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Return ``(placebo_bank, manifest)``: principles deranged across skills.

    The original bank is never mutated -- it is deep-copied first, because the
    real bank is the artifact of record for arm B and a placebo that quietly
    edited it in place would corrupt the very run it is a control for.

    ``manifest`` records the permutation and both token counts, so the claim
    "length-matched" is auditable rather than asserted.
    """
    rng = random.Random(seed)
    placebo = copy.deepcopy(bank)

    skills = list((getattr(placebo, "level_skills", {}) or {}).get(level_name, []))
    if len(skills) < 2:
        raise ValueError(
            f"cannot scramble a {len(skills)}-skill bank for {level_name!r}; "
            "a derangement needs at least 2 skills"
        )

    permutation = derange(len(skills), rng)
    donors = [(s.principle, getattr(s, "example", "")) for s in skills]
    for index, skill in enumerate(skills):
        principle, example = donors[permutation[index]]
        skill.principle = principle
        skill.example = example
    placebo.level_skills[level_name] = skills

    mistake_permutation: list[int] = []
    mistakes = list((getattr(placebo, "level_mistakes", {}) or {}).get(level_name, []))
    if scramble_mistakes and len(mistakes) >= 2:
        mistake_permutation = derange(len(mistakes), rng)
        fixes = [m.how_to_avoid for m in mistakes]
        for index, mistake in enumerate(mistakes):
            mistake.how_to_avoid = fixes[mistake_permutation[index]]
        placebo.level_mistakes[level_name] = mistakes

    manifest = {
        "kind": "scrambled-derangement",
        "seed": seed,
        "level": level_name,
        "n_skills": len(skills),
        "skill_permutation": permutation,
        "n_mistakes": len(mistakes),
        "mistake_permutation": mistake_permutation,
    }
    manifest.update(_token_check(bank, placebo, level_name))
    return placebo, manifest


def _rendered(bank: Any, level_name: str) -> str:
    try:
        return bank.format_skills_as_prompt(level_name)
    except Exception:  # pragma: no cover - defensive; a bank should always render
        return ""


def _token_check(original: Any, placebo: Any, level_name: str) -> dict[str, Any]:
    """Token counts for both banks, plus whether they match within tolerance."""
    before = count_tokens(_rendered(original, level_name))
    after = count_tokens(_rendered(placebo, level_name))
    drift = abs(after - before) / before if before else 0.0
    return {
        "tokens_real": before,
        "tokens_placebo": after,
        "token_drift": round(drift, 4),
        "length_matched": drift <= TOKEN_TOLERANCE,
    }


def assert_scrambled(
    original: Any, placebo: Any, level_name: str = "maze"
) -> dict[str, Any]:
    """Verify a placebo really is one. Returns the check record; raises if not.

    Three things have to hold, and each has failed in some version of this idea:
    every skill's principle must have moved (a shuffle with fixed points leaves
    real advice in place), the rendered token count must match (or "length-matched"
    is false advertising), and the two blocks must not be byte-identical.
    """
    real = list((getattr(original, "level_skills", {}) or {}).get(level_name, []))
    fake = list((getattr(placebo, "level_skills", {}) or {}).get(level_name, []))
    if len(real) != len(fake):
        raise AssertionError(f"skill count changed: {len(real)} -> {len(fake)}")

    unmoved = [
        r.skill_id for r, f in zip(real, fake) if r.principle == f.principle
    ]
    if unmoved:
        raise AssertionError(
            f"{len(unmoved)} skill(s) kept their own principle, so the placebo "
            f"still carries real advice: {unmoved}"
        )
    if {r.principle for r in real} != {f.principle for f in fake}:
        raise AssertionError("the multiset of principles changed; this is not a permutation")

    check = _token_check(original, placebo, level_name)
    if not check["length_matched"]:
        raise AssertionError(
            f"token drift {check['token_drift']:.1%} exceeds {TOKEN_TOLERANCE:.0%}; "
            "the placebo is not length-matched"
        )
    if _rendered(original, level_name) == _rendered(placebo, level_name):
        raise AssertionError("placebo renders identically to the real bank")
    return check


def foreign_bank(
    bank: Any,
    donor_skills: Sequence[Any],
    level_name: str = "maze",
    *,
    seed: int = 0,
) -> tuple[Any, dict[str, Any]]:
    """The alternative control: coherent skills about the wrong task.

    Keeps each donor skill internally consistent, so it cannot mislead through
    incoherence the way :func:`scramble_bank` can. The cost is a register
    mismatch -- advice about a different task reads as off-topic, which is its own
    kind of signal. Neither control dominates; run both when the placebo result is
    ambiguous.

    Takes as many donors as the real bank has skills, so the block stays
    comparable in size. ``donor_skills`` would typically come from a ``hexa_core``
    CORE bank.
    """
    rng = random.Random(seed)
    placebo = copy.deepcopy(bank)
    real = list((getattr(placebo, "level_skills", {}) or {}).get(level_name, []))
    if not real:
        raise ValueError(f"no skills for {level_name!r} to match in size")
    if len(donor_skills) < len(real):
        raise ValueError(
            f"need at least {len(real)} donor skills, got {len(donor_skills)}"
        )

    chosen = rng.sample(list(donor_skills), len(real))
    swapped = [
        replace(
            copy.deepcopy(donor),
            skill_id=original.skill_id,
            source_level=level_name,
            confidence=original.confidence,
        )
        for original, donor in zip(real, chosen)
    ]
    placebo.level_skills[level_name] = swapped

    manifest: dict[str, Any] = {
        "kind": "foreign-donor",
        "seed": seed,
        "level": level_name,
        "n_skills": len(swapped),
        "donor_titles": [getattr(s, "title", "") for s in chosen],
    }
    manifest.update(_token_check(bank, placebo, level_name))
    return placebo, manifest
