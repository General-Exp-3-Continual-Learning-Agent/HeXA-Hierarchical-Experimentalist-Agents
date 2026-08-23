"""Tests for the injection join (`PHASE3_DESIGN.md` §4b, Known issue 4).

Uses a **real** `SkillBank` and **real** CORE `Problem`s, not fakes. The two bugs
this module exists to prevent are both invisible without the real objects:

* the truthiness gate in CORE's own prompt templates, which makes an empty block
  byte-identical to no memory at all;
* parent-problem exclusion failing to fire because `Skill.source_seeds` holds ints on
  HeXA's InterPhyre path and strings on ours.

Neither raises. Both produce numbers that look fine.
"""

from __future__ import annotations

from hexa_core.injection import (
    DEFAULT_INJECTION_BUDGET,
    Injection,
    assert_injection_effective,
    build_injected_prompt,
    filter_parent_problem,
)
from hexa_core.tasks import make_task
from hexa_core.tests import check, check_eq, run_module
from skillrl.core.skill_bank import Mistake, Skill, SkillBank

TASK = "tower-of-hanoi"


def _skill(skill_id: str, level: str, source_seeds: list, confidence: float = 0.8) -> Skill:
    return Skill(
        skill_id=skill_id,
        title=f"Skill {skill_id}",
        principle="Move the largest free disk toward its goal peg before the smaller ones.",
        when_to_apply="When the goal peg is empty and the tower is partially ordered.",
        example="Illustration of the method, not of one problem's answer.",
        source_level=level,
        source_type="contrastive",
        source_seeds=source_seeds,
        generation=0,
        confidence=confidence,
    )


def _bank(*skills: Skill, mistakes: tuple[Mistake, ...] = ()) -> SkillBank:
    bank = SkillBank()
    for skill in skills:
        bank.add_skill(skill)
    for mistake in mistakes:
        bank.add_mistake(mistake)
    return bank


def _problem(index: int = 0):
    """A real CORE problem, so build_prompt is CORE's own template."""
    return make_task(TASK, 5).train_problems()[index]


# ---------------------------------------------------------------- the join exists


def test_injection_changes_the_prompt() -> None:
    """The join must actually add the block to CORE's guidelines slot."""
    problem = _problem()
    bank = _bank(_skill("s1", TASK, ["other-problem"]))
    inj = build_injected_prompt(problem, bank)

    check("prompt differs from baseline", inj.injected)
    check("added_tokens is positive", inj.added_tokens > 0, f"got {inj.added_tokens}")
    check("skill text reached the prompt", "largest free disk" in inj.prompt)
    check(
        "baseline prompt is CORE's untouched prompt",
        inj.baseline_prompt == problem.build_prompt(None),
    )


def test_no_bank_is_byte_identical_to_baseline() -> None:
    """An empty bank must not perturb the prompt at all."""
    problem = _problem()
    inj = build_injected_prompt(problem, _bank())
    check_eq("empty bank -> baseline prompt", inj.prompt, inj.baseline_prompt)
    check_eq("no tokens added", inj.added_tokens, 0)
    check("not flagged injected", not inj.injected)
    check("not flagged starved (bank was empty)", not inj.starved)


# ---------------------------------------------------------------- the silent-degrade trap


def test_starved_is_detected_when_budget_kills_everything() -> None:
    """A bank with content plus a budget too small = a no-memory run, mislabelled.

    This is the state that CORE's truthiness gate makes undetectable downstream, so
    it must be detectable *here*.
    """
    problem = _problem()
    bank = _bank(_skill("s1", TASK, ["other"]))
    inj = build_injected_prompt(problem, bank, budget_tokens=1)

    check_eq("prompt degraded to baseline", inj.prompt, inj.baseline_prompt)
    check("and that is byte-identical", inj.added_tokens == 0)
    check("starved is True", inj.starved, "the bank had a skill but nothing was injected")
    check("injected is False", not inj.injected)


def test_assert_injection_effective_raises_on_starvation() -> None:
    problem = _problem()
    bank = _bank(_skill("s1", TASK, ["other"]))
    good = build_injected_prompt(problem, bank, budget_tokens=DEFAULT_INJECTION_BUDGET)
    bad = build_injected_prompt(problem, bank, budget_tokens=1)

    assert_injection_effective([good])  # must not raise

    try:
        assert_injection_effective([good, bad])
    except AssertionError as exc:
        check("error names the problem", problem.id in str(exc))
        check("error explains the truthiness gate", "truthiness" in str(exc))
    else:
        check("starved injection must raise", False, "no AssertionError raised")


# ---------------------------------------------------------------- parent exclusion


def test_parent_problem_skill_is_excluded() -> None:
    """A skill distilled from this very problem must not be retrieved for it (§5b)."""
    problem = _problem()
    own = _skill("own", TASK, [problem.id], confidence=0.99)
    other = _skill("other", TASK, ["some-other-problem"], confidence=0.5)
    bank = _bank(own, other)

    inj = build_injected_prompt(problem, bank, parent_exclusion=True)
    ids = [s.skill_id for s in inj.block.skills]
    check("parent skill excluded", "own" not in ids, f"retrieved {ids}")
    check("non-parent skill retained", "other" in ids, f"retrieved {ids}")
    check_eq("exclusion counted", inj.excluded_parent, 1)
    check_eq("excluded id recorded", inj.excluded_skill_ids, ["own"])


def test_parent_exclusion_can_be_disabled() -> None:
    """CORE's own default is False; both settings must be reachable and reported."""
    problem = _problem()
    bank = _bank(_skill("own", TASK, [problem.id], confidence=0.99))
    inj = build_injected_prompt(problem, bank, parent_exclusion=False)
    check("parent skill kept when disabled", "own" in [s.skill_id for s in inj.block.skills])
    check_eq("nothing reported excluded", inj.excluded_parent, 0)


def test_exclusion_matches_across_int_and_str_seed_types() -> None:
    """HeXA stores int seeds, we store str problem ids. A type mismatch = no exclusion.

    That failure is silent: exclusion simply never fires, which is indistinguishable
    from there being no parent skills.
    """
    kept, dropped = filter_parent_problem([_skill("a", TASK, [12345])], "12345")
    check_eq("int seed matches str problem id", dropped, ["a"])
    check_eq("nothing kept", kept, [])

    kept2, dropped2 = filter_parent_problem([_skill("b", TASK, ["12345"])], "12345")
    check_eq("str seed matches too", dropped2, ["b"])
    check_eq("and nothing kept", kept2, [])

    kept3, dropped3 = filter_parent_problem([_skill("c", TASK, [])], "12345")
    check_eq("no source_seeds -> not excluded", [s.skill_id for s in kept3], ["c"])
    check_eq("and nothing dropped", dropped3, [])


def test_exclusion_happens_before_budget_walk() -> None:
    """Excluding after the walk would waste budget on a skill that gets dropped.

    With a budget big enough for exactly one skill, the parent must not be the one
    that consumes it.
    """
    problem = _problem()
    parent = _skill("parent", TASK, [problem.id], confidence=0.99)
    usable = _skill("usable", TASK, ["elsewhere"], confidence=0.10)
    bank = _bank(parent, usable)

    inj = build_injected_prompt(problem, bank, budget_tokens=DEFAULT_INJECTION_BUDGET)
    ids = [s.skill_id for s in inj.block.skills]
    check(
        "the lower-confidence non-parent skill still got injected",
        "usable" in ids,
        f"retrieved {ids}: the parent skill (higher confidence) must not have "
        f"consumed the budget before being excluded",
    )
    check("prompt was actually injected", inj.injected)


# ---------------------------------------------------------------- provenance


def test_record_carries_token_provenance() -> None:
    """Added-token figures are heuristic without tiktoken; the record must say which."""
    problem = _problem()
    inj = build_injected_prompt(problem, _bank(_skill("s1", TASK, ["other"])))
    record = inj.to_dict()

    check("counter named", bool(record["token_counter"]))
    check_eq("exactness is a bool", type(record["tokens_exact"]), bool)
    check_eq("added_tokens agrees with the object", record["added_tokens"], inj.added_tokens)
    for key in ("injected", "starved", "budget_bound", "excluded_parent", "problem_id"):
        check(f"record carries {key}", key in record)


def test_budget_bound_is_reported() -> None:
    """Truncation must be visible in the return value, not inferred."""
    problem = _problem()
    bank = _bank(*[_skill(f"s{i}", TASK, ["other"], confidence=0.9) for i in range(8)])
    inj = build_injected_prompt(problem, bank, budget_tokens=120)
    check("something was injected", inj.injected)
    check(
        "budget_bound flags the truncation",
        inj.budget_bound,
        f"dropped {inj.block.dropped_skills} skills but budget_bound is False",
    )


def test_hierarchy_level_selects_retrieval_key() -> None:
    """§3 maps general->level onto task->type via source_level strings."""
    problem = _problem()
    type_skill = _skill("typed", f"{TASK}:deep", ["other"])
    bank = _bank(type_skill)

    flat = build_injected_prompt(problem, bank)
    check("task-level retrieval misses the type-specific skill", not flat.injected)

    typed = build_injected_prompt(problem, bank, hierarchy_level=f"{TASK}:deep")
    check("type-level retrieval finds it", typed.injected)


def test_type_level_retrieval_sees_the_cross_type_tier() -> None:
    """The bug the smoke pilot exposed: skills filed at `task`, read at `task:type`.

    `evolve_skill_bank` is called with the bare task name, so every distilled skill
    lands under ``level_skills["tower-of-hanoi"]``. Retrieval under
    ``hierarchy="task:type"`` asked only for ``"tower-of-hanoi:5"``, and
    ``general_skills`` is empty for single-level runs -- so **no skill ever reached a
    prompt**, with no exception and no starved flag, because a bank that offers nothing
    at the requested level is indistinguishable from an empty bank.
    """
    problem = _problem()
    # Exactly how the loop's bank looks after a round: everything at the task tier.
    bank = _bank(*[_skill(f"s{i}", TASK, ["other"]) for i in range(3)])

    typed = build_injected_prompt(problem, bank, hierarchy_level=f"{TASK}:5")
    check(
        "type-level retrieval reaches the cross-type tier",
        typed.injected,
        "nothing injected: the bank's task-tier skills were invisible at task:type",
    )
    check("added tokens are non-zero", typed.added_tokens > 0, f"{typed.added_tokens}")
    check_eq("all three task-tier skills offered", len(typed.block.skills), 3)


def test_confidence_ranks_across_tiers_with_tier_as_the_tiebreak() -> None:
    """Both tiers compete on confidence -- §4a's policy -- not tier-first.

    Worth pinning explicitly, because "specific beats general" is the intuitive guess
    and it is *not* what happens: `retrieve_skills` sorts the merged candidates by
    confidence, so a high-confidence cross-type skill outranks a low-confidence
    type-specific one. Letting the tier override confidence would be a new design
    decision, not a bug fix. Tier survives only as the stable-sort tiebreak.
    """
    problem = _problem()
    bank = _bank(
        _skill("general-strong", TASK, ["other"], confidence=0.99),
        _skill("specific-weak", f"{TASK}:5", ["other"], confidence=0.2),
    )
    ranked = [
        getattr(s, "skill_id", "")
        for s in build_injected_prompt(
            problem, bank, hierarchy_level=f"{TASK}:5"
        ).block.skills
    ]
    check_eq("confidence ranks, not tier", ranked, ["general-strong", "specific-weak"])

    # At equal confidence the merge order decides, and that puts the specific tier
    # first -- so a tight budget prefers the more particular skill.
    tied = _bank(
        _skill("general-tied", TASK, ["other"], confidence=0.5),
        _skill("specific-tied", f"{TASK}:5", ["other"], confidence=0.5),
    )
    ranked_tied = [
        getattr(s, "skill_id", "")
        for s in build_injected_prompt(
            problem, tied, hierarchy_level=f"{TASK}:5"
        ).block.skills
    ]
    check_eq(
        "specific tier wins the tie", ranked_tied, ["specific-tied", "general-tied"]
    )


def test_tiers_are_merged_without_duplication() -> None:
    problem = _problem()
    shared = _skill("shared", TASK, ["other"])
    bank = _bank(shared)
    # File the same object under both tiers, as a bank edited by hand might.
    bank.level_skills.setdefault(f"{TASK}:5", []).append(shared)

    inj = build_injected_prompt(problem, bank, hierarchy_level=f"{TASK}:5")
    ids = [getattr(s, "skill_id", "") for s in inj.block.skills]
    check_eq("no duplicate", ids, ["shared"])


def test_flat_retrieval_is_untouched_by_the_tier_view() -> None:
    """`level == task_name` must not go through the view at all."""
    problem = _problem()
    bank = _bank(_skill("s0", TASK, ["other"]))
    flat = build_injected_prompt(problem, bank)
    check("flat retrieval still injects", flat.injected)
    check_eq("one skill", len(flat.block.skills), 1)


def test_parent_exclusion_still_fires_across_tiers() -> None:
    """Exclusion must survive the tier merge and the re-render that follows it."""
    problem = _problem()
    parent = _skill("from-parent", TASK, [problem.id])
    other = _skill("safe", f"{TASK}:5", ["other"])
    bank = _bank(parent, other)

    inj = build_injected_prompt(problem, bank, hierarchy_level=f"{TASK}:5")
    check_eq("the parent skill was excluded", inj.excluded_parent, 1)
    check_eq("excluded id recorded", inj.excluded_skill_ids, ["from-parent"])
    ids = [getattr(s, "skill_id", "") for s in inj.block.skills]
    check_eq("only the safe skill rendered", ids, ["safe"])
    check("still injected", inj.injected)


def test_max_specific_is_actually_a_cap() -> None:
    """`max_specific` must cap the block even when exclusion removes nothing.

    Regression: the parent-exclusion over-fetch (`max_specific * 2`) was never trimmed
    back, so with no parent skill to exclude -- the usual case, `excluded_parent` is 0 in
    every completed run -- the block kept twice the requested number. A nominal
    `max_specific=8` injected a whole 10-skill bank, and lowering the knob did nothing
    until it dropped below half. See `WEAKNESSES.md` §11.
    """
    problem = _problem()
    # Ten skills, none of them the parent, so filter_parent_problem removes nothing.
    bank = _bank(*[_skill(f"s{i}", TASK, ["other"]) for i in range(10)])

    for cap in (10, 8, 5, 3, 1):
        inj = build_injected_prompt(
            problem, bank, budget_tokens=100_000, max_specific=cap
        )
        check_eq(f"cap {cap} honoured", len(inj.block.skills), min(cap, 10))
        check_eq(f"cap {cap} excluded nothing", inj.excluded_parent, 0)

    # And the cap must not fight exclusion: a parent skill is still dropped, and the
    # cap applies to what survives.
    parent = _skill("parent", TASK, [problem.id])
    bank2 = _bank(parent, *[_skill(f"t{i}", TASK, ["other"]) for i in range(6)])
    inj2 = build_injected_prompt(problem, bank2, budget_tokens=100_000, max_specific=3)
    check_eq("parent still excluded under a cap", inj2.excluded_parent, 1)
    check_eq("cap applied after exclusion", len(inj2.block.skills), 3)
    check(
        "the excluded parent is not in the block",
        "parent" not in [getattr(s, "skill_id", "") for s in inj2.block.skills],
    )


if __name__ == "__main__":
    raise SystemExit(run_module(__import__(__name__, fromlist=["_"])))
