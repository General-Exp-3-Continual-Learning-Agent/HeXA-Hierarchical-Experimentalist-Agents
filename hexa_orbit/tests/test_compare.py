"""Tests for the placebo bank and the paired arm comparison."""

from __future__ import annotations

import dataclasses

import pytest
from skillrl.core.skill_bank import Mistake, Skill, SkillBank

from hexa_orbit import compare, placebo
from hexa_orbit.actor import InstanceRun, StepRecord


# ── placebo (arm C) ──────────────────────────────────────────────────────

def _bank(n: int = 3) -> SkillBank:
    bank = SkillBank()
    bank.level_skills["maze"] = [
        Skill(
            skill_id=f"m{i}", title=f"Skill {i}",
            principle=f"Principle number {i} with some length to it.",
            when_to_apply=f"trigger {i}", example=f"example {i}", source_level="maze",
        )
        for i in range(n)
    ]
    bank.level_mistakes["maze"] = [
        Mistake(mistake_id=f"e{i}", description=f"mistake {i}",
                why_it_happens=f"because {i}", how_to_avoid=f"fix {i}",
                source_level="maze")
        for i in range(n)
    ]
    return bank


def test_derangement_has_no_fixed_points():
    import random

    for n in range(2, 12):
        order = placebo.derange(n, random.Random(n))
        assert sorted(order) == list(range(n))
        assert all(i != j for i, j in enumerate(order))


def test_a_single_skill_cannot_be_scrambled():
    """Silently returning the identity would make the placebo the real bank."""
    with pytest.raises(ValueError):
        placebo.derange(1, __import__("random").Random(0))
    with pytest.raises(ValueError):
        placebo.scramble_bank(_bank(1))


def test_scramble_is_length_matched_and_moves_every_principle():
    bank = _bank(4)
    fake, manifest = placebo.scramble_bank(bank, seed=3)
    check = placebo.assert_scrambled(bank, fake)
    assert check["length_matched"] and check["token_drift"] == 0.0
    assert manifest["tokens_real"] == manifest["tokens_placebo"]
    assert all(
        real.principle != faked.principle
        for real, faked in zip(bank.level_skills["maze"], fake.level_skills["maze"])
    )


def test_scramble_preserves_titles_triggers_and_the_principle_multiset():
    """Token count, register and apparent specificity fixed; the mapping destroyed."""
    bank = _bank(4)
    fake, _ = placebo.scramble_bank(bank, seed=1)
    real_skills, fake_skills = bank.level_skills["maze"], fake.level_skills["maze"]
    assert [s.title for s in fake_skills] == [s.title for s in real_skills]
    assert [s.when_to_apply for s in fake_skills] == [s.when_to_apply for s in real_skills]
    assert {s.principle for s in fake_skills} == {s.principle for s in real_skills}


def test_scramble_never_mutates_the_real_bank():
    """The real bank is arm B's artifact of record; corrupting it would be fatal."""
    bank = _bank(3)
    before = [s.principle for s in bank.level_skills["maze"]]
    placebo.scramble_bank(bank, seed=5)
    assert [s.principle for s in bank.level_skills["maze"]] == before


def test_scramble_is_deterministic_given_a_seed():
    a, _ = placebo.scramble_bank(_bank(5), seed=11)
    b, _ = placebo.scramble_bank(_bank(5), seed=11)
    c, _ = placebo.scramble_bank(_bank(5), seed=12)
    principles = lambda bank: [s.principle for s in bank.level_skills["maze"]]  # noqa: E731
    assert principles(a) == principles(b)
    assert principles(a) != principles(c)


def test_assert_scrambled_catches_a_shuffle_with_fixed_points():
    bank = _bank(3)
    sneaky = placebo.copy.deepcopy(bank)  # identity "permutation"
    with pytest.raises(AssertionError, match="kept their own principle"):
        placebo.assert_scrambled(bank, sneaky)


def test_foreign_bank_swaps_in_coherent_donors():
    bank = _bank(2)
    donors = [
        Skill(skill_id="x", title="Enumerate cases", principle="List all cases first.",
              when_to_apply="on a combinatorial problem", source_level="matchstick"),
        Skill(skill_id="y", title="Check the invariant", principle="Verify the sum.",
              when_to_apply="before answering", source_level="matchstick"),
    ]
    fake, manifest = placebo.foreign_bank(bank, donors, seed=0)
    assert manifest["kind"] == "foreign-donor"
    # Donor content, but the real bank's ids and confidences.
    assert {s.principle for s in fake.level_skills["maze"]} == {d.principle for d in donors}
    assert [s.skill_id for s in fake.level_skills["maze"]] == ["m0", "m1"]
    assert all(s.source_level == "maze" for s in fake.level_skills["maze"])


# ── the tests themselves ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("b", "c", "expected"),
    [(14, 2, 0.004181), (11, 3, 0.05737), (24, 7, 0.003327), (10, 2, 0.03857)],
)
def test_mcnemar_matches_the_design_doc_table(b, c, expected):
    assert compare.mcnemar_exact(b, c) == pytest.approx(expected, rel=1e-3)


def test_mcnemar_matches_scipys_binomial_test():
    """External validation: McNemar exact IS a two-sided binomial test on b of b+c."""
    from scipy.stats import binomtest

    for b, c in [(1, 0), (3, 1), (14, 2), (24, 7), (30, 30)]:
        assert compare.mcnemar_exact(b, c) == pytest.approx(
            binomtest(b, b + c, 0.5).pvalue, rel=1e-9
        )


def test_mcnemar_is_symmetric_and_safe_on_no_discordance():
    assert compare.mcnemar_exact(0, 0) == 1.0
    assert compare.mcnemar_exact(7, 2) == compare.mcnemar_exact(2, 7)


def test_clopper_pearson_brackets_the_rate():
    low, high = compare.clopper_pearson(8, 81)
    assert low < 8 / 81 < high
    assert compare.clopper_pearson(0, 10)[0] == 0.0
    assert compare.clopper_pearson(10, 10)[1] == 1.0
    assert compare.clopper_pearson(0, 0) == (0.0, 0.0)


# ── pairing ──────────────────────────────────────────────────────────────

def _run(instance_id: str, successes: list[bool], *, leaked: bool = False,
         added_tokens: int = 0, error: str | None = None) -> InstanceRun:
    steps = [
        StepRecord(
            episode_index=i, episode_step=1, total_step=i + 1, observation="o",
            raw_text="\\boxed{up}", reasoning_text="", final_text="\\boxed{up}",
            parsed_action="up", env_action="\\boxed{up}", reward=0.0,
            finish_reason="stop", episode_done=True,
        )
        for i in range(len(successes))
    ]
    return InstanceRun(
        instance_id=instance_id, dataset_seed=1, resolved_seed=1, shortest_path_len=7,
        leaked=leaked, steps=steps, messages=[], metrics={},
        episode_successes=successes, episode_lengths=[7] * len(successes),
        success_count=sum(successes), added_tokens=added_tokens,
        token_counter="stub", error=error,
    )


def test_unpaired_arms_are_refused_not_intersected():
    """The `hexa_core/WEAKNESSES.md:300` failure, made impossible."""
    a = [_run("i1", [True]), _run("i2", [False])]
    b = [_run("i1", [True]), _run("i3", [False])]
    with pytest.raises(compare.Unpaired, match="different instances"):
        compare.pair_runs(a, b)
    assert len(compare.pair_runs(a, b, strict=False)) == 1


def test_a_pair_needs_both_arms_to_have_produced_a_trajectory():
    a = [_run("i1", [True]), _run("i2", [], error="GenerationFailed")]
    b = [_run("i1", [True]), _run("i2", [True])]
    pairs = compare.pair_runs(a, b, strict=False)
    assert [p.instance_id for p in pairs] == ["i1"]


def test_primary_endpoint_uses_clean_instances_only():
    clean = [_run(f"c{i}", [False, False, False], leaked=False) for i in range(10)]
    leaked = [_run(f"l{i}", [True, True, True], leaked=True) for i in range(10)]
    arm_a = clean + leaked
    arm_b = (
        [_run(f"c{i}", [i < 6, True, True], leaked=False) for i in range(10)]
        + [_run(f"l{i}", [True, True, True], leaked=True) for i in range(10)]
    )
    result = compare.compare_arms(arm_a, arm_b)

    assert result["n_clean"] == 10 and result["n_leaked"] == 10
    primary = result["primary"]
    assert primary["n"] == 10                      # clean only
    assert primary["rate_a"] == 0.0 and primary["rate_b"] == 0.6
    assert primary["b_only_wins"] == 6 and primary["a_only_wins"] == 0
    assert primary["p_exact"] == pytest.approx(compare.mcnemar_exact(6, 0))
    # Pooled would have been diluted by the leaked stratum.
    assert result["secondary"]["episode_1_pooled"]["n"] == 20


def test_concordant_pairs_do_not_affect_the_p_value():
    """Adding instances both arms solve must not manufacture significance."""
    base_a = [_run(f"x{i}", [False], leaked=False) for i in range(4)]
    base_b = [_run(f"x{i}", [True], leaked=False) for i in range(4)]
    small = compare.compare_arms(base_a, base_b)["primary"]

    padding_a = [_run(f"p{i}", [True], leaked=False) for i in range(50)]
    padding_b = [_run(f"p{i}", [True], leaked=False) for i in range(50)]
    padded = compare.compare_arms(base_a + padding_a, base_b + padding_b)["primary"]

    assert padded["p_exact"] == small["p_exact"]
    assert padded["discordant"] == small["discordant"] == 4
    assert padded["difference"] < small["difference"]   # the rate diff does dilute


def test_diagnostics_and_added_tokens_are_reported_per_arm():
    a = [_run("i1", [True], added_tokens=0)]
    b = [_run("i1", [True], added_tokens=900)]
    result = compare.compare_arms(a, b)
    assert result["added_tokens"]["no-bank"] == 0
    assert result["added_tokens"]["frozen-bank"] == 900
    assert result["diagnostics"]["no-bank"]["unparsed_rate"] == 0.0


def test_format_comparison_warns_on_a_high_unparsed_rate():
    broken = _run("i1", [False])
    broken.steps[0].raw_text = "no boxed action at all"
    result = compare.compare_arms([broken], [_run("i1", [True])])
    text = compare.format_comparison(result)
    assert "WARNING" in text and "no parseable action" in text
    assert "PRIMARY ENDPOINT" in text


def test_format_flags_an_underpowered_comparison():
    a = [_run(f"x{i}", [False], leaked=False) for i in range(20)]
    b = [_run(f"x{i}", [i < 2], leaked=False) for i in range(20)]
    text = compare.format_comparison(compare.compare_arms(a, b))
    assert "underpowered" in text


def test_leakage_gap_is_undefined_when_a_stratum_is_empty():
    """An empty stratum has rate 0.0, so subtracting reports a spurious -100%."""
    clean = [_run(f"c{i}", [True], leaked=False) for i in range(3)]
    result = compare.compare_arms(clean, clean)
    assert result["descriptive"]["leakage_gap_a"] is None
    assert "note" in result["descriptive"]
    assert "n/a" in compare.format_comparison(result)


def test_leakage_gap_is_reported_when_both_strata_exist():
    a = [_run("c1", [False], leaked=False), _run("l1", [True], leaked=True)]
    b = [_run("c1", [False], leaked=False), _run("l1", [True], leaked=True)]
    result = compare.compare_arms(a, b)
    assert result["descriptive"]["leakage_gap_a"] == pytest.approx(1.0)
    assert "leaked-minus-clean episode 1: no-bank +100.0%" in compare.format_comparison(result)
