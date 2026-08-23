"""Tests for difficulty-weighted sampling and problem revisits (`MATCHSTICK_DIAGNOSIS.md`).

Two properties dominate: the frozen default must stay byte-identical (Hanoi, ZebraLogic
and MathGAP results were produced under it), and neither knob may change the rollout
budget -- a revisit arm that quietly spent more calls would not be comparable to
anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hexa_core.passive_loop import (
    DEFAULT_SAMPLING_UNIFORM_MIX,
    apportion_visits,
    chunk_problems,
    make_schedule,
    plan_rounds,
)
from hexa_core.tests import check, check_eq, check_raises, run_module


@dataclass
class FakeProblem:
    id: str
    metadata: dict = field(default_factory=dict)


@dataclass
class FakeBaselines:
    """Just the one method `apportion_visits` consumes."""

    values: dict

    def b_q(self, problem_id: str) -> float | None:
        return self.values.get(problem_id)


def _problems(n: int) -> list[FakeProblem]:
    return [FakeProblem(f"p{i:03d}") for i in range(n)]


# ── the frozen default ────────────────────────────────────────────────────────


def test_default_schedule_is_unchanged() -> None:
    """visits_per_problem defaults to 1, reproducing the published Matchstick regime."""
    schedule = make_schedule(60, budget=600, round_target=30, admission_enabled=False)
    check_eq("rounds", schedule.rounds, 30)
    check_eq("problems per round", schedule.problems_per_round, 2)
    check_eq("attempts per problem", schedule.attempts_per_problem, 10)
    check_eq("total calls", schedule.total_calls, 600)
    check_eq("visits per problem", schedule.visits_per_problem, 1)


def test_default_plan_is_identical_to_chunk_problems() -> None:
    """The new planner must be a no-op on the frozen path, not merely equivalent."""
    problems = _problems(60)
    old = chunk_problems(problems, 2, 30)
    new = plan_rounds(problems, 2, 30)
    check_eq(
        "same rounds",
        [[p.id for p in c] for c in new],
        [[p.id for p in c] for c in old],
    )


def test_round_robin_ignores_baselines() -> None:
    """Passing b_q must not perturb the control arm."""
    problems = _problems(20)
    baselines = FakeBaselines({p.id: 0.1 if i < 5 else 0.9 for i, p in enumerate(problems)})
    plain = apportion_visits(problems, 40, sampling="round_robin")
    withbq = apportion_visits(problems, 40, baselines=baselines, sampling="round_robin")
    check_eq("uniform either way", plain, withbq)
    check_eq("two visits each", plain, [2] * 20)


# ── budget invariance ─────────────────────────────────────────────────────────


def test_revisits_are_paid_for_out_of_attempts() -> None:
    """More visits must never mean more calls, and must never add rounds."""
    for visits in (1, 2, 3):
        schedule = make_schedule(
            60,
            budget=600,
            round_target=30,
            admission_enabled=False,
            visits_per_problem=visits,
        )
        check(
            f"visits={visits} never overspends ({schedule.total_calls} <= 600)",
            schedule.total_calls <= 600,
        )
        check_eq(f"visits={visits} rounds held fixed", schedule.rounds, 30)
        check_eq(
            f"visits={visits} problem visits",
            schedule.problem_visits,
            60 * visits,
        )


def test_only_dividing_visit_counts_stay_budget_matched() -> None:
    """`k` is floor-divided, so `visits_per_problem` must divide the budget evenly.

    At 600 calls over 30 rounds, visits=2 gives 4x5=20 calls/round exactly, but
    visits=3 gives 6 problems and floor(600/180)=3 attempts -- 540 calls, 60 short.
    A 540-call arm is not budget-matched against CORE's 600, so the shortfall must be
    visible in `slack` rather than discovered afterwards in the logs. This test exists
    to keep it visible, not to forbid it.
    """
    exact = make_schedule(
        60, budget=600, round_target=30, admission_enabled=False, visits_per_problem=2
    )
    check_eq("visits=2 is exact", exact.total_calls, 600)
    check_eq("visits=2 has no slack", exact.slack, 0)
    check_eq("visits=2 shape", (exact.problems_per_round, exact.attempts_per_problem), (4, 5))

    short = make_schedule(
        60, budget=600, round_target=30, admission_enabled=False, visits_per_problem=3
    )
    check_eq("visits=3 underspends", short.total_calls, 540)
    check_eq("visits=3 reports the shortfall", short.slack, 60)


def test_apportionment_sums_exactly() -> None:
    """Largest-remainder, not rounding: the visit slots must be filled exactly."""
    problems = _problems(60)
    baselines = FakeBaselines(
        {p.id: (0.05 if i % 5 == 0 else 0.95) for i, p in enumerate(problems)}
    )
    for mix in (0.0, 0.25, 0.5, 0.75, 1.0):
        counts = apportion_visits(
            problems, 120, baselines=baselines, sampling="difficulty", uniform_mix=mix
        )
        check_eq(f"mix={mix} sums to 120", sum(counts), 120)
        check(f"mix={mix} covers every problem", min(counts) >= 1)


# ── the bias actually biases ──────────────────────────────────────────────────


def test_hard_problems_get_more_visits() -> None:
    """The whole point: low b_q must attract surplus visits."""
    problems = _problems(60)
    hard = {p.id for i, p in enumerate(problems) if i % 5 == 0}  # 12 of 60
    baselines = FakeBaselines(
        {p.id: (0.05 if p.id in hard else 0.95) for p in problems}
    )
    counts = apportion_visits(
        problems, 120, baselines=baselines, sampling="difficulty", uniform_mix=0.5
    )
    hard_visits = sum(c for p, c in zip(problems, counts) if p.id in hard)
    easy_visits = sum(c for p, c in zip(problems, counts) if p.id not in hard)
    check_eq("all slots used", hard_visits + easy_visits, 120)
    check(
        f"hard share {hard_visits}/120 exceeds its 20% population share",
        hard_visits / 120 > 0.20,
    )
    check(
        "every hard problem is visited more than every easy one",
        min(c for p, c in zip(problems, counts) if p.id in hard)
        > max(c for p, c in zip(problems, counts) if p.id not in hard),
    )


def test_uniform_mix_one_is_uniform() -> None:
    """mix=1.0 must reduce to the control allocation, so the knob spans both ends."""
    problems = _problems(60)
    baselines = FakeBaselines({p.id: (i / 60) for i, p in enumerate(problems)})
    counts = apportion_visits(
        problems, 120, baselines=baselines, sampling="difficulty", uniform_mix=1.0
    )
    check_eq("uniform", counts, [2] * 60)


def test_unmeasured_b_q_is_average_not_easy() -> None:
    """A failed baseline must not starve the problem it failed on."""
    problems = _problems(10)
    # Two problems measured hard, two easy, the rest unmeasured.
    baselines = FakeBaselines({"p000": 0.0, "p001": 0.0, "p002": 1.0, "p003": 1.0})
    counts = apportion_visits(
        problems, 30, baselines=baselines, sampling="difficulty", uniform_mix=0.0
    )
    check_eq("sums", sum(counts), 30)
    check("unmeasured beats measured-easy", counts[4] > counts[2])
    check("unmeasured loses to measured-hard", counts[4] < counts[0])


# ── round assignment ──────────────────────────────────────────────────────────


def test_rounds_are_exact_and_duplicate_free() -> None:
    problems = _problems(60)
    baselines = FakeBaselines(
        {p.id: (0.05 if i % 5 == 0 else 0.95) for i, p in enumerate(problems)}
    )
    rounds = plan_rounds(
        problems,
        4,
        30,
        visits_per_problem=2,
        baselines=baselines,
        sampling="difficulty",
        uniform_mix=DEFAULT_SAMPLING_UNIFORM_MIX,
    )
    check_eq("round count", len(rounds), 30)
    check_eq("total visits", sum(len(r) for r in rounds), 120)
    check("every round is full", all(len(r) == 4 for r in rounds))
    for index, chunk in enumerate(rounds):
        check_eq(
            f"round {index} has no repeated problem",
            len({p.id for p in chunk}),
            len(chunk),
        )


def test_no_problem_gets_more_visits_than_there_are_rounds() -> None:
    """Regression: the smoke gate died here.

    A problem cannot appear twice in one round, so `rounds` is a hard ceiling on any
    single problem's count. At n_train=4 / rounds=2 the hardness weighting handed two
    problems 3 visits each and `plan_rounds` had nowhere to put the third.
    """
    problems = _problems(4)
    baselines = FakeBaselines({"p000": 1.0, "p001": 0.5, "p002": 1.0, "p003": 0.5})
    counts = apportion_visits(
        problems, 8, baselines=baselines, sampling="difficulty",
        uniform_mix=0.25, max_per_problem=2,
    )
    check_eq("still sums exactly", sum(counts), 8)
    check(f"no count exceeds 2 rounds: {counts}", max(counts) <= 2)
    # And end to end, the shape that actually crashed the job:
    rounds = plan_rounds(
        problems, 4, 2, visits_per_problem=2, baselines=baselines,
        sampling="difficulty", uniform_mix=0.25,
    )
    check_eq("two full rounds", [len(r) for r in rounds], [4, 4])


def test_infeasible_totals_are_refused_not_asserted() -> None:
    """Asking for more than the rounds can hold is a caller error, caught up front."""
    problems = _problems(4)
    check_raises(
        "total exceeds n * cap",
        ValueError,
        lambda: apportion_visits(
            problems, 12, baselines=FakeBaselines({}), sampling="difficulty",
            max_per_problem=2,
        ),
    )


def test_every_schedule_shape_is_placeable() -> None:
    """Regression: nearest-ideal-round greedy failed on 46 of these 168 shapes.

    It would place visits into the round nearest their ideal position with space left,
    which strands the last visit when the only round with capacity already holds the
    only problem still to place. Nothing about that failure was visible at the shapes
    the first version of this suite happened to test, so the grid is explicit now.
    """
    checked = 0
    for n in (4, 5, 6, 8, 10, 12, 20, 60):
        for rounds in (2, 3, 4, 5, 10, 15, 30):
            for visits in (1, 2, 3):
                if visits > rounds:
                    continue
                width = -(-n * visits // rounds)
                if width > n or rounds * width > n * rounds:
                    continue
                problems = _problems(n)
                baselines = FakeBaselines(
                    {p.id: (0.05 if i % 5 == 0 else 0.95)
                     for i, p in enumerate(problems)}
                )
                plan = plan_rounds(
                    problems, width, rounds, visits_per_problem=visits,
                    baselines=baselines, sampling="difficulty", uniform_mix=0.25,
                )
                shape = f"n={n} rounds={rounds} visits={visits}"
                if len(plan) != rounds or any(len(r) != width for r in plan):
                    check(f"{shape}: every round full", False)
                    continue
                if any(len({p.id for p in r}) != len(r) for r in plan):
                    check(f"{shape}: no duplicate within a round", False)
                    continue
                checked += 1
    check(f"all {checked} schedule shapes placeable", checked == 160)


def test_revisits_are_spread_not_clustered() -> None:
    """A revisit in the very next round teaches nothing -- the bank has barely moved."""
    problems = _problems(60)
    baselines = FakeBaselines({p.id: 0.5 for p in problems})
    rounds = plan_rounds(
        problems, 4, 30, visits_per_problem=2, baselines=baselines,
        sampling="difficulty", uniform_mix=1.0,
    )
    where: dict[str, list[int]] = {}
    for index, chunk in enumerate(rounds):
        for problem in chunk:
            where.setdefault(problem.id, []).append(index)
    gaps = [b - a for visits in where.values() for a, b in zip(visits, visits[1:])]
    check("every problem revisited", all(len(v) == 2 for v in where.values()))
    check(f"minimum gap {min(gaps)} rounds is > 1", min(gaps) > 1)


def test_difficulty_weighted_revisits_are_also_spread() -> None:
    """Regression: capacity-first placement was feasible but clustered.

    All rounds start equally empty, so "emptiest round" resolved to 0, 1, 2, 3 and a
    problem's four visits landed in four consecutive rounds -- median gap 1, which
    teaches nothing because the bank has not moved between them.
    """
    problems = _problems(60)
    baselines = FakeBaselines(
        {p.id: (0.05 if i % 5 == 0 else 0.95) for i, p in enumerate(problems)}
    )
    plan = plan_rounds(
        problems, 4, 30, visits_per_problem=2, baselines=baselines,
        sampling="difficulty", uniform_mix=0.25,
    )
    where: dict[str, list[int]] = {}
    for index, chunk in enumerate(plan):
        for problem in chunk:
            where.setdefault(problem.id, []).append(index)
    gaps = [b - a for v in where.values() for a, b in zip(v, v[1:])]
    check(f"minimum gap {min(gaps)} is > 1", min(gaps) > 1)
    check(f"median gap {sorted(gaps)[len(gaps) // 2]} is >= 4",
          sorted(gaps)[len(gaps) // 2] >= 4)


# ── guards ────────────────────────────────────────────────────────────────────


def test_bad_inputs_are_refused() -> None:
    problems = _problems(10)
    check_raises(
        "unknown sampling",
        ValueError,
        lambda: apportion_visits(problems, 20, sampling="nope"),
    )
    check_raises(
        "mix out of range",
        ValueError,
        lambda: apportion_visits(
            problems, 20, baselines=FakeBaselines({}), sampling="difficulty",
            uniform_mix=1.5,
        ),
    )
    check_raises(
        "cannot cover every problem",
        ValueError,
        lambda: apportion_visits(problems, 5, sampling="round_robin"),
    )
    check_raises(
        "visits below one",
        ValueError,
        lambda: make_schedule(10, budget=100, visits_per_problem=0),
    )


if __name__ == "__main__":
    raise SystemExit(run_module(__import__(__name__, fromlist=["_"])))
