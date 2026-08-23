"""Rollout-accounting checks: the four counters, what ``max_train`` gates, the totals.

Run with::

    PYTHONDONTWRITEBYTECODE=1 python -m hexa_core.tests.test_budget

No CORE import, no filesystem, no network. Cases are top-level ``test_*``
functions driven by :func:`hexa_core.tests.run_module`, so both entry points work.

The single most valuable assertion in this file is the *sign* of what
``max_train`` gates. In CORE the rollout counter is initialised at
``scripts/run_core.py:4499-4502`` -- **after** the baseline-training block that
starts at ``4304`` -- and advances only by
``episode_rollout_count = solve_rollout_count + admission_rollout_count``
(``run_core.py:2489``), gated at ``run_core.py:4526`` / ``5095`` against
``final_rollout = int(args.max_rollouts)`` (``4271``). So the 10 no-memory
baseline samples per training problem (``--num-baseline-train-runs``, default 10,
``run_core.py:4036``) are spent *outside* ``--max-rollouts``. Get that sign
backwards and every budget-matched comparison we publish is silently wrong: we
would be giving HExA 2000 minus the baseline spend where CORE gets a full 2000.
"""

from __future__ import annotations

from typing import Callable

from hexa_core.budget import BudgetExhausted, RolloutBudget
from hexa_core.tests import check, check_eq, check_raises

#: ``--num-baseline-train-runs`` default (`run_core.py:4036`).
N_BASE = 10

#: ``--max-rollouts`` in the paper config.
MAX_ROLLOUTS = 2000

_COUNTERS = ("train_solve", "admission", "baseline", "eval")


def _snapshot(budget: RolloutBudget) -> tuple[int, ...]:
    return tuple(getattr(budget, name) for name in _COUNTERS)


def _charge_methods(budget: RolloutBudget) -> dict[str, Callable[..., None]]:
    return {
        "train_solve": budget.charge_train_solve,
        "admission": budget.charge_admission,
        "baseline": budget.charge_baseline,
        "eval": budget.charge_eval,
    }


def test_categories_are_independent() -> None:
    """Charging one category must leave the other three untouched.

    Distinct amounts per category, so a copy-paste bug in which
    ``charge_admission`` increments ``train_solve`` cannot hide behind equal
    numbers.
    """
    amounts = {"train_solve": 3, "admission": 5, "baseline": 7, "eval": 11}
    budget = RolloutBudget(max_train=100)
    charge = _charge_methods(budget)
    for name, amount in amounts.items():
        before = dict(zip(_COUNTERS, _snapshot(budget)))
        charge[name](amount)
        after = dict(zip(_COUNTERS, _snapshot(budget)))
        assert after[name] == before[name] + amount, (name, before, after)
        for other in _COUNTERS:
            if other != name:
                assert after[other] == before[other], f"charging {name} moved {other}"

    check_eq(
        "charging four categories leaves four independent counters",
        _snapshot(budget),
        (3, 5, 7, 11),
    )
    check_eq("training_rollouts is train_solve + admission only", budget.training_rollouts, 8)
    check_eq("all_in_total spans all four categories", budget.all_in_total, 26)


def test_max_train_gates_train_solve_and_admission() -> None:
    """Both gated counters share one cap, and an overrun charges nothing."""
    budget = RolloutBudget(max_train=10)
    budget.charge_train_solve(6)
    budget.charge_admission(4)
    assert budget.training_rollouts == 10
    assert budget.train_remaining == 0
    assert budget.can_afford_train(1) is False
    assert budget.can_afford_train(0) is True

    for name, method in (
        ("train_solve", budget.charge_train_solve),
        ("admission", budget.charge_admission),
    ):
        before = _snapshot(budget)
        exc = check_raises(f"charge_{name} past max_train raises", BudgetExhausted, method)
        assert "max_train=10" in str(exc), str(exc)
        check(
            f"charge_{name} left the counters untouched when it raised",
            _snapshot(budget) == before,
            f"{_snapshot(budget)} != {before}",
        )

    # A partial fit must be refused outright rather than clamped: a clamped charge
    # would report a spend smaller than the calls actually issued.
    room = RolloutBudget(max_train=10)
    room.charge_train_solve(8)
    assert room.train_remaining == 2
    check_raises(
        "an over-large batch charge is refused, not clamped",
        BudgetExhausted,
        room.charge_train_solve,
        3,
    )
    check_eq("the refused batch charge applied nothing", room.train_solve, 8)


def test_max_train_does_not_gate_baseline_or_eval() -> None:
    """The sign test. Baseline and eval sit outside ``--max-rollouts``, per CORE."""
    # Forward: gated budget already full, ungated charges still succeed.
    full = RolloutBudget(max_train=10)
    full.charge_train_solve(10)
    full.charge_baseline(1000)
    full.charge_eval(1000)
    check(
        "baseline+eval charge against a full training budget does not raise",
        full.baseline == 1000 and full.eval == 1000,
    )
    check_eq("no ungated charge leaked into the gated counter", full.training_rollouts, 10)

    # Reverse: a large baseline spend first must not consume training budget. This
    # is the direction that catches an inverted gate, which the forward case alone
    # would pass.
    reverse = RolloutBudget(max_train=10)
    reverse.charge_baseline(1000)
    check_eq("1000 baseline rollouts leave the full training budget", reverse.train_remaining, 10)
    assert reverse.can_afford_train(10) is True
    for _ in range(10):
        reverse.charge_train_solve()
    check_raises(
        "max_train still applies after a baseline charge",
        BudgetExhausted,
        reverse.charge_train_solve,
    )

    # Degenerate: no training budget at all still permits baseline and eval, which
    # is exactly the Phase 2 no-learning gate's shape.
    gate = RolloutBudget(max_train=0)
    gate.charge_eval(300)
    gate.charge_baseline(50)
    check_eq("max_train=0 still permits 350 ungated rollouts", gate.all_in_total, 350)
    assert gate.training_rollouts == 0
    check_raises("max_train=0 blocks training", BudgetExhausted, gate.charge_train_solve)


def test_max_eval_is_the_only_thing_gating_eval() -> None:
    unlimited = RolloutBudget(max_train=1)
    unlimited.charge_eval(10_000)
    check_eq("eval is ungated with max_eval unset", unlimited.eval, 10_000)

    capped = RolloutBudget(max_train=1, max_eval=5)
    capped.charge_eval(5)
    before = _snapshot(capped)
    exc = check_raises("max_eval is enforced", BudgetExhausted, capped.charge_eval, 1)
    assert "max_eval=5" in str(exc), str(exc)
    assert _snapshot(capped) == before


def test_budget_exhausted_is_a_runtime_error() -> None:
    """Callers (e.g. ``baseline_runner``) catch ``RuntimeError`` at the top level."""
    check("BudgetExhausted is a RuntimeError", issubclass(BudgetExhausted, RuntimeError))


def test_negative_and_zero_charges() -> None:
    """Zero is a no-op; negative is a ``ValueError``, never a silent refund.

    A negative charge would let a run exceed ``max_train`` while still reporting
    ``training_rollouts <= max_train``, which destroys the one property this module
    exists to provide.
    """
    budget = RolloutBudget(max_train=10)
    budget.charge_train_solve(4)
    before = _snapshot(budget)
    for method in _charge_methods(budget).values():
        method(0)
    check("charging zero is a no-op on every counter", _snapshot(budget) == before)

    for name, method in _charge_methods(budget).items():
        # BudgetExhausted here would also be wrong: a negative charge is a caller
        # bug, not an exhausted budget, and check_raises reports the difference.
        check_raises(f"charge_{name}(-1) raises ValueError", ValueError, method, -1)
    assert _snapshot(budget) == before

    check_raises(
        "can_afford_train(-1) does not green-light a nonsense charge",
        ValueError,
        budget.can_afford_train,
        -1,
    )

    for kwargs in ({"max_train": -1}, {"max_train": 1, "max_eval": -1}):
        check_raises(f"RolloutBudget rejects {kwargs}", ValueError, RolloutBudget, **kwargs)


def test_report_is_internally_consistent() -> None:
    """Every derived field in ``report()`` must agree with the properties, and the
    two totals must be impossible to mistake for one another."""
    states: list[RolloutBudget] = []

    states.append(RolloutBudget(max_train=2000))

    mixed = RolloutBudget(max_train=2000, max_eval=500)
    mixed.charge_train_solve(1200)
    mixed.charge_admission(300)
    mixed.charge_baseline(100)
    mixed.charge_eval(100)
    states.append(mixed)

    spent = RolloutBudget(max_train=50)
    spent.charge_train_solve(50)
    spent.charge_baseline(500)
    try:
        spent.charge_admission(1)
    except BudgetExhausted:
        pass
    states.append(spent)

    for budget in states:
        report = budget.report()
        assert set(report) == {
            "training_rollouts",
            "max_train",
            "train_remaining",
            "train_solve",
            "admission",
            "baseline",
            "eval",
            "max_eval",
            "all_in_total",
        }, sorted(report)

        assert report["training_rollouts"] == report["train_solve"] + report["admission"]
        assert report["training_rollouts"] == budget.training_rollouts
        assert report["all_in_total"] == (
            report["train_solve"] + report["admission"] + report["baseline"] + report["eval"]
        )
        assert report["all_in_total"] == budget.all_in_total
        assert report["train_remaining"] == report["max_train"] - report["training_rollouts"]
        assert report["train_remaining"] == budget.train_remaining
        assert report["max_train"] == budget.max_train
        assert report["max_eval"] == budget.max_eval

        # These get json.dump'd into the run manifest; a numpy scalar or a float
        # would either fail to serialise or print as "2000.0".
        for key, value in report.items():
            if key == "max_eval" and value is None:
                continue
            assert type(value) is int, (key, type(value), value)

        if report["baseline"] > 0 or report["eval"] > 0:
            assert report["all_in_total"] != report["training_rollouts"], (
                "the CORE-comparable figure and the honest total collapsed into one number"
            )

        # A progress line must not be able to show the all-in total next to
        # max_train: the gated count is printed as a fraction of its cap, the
        # honest total under a separate label.
        text = str(budget)
        assert f"train={budget.training_rollouts}/{budget.max_train}" in text, text
        assert f"all-in={budget.all_in_total}" in text, text

    check(
        f"report() is self-consistent, int-valued and two-figured across {len(states)} states",
        True,
    )


def test_documented_arithmetic_2050_2100_3000() -> None:
    """The headline figures: 2000 training + 10 * |D_train| baseline per run."""
    for n_train, expected in ((5, 2050), (10, 2100), (100, 3000)):
        # The literals and the formula check each other, so a typo in either shows.
        assert expected == MAX_ROLLOUTS + N_BASE * n_train

        budget = RolloutBudget(max_train=MAX_ROLLOUTS)
        budget.charge_baseline(N_BASE * n_train)

        # A realistic split of the gated budget; only the sum is fixed by CORE.
        budget.charge_train_solve(1500)
        budget.charge_admission(500)
        assert budget.training_rollouts == MAX_ROLLOUTS
        check_raises(
            f"n={n_train}: the training budget stops at 2000",
            BudgetExhausted,
            budget.charge_train_solve,
        )

        check_eq(
            f"n={n_train}: 2000 training + {N_BASE}*{n_train} baseline == {expected}",
            budget.train_solve + budget.admission + budget.baseline,
            expected,
        )
        assert budget.all_in_total == expected, budget.report()
        check_eq(
            f"n={n_train}: the CORE-comparable figure is unmoved by the baseline spend",
            budget.report()["training_rollouts"],
            MAX_ROLLOUTS,
        )

        # Eval belongs in the honest total and in neither published figure.
        budget.charge_eval(100)
        assert budget.training_rollouts == MAX_ROLLOUTS
        check_eq(
            f"n={n_train}: 100 eval rollouts land in all_in_total only",
            budget.all_in_total,
            expected + 100,
        )


if __name__ == "__main__":
    import sys

    from hexa_core.tests import run_module

    sys.exit(run_module(sys.modules[__name__]))
