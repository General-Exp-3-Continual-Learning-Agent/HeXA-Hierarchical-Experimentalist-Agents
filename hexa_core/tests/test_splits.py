"""Split invariants for all four CORE tasks at all three regimes.

Splits are the load-bearing precondition for every number in the comparison. A
single leaked eval item invalidates a run (hard rule 4), and an eval set that
silently reseeds makes two runs incomparable -- both failures are invisible in the
accuracy figure itself, so they get asserted mechanically here.

Run with::

    PYTHONDONTWRITEBYTECODE=1 python -m hexa_core.tests.test_splits

No network, no GPU. Constructs CORE tasks in process and reads CORE's data files
read-only.
"""

from __future__ import annotations

import json

from hexa_core.adapter import EVAL_ITEMS
from hexa_core.tasks import REGIMES, TASK_NAMES, TASK_SPECS, make_task
from hexa_core.tests import cached_task, check, check_eq, core_module, fresh_task

#: CORE's own ``.name`` for each registry key. They differ for Hanoi, whose
#: ``variant="state"`` instance calls itself ``tower-of-hanoi-state`` -- anything
#: joining our logs to CORE's has to translate, so it is pinned rather than assumed.
CORE_TASK_NAMES = {
    "matchstick": "matchstick",
    "mathgap": "mathgap",
    "zebralogic": "zebralogic",
    "tower-of-hanoi": "tower-of-hanoi-state",
}


def test_sizes_and_no_overlap() -> None:
    for name in TASK_NAMES:
        for n in REGIMES:
            task = cached_task(name, n)
            train_ids = [p.id for p in task.train_problems()]
            eval_ids = [p.id for p in task.eval_problems()]

            check_eq(f"{name} n={n}: eval size", len(eval_ids), EVAL_ITEMS)
            # CORE soft-degrades (prints "Warning: Not enough ... items" and returns
            # a shorter split) rather than raising, so an exact size check is the
            # only thing standing between us and a silently mis-sized regime.
            check_eq(f"{name} n={n}: train size", len(train_ids), n)
            check(
                f"{name} n={n}: no duplicate ids within a split",
                len(set(train_ids)) == len(train_ids) and len(set(eval_ids)) == len(eval_ids),
            )
            overlap = sorted(set(train_ids) & set(eval_ids))
            check(f"{name} n={n}: train/eval disjoint by id", not overlap, f"leaked {overlap[:5]}")


def test_eval_ids_identical_across_regimes() -> None:
    """Eval sets must not move with the regime -- except where CORE makes them move.

    Three of the four tasks draw eval from a fixed file, a fixed shuffle slice or a
    disjoint disk count, so their eval id list is regime-invariant and cross-regime
    numbers are directly comparable. MathGAP is not: it slices one shuffled pool as
    ``train = data[:n]`` / ``eval = data[n : n + 100]``, so the eval window slides.
    That is asserted positively in :func:`test_mathgap_eval_window_slides` rather
    than skipped, so the difference stays a recorded property of the task.
    """
    for name in TASK_NAMES:
        if not TASK_SPECS[name].eval_split_regime_invariant:
            continue
        reference = [p.id for p in cached_task(name, REGIMES[0]).eval_problems()]
        for n in REGIMES[1:]:
            check_eq(
                f"{name}: eval id list at n={n} equals n={REGIMES[0]}",
                [p.id for p in cached_task(name, n).eval_problems()],
                reference,
            )


def test_mathgap_eval_window_slides() -> None:
    """MathGAP's declared regime-dependence is real, and its splits stay disjoint.

    Two claims, because only the pair is safe. The eval sets at different regimes
    *do* differ (so no HeXA-vs-baseline comparison may straddle a regime), and yet
    train and eval remain disjoint at every regime (so the dependence costs
    comparability, never contamination). If CORE ever changed the slicing to a
    fixed window this test would fail loudly and the spec flag would be wrong --
    which is the point of asserting it rather than trusting the flag.
    """
    check(
        "mathgap: declared eval_split_regime_invariant=False",
        TASK_SPECS["mathgap"].eval_split_regime_invariant is False,
    )
    by_regime = {n: [p.id for p in cached_task("mathgap", n).eval_problems()] for n in REGIMES}
    for n in REGIMES[1:]:
        check(
            f"mathgap: eval ids at n={n} differ from n={REGIMES[0]} (window slid)",
            by_regime[n] != by_regime[REGIMES[0]],
        )
    # The slide is exactly `num_train_items` forward through one shuffled pool, so
    # eval@5 and eval@100 overlap heavily. Disjointness that matters is train/eval
    # *within* a regime, already covered by test_sizes_and_no_overlap; re-check it
    # here against the pool bound to catch an off-by-one in the slice.
    spec_pool = TASK_SPECS["mathgap"].pool_size
    for n in REGIMES:
        train_ids = {p.id for p in cached_task("mathgap", n).train_problems()}
        check(
            f"mathgap n={n}: train/eval disjoint despite consecutive slices",
            not (train_ids & set(by_regime[n])),
        )
        check(
            f"mathgap n={n}: {n} train + {EVAL_ITEMS} eval fits the {spec_pool}-row pool",
            n + EVAL_ITEMS <= spec_pool,
        )


def test_mathgap_rejects_oversized_regime() -> None:
    """An over-large regime must raise, not soft-degrade to a short eval split."""
    too_big = TASK_SPECS["mathgap"].pool_size - EVAL_ITEMS + 1
    try:
        make_task("mathgap", too_big)
    except ValueError as exc:
        check(
            f"mathgap n={too_big}: raises and names the pool",
            str(TASK_SPECS['mathgap'].pool_size) in str(exc),
            f"message was {exc!r}",
        )
    else:
        check(f"mathgap n={too_big}: raises instead of degrading", False)


def test_eval_ids_stable_across_independent_construction() -> None:
    """Two independent constructions with identical args must agree exactly.

    Catches accidental reseeding: a module-level ``random.seed`` being consumed by
    an earlier import, or a ``random.shuffle`` on the global RNG instead of a local
    ``Random(0)``. Deliberately bypasses ``cached_task``.
    """
    for name in TASK_NAMES:
        first = [p.id for p in fresh_task(name, 10).eval_problems()]
        second = [p.id for p in fresh_task(name, 10).eval_problems()]
        check_eq(f"{name}: eval ids stable across two constructions", second, first)
        # And stable against a construction at a different regime interleaved
        # between them, which is what a real sweep does.
        fresh_task(name, 5)
        third = [p.id for p in fresh_task(name, 10).eval_problems()]
        check_eq(f"{name}: eval ids stable after an interleaved construction", third, first)


def test_train_ids_nested_by_set() -> None:
    """Train sets nest by *set* across regimes -- but 5 is NOT a prefix of 10.

    Measured: ``train@10[:5] != train@5`` for matchstick and Hanoi. Nothing
    downstream may assume regime-nested prefixes (e.g. reusing per-problem
    baselines ``b_q`` positionally instead of by id).
    """
    for name in TASK_NAMES:
        sets = {n: {p.id for p in cached_task(name, n).train_problems()} for n in REGIMES}
        check(f"{name}: train@5 subset of train@10", sets[5] <= sets[10],
              f"missing {sorted(sets[5] - sets[10])[:5]}")
        check(f"{name}: train@10 subset of train@100", sets[10] <= sets[100])


def test_hanoi_disk_counts() -> None:
    for n in REGIMES:
        task = cached_task("tower-of-hanoi", n)
        train_disks = {p.metadata["num_disks"] for p in task.train_problems()}
        eval_disks = {p.metadata["num_disks"] for p in task.eval_problems()}
        check_eq(f"hanoi n={n}: train disks", train_disks, {5})
        check_eq(f"hanoi n={n}: eval disks", eval_disks, {6})
        check(f"hanoi n={n}: train/eval disk counts disjoint", not (train_disks & eval_disks))


def test_zebralogic_eval_all_xlarge() -> None:
    """Every eval item must be an ``xlarge`` puzzle.

    ``ZebraLogicTask`` builds its ``TaskItem``s with an empty ``metadata`` dict, so
    the size group is not recoverable from the adapter and has to be re-derived from
    the pool file with CORE's own classifier -- honouring an explicit ``size_group``
    column first, exactly as ``ZebraLogicTask._load_pool`` does.
    """
    zl = core_module("tasks.single_turn.zebralogic_task")
    with open(zl.DATA_PATH) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    group_by_id = {
        f"zebralogic-{row['id']}": str(
            row.get("size_group") or zl._size_group_from_size(str(row.get("size", "")).strip())
        ).strip().lower()
        for row in rows
    }

    task = cached_task("zebralogic", 10)
    check_eq("zebralogic: task size_group", task.core_task.size_group, "xlarge")

    eval_ids = [p.id for p in task.eval_problems()]
    unresolved = [i for i in eval_ids if i not in group_by_id]
    check("zebralogic: every eval id resolves in the pool file", not unresolved, f"{unresolved[:5]}")
    wrong = sorted({group_by_id[i] for i in eval_ids if group_by_id.get(i) != "xlarge"})
    check(f"zebralogic: all {len(eval_ids)} eval items are xlarge", not wrong, f"also saw {wrong}")

    train_wrong = sorted(
        {group_by_id.get(p.id) for p in task.train_problems() if group_by_id.get(p.id) != "xlarge"}
    )
    check("zebralogic: all train items are xlarge too", not train_wrong, f"also saw {train_wrong}")


def test_split_labels_and_task_names() -> None:
    for name in TASK_NAMES:
        task = cached_task(name, 5)
        splits = {p.split for p in task.train_problems()} | {
            f"eval:{p.split}" for p in task.eval_problems()
        }
        check_eq(f"{name}: split labels", splits, {"train", "eval:eval"})

        task_names = {p.task_name for p in task.train_problems() + task.eval_problems()}
        check_eq(f"{name}: Problem.task_name is the registry key", task_names, {name})
        check_eq(
            f"{name}: manifest core_task_name",
            task.manifest()["core_task_name"],
            CORE_TASK_NAMES[name],
        )


def test_unknown_task_and_bad_regime_are_rejected() -> None:
    """A typo'd task name or a nonsensical regime must fail at construction.

    ``make_task`` is the only door to a task, so it is the only place a silent
    fallback could hide.
    """
    for bad in ("hanoi", "matchsticks", ""):
        try:
            make_task(bad, 5)
        except ValueError:
            check(f"make_task({bad!r}) rejected", True)
        else:
            check(f"make_task({bad!r}) rejected", False, "constructed a task anyway")
    for bad_n in (0, -1, True):
        try:
            make_task("matchstick", bad_n)  # type: ignore[arg-type]
        except ValueError:
            check(f"make_task('matchstick', {bad_n!r}) rejected", True)
        else:
            check(f"make_task('matchstick', {bad_n!r}) rejected", False, "constructed a task anyway")


if __name__ == "__main__":
    import sys

    from hexa_core.tests import run_module

    sys.exit(run_module(sys.modules[__name__]))
