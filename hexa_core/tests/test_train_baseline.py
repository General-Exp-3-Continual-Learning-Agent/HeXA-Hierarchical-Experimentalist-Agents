"""Tests for the b_q producer (`PHASE3_DESIGN.md` §2b, M1).

Fake LLM, real everything else. The properties that matter are all structural: the
prompt must carry no guidelines, the seed namespace must be 1000 not 2000, the calls
must be charged outside `max_train`, and an unmeasured baseline must be
distinguishable from a measured 0.0.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from hexa_core.seeds import CORE_BASELINE_SEED_OFFSET, CORE_EVAL_SEED_OFFSET
from hexa_core.budget import RolloutBudget
from hexa_core.tasks import make_task
from hexa_core.tests import check, check_eq, run_module
from hexa_core.train_baseline import (
    DEFAULT_N_BASE,
    ProblemBaseline,
    TrainBaselines,
    attach_baselines,
    baseline_seed,
    estimate_train_baselines,
)
from hexa_core.trajectory import Attempt

TASK = "tower-of-hanoi"


def _task():
    return make_task(TASK, 5)


def _always_wrong(prompt: str, seed: int | None = None) -> str:
    return "moves = [[0, 0, 0]]"


# ---------------------------------------------------------------- the baseline property


def test_prompts_carry_no_guidelines() -> None:
    """b_q must be a NO-memory rate, or U = r - b_q measures a skill against itself."""
    task = _task()
    problems = task.train_problems(5)
    expected = {p.build_prompt(None) for p in problems}
    seen: list[str] = []

    estimate_train_baselines(
        task, lambda prompt, seed=None: seen.append(prompt) or "x", n_base=2
    )
    check_eq("every prompt is CORE's untouched prompt", set(seen), expected)


def test_b_q_reflects_measured_successes() -> None:
    task = _task()
    problems = task.train_problems(5)
    reference = task.core_task.format_reference_answer_for_prompt(
        problems[0].all_valid_outputs
    )
    check("CORE gave a reference answer", bool(reference))

    # Correct on the first problem only.
    def generate(prompt: str, seed: int | None = None) -> str:
        return str(reference) if problems[0].problem_text[:50] in prompt else "moves = [[0,0,0]]"

    result = estimate_train_baselines(task, generate, n_base=4)
    check_eq("solved problem has b_q 1.0", result.b_q(problems[0].id), 1.0)
    check_eq("unsolved problem has b_q 0.0", result.b_q(problems[1].id), 0.0)


# ---------------------------------------------------------------- seed namespace


def test_uses_baseline_namespace_1000_not_eval_2000() -> None:
    """CORE: 1000 = baseline, 2000 = eval. Reusing 2000 makes these replays."""
    from hexa_core.seeds import core_generation_seed

    check_eq("namespace constant is 1000", CORE_BASELINE_SEED_OFFSET, 1000)
    check_eq("eval namespace is 2000", CORE_EVAL_SEED_OFFSET, 2000)

    ours = baseline_seed(problem_index=3, run_index=2, llm_seed=42)
    expected = core_generation_seed(42, 1000, 2, 3)
    check_eq("seed derives from namespace 1000", ours, expected)

    eval_equivalent = core_generation_seed(42, 2000, 2, 3)
    check("baseline seed differs from the eval seed", ours != eval_equivalent)


def test_seeds_are_distinct_per_run_and_problem() -> None:
    seeds = {
        baseline_seed(p, r) for p in range(5) for r in range(DEFAULT_N_BASE)
    }
    check_eq("50 distinct seeds for 5x10", len(seeds), 50)


def test_seeds_are_recorded_for_reproducibility() -> None:
    task = _task()
    result = estimate_train_baselines(task, _always_wrong, n_base=3)
    record = next(iter(result.baselines.values()))
    check_eq("3 seeds recorded", len(record.seeds), 3)
    check_eq("they match the derivation", record.seeds[0], baseline_seed(0, 0))


# ---------------------------------------------------------------- budget accounting


def test_charged_to_baseline_outside_max_train() -> None:
    """CORE excludes baseline estimation from --max-rollouts; admission is included."""
    task = _task()
    budget = RolloutBudget(max_train=10)
    estimate_train_baselines(task, _always_wrong, n_base=DEFAULT_N_BASE, budget=budget)

    check_eq("5 problems x 10 = 50 baseline calls", budget.baseline, 50)
    check_eq("nothing charged to train_solve", budget.train_solve, 0)
    check_eq("nothing charged to admission", budget.admission, 0)
    # The load-bearing assertion: 50 calls with max_train=10 must NOT have raised,
    # because baseline sits outside the gate.
    check_eq("training budget untouched", budget.training_rollouts, 0)


def test_core_totals_match() -> None:
    """n_base * |D_train| = 50 / 100 / 1000, giving CORE's 2050 / 2100 / 3000."""
    for n_train, expected in ((5, 50), (10, 100), (100, 1000)):
        check_eq(f"n={n_train} baseline calls", DEFAULT_N_BASE * n_train, expected)
        check_eq(f"n={n_train} grand total", 2000 + DEFAULT_N_BASE * n_train,
                 {5: 2050, 10: 2100, 100: 3000}[n_train])


# ---------------------------------------------------------------- unmeasured vs zero


def test_unmeasured_is_distinguishable_from_measured_zero() -> None:
    """Both read as 0.0; conflating them makes every skill look maximally useful."""
    measured = ProblemBaseline("p1", TASK, n_base=10, successes=0)
    check_eq("measured zero has b_q 0.0", measured.b_q, 0.0)
    check("measured zero is usable", measured.usable)

    all_errored = ProblemBaseline("p2", TASK, n_base=10, successes=0, errors=10)
    check_eq("all-errored also reads 0.0", all_errored.b_q, 0.0)
    check("but is NOT usable", not all_errored.usable)

    baselines = TrainBaselines(TASK, 10, {"p1": measured, "p2": all_errored})
    check_eq("usable one returns a value", baselines.b_q("p1"), 0.0)
    check_eq("unusable one returns None", baselines.b_q("p2"), None)
    check_eq("absent one returns None", baselines.b_q("p3"), None)


def test_attach_baselines_degrades_to_pre_m1_when_unknown() -> None:
    """Attempt.utility must fall back to raw reward, not silently use b_q = 0."""
    known = Attempt("p1", TASK, "prompt", "raw", "ans", True, 1.0)
    unknown = Attempt("p9", TASK, "prompt", "raw", "ans", True, 1.0)
    baselines = TrainBaselines(
        TASK, 10, {"p1": ProblemBaseline("p1", TASK, 10, 6)}
    )

    (known_out, unknown_out), stamped = attach_baselines([known, unknown], baselines)
    check_eq("one stamped", stamped, 1)
    # U uses float(correct), NOT the +/-1 reward -- see Attempt.utility.
    check_eq("known gets U = correct - b_q", known_out.utility, 1.0 - 0.6)
    check("known reports a baseline", known_out.has_baseline)
    check_eq("unknown keeps raw reward", unknown_out.utility, 1.0)
    check("unknown reports no baseline", not unknown_out.has_baseline)
    check("originals are untouched (Attempt is frozen)", not known.has_baseline)


def test_utility_lands_in_the_domain_the_map_expects() -> None:
    """M1's arithmetic reuses _compute_initial_confidence's [-1,1] -> [0.1,1.0] map."""
    for reward in (1.0, -1.0):
        for b_q in (0.0, 0.5, 1.0):
            attempt = Attempt("p", TASK, "x", "y", "z", reward > 0, reward, baseline=b_q)
            check(
                f"U in [-1,1] for r={reward} b={b_q}",
                -1.0 <= attempt.utility <= 1.0,
                f"got {attempt.utility}",
            )


# ---------------------------------------------------------------- split discipline


def test_eval_problems_are_refused() -> None:
    """b_q is a training quantity; the eval-split equivalent is baseline_runner."""

    class EvalAsTrain:
        name = TASK

        def __init__(self) -> None:
            self._task = make_task(TASK, 5)

        def train_problems(self, n=None):
            return self._task.eval_problems()[:2]

        def eval_problems(self):
            return self._task.eval_problems()

    try:
        estimate_train_baselines(EvalAsTrain(), _always_wrong, n_base=1)
    except ValueError as exc:
        check("error points at baseline_runner", "baseline_runner" in str(exc))
        check("error names the namespaces", "1000" in str(exc) and "2000" in str(exc))
    else:
        check("eval-split problems must raise", False, "no ValueError")


# ---------------------------------------------------------------- resume


def test_resume_skips_completed_problems() -> None:
    """A 1000-call n=100 pass must not restart from scratch after an interruption."""
    task = _task()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = Path(tmp) / "b_q.json"

        calls: list[str] = []
        estimate_train_baselines(
            task,
            lambda prompt, seed=None: calls.append(prompt) or "moves = [[0,0,0]]",
            n_base=2,
            output_path=path,
        )
        first = len(calls)
        check_eq("5 problems x 2 runs", first, 10)
        check("file written", path.is_file())

        calls.clear()
        again = estimate_train_baselines(
            task,
            lambda prompt, seed=None: calls.append(prompt) or "moves = [[0,0,0]]",
            n_base=2,
            output_path=path,
        )
        check_eq("no calls on resume", len(calls), 0)
        check_eq("all 5 baselines present", len(again.baselines), 5)


def test_changing_n_base_invalidates_the_cache() -> None:
    """A cache computed at a different n_base is a different measurement."""
    task = _task()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = Path(tmp) / "b_q.json"
        estimate_train_baselines(task, _always_wrong, n_base=2, output_path=path)

        calls: list[str] = []
        estimate_train_baselines(
            task,
            lambda prompt, seed=None: calls.append(prompt) or "moves = [[0,0,0]]",
            n_base=4,
            output_path=path,
        )
        check_eq("recomputed at the new n_base", len(calls), 20)
        reloaded = TrainBaselines.from_dict(json.loads(path.read_text()))
        check_eq("stored n_base updated", reloaded.n_base, 4)


def test_round_trip_through_json() -> None:
    task = _task()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = Path(tmp) / "b_q.json"
        original = estimate_train_baselines(
            task, _always_wrong, n_base=2, output_path=path
        )
        reloaded = TrainBaselines.from_dict(json.loads(path.read_text()))
        check_eq("task name survives", reloaded.task_name, original.task_name)
        check_eq("problem count survives", len(reloaded.baselines), len(original.baselines))
        for problem_id, record in original.baselines.items():
            check_eq(f"{problem_id} b_q survives", reloaded.b_q(problem_id), record.b_q)


# ---------------------------------------------------------------- concurrency


def test_concurrent_matches_sequential_exactly() -> None:
    """Fan-out must change only wall-clock, never a number.

    ``b_q`` feeds every utility estimate, so a concurrency bug here would silently
    shift confidence across the whole bank rather than announce itself.
    """
    task = _task()
    problems = task.train_problems(5)
    reference = task.core_task.format_reference_answer_for_prompt(
        problems[0].all_valid_outputs
    )

    def generate(prompt: str, seed: int | None = None) -> str:
        # Deterministic but problem-dependent, so the tallies differ per problem.
        return str(reference) if problems[0].problem_text[:50] in prompt else "moves = [[0,0,0]]"

    serial = estimate_train_baselines(task, generate, n_base=4, concurrency=1)
    parallel = estimate_train_baselines(task, generate, n_base=4, concurrency=8)

    check_eq("same problems measured", set(serial.baselines), set(parallel.baselines))
    for problem_id, record in serial.baselines.items():
        other = parallel.baselines[problem_id]
        check_eq(f"{problem_id} b_q", other.b_q, record.b_q)
        check_eq(f"{problem_id} successes", other.successes, record.successes)
        check_eq(f"{problem_id} errors", other.errors, record.errors)
        # Seeds are derived from (problem_index, run_index), so they must be
        # order-independent -- if fan-out reshuffled them the pass is not reproducible.
        check_eq(f"{problem_id} seeds", other.seeds, record.seeds)


def test_concurrent_charges_the_same_budget() -> None:
    task = _task()
    serial_budget = RolloutBudget(max_train=10)
    parallel_budget = RolloutBudget(max_train=10)

    estimate_train_baselines(task, _always_wrong, n_base=4, budget=serial_budget)
    estimate_train_baselines(
        task, _always_wrong, n_base=4, budget=parallel_budget, concurrency=8
    )
    check_eq("identical baseline charge", parallel_budget.baseline, serial_budget.baseline)
    check_eq("5 problems x 4 runs", parallel_budget.baseline, 20)
    check_eq("still nothing on train_solve", parallel_budget.train_solve, 0)


def test_concurrent_pass_is_still_resumable() -> None:
    """Batched commits: an interruption must not cost the whole pass."""
    task = _task()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = Path(tmp) / "b_q.json"
        estimate_train_baselines(
            task, _always_wrong, n_base=2, output_path=path, concurrency=8
        )
        calls: list[str] = []
        again = estimate_train_baselines(
            task,
            lambda prompt, seed=None: calls.append(prompt) or "moves = [[0,0,0]]",
            n_base=2,
            output_path=path,
            concurrency=8,
        )
        check_eq("no calls on resume", len(calls), 0)
        check_eq("all 5 baselines present", len(again.baselines), 5)


if __name__ == "__main__":
    raise SystemExit(run_module(__import__(__name__, fromlist=["_"])))
