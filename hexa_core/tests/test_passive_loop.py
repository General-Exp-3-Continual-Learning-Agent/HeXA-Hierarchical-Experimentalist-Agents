"""Tests for the curriculum loop (`PHASE3_DESIGN.md` §1, build-order step 4).

The LLM is faked; everything else is real -- real `CoreTask`, real CORE prompts, real
verifiers, real `SkillBank`. The properties under test are structural (does the
curriculum stay disjoint, is the budget exactly spent, is eval frozen, does the
contamination guard actually run) and a fake task would let all of them pass vacuously.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hexa_core.adapter import EvalContamination
from hexa_core.passive_loop import (
    DEFAULT_TRAIN_BUDGET,
    chunk_problems,
    make_schedule,
    run_passive_loop,
)
from hexa_core.tasks import make_task
from hexa_core.tests import check, check_eq, run_module
from skillrl.core.skill_bank import Skill, SkillBank

TASK = "tower-of-hanoi"

# The three regimes CORE ships, and the schedule §1a derives for each:
#   n -> (rounds, problems_per_round, k, admission_reserve)
# `k` is derived from `budget - admission_reserve`, not `budget`, because CORE charges
# admission trials against `--max-rollouts`. Solve + reserve is exactly 2000 in all
# three -- see test_admission_reserve_is_inside_the_budget.
REGIMES = {
    100: (10, 10, 18, 200),
    10: (10, 1, 180, 200),
    5: (5, 1, 380, 100),
}


def _skill(skill_id: str, level: str, source_seeds: list) -> Skill:
    return Skill(
        skill_id=skill_id,
        title=f"Skill {skill_id}",
        principle="Move the largest free disk toward its goal peg first.",
        when_to_apply="When the goal peg is empty.",
        example="Method illustration, not a specific answer.",
        source_level=level,
        source_type="contrastive",
        source_seeds=source_seeds,
        generation=0,
        confidence=0.8,
    )


# ---------------------------------------------------------------- the schedule


def test_every_regime_spends_exactly_the_budget() -> None:
    """§1a's table must sum to 2000 in every regime -- the earlier draft did not."""
    for n_train, (rounds, ppr, k, reserve) in REGIMES.items():
        schedule = make_schedule(n_train)
        check_eq(f"n={n_train} rounds", schedule.rounds, rounds)
        check_eq(f"n={n_train} problems/round", schedule.problems_per_round, ppr)
        check_eq(f"n={n_train} attempts/problem", schedule.attempts_per_problem, k)
        check_eq(f"n={n_train} admission reserve", schedule.admission_reserve, reserve)
        check_eq(
            f"n={n_train} solve+reserve", schedule.total_planned, DEFAULT_TRAIN_BUDGET
        )
        check_eq(f"n={n_train} no unspent slack", schedule.slack, 0)


def test_admission_reserve_is_inside_the_budget() -> None:
    """CORE charges admission against --max-rollouts, so the schedule must reserve it.

    Regression for the smoke-pilot failure: `k` was derived from the whole budget, so
    round 0's admission trials pushed round 1 past `max_train` and the run died at the
    first boundary that admitted a skill.
    """
    for n_train in REGIMES:
        schedule = make_schedule(n_train)
        check(
            f"n={n_train} reserve is non-zero when the gate is on",
            schedule.admission_reserve > 0,
        )
        check_eq(
            f"n={n_train} k derives from solve_budget",
            schedule.attempts_per_problem,
            schedule.solve_budget // (schedule.rounds * schedule.problems_per_round),
        )
        # The property that actually failed: worst case must fit.
        check(
            f"n={n_train} worst case fits max_train",
            schedule.total_planned <= DEFAULT_TRAIN_BUDGET,
            f"planned {schedule.total_planned}",
        )
        check_eq(
            f"n={n_train} reserve covers cap x rounds x trials",
            schedule.admission_reserve,
            schedule.rounds * schedule.admission_cap_per_round,
        )


def test_gate_off_reserves_nothing_and_buys_more_attempts() -> None:
    """A6's two arms differ in k, not only in config -- that must be visible."""
    on = make_schedule(10)
    off = make_schedule(10, admission_enabled=False)
    check_eq("gate off reserves nothing", off.admission_reserve, 0)
    check_eq("gate off caps nothing", off.admission_cap_per_round, 0)
    check("gate off gets more attempts", off.attempts_per_problem > on.attempts_per_problem)
    check_eq("gate off still spends the budget", off.total_planned, DEFAULT_TRAIN_BUDGET)


def test_reserve_is_clamped_so_small_budgets_still_schedule() -> None:
    """The smoke run's budget=40 must not be eaten whole by a 20/round reserve."""
    schedule = make_schedule(4, budget=40, round_target=2)
    check_eq("2 rounds", schedule.rounds, 2)
    check("reserve is clamped to <= 20% of budget", schedule.admission_reserve <= 8)
    check("k is still positive", schedule.attempts_per_problem >= 1)
    check(
        "worst case fits",
        schedule.total_planned <= 40,
        f"planned {schedule.total_planned}",
    )


def test_rounds_cannot_exceed_the_training_set() -> None:
    """Streaming consumes an unseen problem per round, so n=5 admits at most 5."""
    check_eq("n=5 capped at 5 rounds", make_schedule(5).rounds, 5)
    check_eq("n=3 capped at 3 rounds", make_schedule(3).rounds, 3)
    check_eq("n=1 still valid", make_schedule(1).rounds, 1)


def test_impossible_schedule_raises_rather_than_rounding_to_zero() -> None:
    """Two distinct ways a budget can be too small; both must raise, not round to 0.

    With the gate off the binding constraint is `k < 1`. With it on, the reserve can
    fail first -- a budget that cannot afford one trial per round is unschedulable
    before `k` is even computed, and the message must say which of the two it was.
    """
    try:
        make_schedule(100, budget=10, round_target=10, admission_enabled=False)
    except ValueError as exc:
        check("gate off: error explains the arithmetic", "attempts per problem" in str(exc))
    else:
        check("k<1 must raise", False, "no ValueError")

    try:
        make_schedule(100, budget=10, round_target=10)
    except ValueError as exc:
        check(
            "gate on: error names the admission reserve, not k",
            "admission-test" in str(exc),
            str(exc),
        )
    else:
        check("an unaffordable reserve must raise", False, "no ValueError")


# ---------------------------------------------------------------- the curriculum


def test_chunks_are_disjoint_and_cover_the_set() -> None:
    """Disjointness is what makes the per-round held-out signal real."""
    problems = make_task(TASK, 10).train_problems()
    schedule = make_schedule(10)
    chunks = chunk_problems(problems, schedule.problems_per_round, schedule.rounds)

    seen: set[str] = set()
    for chunk in chunks:
        ids = {p.id for p in chunk}
        check(f"chunk disjoint from earlier ones", not (ids & seen), f"overlap: {ids & seen}")
        seen |= ids
    check_eq("all problems covered", len(seen), 10)
    check_eq("one chunk per round", len(chunks), schedule.rounds)


def test_each_round_attempts_only_its_own_chunk() -> None:
    """A round must never touch a future chunk's problems."""
    task = make_task(TASK, 5)
    result = run_passive_loop(
        task,
        lambda prompt: "moves = [[0, 0, 0]]",
        n_train=5,
        budget_calls=20,
        round_target=5,
        evaluate=False,
    )
    for record in result.rounds:
        attempted = {a.problem_id for a in record.attempts}
        check_eq(
            f"round {record.round_index} touched only its chunk",
            attempted,
            set(record.problem_ids),
        )


def test_streaming_never_triggers_parent_exclusion() -> None:
    """§1's structural claim, asserted rather than assumed.

    The bank only meets problems it was not distilled from, so exclusion must never
    fire. If it does, a future-chunk problem reached retrieval.
    """
    task = make_task(TASK, 5)
    problems = task.train_problems(5)

    # A bank whose skills descend from the FIRST chunk's problem. Under a correct
    # curriculum, later rounds never revisit it, so exclusion has nothing to remove.
    bank = SkillBank()
    bank.add_skill(_skill("from-round0", TASK, [problems[0].id]))

    result = run_passive_loop(
        task,
        lambda prompt: "moves = [[0, 0, 0]]",
        bank=bank,
        n_train=5,
        budget_calls=20,
        round_target=5,
        evaluate=False,
    )
    # Round 0 does re-meet its own problem (the bank was pre-seeded), so exclusion is
    # expected exactly there and nowhere after it.
    later = sum(r.excluded_parent for r in result.rounds[1:])
    check_eq("no exclusion after the parent's own round", later, 0)


# ---------------------------------------------------------------- budget accounting


def test_train_solve_is_charged_per_attempt() -> None:
    task = make_task(TASK, 5)
    result = run_passive_loop(
        task,
        lambda prompt: "moves = [[0, 0, 0]]",
        n_train=5,
        budget_calls=20,
        round_target=5,
        evaluate=False,
    )
    expected = result.schedule.total_calls
    check_eq("train_solve matches the schedule", result.budget.train_solve, expected)
    check_eq("no admission charged without evolve", result.budget.admission, 0)
    check_eq("no eval charged when evaluate=False", result.budget.eval, 0)


def test_eval_is_charged_separately_from_training() -> None:
    """CORE charges eval outside --max-rollouts; conflating them would hide cost."""
    task = make_task(TASK, 5)
    result = run_passive_loop(
        task,
        lambda prompt: "moves = [[0, 0, 0]]",
        n_train=5,
        budget_calls=10,
        round_target=5,
        evaluate=True,
    )
    # Two arms by default: with-memory and the paired no-memory control, 100 each.
    check_eq("both eval arms charged", result.budget.eval, 200)
    check_eq("eval not counted as training", result.budget.training_rollouts, 10)
    check_eq("eval_n is the split size, not the call count", result.eval_n, 100)
    check("no-memory arm produced a number", result.eval_accuracy_no_memory is not None)

    single = run_passive_loop(
        task,
        lambda prompt: "moves = [[0, 0, 0]]",
        n_train=5,
        budget_calls=10,
        round_target=5,
        evaluate=True,
        eval_no_memory=False,
    )
    check_eq("one arm charges 100", single.budget.eval, 100)
    check_eq("and reports no control", single.eval_accuracy_no_memory, None)


def test_eval_arms_are_paired_on_problem_and_seed() -> None:
    """The two arms must differ ONLY in the injected guidelines.

    Same problems, same CORE eval seeds (namespace 2000). Unpaired, the comparison
    would be two independent samples and would need far more runs to detect an effect
    of the size we expect.
    """
    import json

    task = make_task(TASK, 5)
    bank = SkillBank()
    bank.add_skill(_skill("s0", TASK, ["other"]))

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        run_passive_loop(
            task,
            lambda prompt, seed=None: "moves = [[0, 0, 0]]",
            bank=bank,
            n_train=5,
            budget_calls=10,
            round_target=5,
            evaluate=True,
            output_dir=tmp,
        )
        rows = {}
        for arm in ("with_memory", "no_memory"):
            path = Path(tmp) / f"eval_{arm}.jsonl"
            check(f"{arm} record written", path.is_file())
            rows[arm] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

        check_eq(
            "same problems, same order",
            [r["problem_id"] for r in rows["no_memory"]],
            [r["problem_id"] for r in rows["with_memory"]],
        )
        check_eq(
            "same seeds",
            [r["seed"] for r in rows["no_memory"]],
            [r["seed"] for r in rows["with_memory"]],
        )
        # And the seeds are CORE's eval namespace, so an external baseline pass lines up.
        from hexa_core.seeds import CORE_EVAL_SEED_OFFSET, core_generation_seed

        check_eq(
            "seed derives from namespace 2000",
            rows["with_memory"][0]["seed"],
            core_generation_seed(42, CORE_EVAL_SEED_OFFSET, 0, 0),
        )
        check(
            "only the memory arm added tokens",
            all(r["added_tokens"] == 0 for r in rows["no_memory"]),
        )


# ---------------------------------------------------------------- eval discipline


def test_contamination_guard_runs_before_eval() -> None:
    """Hard rule 4 must be enforced mechanically, not by inspection."""

    class LeakyTask:
        """A task whose 'training' set is drawn from the eval split."""

        name = TASK

        def __init__(self) -> None:
            self._task = make_task(TASK, 5)

        def train_problems(self, n=None):
            return self._task.eval_problems()[:2]

        def eval_problems(self):
            return self._task.eval_problems()

    try:
        run_passive_loop(
            LeakyTask(),
            lambda prompt: "moves = [[0, 0, 0]]",
            budget_calls=4,
            round_target=2,
            evaluate=True,
        )
    except EvalContamination as exc:
        check("guard names the leaked count", "attempted during training" in str(exc))
    else:
        check("leaked eval items must raise", False, "no EvalContamination")


def test_eval_does_not_grow_the_bank() -> None:
    """Frozen eval: no bank writes, mirroring run_core.py:1604."""
    task = make_task(TASK, 5)
    bank = SkillBank()
    bank.add_skill(_skill("s1", TASK, ["elsewhere"]))
    before = len(bank.level_skills.get(TASK, []))

    run_passive_loop(
        task,
        lambda prompt: "moves = [[0, 0, 0]]",
        bank=bank,
        n_train=5,
        budget_calls=10,
        round_target=5,
        evaluate=True,
    )
    check_eq("bank unchanged by eval", len(bank.level_skills.get(TASK, [])), before)


def test_eval_attempts_are_not_returned_as_trajectories() -> None:
    """Eval rollouts must not be reachable by a later distillation pass."""
    task = make_task(TASK, 5)
    result = run_passive_loop(
        task,
        lambda prompt: "moves = [[0, 0, 0]]",
        n_train=5,
        budget_calls=10,
        round_target=5,
        evaluate=True,
    )
    eval_ids = {p.id for p in task.eval_problems()}
    stored = {a.problem_id for r in result.rounds for a in r.attempts}
    check("no eval problem among stored attempts", not (stored & eval_ids))
    check("all stored attempts are train-split", all(
        a.split == "train" for r in result.rounds for a in r.attempts
    ))


# ---------------------------------------------------------------- no-evolve baseline


def test_no_evolve_measures_the_no_memory_baseline() -> None:
    """With evolve=None the loop must add nothing -- the harness's own null test."""
    task = make_task(TASK, 5)
    problems = task.train_problems(5)
    baseline_prompts = {p.id: p.build_prompt(None) for p in problems}
    seen: dict[str, str] = {}

    def generate(prompt: str) -> str:
        return "moves = [[0, 0, 0]]"

    result = run_passive_loop(
        task, generate, n_train=5, budget_calls=10, round_target=5, evaluate=False
    )
    for record in result.rounds:
        for attempt in record.attempts:
            check_eq(
                f"{attempt.problem_id} prompt is CORE's untouched prompt",
                attempt.prompt,
                baseline_prompts[attempt.problem_id],
            )
            check_eq("no tokens added", attempt.added_tokens, 0)


def test_round_dirs_are_written_where_the_loader_reads() -> None:
    """The trajectory writer / loader integration, end to end."""
    from hexa_core.trajectory import load_attempt_trajectories

    task = make_task(TASK, 5)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        out = Path(tmp)
        run_passive_loop(
            task,
            lambda prompt: "moves = [[0, 0, 0]]",
            n_train=5,
            budget_calls=10,
            round_target=5,
            evaluate=False,
            output_dir=out,
        )
        round0 = out / "round0"
        check("round0 dir written", round0.is_dir())
        successes, failures = load_attempt_trajectories(round0)
        check("loader reads what the loop wrote", (len(successes) + len(failures)) > 0)
        check("run summary written", (out / "run_summary.json").is_file())


def test_binary_reward_is_used() -> None:
    """M2: +1 / -1, not HeXA's shaped scale."""
    task = make_task(TASK, 5)
    result = run_passive_loop(
        task,
        lambda prompt: "moves = [[0, 0, 0]]",
        n_train=5,
        budget_calls=10,
        round_target=5,
        evaluate=False,
    )
    rewards = {a.reward for r in result.rounds for a in r.attempts}
    check("rewards are binary", rewards <= {1.0, -1.0}, f"got {rewards}")


def test_baselines_reach_the_attempts_the_evolver_reads() -> None:
    """M1 is inert unless b_q is stamped onto the attempts before distillation.

    Regression for the Stage 3a smoke pilot, which measured b_q, printed it, wrote it
    to disk -- and then scored every skill against the binary outcome, because
    `attach_baselines` had no call site. The symptom was a confidence of exactly
    0.1875 = (3 - 13)/16 mapped from [-1,1], i.e. the binary fallback, on a run whose
    b_q values were 0.0 and 0.5.
    """
    from hexa_core.train_baseline import ProblemBaseline, TrainBaselines

    task = make_task(TASK, 4)
    problems = task.train_problems(4)
    baselines = TrainBaselines(
        TASK,
        10,
        {p.id: ProblemBaseline(p.id, TASK, 10, 3) for p in problems},
    )

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        result = run_passive_loop(
            task,
            lambda prompt: "moves = [[0, 0, 0]]",
            n_train=4,
            budget_calls=20,
            round_target=2,
            evaluate=False,
            output_dir=tmp,
            baselines=baselines,
        )
        check("every attempt was stamped", result.baselines_stamped > 0)
        check_eq("none missing", result.baselines_missing, 0)

        # The written trajectories are what the evolver actually reads.
        import json

        rows = [
            json.loads(line)
            for line in (Path(tmp) / "round0" / "attempts.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
        check("b_q present on disk", all(r["baseline"] == 0.3 for r in rows))

        # The assertion that matters: what the *evolver* sees. `to_dict` is the disk
        # record; `to_teacher_dict` (via the rebound loader) is what confidence reads.
        from hexa_core.trajectory import load_attempt_trajectories

        # The loader returns (successes, failures), matching `distill.load_trajectories`.
        succeeded, failed = load_attempt_trajectories(str(Path(tmp) / "round0"))
        trajectories = succeeded + failed
        check_eq("loader found every attempt", len(trajectories), len(rows))
        check(
            "teacher dict carries b_q",
            all(t["baseline"] == 0.3 for t in trajectories),
        )
        check(
            "utility is float(correct) - b_q, not the raw reward",
            all(
                abs(t["utility"] - (float(t["success"]) - 0.3)) < 1e-9
                for t in trajectories
            ),
        )

        # And end to end: M1 confidence must now differ from the binary fallback.
        from hexa_core.hexa_bridge import compute_initial_confidence_core

        seed = trajectories[0]["seed"]
        with_bq = compute_initial_confidence_core([seed], trajectories)
        stripped = [dict(t, baseline=None) for t in trajectories]
        without = compute_initial_confidence_core([seed], stripped)
        check(
            "b_q actually changes the confidence",
            abs(with_bq - without) > 1e-9,
            f"both {with_bq}: b_q is being ignored downstream",
        )


def test_without_baselines_it_degrades_rather_than_assuming_zero() -> None:
    """No b_q must mean pre-M1 behaviour, never a silent b_q = 0."""
    task = make_task(TASK, 4)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        result = run_passive_loop(
            task,
            lambda prompt: "moves = [[0, 0, 0]]",
            n_train=4,
            budget_calls=20,
            round_target=2,
            evaluate=False,
            output_dir=tmp,
        )
        check_eq("nothing stamped", result.baselines_stamped, 0)
        for attempt in result.rounds[0].attempts:
            check_eq("baseline stays None", attempt.baseline, None)
            check_eq("utility falls back to reward", attempt.utility, attempt.reward)


if __name__ == "__main__":
    raise SystemExit(run_module(__import__(__name__, fromlist=["_"])))
