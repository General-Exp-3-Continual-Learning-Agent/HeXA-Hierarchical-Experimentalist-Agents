"""Tests for the admission gate (`PHASE3_DESIGN.md` §5c, M5).

Uses real `Problem`s and the real `SkillBank`, with only the LLM call faked -- the
gate's correctness is about budget accounting, which problem gets probed, and how
un-judgeable candidates are handled, none of which a fake problem would exercise.
"""

from __future__ import annotations

from hexa_core.admission import (
    AdmissionResult,
    accept_rate,
    admit_skills,
    test_skill_admission,
)
from hexa_core.budget import BudgetExhausted, RolloutBudget
from hexa_core.tasks import make_task
from hexa_core.tests import check, check_eq, run_module
from skillrl.core.skill_bank import Skill

TASK = "tower-of-hanoi"


def _task():
    return make_task(TASK, 5)


def _skill(skill_id: str, source_seeds: list, level: str = TASK) -> Skill:
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


def _correct_answer_for(task, problem) -> str:
    """A genuinely correct answer, from CORE's own reference formatter.

    Same source `test_verifiers` uses. Deliberately not a hand-written move list: the
    point is that the gate admits on an answer CORE itself certifies as correct, so the
    admit path is exercised through the real verifier rather than a stub of it.
    """
    reference = task.core_task.format_reference_answer_for_prompt(problem.all_valid_outputs)
    return "" if reference is None else str(reference)


# ---------------------------------------------------------------- verdicts


def test_skill_passing_its_trial_is_admitted() -> None:
    task = _task()
    problems = task.train_problems()
    parent = problems[0]
    reference = _correct_answer_for(task, parent)
    check("CORE produced a reference answer", bool(reference))

    result = test_skill_admission(
        _skill("s1", [parent.id]), problems, lambda prompt: reference
    )
    check("admitted on a correct answer", result.admitted, result.reason)
    check_eq("probed the parent problem", result.problem_id, parent.id)
    check_eq("one trial performed", result.trials, 1)
    check_eq("reason recorded", result.reason, "passed")


def test_skill_failing_its_trial_is_rejected() -> None:
    task = _task()
    problems = task.train_problems()
    parent = problems[0]

    result = test_skill_admission(
        _skill("s1", [parent.id]), problems, lambda prompt: "moves = [[9, 9, 9]]"
    )
    check("rejected on a wrong answer", not result.admitted)
    check_eq("reason recorded", result.reason, "failed-trial")
    check_eq("successes counted", result.successes, 0)


def test_the_parent_problem_is_the_one_probed() -> None:
    """Not problems[0], and not a random one -- the skill's own source problem."""
    task = _task()
    problems = task.train_problems()
    parent = problems[2]

    seen: list[str] = []

    def generate(prompt: str) -> str:
        seen.append(prompt)
        return "moves = [[0, 0, 0]]"

    result = test_skill_admission(_skill("s1", [parent.id]), problems, generate)
    check_eq("result names the parent", result.problem_id, parent.id)
    check(
        "the parent's own problem text was in the prompt",
        parent.problem_text[:60] in seen[0],
    )


# ---------------------------------------------------------------- budget accounting


def test_trials_are_charged_to_admission_inside_max_train() -> None:
    """CORE charges admission against --max-rollouts; baseline sits outside it."""
    task = _task()
    problems = task.train_problems()
    budget = RolloutBudget(max_train=10)

    test_skill_admission(
        _skill("s1", [problems[0].id]),
        problems,
        lambda p: "moves = [[0,0,0]]",
        budget=budget,
        trials=3,
    )
    check_eq("3 admission rollouts charged", budget.admission, 3)
    check_eq("counted toward training total", budget.training_rollouts, 3)
    check_eq("no train_solve charged", budget.train_solve, 0)
    # The load-bearing half: CORE charges admission against --max-rollouts but leaves
    # baseline estimation outside it. Charging admission to `baseline` would silently
    # hand the method extra training budget.
    check_eq("no baseline charged", budget.baseline, 0)


def test_budget_exhaustion_raises_rather_than_silently_skipping() -> None:
    """A truncated admission pass must not look like a low accept rate."""
    task = _task()
    problems = task.train_problems()
    budget = RolloutBudget(max_train=1)

    try:
        test_skill_admission(
            _skill("s1", [problems[0].id]),
            problems,
            lambda p: "moves = [[0,0,0]]",
            budget=budget,
            trials=5,
        )
    except BudgetExhausted:
        check("BudgetExhausted raised", True)
    else:
        check("exhausted budget must raise", False, "no BudgetExhausted")


# ---------------------------------------------------------------- un-judgeable cases


def test_skill_with_no_resolvable_parent_is_not_judged() -> None:
    task = _task()
    problems = task.train_problems()
    calls: list[str] = []

    result = test_skill_admission(
        _skill("orphan", ["a-problem-that-does-not-exist"]),
        problems,
        lambda p: calls.append(p) or "x",
    )
    check_eq("no trial run", result.trials, 0)
    check_eq("no LLM call made", len(calls), 0)
    check_eq("reason recorded", result.reason, "no-parent-problem-available")
    check("default admits (pre-M5 behaviour)", result.admitted)

    strict = test_skill_admission(
        _skill("orphan", ["nope"]), problems, lambda p: "x", admit_untestable=False
    )
    check("admit_untestable=False rejects instead", not strict.admitted)


def test_skill_too_big_for_budget_is_not_scored() -> None:
    """If the skill never reached the prompt, the trial would score no-memory."""
    task = _task()
    problems = task.train_problems()
    calls: list[str] = []

    result = test_skill_admission(
        _skill("s1", [problems[0].id]),
        problems,
        lambda p: calls.append(p) or "x",
        injection_budget_tokens=1,
    )
    check_eq("no trial run", result.trials, 0)
    check_eq("no LLM call made", len(calls), 0)
    check_eq("reason recorded", result.reason, "skill-did-not-fit-injection-budget")
    check("injected flag is False", not result.injected)


# ---------------------------------------------------------------- batch + A6


def test_gate_disabled_admits_everything_but_says_so() -> None:
    """A6: the two arms must be distinguishable in the logs, not just the config."""
    task = _task()
    problems = task.train_problems()
    skills = [_skill(f"s{i}", [problems[0].id]) for i in range(3)]
    calls: list[str] = []

    admitted, results = admit_skills(
        skills, problems, lambda p: calls.append(p) or "wrong", enabled=False
    )
    check_eq("all admitted", len(admitted), 3)
    check_eq("no LLM calls", len(calls), 0)
    check("reason distinguishes the arm", all(r.reason == "gate-disabled" for r in results))


def test_accept_rate_excludes_unjudged() -> None:
    """Folding un-judged candidates in would report the default, not the behaviour."""
    results = [
        AdmissionResult("a", "p1", True, 1, 1, "passed"),
        AdmissionResult("b", "p2", False, 1, 0, "failed-trial"),
        AdmissionResult("c", "", True, 0, 0, "no-parent-problem-available"),
    ]
    summary = accept_rate(results)
    check_eq("candidates counted", summary["candidates"], 3)
    check_eq("only 2 judged", summary["judged"], 2)
    check_eq("rate over judged only", summary["accept_rate_judged"], 0.5)
    check_eq("unjudged counted", summary["unjudged"], 1)
    check_eq(
        "unjudged reason broken out",
        summary["unjudged_reasons"]["no-parent-problem-available"],
        1,
    )
    check_eq("admission rollouts summed", summary["admission_rollouts"], 2)


def test_only_the_candidate_skill_is_in_context() -> None:
    """The gate must judge the skill, not the bank it is joining."""
    task = _task()
    problems = task.train_problems()
    parent = problems[0]
    seen: list[str] = []

    test_skill_admission(
        _skill("candidate", [parent.id]),
        problems,
        lambda p: seen.append(p) or "moves = [[0,0,0]]",
    )
    check("candidate skill present", "Skill candidate" in seen[0])
    check("no other skill leaked in", "Skill other" not in seen[0])


def test_max_tests_honours_the_admission_reserve() -> None:
    """The schedule reserves N trials per round; the gate must not spend more.

    Overflow is *rejected*, not waved through. That direction is deliberate: it can
    only cost HeXA a skill it might have kept, never grant one it did not earn.
    """
    task = _task()
    problems = task.train_problems()
    parent = problems[0]
    budget = RolloutBudget(max_train=100)
    calls: list[str] = []

    candidates = [_skill(f"s{i}", [parent.id]) for i in range(5)]
    admitted, results = admit_skills(
        candidates,
        problems,
        lambda p: calls.append(p) or "moves = [[0,0,0]]",
        budget=budget,
        trials=1,
        max_tests=2,
    )

    check_eq("only 2 candidates judged", len(calls), 2)
    check_eq("only 2 charged to admission", budget.admission, 2)
    check_eq("all 5 recorded", len(results), 5)

    capped = [r for r in results if r.reason == "admission-budget-capped"]
    check_eq("3 capped", len(capped), 3)
    check("capped candidates are rejected, not admitted", all(not r.admitted for r in capped))
    check("no capped skill reached the bank", all(s.skill_id not in {r.skill_id for r in capped} for s in admitted))

    summary = accept_rate(results)
    check_eq("accept rate is over judged only", summary["judged"], 2)
    check_eq(
        "the cap is visible in the log",
        summary["unjudged_reasons"]["admission-budget-capped"],
        3,
    )


def test_no_cap_judges_everything() -> None:
    task = _task()
    problems = task.train_problems()
    calls: list[str] = []
    admit_skills(
        [_skill(f"s{i}", [problems[0].id]) for i in range(4)],
        problems,
        lambda p: calls.append(p) or "moves = [[0,0,0]]",
        trials=1,
        max_tests=None,
    )
    check_eq("all 4 judged when uncapped", len(calls), 4)


def test_concurrent_admission_matches_sequential() -> None:
    """Fan-out changes wall-clock only: same verdicts, same order, same charge."""
    task = _task()
    problems = task.train_problems()
    parents = [p.id for p in problems]

    # Alternate pass/fail so a reordering would be visible in the verdicts.
    def generate(prompt: str) -> str:
        return "moves = [[0,0,0]]" if "s0" in prompt or "s2" in prompt else "nonsense"

    candidates = [_skill(f"s{i}", [parents[i % len(parents)]]) for i in range(6)]

    serial_budget = RolloutBudget(max_train=100)
    par_budget = RolloutBudget(max_train=100)
    serial_admitted, serial_results = admit_skills(
        candidates, problems, generate, budget=serial_budget, trials=1
    )
    par_admitted, par_results = admit_skills(
        candidates, problems, generate, budget=par_budget, trials=1, concurrency=6
    )

    check_eq(
        "same admitted skills, same order",
        [s.skill_id for s in par_admitted],
        [s.skill_id for s in serial_admitted],
    )
    check_eq(
        "results stay in candidate order",
        [r.skill_id for r in par_results],
        [r.skill_id for r in serial_results],
    )
    check_eq(
        "same verdicts",
        [r.admitted for r in par_results],
        [r.admitted for r in serial_results],
    )
    check_eq("same admission charge", par_budget.admission, serial_budget.admission)


def test_concurrent_admission_respects_the_cap() -> None:
    """The reserve is a hard ceiling; fan-out must not spend past it."""
    task = _task()
    problems = task.train_problems()
    budget = RolloutBudget(max_train=100)
    calls: list[str] = []

    admit_skills(
        [_skill(f"s{i}", [problems[0].id]) for i in range(8)],
        problems,
        lambda p: calls.append(p) or "moves = [[0,0,0]]",
        budget=budget,
        trials=1,
        max_tests=3,
        concurrency=8,
    )
    check_eq("cap honoured under fan-out", len(calls), 3)
    check_eq("charge matches the cap", budget.admission, 3)


def test_concurrent_admission_cannot_overshoot_max_train() -> None:
    """`charge_*` is check-then-increment; unguarded threads could both pass it."""
    task = _task()
    problems = task.train_problems()
    budget = RolloutBudget(max_train=4)

    try:
        admit_skills(
            [_skill(f"s{i}", [problems[0].id]) for i in range(12)],
            problems,
            lambda p: "moves = [[0,0,0]]",
            budget=budget,
            trials=1,
            concurrency=12,
        )
    except BudgetExhausted:
        pass  # Exhaustion is the correct outcome; overshooting silently is not.
    check(
        "never charged past max_train",
        budget.training_rollouts <= 4,
        f"training_rollouts={budget.training_rollouts} exceeded max_train=4",
    )


if __name__ == "__main__":
    raise SystemExit(run_module(__import__(__name__, fromlist=["_"])))
