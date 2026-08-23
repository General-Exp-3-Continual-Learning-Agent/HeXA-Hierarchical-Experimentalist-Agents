"""The admission gate (M5): test a distilled skill before it enters the bank.

`PHASE3_DESIGN.md` §5c. CORE tests each distilled lesson alone, in context, on its
originating problem, and admits it only if that attempt succeeds
(`run_admission_trials`, ``--lesson-admission-trials 1``). **HeXA has no accept/reject
gate at all** -- the evolver's output goes straight into the bank. This is CORE's
cheapest structural advantage and ablation A6 exists to find out whether it is what
carries their performance, so it is a flag (default on), not a hard-wired behaviour.

Three details that are easy to get wrong and consequential if you do.

**1. Admission trials are charged, and charged to the right counter.** CORE counts
them against ``--max-rollouts`` (`_count_admission_rollouts`, `run_core.py:2837`)
while baseline estimation sits *outside* it. `RolloutBudget.charge_admission` is
inside ``max_train`` for exactly this reason. Charging these to `charge_baseline`
would silently buy the method extra budget.

**2. Admission tests a skill on its own parent problem -- which retrieval forbids.**
§5b's parent-problem exclusion exists to stop a skill being *credited* for the problem
it was distilled from. Admission deliberately does the opposite: it probes that exact
pair once. These are consistent because admission is a one-off yes/no probe, not
utility accumulation -- but only as long as the trial does **not** write to the
problem table or update utilities. `AdmissionResult` therefore returns the outcome and
nothing else mutates.

**3. Seed namespace.** CORE uses 5000/5001 for admission, distinct from 1000
(baseline), 2000 (eval), 3000 (train-solve) and 7000 (sampler). Reusing the train-solve
namespace would make the admission attempt a *replay* of an attempt already made,
which passes trivially and measures nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from hexa_core.adapter import Problem
from hexa_core.budget import RolloutBudget
from hexa_core.injection import build_injected_prompt

# CORE's admission seed namespaces (`run_core.py`), kept distinct from train-solve so
# the trial is a fresh sample rather than a replay.
ADMISSION_SEED_BASE = 5000
ADMISSION_SEED_ALT = 5001

DEFAULT_ADMISSION_TRIALS = 1


@dataclass
class AdmissionResult:
    """Outcome of testing one skill on one problem.

    Records the *reason* alongside the verdict because "not admitted" has three very
    different causes -- the skill failed the trial, the trial could not be run, or the
    gate was disabled -- and an accept-rate statistic that conflates them is
    uninterpretable.
    """

    skill_id: str
    problem_id: str
    admitted: bool
    trials: int
    successes: int
    reason: str
    injected: bool = True
    raw_texts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "problem_id": self.problem_id,
            "admitted": self.admitted,
            "trials": self.trials,
            "successes": self.successes,
            "reason": self.reason,
            "injected": self.injected,
        }


def _parent_problem_ids(skill: Any) -> list[str]:
    """Problem ids this skill was distilled from, as strings.

    Coerced for the same reason `injection._skill_parents` coerces: HeXA's InterPhyre
    path stores ints, ours stores CORE problem-id strings, and a type mismatch here
    would silently make every skill untestable and thus auto-admitted-or-rejected
    depending on `admit_untestable`.
    """
    return [str(seed) for seed in (getattr(skill, "source_seeds", None) or [])]


def test_skill_admission(
    skill: Any,
    problems: Sequence[Problem],
    generate: Callable[[str], str | None],
    *,
    budget: RolloutBudget | None = None,
    trials: int = DEFAULT_ADMISSION_TRIALS,
    require_all: bool = False,
    injection_budget_tokens: int = 1000,
    admit_untestable: bool = True,
) -> AdmissionResult:
    """Test one skill in isolation on its originating problem.

    ``generate`` takes a prompt and returns the model's scored text (``raw_text``), or
    None on failure -- the same contract as `VLLMClient.generate`.

    ``problems`` is searched for the skill's parent problem. Only the parent is used:
    testing on a different problem would measure generalisation, which is what the
    held-out eval set is for, not admission.

    ``admit_untestable`` decides what happens when no parent problem is available (a
    skill from a merge whose ``source_seeds`` no longer resolve). Default True --
    admitting is the pre-M5 HeXA behaviour, so an unresolvable parent degrades to "no
    gate" rather than silently pruning skills the gate never actually judged. The
    ``reason`` field records it either way; the accept rate must be reported excluding
    these.

    ``require_all`` selects unanimity over any-success across trials. CORE ships one
    trial, where the distinction is vacuous; it matters only if trials is raised.
    """
    parents = _parent_problem_ids(skill)
    by_id = {problem.id: problem for problem in problems}
    parent = next((by_id[pid] for pid in parents if pid in by_id), None)
    skill_id = getattr(skill, "skill_id", "<unknown>")

    if parent is None:
        return AdmissionResult(
            skill_id=skill_id,
            problem_id="",
            admitted=admit_untestable,
            trials=0,
            successes=0,
            reason="no-parent-problem-available",
            injected=False,
        )

    # A bank holding ONLY this skill: the gate must judge the skill, not the bank it
    # happens to be joining. Parent exclusion is off by construction here -- probing
    # the parent pair is the entire point (see module docstring, point 2).
    solo = _solo_bank(skill)
    injection = build_injected_prompt(
        parent,
        solo,
        budget_tokens=injection_budget_tokens,
        parent_exclusion=False,
    )

    if not injection.injected:
        # The skill never reached the prompt, so the trial would measure the no-memory
        # baseline and call it the skill's verdict. Refuse to score it.
        return AdmissionResult(
            skill_id=skill_id,
            problem_id=parent.id,
            admitted=admit_untestable,
            trials=0,
            successes=0,
            reason="skill-did-not-fit-injection-budget",
            injected=False,
        )

    successes = 0
    raw_texts: list[str] = []
    performed = 0
    for trial in range(trials):
        if budget is not None:
            # Inside max_train, matching CORE. Raises BudgetExhausted rather than
            # quietly skipping, so a truncated admission pass cannot masquerade as a
            # low accept rate.
            budget.charge_admission(1)
        performed += 1
        raw = generate(injection.prompt)
        raw_texts.append(raw or "")
        if raw and parent.verify(raw):
            successes += 1

    admitted = successes == performed if require_all else successes > 0
    return AdmissionResult(
        skill_id=skill_id,
        problem_id=parent.id,
        admitted=admitted,
        trials=performed,
        successes=successes,
        reason="passed" if admitted else "failed-trial",
        raw_texts=raw_texts,
    )


def _solo_bank(skill: Any) -> Any:
    """A SkillBank containing exactly one skill.

    Built through `SkillBank.add_skill` rather than by poking `level_skills` directly,
    so the skill is filed under whatever key the real bank would file it under -- if
    that logic ever changes, admission follows it instead of drifting.
    """
    from skillrl.core.skill_bank import SkillBank

    bank = SkillBank()
    bank.add_skill(skill)
    return bank


def admit_skills(
    skills: Iterable[Any],
    problems: Sequence[Problem],
    generate: Callable[[str], str | None],
    *,
    budget: RolloutBudget | None = None,
    trials: int = DEFAULT_ADMISSION_TRIALS,
    enabled: bool = True,
    max_tests: int | None = None,
    concurrency: int = 1,
    **kwargs: Any,
) -> tuple[list[Any], list[AdmissionResult]]:
    """Gate a batch of candidate skills. Returns (admitted_skills, all_results).

    ``enabled=False`` is ablation A6: every candidate passes, and the results still
    record ``reason="gate-disabled"`` so the two arms are distinguishable in the logs
    rather than only in the config.

    ``max_tests`` caps how many candidates are *judged* in this call, so the round
    cannot overrun the admission reserve the schedule set aside (`passive_loop.
    make_schedule`). Overflow candidates are **rejected**, not waved through: that is
    the conservative direction, since it can only cost HeXA skills it might have kept
    and never manufactures a skill it did not earn. They are recorded with
    ``reason="admission-budget-capped"`` so a binding cap is visible in the logs
    rather than inferable from a bank that came out smaller than expected.

    ``concurrency`` fans out across candidates, which are independent by construction:
    each is tested alone against its own parent problem, and nothing here writes to the
    bank. Sequential by default. At 20 candidates x 10 rounds this is ~200 strictly
    sequential calls a run -- hours of a near-idle server, which is what the cluster's
    underutilisation reaper reacts to. Results stay in candidate order regardless of
    completion order, so the admitted list is deterministic.
    """
    candidates = list(skills)

    # Decide the cheap verdicts up front, in order, so the cap is applied to the same
    # candidates whether or not the fan-out is on.
    plan: list[tuple[int, Any, AdmissionResult | None]] = []
    judged = 0
    for index, skill in enumerate(candidates):
        skill_id = getattr(skill, "skill_id", "<unknown>")
        if not enabled:
            plan.append((index, skill, AdmissionResult(
                skill_id=skill_id, problem_id="", admitted=True, trials=0,
                successes=0, reason="gate-disabled", injected=False,
            )))
        elif max_tests is not None and judged >= max_tests:
            plan.append((index, skill, AdmissionResult(
                skill_id=skill_id, problem_id="", admitted=False, trials=0,
                successes=0, reason="admission-budget-capped", injected=False,
            )))
        else:
            plan.append((index, skill, None))
            # Counted optimistically: a candidate that turns out to be untestable
            # spends no call, so the cap can only under-spend the reserve, never
            # over-spend it. Erring the other way would reintroduce the overrun.
            judged += 1

    to_test = [(index, skill) for index, skill, verdict in plan if verdict is None]

    def run_one(item: tuple[int, Any]) -> tuple[int, AdmissionResult]:
        return item[0], test_skill_admission(
            item[1], problems, generate, budget=budget, trials=trials, **kwargs
        )

    if concurrency > 1 and len(to_test) > 1:
        # `RolloutBudget.charge_*` is a check-then-increment on a plain int, so
        # concurrent callers could both pass the check and push the total past
        # max_train. Serialise just the charging; the generation stays parallel.
        from concurrent.futures import ThreadPoolExecutor

        guarded = _LockedBudget(budget) if budget is not None else None

        def run_guarded(item: tuple[int, Any]) -> tuple[int, AdmissionResult]:
            return item[0], test_skill_admission(
                item[1], problems, generate, budget=guarded, trials=trials, **kwargs
            )

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            tested = dict(pool.map(run_guarded, to_test))
    else:
        tested = dict(run_one(item) for item in to_test)

    admitted: list[Any] = []
    results: list[AdmissionResult] = []
    for index, skill, verdict in plan:
        result = verdict if verdict is not None else tested[index]
        results.append(result)
        if result.admitted:
            admitted.append(skill)

    return admitted, results


class _LockedBudget:
    """Serialise `RolloutBudget` charging across threads.

    The counters are plain ints and `_charge_train` checks the limit *before*
    incrementing, so two threads can both observe room and jointly overshoot
    `max_train` -- turning the budget from a guarantee into a suggestion. Everything
    else delegates unchanged.
    """

    def __init__(self, budget: RolloutBudget) -> None:
        import threading

        self._budget = budget
        self._lock = threading.Lock()

    def charge_admission(self, n: int = 1) -> None:
        with self._lock:
            self._budget.charge_admission(n)

    def charge_train_solve(self, n: int = 1) -> None:
        with self._lock:
            self._budget.charge_train_solve(n)

    def charge_baseline(self, n: int = 1) -> None:
        with self._lock:
            self._budget.charge_baseline(n)

    def charge_eval(self, n: int = 1) -> None:
        with self._lock:
            self._budget.charge_eval(n)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._budget, name)


def accept_rate(results: Sequence[AdmissionResult]) -> dict[str, Any]:
    """Accept-rate summary for the run log.

    Reports the rate over **judged** candidates only. Skills that were auto-admitted
    because no parent was available or the block did not fit were never judged, so
    folding them in would inflate the rate toward whatever `admit_untestable` was set
    to -- reporting the gate's default instead of its behaviour.
    """
    judged = [r for r in results if r.trials > 0]
    return {
        "candidates": len(results),
        "judged": len(judged),
        "admitted_of_judged": sum(1 for r in judged if r.admitted),
        "accept_rate_judged": (
            sum(1 for r in judged if r.admitted) / len(judged) if judged else None
        ),
        "unjudged": len(results) - len(judged),
        "unjudged_reasons": {
            reason: sum(1 for r in results if r.trials == 0 and r.reason == reason)
            for reason in sorted({r.reason for r in results if r.trials == 0})
        },
        "admission_rollouts": sum(r.trials for r in results),
    }
