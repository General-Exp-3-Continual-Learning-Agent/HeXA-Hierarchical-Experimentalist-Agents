"""The HExA-passive curriculum loop (`PHASE3_DESIGN.md` §1, build-order step 4).

A round is a chunk of **previously unseen** training problems, each attempted `k`
times. Distil from chunk 1, apply the bank to chunk 2, and so on. This is HeXA's
published shape -- InterPhyre's round boundary also falls after a set of unseen items --
rather than an adaptation of it, and it buys two things an attempt-paced loop cannot:

* a **held-out signal at every round boundary**, not only at final eval. An
  attempt-paced loop re-attempts the problems its skills were distilled from, so
  within-training improvement cannot separate learning from memorising.
* **structural immunity to §5b parent contamination.** The bank only ever meets
  problems it was not distilled from, so `--parent-exclusion` should never fire under
  pure streaming. `PassiveRunResult.streaming_leak` asserts that rather than assuming
  it: a non-zero count means the curriculum leaked and is a bug, not a statistic.

Eval is frozen exactly as CORE freezes it (`run_core.py:1604`): no reflection, no
evolution, no admission, no bank writes, and eval rollouts are never stored. The
contamination guard runs **before** each evaluation, because a leaked eval item is the
one error that cannot be detected after the fact -- it just makes the method look
better.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from hexa_core.adapter import Problem, assert_no_eval_contamination
from hexa_core.admission import accept_rate, admit_skills
from hexa_core.seeds import CORE_EVAL_SEED_OFFSET, core_generation_seed
from hexa_core.budget import BudgetExhausted, RolloutBudget
from hexa_core.injection import (
    DEFAULT_INJECTION_BUDGET,
    DEFAULT_MAX_INJECTED_SKILLS,
    Injection,
    build_injected_prompt,
)
from hexa_core.train_baseline import attach_baselines
from hexa_core.trajectory import Attempt, round_summary, write_attempts

# CORE's `--max-rollouts` for the sample-efficiency grid.
DEFAULT_TRAIN_BUDGET = 2000
DEFAULT_ROUND_TARGET = 10

# Admission trials are charged INSIDE `max_train`, matching CORE (`PHASE3_DESIGN.md`
# §5c). So the schedule cannot hand the whole budget to solve calls: every admitted
# skill would then eat budget already promised to a later round. `make_schedule`
# reserves `rounds * cap * trials` up front and derives `k` from what remains.
#
# The reserve is per-round-uniform on purpose. The obvious alternative -- keep the
# planned `k` and let later rounds absorb the shortfall -- is the *worst* allocation
# available: it starves precisely the rounds whose bank is most developed, which is
# where the learning signal lives.
DEFAULT_ADMISSION_CAP = 20

# The reserve may never swallow more than this fraction of the budget, so a small
# `budget` (a smoke run) still yields k >= 1. When the ceiling binds, the effective
# cap is derived back down from it rather than the reserve silently under-covering.
ADMISSION_RESERVE_MAX_FRACTION = 5  # i.e. budget // 5 == 20%

# CORE seed namespaces. Distinct namespaces are not cosmetic: reusing train-solve's for
# admission would make the admission attempt a replay of an attempt already made.
SEED_TRAIN_SOLVE = 3000


@dataclass(frozen=True)
class Schedule:
    """The derived curriculum schedule. See `PHASE3_DESIGN.md` §1a.

    Derived rather than configured so it cannot drift from the budget. `k` is
    floor-divided, so `total_calls` may fall a little short of `budget` when
    `rounds * problems_per_round` does not divide it -- reported as `slack` instead of
    being silently absorbed, because an unspent budget is a real difference from CORE.

    `budget` is CORE's `--max-rollouts`, which covers solve **and** admission. So the
    quantity `k` is derived from is `solve_budget = budget - admission_reserve`, not
    `budget`. `total_planned` is the ceiling the run can reach; actual spend is lower
    whenever fewer than `admission_cap` skills are proposed in a round.
    """

    n_train: int
    rounds: int
    problems_per_round: int
    attempts_per_problem: int
    budget: int
    admission_reserve: int = 0
    admission_cap_per_round: int = 0
    #: How many times each training problem is scheduled. ``1`` is HeXA's original
    #: streaming regime: `rounds * problems_per_round == n_train`, every problem seen
    #: once and never again. ``>1`` re-presents problems in later rounds so a skill
    #: distilled from a failure can be *validated* on the same problem after the bank
    #: has changed -- the mechanism CORE gets for free from its ~6 episodes/problem.
    #: Raising it does not change `total_calls`: the extra visits are paid for out of
    #: `attempts_per_problem`, which `make_schedule` derives downward to compensate.
    visits_per_problem: int = 1

    @property
    def total_calls(self) -> int:
        """Planned *solve* calls. Kept under this name because it is the figure the
        curriculum table quotes; `total_planned` is the budget-comparable one."""
        return self.rounds * self.problems_per_round * self.attempts_per_problem

    @property
    def problem_visits(self) -> int:
        """Total (problem, round) slots the curriculum must fill."""
        return self.n_train * self.visits_per_problem

    @property
    def solve_budget(self) -> int:
        return self.budget - self.admission_reserve

    @property
    def total_planned(self) -> int:
        """Worst-case charge against `max_train`: every reserved trial gets used."""
        return self.total_calls + self.admission_reserve

    @property
    def slack(self) -> int:
        return self.budget - self.total_planned

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_train": self.n_train,
            "rounds": self.rounds,
            "problems_per_round": self.problems_per_round,
            "attempts_per_problem": self.attempts_per_problem,
            "visits_per_problem": self.visits_per_problem,
            "problem_visits": self.problem_visits,
            "budget": self.budget,
            "solve_budget": self.solve_budget,
            "admission_reserve": self.admission_reserve,
            "admission_cap_per_round": self.admission_cap_per_round,
            "total_calls": self.total_calls,
            "total_planned": self.total_planned,
            "slack": self.slack,
        }


def make_schedule(
    n_train: int,
    *,
    budget: int = DEFAULT_TRAIN_BUDGET,
    round_target: int = DEFAULT_ROUND_TARGET,
    admission_cap_per_round: int = DEFAULT_ADMISSION_CAP,
    admission_trials: int = 1,
    admission_enabled: bool = True,
    visits_per_problem: int = 1,
) -> Schedule:
    """Derive the schedule for a regime.

    ``rounds = min(round_target, n_train)`` because pure streaming consumes at least
    one unseen problem per round -- 5 training problems admit at most 5 rounds. That is
    a property of the regime, not a knob; manufacturing more requires `cycle_chunks`.

    Admission is reserved before ``k`` is derived, because CORE charges admission
    against ``--max-rollouts``. Without the reserve the schedule promises the whole
    budget to solve calls and the run dies at the first round boundary that admits a
    skill -- which is exactly how this was found.
    """
    if n_train < 1:
        raise ValueError(f"n_train must be >= 1, got {n_train}")
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    if visits_per_problem < 1:
        raise ValueError(
            f"visits_per_problem must be >= 1, got {visits_per_problem}"
        )

    rounds = max(1, min(round_target, n_train))
    # `visits_per_problem` widens each round rather than adding rounds, so `rounds`
    # -- and therefore the number of distillation steps -- is held fixed. That keeps
    # a revisit arm comparable to the baseline on everything except *when* a problem
    # is seen: same budget, same rounds, same evolver call count.
    problems_per_round = math.ceil(n_train * visits_per_problem / rounds)

    if not admission_enabled or admission_cap_per_round < 1 or admission_trials < 1:
        # A6 (gate off) tests nothing, so it reserves nothing and gets a larger k.
        # That is a real difference between the arms and is recorded in the schedule.
        reserve = 0
        cap = 0
    else:
        wanted = rounds * admission_cap_per_round * admission_trials
        ceiling = budget // ADMISSION_RESERVE_MAX_FRACTION
        reserve = min(wanted, ceiling)
        cap = reserve // (rounds * admission_trials)
        if cap < 1:
            raise ValueError(
                f"budget={budget} leaves no room to admission-test even one skill "
                f"per round across {rounds} rounds at {admission_trials} trial(s). "
                f"Raise the budget, lower round_target, or disable the gate."
            )
        # Re-derive the reserve from the (possibly clamped) cap so the two agree
        # exactly; otherwise the reserve over-books and shows up as phantom slack.
        reserve = rounds * cap * admission_trials

    attempts = (budget - reserve) // (rounds * problems_per_round)
    if attempts < 1:
        raise ValueError(
            f"budget={budget} is too small for {rounds} rounds x "
            f"{problems_per_round} problems after reserving {reserve} for admission: "
            f"it yields {attempts} attempts per problem. Raise the budget or lower "
            f"round_target."
        )
    return Schedule(
        n_train,
        rounds,
        problems_per_round,
        attempts,
        budget,
        reserve,
        cap,
        visits_per_problem,
    )


def chunk_problems(
    problems: Sequence[Problem], problems_per_round: int, rounds: int
) -> list[list[Problem]]:
    """Split the training set into consecutive, disjoint curriculum chunks.

    Disjointness is what makes the held-out-per-round property true, so it is produced
    by construction here rather than checked later.
    """
    chunks: list[list[Problem]] = []
    for index in range(rounds):
        start = index * problems_per_round
        chunk = list(problems[start : start + problems_per_round])
        if chunk:
            chunks.append(chunk)
    return chunks


#: How the curriculum allocates its `problem_visits` slots across training problems.
#:
#: ``"round_robin"``  HeXA's original regime: consecutive disjoint chunks in index
#:                    order, every problem exactly `visits_per_problem` times. This is
#:                    the frozen default -- Hanoi, ZebraLogic and MathGAP results were
#:                    produced under it and must keep reproducing under it.
#: ``"difficulty"``   Slots beyond the first are apportioned by measured hardness
#:                    ``1 - b_q``, mixed with the uniform allocation at
#:                    `sampling_uniform_mix`. Mirrors CORE's
#:                    ``--enable-biased-problem-sampling`` /
#:                    ``--biased-problem-sampling-uniform-mix``.
PROBLEM_SAMPLINGS = ("round_robin", "difficulty")

#: CORE's published value for ``--biased-problem-sampling-uniform-mix``. Kept as the
#: default so the biased arm starts from their setting rather than a tuned one.
DEFAULT_SAMPLING_UNIFORM_MIX = 0.5


def apportion_visits(
    problems: Sequence[Problem],
    total_visits: int,
    *,
    baselines: Any = None,
    sampling: str = "round_robin",
    uniform_mix: float = DEFAULT_SAMPLING_UNIFORM_MIX,
    max_per_problem: int | None = None,
) -> list[int]:
    """How many rounds each problem appears in. Returns counts aligned to `problems`.

    Every problem gets **at least one** visit before any problem gets a second, so
    coverage of the training split is preserved exactly and the arms stay comparable on
    "trained on the same |D_train| problems". Only the surplus is biased.

    Hardness is ``1 - b_q``, the same per-problem baseline M1 already pays for, so the
    bias costs no extra rollouts. Problems whose ``b_q`` is unmeasured or unusable fall
    back to the mean hardness rather than to 0 -- an unmeasured problem is unknown, not
    easy, and scoring it 0 would systematically starve exactly the problems whose
    baseline pass failed.

    ``max_per_problem`` caps any single problem's count, and callers scheduling into
    rounds **must** pass ``rounds``: a problem cannot appear twice in one round, so an
    uncapped weighting is free to hand out more visits than there are rounds to hold
    them. Found by the smoke gate at ``n_train=4, rounds=2``, where the weighting gave
    two problems 3 visits each and `plan_rounds` then had nowhere to put the third.
    Excess above the cap is redistributed to the problems still under it, by weight,
    so the total is preserved exactly rather than silently shrinking.
    """
    n = len(problems)
    if n == 0:
        return []
    if sampling not in PROBLEM_SAMPLINGS:
        raise ValueError(
            f"sampling must be one of {PROBLEM_SAMPLINGS}, got {sampling!r}"
        )
    if not 0.0 <= uniform_mix <= 1.0:
        raise ValueError(f"uniform_mix must be in [0, 1], got {uniform_mix}")
    if total_visits < n:
        raise ValueError(
            f"total_visits={total_visits} cannot cover {n} problems at least once"
        )
    cap = total_visits if max_per_problem is None else max_per_problem
    if cap < 1:
        raise ValueError(f"max_per_problem must be >= 1, got {max_per_problem}")
    if total_visits > n * cap:
        raise ValueError(
            f"total_visits={total_visits} exceeds what {n} problems can hold at "
            f"{cap} visit(s) each ({n * cap})"
        )

    base = [1] * n
    surplus = total_visits - n
    if surplus == 0:
        return base

    if sampling == "round_robin" or baselines is None:
        # Spread the surplus evenly, earlier problems first. With
        # `total_visits % n == 0` this is exactly the uniform allocation. The modulo
        # walk cannot breach `cap` because it never gets more than one round ahead of
        # the least-visited problem.
        for i in range(surplus):
            base[i % n] += 1
        return base

    measured = [baselines.b_q(p.id) for p in problems]
    known = [b for b in measured if b is not None]
    fallback = 1.0 - (sum(known) / len(known)) if known else 1.0
    hardness = [
        fallback if b is None else max(0.0, min(1.0, 1.0 - b)) for b in measured
    ]
    total_hardness = sum(hardness)
    if total_hardness <= 0.0:
        # Nothing is hard (or every b_q is 1.0): biasing is meaningless, so fall back
        # to uniform rather than dividing by zero or silently picking index order.
        for i in range(surplus):
            base[i % n] += 1
        return base

    weights = [
        uniform_mix / n + (1.0 - uniform_mix) * (h / total_hardness) for h in hardness
    ]

    # Capped largest-remainder apportionment. Deterministic, sums to `surplus` exactly
    # rather than approximately (so `total_calls` stays exactly the budget), and no
    # problem exceeds `cap`. Problems that would breach the cap are pinned there and
    # their weight is removed, then the rest is re-apportioned -- repeated until
    # nothing breaches, which terminates because each pass pins at least one problem.
    headroom = [cap - 1] * n  # each problem already holds its guaranteed first visit
    granted = [0] * n
    eligible = {i for i in range(n) if headroom[i] > 0}
    remaining = surplus
    while remaining > 0 and eligible:
        weight_sum = sum(weights[i] for i in eligible)
        if weight_sum <= 0.0:  # pragma: no cover - guarded by total_hardness above
            break
        exact = {i: weights[i] * remaining / weight_sum for i in eligible}
        breached = [i for i in eligible if exact[i] >= headroom[i]]
        if breached:
            for i in breached:
                granted[i] += headroom[i]
                remaining -= headroom[i]
                headroom[i] = 0
                eligible.discard(i)
            continue
        for i in eligible:
            granted[i] += int(exact[i])
            remaining -= int(exact[i])
        order = sorted(eligible, key=lambda i: (-(exact[i] % 1.0), i))
        for i in order[:remaining]:
            granted[i] += 1
        remaining = 0

    if remaining > 0:  # pragma: no cover - the n*cap guard above makes this impossible
        raise AssertionError(
            f"{remaining} visit(s) could not be apportioned within cap {cap}"
        )
    return [b + g for b, g in zip(base, granted)]


def plan_rounds(
    problems: Sequence[Problem],
    problems_per_round: int,
    rounds: int,
    *,
    visits_per_problem: int = 1,
    baselines: Any = None,
    sampling: str = "round_robin",
    uniform_mix: float = DEFAULT_SAMPLING_UNIFORM_MIX,
) -> list[list[Problem]]:
    """Build the per-round problem lists.

    With the defaults (`visits_per_problem=1`, ``sampling="round_robin"``) this
    delegates to :func:`chunk_problems` and is byte-identical to the frozen behaviour.

    Otherwise each problem's visits are spread as evenly as possible across the round
    range, because a revisit is only useful if the bank has had time to change: two
    consecutive rounds on the same problem teach almost nothing that one round of twice
    the width would not. A problem never appears twice within a round.
    """
    if visits_per_problem == 1 and sampling == "round_robin":
        return chunk_problems(problems, problems_per_round, rounds)

    counts = apportion_visits(
        problems,
        rounds * problems_per_round,
        baselines=baselines,
        sampling=sampling,
        uniform_mix=uniform_mix,
        # A problem cannot appear twice in one round, so `rounds` is a hard ceiling on
        # any single problem's visit count.
        max_per_problem=rounds,
    )

    # Assigning visits to rounds is a degree-constrained bipartite problem, and the
    # obvious greedies fail on it:
    #
    #   * nearest-ideal-round-with-space paints itself into corners where the only
    #     round left with capacity already holds the only problem left to place (found
    #     by the smoke gate, then reproduced on 46 of 168 schedule shapes);
    #   * most-remaining-capacity-first is feasible but clusters, because early on all
    #     rounds are equally empty and the tie resolves to rounds 0, 1, 2, 3 -- giving a
    #     median revisit gap of 1, which defeats the point of revisiting at all.
    #
    # So walk rounds in order and fill each with the `problems_per_round` problems that
    # have the most visits still owed. Taking the largest remaining counts at every step
    # is exactly the classic feasible schedule: it preserves both
    # `sum(remaining) == width * rounds_left` and `max(remaining) <= rounds_left`, the
    # two conditions the cap and the total already establish. Ties -- which is where
    # spreading actually happens -- go to whichever problem has gone longest unseen.
    remaining = list(counts)
    last_seen = [-(10**9)] * len(counts)
    buckets: list[list[int]] = []
    for round_index in range(rounds):
        rounds_left = rounds - round_index
        candidates = sorted(
            (i for i in range(len(counts)) if remaining[i] > 0),
            key=lambda i: (-remaining[i], last_seen[i], i),
        )
        chosen = candidates[:problems_per_round]
        if len(chosen) < problems_per_round or any(
            remaining[i] > rounds_left for i in range(len(counts))
        ):
            raise AssertionError(  # pragma: no cover - guarded by the cap and total
                f"round {round_index} of {rounds} cannot be filled to width "
                f"{problems_per_round}: {sum(remaining)} visit(s) owed across "
                f"{len(candidates)} problem(s), {rounds_left} round(s) left"
            )
        for index in chosen:
            remaining[index] -= 1
            last_seen[index] = round_index
        buckets.append(chosen)

    return [[problems[i] for i in bucket] for bucket in buckets if bucket]


@dataclass
class RoundRecord:
    """One round's outcome, with the diagnostics §8 requires."""

    round_index: int
    problem_ids: list[str]
    attempts: list[Attempt]
    summary: dict[str, Any]
    injections: list[dict[str, Any]]
    n_skills_before: int
    n_skills_after: int
    admission: dict[str, Any]
    excluded_parent: int
    starved_injections: int
    budget_bound_injections: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_index,
            "problem_ids": self.problem_ids,
            "summary": self.summary,
            "n_skills_before": self.n_skills_before,
            "n_skills_after": self.n_skills_after,
            "admission": self.admission,
            "excluded_parent": self.excluded_parent,
            "starved_injections": self.starved_injections,
            "budget_bound_injections": self.budget_bound_injections,
        }


@dataclass
class PassiveRunResult:
    """Everything a run produced, plus the assertions that make it trustworthy."""

    task_name: str
    schedule: Schedule
    rounds: list[RoundRecord] = field(default_factory=list)
    attempted_ids: set[str] = field(default_factory=set)
    budget: RolloutBudget | None = None
    eval_accuracy: float | None = None
    eval_n: int = 0
    eval_truncated: int = 0
    eval_accuracy_no_memory: float | None = None
    eval_truncated_no_memory: int = 0
    baselines_stamped: int = 0
    baselines_missing: int = 0
    stopped_early: str | None = None

    @property
    def excluded_parent_total(self) -> int:
        return sum(r.excluded_parent for r in self.rounds)

    @property
    def streaming_leak(self) -> bool:
        """True if parent exclusion fired -- under pure streaming, a bug.

        The curriculum guarantees the bank never meets a problem it was distilled
        from. If exclusion fires anyway, a future-chunk problem reached retrieval.
        """
        return self.excluded_parent_total > 0

    @property
    def starved_total(self) -> int:
        return sum(r.starved_injections for r in self.rounds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "schedule": self.schedule.to_dict(),
            "rounds": [r.to_dict() for r in self.rounds],
            "n_attempted_problems": len(self.attempted_ids),
            "excluded_parent_total": self.excluded_parent_total,
            "streaming_leak": self.streaming_leak,
            "starved_injections_total": self.starved_total,
            "rounds_completed": len(self.rounds),
            "stopped_early": self.stopped_early,
            "eval_accuracy": self.eval_accuracy,
            "eval_truncated": self.eval_truncated,
            "eval_accuracy_no_memory": self.eval_accuracy_no_memory,
            "eval_truncated_no_memory": self.eval_truncated_no_memory,
            "eval_delta": (
                None
                if self.eval_accuracy is None or self.eval_accuracy_no_memory is None
                else round(self.eval_accuracy - self.eval_accuracy_no_memory, 4)
            ),
            "baselines_stamped": self.baselines_stamped,
            "baselines_missing": self.baselines_missing,
            "eval_n": self.eval_n,
            "budget": self.budget.report() if self.budget else None,
        }


def run_passive_loop(
    task: Any,
    generate: Callable[[str], str | None],
    *,
    evolve: Callable[[str, Any, Path, int], Any] | None = None,
    bank: Any = None,
    n_train: int | None = None,
    budget_calls: int = DEFAULT_TRAIN_BUDGET,
    round_target: int = DEFAULT_ROUND_TARGET,
    injection_budget_tokens: int = DEFAULT_INJECTION_BUDGET,
    max_injected_skills: int = DEFAULT_MAX_INJECTED_SKILLS,
    parent_exclusion: bool = True,
    admission_enabled: bool = True,
    admission_trials: int = 1,
    hierarchy: str = "task",
    output_dir: str | Path | None = None,
    evaluate: bool = True,
    strict_injection: bool = False,
    concurrency: int = 1,
    generate_meta: Callable[[str], tuple[str, str]] | None = None,
    baselines: Any = None,
    eval_no_memory: bool = True,
    run_index: int = 0,
    visits_per_problem: int = 1,
    sampling: str = "round_robin",
    sampling_uniform_mix: float = DEFAULT_SAMPLING_UNIFORM_MIX,
) -> PassiveRunResult:
    """Run the curriculum loop over one task's training split, then evaluate.

    ``generate`` maps a prompt to scored text (``raw_text``) or None -- the same
    contract as `VLLMClient.generate`.

    ``evolve`` maps ``(level_name, prev_bank, round_dir, round_index)`` to a new bank.
    Left None, no distillation happens and the loop measures the no-memory baseline
    through the exact same code path -- which is the cheapest possible check that the
    harness itself adds nothing.

    ``hierarchy`` is ``"task"`` (flat) or ``"task:type"`` (§3), which keys retrieval on
    the problem's declared type from `Problem.metadata`.

    ``baselines`` is a `TrainBaselines` from `estimate_train_baselines`. Without it
    `Attempt.baseline` stays None and M1 degrades to the binary reward -- which is the
    documented pre-M1 fallback, but as a *default* it silently discards the b_q pass
    the caller just paid 10 x |D_train| calls for. The Stage 3a smoke pilot did exactly
    that: it measured b_q, printed it, wrote it to disk, and then scored every skill
    against the binary outcome anyway.
    """
    problems = task.train_problems(n_train)
    schedule = make_schedule(
        len(problems),
        budget=budget_calls,
        round_target=round_target,
        admission_trials=admission_trials,
        admission_enabled=admission_enabled and evolve is not None,
        visits_per_problem=visits_per_problem,
    )
    if sampling != "round_robin" and baselines is None:
        # Difficulty weighting reads `b_q`. Without baselines `apportion_visits` would
        # silently degrade to uniform, producing a run labelled "difficulty" that is
        # the control arm -- the same class of silent-fallback bug the b_q docstring
        # above records. Refuse instead.
        raise ValueError(
            f"sampling={sampling!r} needs `baselines` to weight by b_q, but none were "
            f"passed. Run estimate_train_baselines first, or use 'round_robin'."
        )
    budget = RolloutBudget(max_train=budget_calls)
    result = PassiveRunResult(
        task_name=getattr(task, "name", "?"), schedule=schedule, budget=budget
    )

    out_dir = Path(output_dir) if output_dir else None
    chunks = plan_rounds(
        problems,
        schedule.problems_per_round,
        schedule.rounds,
        visits_per_problem=visits_per_problem,
        baselines=baselines,
        sampling=sampling,
        uniform_mix=sampling_uniform_mix,
    )

    for round_index, chunk in enumerate(chunks):
        n_before = _bank_size(bank)
        attempts: list[Attempt] = []
        injection_records: list[dict[str, Any]] = []
        excluded = starved = budget_bound = 0

        # Build every injection for the round first. This is pure and cheap, and
        # crucially it is safe to do up front: the bank is FROZEN for the duration of a
        # round (it only changes at the round boundary), so every attempt in the round
        # sees the same bank by construction. That is exactly what makes the
        # generations independent and therefore concurrent -- without it, attempt i+1
        # could see a bank attempt i had changed and the round would not be a round.
        work: list[tuple[Problem, Injection]] = []
        for problem in chunk:
            level = _retrieval_level(problem, hierarchy)
            for _ in range(schedule.attempts_per_problem):
                injection = build_injected_prompt(
                    problem,
                    bank,
                    budget_tokens=injection_budget_tokens,
                    max_specific=max_injected_skills,
                    hierarchy_level=level,
                    parent_exclusion=parent_exclusion,
                )
                excluded += injection.excluded_parent
                starved += int(injection.starved)
                budget_bound += int(injection.budget_bound)
                injection_records.append(injection.to_dict())

                if strict_injection and injection.starved:
                    raise AssertionError(
                        f"round {round_index}, problem {problem.id}: the bank offered "
                        f"candidates but none reached the prompt, so this attempt is a "
                        f"no-memory rollout wearing a with-memory label. Raise "
                        f"injection_budget_tokens."
                    )
                work.append((problem, injection))

        # Charged before dispatch, for the whole batch: BudgetExhausted must fire
        # *before* spending the calls, not halfway through a concurrent fan-out where
        # some are already in flight.
        #
        # With the admission reserve in place this should be unreachable. It is caught
        # rather than raised anyway: an overrun is a scheduling defect, and losing a
        # multi-thousand-call run to it -- plus every round already completed -- costs
        # far more than stopping at a clean boundary and recording why. `stopped_early`
        # makes the truncation loud, so a short run can never be mistaken for a
        # full-budget one.
        try:
            budget.charge_train_solve(len(work))
        except BudgetExhausted as exc:
            result.stopped_early = (
                f"round {round_index} of {schedule.rounds} could not be afforded: {exc}"
            )
            break
        generated = _generate_many(
            [inj.prompt for _, inj in work], generate, concurrency, generate_meta
        )

        # Verification and record-building stay sequential and in submission order, so
        # the attempts list is deterministic regardless of completion order.
        for (problem, injection), (raw, finish_reason) in zip(work, generated):
            correct = bool(problem.verify(raw)) if raw else False
            attempts.append(
                Attempt(
                    problem_id=problem.id,
                    task_name=problem.task_name,
                    prompt=injection.prompt,
                    raw_text=raw,
                    parsed_answer=problem.parse_answer(raw) if raw else "",
                    correct=correct,
                    # Binary, matching M2. CORE's three tasks all return
                    # float(bool(is_correct)) from _reward_from_metrics.
                    reward=1.0 if correct else -1.0,
                    round=round_index,
                    added_tokens=injection.added_tokens,
                    finish_reason=finish_reason,
                    split="train",
                )
            )
            result.attempted_ids.add(problem.id)

        # Stamp b_q BEFORE the attempts are written and handed to the evolver: the
        # trajectory dicts carry `baseline`, and that is what M1's confidence reads.
        if baselines is not None:
            attempts, stamped = attach_baselines(attempts, baselines)
            result.baselines_stamped += stamped
            result.baselines_missing += len(attempts) - stamped
            if stamped == 0:
                # b_q was measured and paid for, and none of it reached the attempts.
                # Silently this degrades every downstream consumer to the binary
                # reward -- the evolver's ranking, the reward labels in its prompt,
                # and skill confidence -- while the run header still says "M1".
                # Exactly this happened once already, when `attach_baselines` had no
                # call site at all.
                raise RuntimeError(
                    f"round {round_index}: baselines were supplied but none of the "
                    f"{len(attempts)} attempts matched a measured problem. The "
                    f"evolver would silently fall back to the binary reward. Check "
                    f"that b_q was estimated on THIS task's training split."
                )

        round_dir = _write_round(out_dir, round_index, attempts)

        # ---- distil / evolve, then gate ------------------------------------
        admission_summary: dict[str, Any] = {"skipped": True}
        if evolve is not None and round_dir is not None:
            candidate_bank = evolve(
                getattr(task, "name", "?"), bank, round_dir, round_index
            )
            if candidate_bank is not None:
                new_skills = _new_skills(bank, candidate_bank)
                admitted, admission_results = admit_skills(
                    new_skills,
                    chunk,
                    generate,
                    budget=budget,
                    trials=admission_trials,
                    enabled=admission_enabled,
                    # Honour the reserve. Without this the cap is a number in a
                    # manifest rather than a property of the run.
                    max_tests=schedule.admission_cap_per_round or None,
                    concurrency=concurrency,
                    injection_budget_tokens=injection_budget_tokens,
                )
                bank = _apply_admission(candidate_bank, new_skills, admitted)
                admission_summary = accept_rate(admission_results)

        result.rounds.append(
            RoundRecord(
                round_index=round_index,
                problem_ids=[p.id for p in chunk],
                attempts=attempts,
                summary=round_summary(attempts),
                injections=injection_records,
                n_skills_before=n_before,
                n_skills_after=_bank_size(bank),
                admission=admission_summary,
                excluded_parent=excluded,
                starved_injections=starved,
                budget_bound_injections=budget_bound,
            )
        )

    if evaluate:
        # BEFORE the eval, never after: a leaked eval item cannot be detected from the
        # numbers afterwards, it only makes the method look better.
        assert_no_eval_contamination(task, result.attempted_ids)
        result.eval_accuracy, result.eval_n, result.eval_truncated = _frozen_eval(
            task,
            bank,
            generate,
            budget=budget,
            injection_budget_tokens=injection_budget_tokens,
            max_injected_skills=max_injected_skills,
            hierarchy=hierarchy,
            parent_exclusion=parent_exclusion,
            concurrency=concurrency,
            generate_meta=generate_meta,
            with_memory=True,
            run_index=run_index,
            record_path=(out_dir / "eval_with_memory.jsonl") if out_dir else None,
        )
        if eval_no_memory:
            # The paired control: identical problems, identical seeds, no bank. Run
            # through the SAME code path, so the difference is the memory and nothing
            # else -- a baseline measured elsewhere measures the elsewhere too.
            (
                result.eval_accuracy_no_memory,
                _,
                result.eval_truncated_no_memory,
            ) = _frozen_eval(
                task,
                None,
                generate,
                budget=budget,
                injection_budget_tokens=injection_budget_tokens,
                max_injected_skills=max_injected_skills,
                hierarchy=hierarchy,
                parent_exclusion=parent_exclusion,
                concurrency=concurrency,
                generate_meta=generate_meta,
                with_memory=False,
                run_index=run_index,
                record_path=(out_dir / "eval_no_memory.jsonl") if out_dir else None,
            )

    if out_dir is not None:
        (out_dir / "run_summary.json").write_text(
            json.dumps(result.to_dict(), indent=2, default=str)
        )
    return result


def _generate_many(
    prompts: Sequence[str],
    generate: Callable[[str], str | None],
    concurrency: int,
    generate_meta: Callable[[str], tuple[str, str]] | None = None,
    seeds: Sequence[int] | None = None,
) -> list[tuple[str, str]]:
    """Generate for many prompts, preserving input order.

    ``concurrency <= 1`` runs inline, which keeps the sequential path (and the tests
    that count call order) free of thread machinery. Above that, a thread pool -- the
    work is network-bound on the vLLM endpoint, so threads are the right tool and
    `ThreadPoolExecutor.map` already yields results in submission order.

    A failed generation becomes ``""`` rather than propagating, matching the sequential
    path: one 503 in a 200-call round must not discard the other 199.

    Returns ``(raw_text, finish_reason)`` pairs. The finish reason is not decoration:
    Hanoi truncated 9/20 at the reproduction gate, truncation only ever *depresses*
    accuracy, and without it a low number cannot be separated from a token-limit
    artefact. Callers that pass a plain ``generate`` get ``""`` and lose only the
    diagnostic, never a result.
    """
    def call(item: tuple[str, int | None]) -> tuple[str, str]:
        prompt, seed = item
        fn = generate_meta if generate_meta is not None else generate
        # Seeding is what makes the with-memory and no-memory arms paired, so it is
        # passed when the callable accepts it and silently skipped when it does not
        # (tests use one-argument fakes).
        if seed is not None:
            try:
                result = fn(prompt, seed=seed)
            except TypeError as exc:
                if "seed" not in str(exc):
                    raise
                result = fn(prompt)
        else:
            result = fn(prompt)
        if generate_meta is not None:
            text, reason = result
            return (text or ""), (reason or "")
        return (result or ""), ""

    items = list(zip(prompts, seeds if seeds is not None else [None] * len(prompts)))

    if concurrency <= 1:
        return [call(item) for item in items]

    from concurrent.futures import ThreadPoolExecutor

    def one(item: tuple[str, int | None]) -> tuple[str, str]:
        try:
            return call(item)
        except Exception:
            return "", "error"

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(one, items))


def _frozen_eval(
    task: Any,
    bank: Any,
    generate: Callable[[str], str | None],
    *,
    budget: RolloutBudget,
    injection_budget_tokens: int,
    hierarchy: str,
    parent_exclusion: bool,
    max_injected_skills: int = DEFAULT_MAX_INJECTED_SKILLS,
    concurrency: int = 1,
    generate_meta: Callable[[str], tuple[str, str]] | None = None,
    with_memory: bool = True,
    run_index: int = 0,
    llm_seed: int = 42,
    record_path: Path | None = None,
) -> tuple[float, int, int]:
    """Evaluate on the held-out split with the bank frozen.

    Mirrors CORE's `evaluate_memory_on(register_problem=False, update_utilities=False,
    mark_use=False, deterministic=True)`. Nothing here writes to the bank, and eval
    attempts are deliberately not returned as `Attempt`s so they cannot be fed to a
    later distillation pass by accident.

    **Seeded from CORE's eval namespace (2000), per (run, item).** This is what makes
    the with-memory and no-memory arms *paired*: same problem, same seed, the injected
    guidelines the only difference. Unseeded, the two arms would be independent samples
    and the comparison would need far more runs to say anything. `baseline_runner`
    derives its eval seeds identically, so an external baseline pass lines up too.

    ``with_memory=False`` runs the same problems and seeds with no bank at all, which
    is the no-memory arm. It is the same code path deliberately: a baseline measured
    through a different path measures the path as well as the memory.
    """
    problems = task.eval_problems()
    correct = 0
    truncated = 0
    records: list[dict[str, Any]] = []

    # The bank is frozen at eval by definition, so every injection is independent and
    # the whole split can be built up front and generated concurrently.
    injections = [
        build_injected_prompt(
            problem,
            bank if with_memory else None,
            budget_tokens=injection_budget_tokens,
            max_specific=max_injected_skills,
            hierarchy_level=_retrieval_level(problem, hierarchy),
            parent_exclusion=parent_exclusion,
        )
        for problem in problems
    ]
    seeds = [
        core_generation_seed(llm_seed, CORE_EVAL_SEED_OFFSET, run_index, item_index)
        for item_index in range(len(problems))
    ]
    budget.charge_eval(len(problems))
    generated = _generate_many(
        [inj.prompt for inj in injections],
        generate,
        concurrency,
        generate_meta,
        seeds=seeds,
    )

    for problem, injection, seed, (raw, finish_reason) in zip(
        problems, injections, seeds, generated
    ):
        if finish_reason == "length":
            truncated += 1
        is_correct = bool(raw and problem.verify(raw))
        correct += int(is_correct)
        records.append({
            "problem_id": problem.id,
            "seed": seed,
            "with_memory": with_memory,
            "correct": is_correct,
            "added_tokens": injection.added_tokens,
            "finish_reason": finish_reason,
            "parsed_answer": problem.parse_answer(raw) if raw else "",
        })

    if record_path is not None:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n"
        )
    return (
        (correct / len(problems) if problems else 0.0),
        len(problems),
        truncated,
    )


def _retrieval_level(problem: Problem, hierarchy: str) -> str:
    """The retrieval key. §3 maps HeXA's general->level onto task->type.

    Falls back to the bare task name when the problem declares no type, so a task
    without internal structure degrades to flat retrieval rather than keying on the
    string ``"matchstick:None"`` and matching nothing.
    """
    if hierarchy != "task:type":
        return problem.task_name
    problem_type = (
        problem.metadata.get("type")
        or problem.metadata.get("size_group")
        or problem.metadata.get("num_disks")
    )
    return f"{problem.task_name}:{problem_type}" if problem_type else problem.task_name


def _write_round(
    out_dir: Path | None, round_index: int, attempts: Sequence[Attempt]
) -> Path | None:
    """Persist a round's attempts where the (rebound) trajectory loader will find them.

    `evolve_skill_bank` takes a *directory* and calls `load_trajectories` on it, which
    `hexa_bridge.patch(load_trajectories_fn=load_attempt_trajectories)` rebinds to read
    ``*.jsonl``. Writing the file is therefore the whole integration -- no format
    conversion, no fork.
    """
    if out_dir is None:
        return None
    round_dir = Path(out_dir) / f"round{round_index}"
    write_attempts(round_dir / "attempts.jsonl", attempts)
    return round_dir


def _bank_size(bank: Any) -> int:
    if bank is None:
        return 0
    total = len(getattr(bank, "general_skills", []) or [])
    for skills in (getattr(bank, "level_skills", {}) or {}).values():
        total += len(skills)
    return total


def _all_skills(bank: Any) -> list[Any]:
    if bank is None:
        return []
    out = list(getattr(bank, "general_skills", []) or [])
    for skills in (getattr(bank, "level_skills", {}) or {}).values():
        out.extend(skills)
    return out


def _new_skills(previous: Any, candidate: Any) -> list[Any]:
    """Skills present in *candidate* but not *previous*, by skill_id.

    Only new skills face the gate. Re-gating a retained skill every round would charge
    admission rollouts repeatedly for a decision already made, quietly eating the
    training budget.
    """
    seen = {getattr(s, "skill_id", None) for s in _all_skills(previous)}
    return [s for s in _all_skills(candidate) if getattr(s, "skill_id", None) not in seen]


def _apply_admission(candidate: Any, proposed: Sequence[Any], admitted: Sequence[Any]) -> Any:
    """Remove rejected new skills from the candidate bank, in place.

    Uses `SkillBank.remove_skill` so removal follows the bank's own bookkeeping rather
    than this module reaching into its internals.
    """
    admitted_ids = {getattr(s, "skill_id", None) for s in admitted}
    for skill in proposed:
        skill_id = getattr(skill, "skill_id", None)
        if skill_id not in admitted_ids:
            remove = getattr(candidate, "remove_skill", None)
            if remove is not None:
                remove(skill_id)
    return candidate
