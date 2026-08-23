"""The ``b_q`` producer: per-problem no-memory baselines on the TRAIN split (M1).

`PHASE3_DESIGN.md` §2b. This exists for HeXA, not for CORE reproduction. HeXA's skill
confidence is currently the mean reward of the trajectories a skill was distilled from
(`distill._compute_initial_confidence`), which on CORE conflates *task difficulty* with
*skill quality*: Matchstick's base rate is ~0.68 and Hanoi's ~0.18, so a Matchstick
skill scores ~0.68 whether or not it contributed anything, and a genuinely useful Hanoi
skill scores ~0.18. Confidence drives retrieval ranking, pruning and admission, so all
three would be consuming a number that mostly encodes which task this is.

M1 replaces it with baseline-relative utility ``U(q) = r - b_q`` -- "did this skill beat
what the model does unaided *on this problem*". With ``r`` in {0,1} and ``b_q`` in
[0,1], ``U`` lands in [-1,1], exactly the domain `_compute_initial_confidence` already
maps to [0.1, 1.0], so its affine map applies unchanged.

Why this is not `baseline_runner`
---------------------------------
They look similar and are not interchangeable:

===================  ==========================  =============================
                     ``baseline_runner``         this module
===================  ==========================  =============================
split                **eval** (100 held out)     **train** (5 / 10 / 100)
granularity          aggregate accuracy          **per problem**
purpose              calibrate provider vs paper feed HeXA's confidence
seed namespace       2000                        **1000**
budget counter       ``charge_eval``             ``charge_baseline``
===================  ==========================  =============================

`baseline_runner._prepare_problems` hard-raises on any problem whose ``split !=
"eval"``, deliberately, so eval contamination is impossible -- which is exactly why it
cannot be reused here.

Two accounting details that matter for CORE parity
--------------------------------------------------
* **Seed namespace 1000, not 2000.** CORE uses 1000 for baseline estimation and 2000
  for eval (`run_core.py:1193` vs `1462`). Reusing 2000 would make these calls replays
  of eval seeds on train problems -- correlated in a way nothing downstream expects.
* **Charged to ``charge_baseline``, which is outside ``max_train``.** CORE does not
  count baseline estimation against ``--max-rollouts``; it counts admission trials but
  not this. Charging to ``charge_train_solve`` would silently take 50/100/1000 calls
  away from the training budget and understate what the method spent.

Totals therefore come out at 2050 / 2100 / 3000 for n = 5 / 10 / 100, matching CORE.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from hexa_core.adapter import Problem
from hexa_core.seeds import CORE_BASELINE_SEED_OFFSET, core_generation_seed
from hexa_core.budget import RolloutBudget

#: CORE's ``--num-baseline-train-runs`` default (`run_core.py:4036`).
DEFAULT_N_BASE = 10

#: CORE's ``--llm-seed`` default, so seeds are reproducible across our runs.
DEFAULT_LLM_SEED = 42


@dataclass
class ProblemBaseline:
    """``b_q`` for one training problem, with the evidence behind it."""

    problem_id: str
    task_name: str
    n_base: int
    successes: int
    seeds: list[int] = field(default_factory=list)
    errors: int = 0

    @property
    def b_q(self) -> float:
        """The no-memory success rate. Zero samples yields 0.0, but see `usable`."""
        return self.successes / self.n_base if self.n_base else 0.0

    @property
    def usable(self) -> bool:
        """False when no sample actually landed.

        Distinguishing this from a genuine ``b_q = 0.0`` matters: an unattempted
        problem and a never-solved problem both read as 0.0, and the first would make
        every skill from that problem look maximally useful (``U = r - 0``).
        """
        return self.n_base > 0 and self.errors < self.n_base

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "task_name": self.task_name,
            "n_base": self.n_base,
            "successes": self.successes,
            "b_q": self.b_q,
            "usable": self.usable,
            "errors": self.errors,
            "seeds": self.seeds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProblemBaseline:
        return cls(
            problem_id=data["problem_id"],
            task_name=data["task_name"],
            n_base=data["n_base"],
            successes=data["successes"],
            seeds=list(data.get("seeds", [])),
            errors=data.get("errors", 0),
        )


@dataclass
class TrainBaselines:
    """All ``b_q`` values for one task's training split."""

    task_name: str
    n_base: int
    baselines: dict[str, ProblemBaseline] = field(default_factory=dict)

    def b_q(self, problem_id: str) -> float | None:
        """``b_q`` for a problem, or None if unmeasured or unusable.

        Returns None rather than a default so callers must decide explicitly.
        `Attempt.utility` already degrades to the raw reward when ``baseline is None``,
        which is precisely HeXA's pre-M1 behaviour -- a visible, documented fallback
        rather than a silent 0.0 that would inflate every utility.
        """
        record = self.baselines.get(problem_id)
        return record.b_q if record is not None and record.usable else None

    @property
    def mean_b_q(self) -> float | None:
        usable = [r.b_q for r in self.baselines.values() if r.usable]
        return sum(usable) / len(usable) if usable else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "n_base": self.n_base,
            "n_problems": len(self.baselines),
            "n_usable": sum(1 for r in self.baselines.values() if r.usable),
            "mean_b_q": self.mean_b_q,
            "baselines": {k: v.to_dict() for k, v in self.baselines.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainBaselines:
        return cls(
            task_name=data["task_name"],
            n_base=data["n_base"],
            baselines={
                key: ProblemBaseline.from_dict(value)
                for key, value in data.get("baselines", {}).items()
            },
        )


def baseline_seed(problem_index: int, run_index: int, llm_seed: int = DEFAULT_LLM_SEED) -> int:
    """CORE's baseline-namespace seed for one (problem, run) pair.

    Namespace 1000, matching CORE's baseline estimation rather than its eval pass.
    """
    return core_generation_seed(
        llm_seed, CORE_BASELINE_SEED_OFFSET, run_index, problem_index
    )


def estimate_train_baselines(
    task: Any,
    generate: Callable[..., str | None],
    *,
    n_train: int | None = None,
    n_base: int = DEFAULT_N_BASE,
    budget: RolloutBudget | None = None,
    llm_seed: int = DEFAULT_LLM_SEED,
    output_path: str | Path | None = None,
    resume: bool = True,
    concurrency: int = 1,
) -> TrainBaselines:
    """Estimate ``b_q`` for every training problem with no memory injected.

    ``generate`` takes ``(prompt, seed=...)`` and returns scored text or None. A plain
    one-argument callable is accepted too, for tests and for clients without seeding.

    **Every call uses ``problem.build_prompt(None)``** -- CORE's untouched prompt. That
    is what makes the number a *baseline*: any injected guidelines would make ``b_q``
    a with-memory rate, and ``U = r - b_q`` would then be measuring a skill against
    itself.

    Resumable per problem, like `baseline_runner`: the 1000-call n=100 pass should not
    restart from scratch after an interruption.

    ``concurrency`` fans out across (problem, run) pairs, which are mutually
    independent -- no bank, no memory, no ordering. Sequential by default so the tests
    that count call order stay meaningful, but the pilot must raise it: 10 x 100 = 1000
    strictly sequential calls at ~45s each is ~12 hours of a server sitting at one
    request in flight, which is both unusable and how the cluster's underutilisation
    reaper found us.

    Problems are still committed in batches rather than all at once, so an interruption
    loses at most one batch of work instead of the whole pass.
    """
    problems = _assert_train_split(task.train_problems(n_train))
    result = TrainBaselines(task_name=getattr(task, "name", "?"), n_base=n_base)

    path = Path(output_path) if output_path else None
    if path is not None and resume and path.is_file():
        cached = TrainBaselines.from_dict(json.loads(path.read_text()))
        if cached.n_base == n_base:
            result.baselines.update(cached.baselines)

    pending = [
        (index, problem)
        for index, problem in enumerate(problems)
        if not (
            (existing := result.baselines.get(problem.id)) is not None
            and existing.n_base == n_base
            and existing.usable
        )
    ]

    # Size a batch so its fan-out fills `concurrency` without overshooting it.
    per_batch = max(1, concurrency // max(1, n_base)) if concurrency > 1 else 1

    for start in range(0, len(pending), per_batch):
        batch = pending[start : start + per_batch]

        jobs: list[tuple[str, str, int]] = []          # (problem_id, prompt, seed)
        seeds_by_problem: dict[str, list[int]] = {}
        for problem_index, problem in batch:
            # CORE's own prompt, with NO guidelines. See the docstring.
            prompt = problem.build_prompt(None)
            seeds = [
                baseline_seed(problem_index, run_index, llm_seed)
                for run_index in range(n_base)
            ]
            seeds_by_problem[problem.id] = seeds
            jobs.extend((problem.id, prompt, seed) for seed in seeds)

        if budget is not None:
            # Outside max_train, matching CORE. Charged for the whole batch before
            # dispatch, so a limit cannot trip midway through a live fan-out.
            budget.charge_baseline(len(jobs))

        raw_texts = _call_many(generate, jobs, concurrency)

        by_id = {problem.id: problem for _, problem in batch}
        tallies: dict[str, list[int]] = {pid: [0, 0] for pid in seeds_by_problem}
        for (problem_id, _, _), raw in zip(jobs, raw_texts):
            if not raw:
                tallies[problem_id][1] += 1                       # errors
            elif by_id[problem_id].verify(raw):
                tallies[problem_id][0] += 1                       # successes

        for problem_id, (successes, errors) in tallies.items():
            problem = by_id[problem_id]
            result.baselines[problem_id] = ProblemBaseline(
                problem_id=problem_id,
                task_name=problem.task_name,
                n_base=n_base,
                successes=successes,
                seeds=seeds_by_problem[problem_id],
                errors=errors,
            )
        if path is not None:
            _write(path, result)

    if path is not None:
        _write(path, result)
    return result


def _call_many(
    generate: Callable[..., str | None],
    jobs: Sequence[tuple[str, str, int]],
    concurrency: int,
) -> list[str]:
    """Run ``(problem_id, prompt, seed)`` jobs, preserving input order.

    Order is preserved so the ``zip`` against ``jobs`` is sound regardless of
    completion order. A failed call yields ``""`` and is counted as an error rather
    than propagating: one 503 must not discard a whole batch.
    """
    if concurrency <= 1:
        return [_call(generate, prompt, seed) for _, prompt, seed in jobs]

    from concurrent.futures import ThreadPoolExecutor

    def one(job: tuple[str, str, int]) -> str:
        try:
            return _call(generate, job[1], job[2])
        except Exception:
            return ""

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(one, jobs))


def _call(generate: Callable[..., str | None], prompt: str, seed: int) -> str:
    """Call *generate*, passing a seed when it accepts one.

    Seeding is what makes the pass reproducible, so a client that supports it must get
    it; a fake that does not must still work. TypeError is narrowed to the seed keyword
    so a genuine TypeError inside the client is not swallowed as "no seed support".
    """
    try:
        return generate(prompt, seed=seed) or ""
    except TypeError as exc:
        if "seed" not in str(exc):
            raise
        return generate(prompt) or ""


def _assert_train_split(problems: Sequence[Problem]) -> list[Problem]:
    """Refuse anything that is not a training problem.

    The mirror of `baseline_runner._prepare_problems`, which refuses non-eval problems.
    Measuring ``b_q`` on an eval item would both contaminate the eval split and produce
    a baseline for a problem no skill can ever be distilled from.
    """
    wrong = [p.id for p in problems if p.split != "train"]
    if wrong:
        raise ValueError(
            f"estimate_train_baselines requires train-split problems; got "
            f"{len(wrong)} with a different split: {wrong[:5]}. b_q is a "
            f"training-time quantity (seed namespace 1000); the eval-split "
            f"equivalent is baseline_runner.run_baseline_eval (namespace 2000)."
        )
    return list(problems)


def _write(path: Path, result: TrainBaselines) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, default=str))


def attach_baselines(
    attempts: Iterable[Any], baselines: TrainBaselines
) -> tuple[list[Any], int]:
    """Return ``(attempts_with_b_q, n_stamped)``.

    `Attempt` is a **frozen** dataclass, so this returns new objects via
    `dataclasses.replace` rather than mutating -- an in-place version raises
    `FrozenInstanceError`. Frozen is the right choice for a recorded rollout (an
    attempt is evidence and should not be editable after the fact), so the copy is
    deliberate rather than a workaround.

    `Attempt.utility` then yields ``float(correct) - b_q`` instead of the raw reward,
    which is the whole of M1 on the consumption side. Attempts whose problem has no
    usable baseline are passed through untouched, so they degrade to HeXA's pre-M1
    behaviour rather than silently taking ``b_q = 0`` -- which would make every skill
    from an unmeasured problem look maximally useful.
    """
    from dataclasses import replace

    out: list[Any] = []
    stamped = 0
    for attempt in attempts:
        value = baselines.b_q(attempt.problem_id)
        if value is None:
            out.append(attempt)
            continue
        out.append(replace(attempt, baseline=value))
        stamped += 1
    return out, stamped
