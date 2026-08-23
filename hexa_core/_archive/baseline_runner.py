"""Phase 2: the no-learning reproduction gate.

One LLM call per problem, with **empty guidelines**, over a task's 100 held-out
eval problems, repeated for N independent runs. No memory, no bank, no retrieval,
nothing learned -- so the prompt is CORE's base solve prompt and the measured
accuracy is the number every later HExA result has to beat.

Three parity details are load-bearing, and each one is *checked* here rather than
commented:

* **Empty guidelines means CORE's baseline prompt byte for byte.** CORE's own
  no-memory pass calls ``task.build_solve_prompt(item.problem, None)``
  (``scripts/run_core.py:1189``); ``None``, not ``""``, is CORE's empty value
  because ``_format_lesson_context_from_objects([])`` returns ``None``. Before any
  API call the runner asserts ``problem.build_prompt(None) == problem.prompt`` for
  every problem.
* **The scored text is ``reasoning_text + final_text``**, concatenated with no
  separator (``src/model/nvidia_client.py:141``, consumed at
  ``scripts/run_core.py:433``). Every response is checked against that identity, so
  a server configured without ``--reasoning-parser openai_gptoss`` fails on call
  one instead of after 900 calls and an unexplained accuracy miss.
* **Seeds are CORE's.** ``seed = _compute_generation_seed(llm_seed, 2000, run_idx,
  item_idx)``, so run 1 here issues byte-identical seeds to a CORE eval pass.

Every generation is charged to :attr:`RolloutBudget.eval`. Results are written one
JSON file per (problem, run) and reused on a second invocation, so an interrupted
900-call gate resumes for free.

Because we serve ``gpt-oss-120b`` on local vLLM and the published numbers are from
NVIDIA NIM, the provider does not match: the output of this module is
**calibration** data. See :data:`PROVIDER_CAVEAT`.

One deliberate divergence from CORE: CORE raises and aborts the whole eval when an
API call fails; we retry, then record the failure and mark the task ``ERROR``. A
900-call run should not lose forty minutes to one 503, but an errored run must
never be quotable -- which is why ``ERROR`` short-circuits the gate verdict.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from math import hypot, isnan, nan, sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from .budget import RolloutBudget
from .logging_schema import RunManifest, ValidationLogger

if TYPE_CHECKING:  # runtime-decoupled on purpose: see module docstring
    from .adapter import Problem, Task

# -- CORE's eval configuration, every value traced to a line in CORE ---------

# CORE gives each phase its own seed namespace, so the same (run, item) pair draws
# a different generation seed in each. All five were read off the call sites; they
# are unrelated numbers, not offsets from a common base, so picking the wrong one
# silently samples from a different distribution than CORE did.
#
#: Eval / validation passes (``run_core.py:1421``, used at ``:1462``).
CORE_EVAL_SEED_OFFSET = 2000
#: ``run_baseline_training``'s no-memory passes -- the n_base=10 samples that
#: estimate the per-problem baseline b_q (``run_core.py:1193``). **M1 must use
#: this one**, not :data:`CORE_EVAL_SEED_OFFSET`; b_q is a training-item quantity.
CORE_BASELINE_SEED_OFFSET = 1000
#: Training solve attempts (``run_core.py:1993``).
CORE_TRAIN_SOLVE_SEED_OFFSET = 3000
#: Reflection generation (``run_core.py:2603``).
CORE_REFLECTION_SEED_OFFSET = 4000
#: Lesson-admission trials, in ``_compute_admission_seed``: 5000 for a global
#: trial, 5001 when the trial is per-lesson.
CORE_ADMISSION_SEED_OFFSETS = (5000, 5001)
#: The biased training-item sampler (``_sample_biased_training_item``). Unlike
#: every constant above, this one is applied to ``args.seed`` rather than
#: ``args.llm_seed`` -- it seeds problem *selection*, not generation.
CORE_BIASED_SAMPLER_SEED_OFFSET = 7000
DEFAULT_LLM_SEED = 42  # --llm-seed          (run_core.py:3991)
DEFAULT_MAX_TOKENS = 32768  # --eval-tokens       (run_core.py:3995)
DEFAULT_TEMPERATURE = 0.6  # --eval-temperature  (run_core.py:4003)
DEFAULT_TOP_P = 0.95  # --eval-top-p        (run_core.py:4006)
DEFAULT_CONCURRENCY = 10  # --num-eval-batch-calls (run_core.py:3957)
DEFAULT_REASONING_EFFORT = "high"  # nvidia_client.py:102, for any *gpt-oss-120b* model
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_SEM_TOLERANCE = 2.0

EVAL_ITEMS = 100

PROVIDER_CAVEAT = (
    "CALIBRATION RUN: gpt-oss-120b served on local vLLM, while the published "
    "numbers were produced on NVIDIA NIM. The provider does not match, so the "
    "paper's values are reference points only and are NOT directly citable from "
    "this run."
)

RUN_DESCRIPTION = "no-learning baseline: empty guidelines, no memory, no retrieval"


@dataclass(frozen=True)
class PaperTarget:
    """A published no-learning accuracy and its reported spread."""

    accuracy: float
    sem: float
    provider: str = "NVIDIA NIM"


#: Table 1 no-learning column of the CORE paper (gpt-oss-120b on NVIDIA NIM).
PAPER_TARGETS: Mapping[str, PaperTarget] = {
    "matchstick": PaperTarget(0.681, 0.007),
    "zebralogic": PaperTarget(0.509, 0.014),
    "tower-of-hanoi": PaperTarget(0.179, 0.007),
}


def core_generation_seed(base_seed: int, *components: int) -> int:
    """CORE's per-call seed derivation.

    Transcribed from ``core-reasoning/scripts/run_core.py:1063-1067`` rather than
    imported, because ``run_core.py`` calls ``os.chdir(ROOT)`` at import time
    (line 24) and importing it would silently move our process into the CORE
    checkout. Four lines of pure arithmetic -- no verifier, prompt or generator is
    duplicated here.
    """
    seed = int(base_seed)
    for component in components:
        seed = (seed * 6364136223846793005 + int(component) + 1442695040888963407) % (2**31 - 1)
    return seed


# -- records ----------------------------------------------------------------


@dataclass(frozen=True)
class GenerationConfig:
    """Everything that, if changed, invalidates a cached per-problem result."""

    model: str
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    max_tokens: int = DEFAULT_MAX_TOKENS
    llm_seed: int = DEFAULT_LLM_SEED
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT
    system_prompt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "llm_seed": self.llm_seed,
            "reasoning_effort": self.reasoning_effort,
            "system_prompt": self.system_prompt,
        }

    def fingerprint(self, *, prompt: str, seed: int, eval_ids_sha256: str) -> str:
        """Identity of one cached call: config + the exact prompt + the exact seed.

        Resumability's failure mode is a stale result from a different model or
        temperature silently poisoning the gate, so the cache key covers every
        input rather than just the problem id.
        """
        payload = json.dumps(
            {
                **self.to_dict(),
                "seed": seed,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "eval_ids_sha256": eval_ids_sha256,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProblemResult:
    """One graded generation."""

    problem_id: str
    run: int  # 1-based, matching CORE's `all_run_results[i]["run"]`
    item_idx: int  # position in CORE's eval order; part of the seed
    seed: int
    is_correct: bool
    verifier_score: float
    prediction: str
    finish_reason: str | None
    raw_chars: int
    reasoning_chars: int
    final_chars: int
    elapsed_s: float
    usage: dict[str, Any] | None = None
    error: str | None = None
    attempts: int = 1
    fingerprint: str | None = None
    reused: bool = field(default=False, compare=False)

    @property
    def truncated(self) -> bool:
        """CORE's ``primary_length_count`` rule (``run_core.py:1888``)."""
        return self.finish_reason == "length"

    @property
    def empty_parse(self) -> bool:
        """CORE's ``missing_usable_answer_count`` rule (``run_core.py:1882``)."""
        return not str(self.prediction or "").strip()

    @property
    def no_reasoning(self) -> bool:
        """No chain of thought came back -- the ``--reasoning-parser`` smoke signal."""
        return self.reasoning_chars == 0

    def to_core_detail(self) -> dict[str, Any]:
        """A ``details`` entry shaped so CORE's own summarizer works on our logs.

        ``_summarize_validation_attempt_diagnostics`` (``run_core.py:1865``) reads
        exactly ``prediction``, ``primary_finish_reason`` and ``prediction_source``,
        which is where our truncation and empty-parse counts come from -- mirrored,
        not reinvented. ``prediction_source`` is always ``"primary"``: we have no
        fallback-generation path.
        """
        return {
            "problem_id": self.problem_id,
            "prediction": self.prediction,
            "verifier_score": self.verifier_score,
            "is_correct": self.is_correct,
            "prediction_source": "primary",
            "primary_finish_reason": self.finish_reason,
            "finish_reason": self.finish_reason,
            "seed": self.seed,
            "item_idx": self.item_idx,
            "elapsed_s": self.elapsed_s,
            "reasoning_chars": self.reasoning_chars,
            "final_chars": self.final_chars,
            "error": self.error,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "run": self.run,
            "item_idx": self.item_idx,
            "seed": self.seed,
            "is_correct": self.is_correct,
            "verifier_score": self.verifier_score,
            "prediction": self.prediction,
            "finish_reason": self.finish_reason,
            "raw_chars": self.raw_chars,
            "reasoning_chars": self.reasoning_chars,
            "final_chars": self.final_chars,
            "elapsed_s": self.elapsed_s,
            "usage": self.usage,
            "error": self.error,
            "attempts": self.attempts,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any], *, reused: bool = True) -> ProblemResult:
        return cls(
            problem_id=str(payload["problem_id"]),
            run=int(payload["run"]),
            item_idx=int(payload["item_idx"]),
            seed=int(payload["seed"]),
            is_correct=bool(payload["is_correct"]),
            verifier_score=float(payload["verifier_score"]),
            prediction=str(payload.get("prediction") or ""),
            finish_reason=payload.get("finish_reason"),
            raw_chars=int(payload.get("raw_chars", 0)),
            reasoning_chars=int(payload.get("reasoning_chars", 0)),
            final_chars=int(payload.get("final_chars", 0)),
            elapsed_s=float(payload.get("elapsed_s", 0.0)),
            usage=payload.get("usage"),
            error=payload.get("error"),
            attempts=int(payload.get("attempts", 1)),
            fingerprint=payload.get("fingerprint"),
            reused=reused,
        )


@dataclass
class RunResult:
    """One full pass over the problem list."""

    run: int
    results: list[ProblemResult]
    wall_clock_s: float

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def correct(self) -> int:
        return sum(1 for r in self.results if r.is_correct)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def mean_verifier_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.verifier_score for r in self.results) / self.total

    @property
    def truncated(self) -> int:
        return sum(1 for r in self.results if r.truncated)

    @property
    def empty_parses(self) -> int:
        return sum(1 for r in self.results if r.empty_parse)

    @property
    def no_reasoning(self) -> int:
        return sum(1 for r in self.results if r.no_reasoning)

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.error)

    @property
    def reused(self) -> int:
        return sum(1 for r in self.results if r.reused)

    def to_core_run_dict(self) -> dict[str, Any]:
        """CORE's per-run aggregate shape (``run_core.py:1584-1592``)."""
        return {
            "run": self.run,
            "accuracy": self.accuracy,
            "mean_verifier_score": self.mean_verifier_score,
            "correct": self.correct,
            "total": self.total,
            "details": [r.to_core_detail() for r in self.results],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run,
            "accuracy": self.accuracy,
            "mean_verifier_score": self.mean_verifier_score,
            "correct": self.correct,
            "total": self.total,
            "truncated": self.truncated,
            "empty_parses": self.empty_parses,
            "no_reasoning": self.no_reasoning,
            "errors": self.errors,
            "reused": self.reused,
            "wall_clock_s": self.wall_clock_s,
        }


@dataclass
class TaskBaselineResult:
    """Aggregate over all runs for one task."""

    task_name: str
    core_task_name: str | None
    n_problems: int
    runs: list[RunResult]
    eval_ids_sha256: str
    config: dict[str, Any]
    wall_clock_s: float
    budget_report: dict[str, int | None] = field(default_factory=dict)
    task_manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def accuracies(self) -> list[float]:
        return [r.accuracy for r in self.runs]

    @property
    def num_runs(self) -> int:
        return len(self.runs)

    @property
    def mean_accuracy(self) -> float:
        """Mean of the per-run accuracies -- CORE's aggregation (``run_core.py:1601``)."""
        return float(statistics.fmean(self.accuracies)) if self.runs else nan

    @property
    def population_std(self) -> float:
        """``np.std`` over run accuracies: CORE's ``val_std_with_mem``, NOT a SEM."""
        if not self.runs:
            return nan
        return float(statistics.pstdev(self.accuracies)) if len(self.runs) > 1 else 0.0

    @property
    def sem(self) -> float:
        """Standard error of the mean across runs (sample std, ``ddof=1``)."""
        if len(self.runs) < 2:
            return nan
        return float(statistics.stdev(self.accuracies)) / sqrt(len(self.runs))

    @property
    def binomial_se(self) -> float:
        """Pooled-attempt binomial SE: the floor used when the across-run SEM is unusable.

        With three runs the across-run SEM is a 2-dof estimate and can collapse to
        exactly 0 when all runs happen to land on the same accuracy. Both bases are
        always reported and the gate names the one it used.
        """
        attempts = sum(r.total for r in self.runs)
        if attempts == 0:
            return nan
        p = sum(r.correct for r in self.runs) / attempts
        return sqrt(max(p * (1.0 - p), 0.0) / attempts)

    @property
    def mean_verifier_score(self) -> float:
        scores = [r.mean_verifier_score for r in self.runs]
        return float(statistics.fmean(scores)) if scores else nan

    @property
    def verifier_score_std(self) -> float:
        scores = [r.mean_verifier_score for r in self.runs]
        if not scores:
            return nan
        return float(statistics.pstdev(scores)) if len(scores) > 1 else 0.0

    @property
    def truncated(self) -> int:
        return sum(r.truncated for r in self.runs)

    @property
    def empty_parses(self) -> int:
        return sum(r.empty_parses for r in self.runs)

    @property
    def no_reasoning(self) -> int:
        return sum(r.no_reasoning for r in self.runs)

    @property
    def errors(self) -> int:
        return sum(r.errors for r in self.runs)

    @property
    def reused(self) -> int:
        return sum(r.reused for r in self.runs)

    @property
    def attempts(self) -> int:
        return sum(r.total for r in self.runs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "core_task_name": self.core_task_name,
            "description": RUN_DESCRIPTION,
            "provider_caveat": PROVIDER_CAVEAT,
            "n_problems": self.n_problems,
            "num_runs": self.num_runs,
            "eval_ids_sha256": self.eval_ids_sha256,
            "config": self.config,
            "accuracies": self.accuracies,
            "mean_accuracy": self.mean_accuracy,
            "sem_across_runs": self.sem,
            "binomial_se": self.binomial_se,
            # CORE's key name, CORE's definition (population std). Never a SEM.
            "val_std_with_mem": self.population_std,
            "mean_verifier_score": self.mean_verifier_score,
            "verifier_score_std": self.verifier_score_std,
            "attempts": self.attempts,
            "truncated_length": self.truncated,
            "empty_parses": self.empty_parses,
            "no_reasoning": self.no_reasoning,
            "errors": self.errors,
            "reused": self.reused,
            "wall_clock_s": self.wall_clock_s,
            "budget": self.budget_report,
            "runs": [r.to_dict() for r in self.runs],
        }


# -- gate comparison --------------------------------------------------------


@dataclass(frozen=True)
class GateComparison:
    """Measured versus published, with the verdict and how it was reached."""

    task_name: str
    measured: float
    measured_sem: float
    sem_basis: str
    target: PaperTarget | None
    delta: float
    combined_sem: float
    delta_in_sems: float
    verdict: str  # "PASS*" | "MISS*" | "ERROR" | "NO-TARGET"
    tolerance_sems: float
    errors: int = 0

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS*"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "measured": self.measured,
            "measured_sem": self.measured_sem,
            "sem_basis": self.sem_basis,
            "paper_accuracy": None if self.target is None else self.target.accuracy,
            "paper_sem": None if self.target is None else self.target.sem,
            "paper_provider": None if self.target is None else self.target.provider,
            "delta": self.delta,
            "combined_sem": self.combined_sem,
            "delta_in_sems": self.delta_in_sems,
            "tolerance_sems": self.tolerance_sems,
            "verdict": self.verdict,
            "errors": self.errors,
            "caveat": PROVIDER_CAVEAT,
        }


#: Printed under every gate table. The ``*`` on the verdicts points here: nothing
#: in this module is allowed to claim a bare "PASS".
VERDICT_FOOTNOTE = (
    "* consistent with the published value within {tol:g} SEM on a DIFFERENT "
    "provider: calibration evidence, not a reproduction claim."
)


def compare_to_paper(
    result: TaskBaselineResult, *, tolerance_sems: float = DEFAULT_SEM_TOLERANCE
) -> GateComparison:
    """Compare one task's measured accuracy against its published value.

    ``delta`` is measured minus paper, and ``delta_in_sems`` divides it by the
    quadrature-combined SEM of the two. A task with any failed API call is
    ``ERROR`` regardless of the arithmetic -- an incomplete run can neither pass
    nor miss.
    """
    target = PAPER_TARGETS.get(result.task_name)
    measured = result.mean_accuracy

    across = result.sem
    if isnan(across) or across <= 0.0:
        measured_sem, sem_basis = result.binomial_se, f"binomial (pooled n={result.attempts})"
    else:
        measured_sem, sem_basis = across, f"across-run (n={result.num_runs})"

    if target is None:
        return GateComparison(
            task_name=result.task_name,
            measured=measured,
            measured_sem=measured_sem,
            sem_basis=sem_basis,
            target=None,
            delta=nan,
            combined_sem=nan,
            delta_in_sems=nan,
            verdict="NO-TARGET",
            tolerance_sems=tolerance_sems,
            errors=result.errors,
        )

    delta = measured - target.accuracy
    combined = hypot(0.0 if isnan(measured_sem) else measured_sem, target.sem)
    delta_in_sems = delta / combined if combined > 0 else nan

    if result.errors > 0:
        verdict = "ERROR"
    elif not result.runs:
        verdict = "ERROR"
    elif isnan(delta_in_sems):
        verdict = "MISS*"
    else:
        verdict = "PASS*" if abs(delta_in_sems) <= tolerance_sems else "MISS*"

    return GateComparison(
        task_name=result.task_name,
        measured=measured,
        measured_sem=measured_sem,
        sem_basis=sem_basis,
        target=target,
        delta=delta,
        combined_sem=combined,
        delta_in_sems=delta_in_sems,
        verdict=verdict,
        tolerance_sems=tolerance_sems,
        errors=result.errors,
    )


def summarize_gate(
    results: Iterable[TaskBaselineResult], *, tolerance_sems: float = DEFAULT_SEM_TOLERANCE
) -> list[GateComparison]:
    """All the statistics the CLI needs, so the CLI does none of its own."""
    return [compare_to_paper(r, tolerance_sems=tolerance_sems) for r in results]


# -- on-disk layout ---------------------------------------------------------

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def _slug(problem_id: str) -> str:
    """Path-safe, collision-free filename stem for a CORE problem id."""
    digest = hashlib.sha1(problem_id.encode("utf-8")).hexdigest()[:8]
    return f"{_UNSAFE.sub('_', problem_id)[:80]}-{digest}"


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Write via a temp file so a killed job cannot leave valid-looking truncated JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


class _Reporter:
    """Serialised stdout, so ten worker threads do not interleave mid-line."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._lock = threading.Lock()

    def __call__(self, message: str, *, force: bool = False) -> None:
        if not (self.enabled or force):
            return
        with self._lock:
            print(message, flush=True)


# -- the runner -------------------------------------------------------------


def run_baseline_eval(
    task: Task,
    client: Any,
    *,
    output_dir: str | Path,
    runs: int = 3,
    limit: int | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    llm_seed: int = DEFAULT_LLM_SEED,
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
    system_prompt: str | None = None,
    model: str | None = None,
    task_seed: int = 42,
    budget: RolloutBudget | None = None,
    logger: ValidationLogger | None = None,
    resume: bool = True,
    write_manifest: bool = True,
    log_core_validation: bool = True,
    provider_info: Mapping[str, Any] | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    progress: bool = True,
    tolerance_sems: float = DEFAULT_SEM_TOLERANCE,
) -> TaskBaselineResult:
    """Run the no-learning baseline for one task and write its logs.

    Args:
        task: anything satisfying :class:`hexa_core.adapter.Task`.
        client: anything with ``generate_structured(prompt=, max_new_tokens=,
            temperature=, top_p=, system_prompt=, seed=, reasoning_effort=)``
            returning an object with ``raw_text`` / ``reasoning_text`` /
            ``final_text`` / ``finish_reason`` / ``usage`` -- i.e. CORE's
            ``NvidiaGeneration`` shape.
        output_dir: gate root; this task writes under ``<root>/<task>/n<N>/``.
        task_seed: the seed ``make_task`` was called with, recorded in the manifest.
            Only Tower of Hanoi's split actually depends on it.
        limit: use only the first ``limit`` eval problems, in CORE order. For smoke
            tests -- ``limit`` is part of the results path, so smoke results can
            never be reused by the full gate.
        budget: defaults to one that caps eval at exactly ``runs * n_problems``, so
            a runaway loop raises :class:`~hexa_core.budget.BudgetExhausted` instead
            of burning GPU-days.

    Only calls actually issued are charged to the budget; results replayed from
    disk are counted separately as ``reused``.
    """
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    say = _Reporter(progress)
    task_name = getattr(task, "name", "unknown-task")

    problems = _prepare_problems(task, limit=limit, task_name=task_name)
    eval_ids_sha256 = hashlib.sha256(
        "\n".join(p.id for p in problems).encode("utf-8")
    ).hexdigest()

    cfg = GenerationConfig(
        model=model or getattr(client, "model", None) or "unknown-model",
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        llm_seed=llm_seed,
        reasoning_effort=reasoning_effort,
        system_prompt=system_prompt,
    )

    if budget is None:
        budget = RolloutBudget(max_train=0, max_eval=runs * len(problems))

    run_dir_root = Path(output_dir) / task_name / f"n{len(problems)}"
    run_dir_root.mkdir(parents=True, exist_ok=True)

    planned = [
        (run_idx, item_idx, problem)
        for run_idx in range(runs)
        for item_idx, problem in enumerate(problems)
    ]
    cached = sum(
        1
        for run_idx, item_idx, problem in planned
        if resume
        and _read_cached(
            _result_path(run_dir_root, run_idx, problem.id),
            expected_fingerprint=cfg.fingerprint(
                prompt=problem.prompt,
                seed=core_generation_seed(llm_seed, CORE_EVAL_SEED_OFFSET, run_idx, item_idx),
                eval_ids_sha256=eval_ids_sha256,
            ),
            say=say,
            warn=False,
        )
        is not None
    )

    say(
        f"\n=== {task_name}: {RUN_DESCRIPTION} ===\n"
        f"    problems={len(problems)}  runs={runs}  concurrency={concurrency}\n"
        f"    model={cfg.model}  temp={cfg.temperature}  top_p={cfg.top_p}  "
        f"max_tokens={cfg.max_tokens}  effort={cfg.reasoning_effort}\n"
        f"    results dir={run_dir_root}\n"
        f"    {cached}/{len(planned)} already on disk; "
        f"{len(planned) - cached} generation call(s) will be issued"
    )

    lock = threading.Lock()
    started = time.time()
    run_results: list[RunResult] = []

    for run_idx in range(runs):
        run_started = time.time()
        run_dir = run_dir_root / f"run{run_idx + 1}"
        run_dir.mkdir(parents=True, exist_ok=True)

        def solve(indexed: tuple[int, Problem], _run_idx: int = run_idx) -> ProblemResult:
            item_idx, problem = indexed
            return _solve_one(
                problem,
                client=client,
                run_idx=_run_idx,
                item_idx=item_idx,
                cfg=cfg,
                eval_ids_sha256=eval_ids_sha256,
                path=_result_path(run_dir_root, _run_idx, problem.id),
                budget=budget,
                lock=lock,
                resume=resume,
                max_attempts=max_attempts,
                say=say,
                n_total=len(problems),
                task_name=task_name,
            )

        indexed = list(enumerate(problems))
        # CORE walks fixed windows of `--num-eval-batch-calls` with a barrier per
        # window (run_core.py:1429-1470). Each call's seed is a pure function of
        # (run_idx, item_idx) and the calls are independent, so a rolling pool is
        # numerically identical and never idles on a slow straggler.
        if concurrency == 1 or len(indexed) == 1:
            results = [solve(pair) for pair in indexed]
        else:
            with ThreadPoolExecutor(max_workers=min(concurrency, len(indexed))) as pool:
                results = list(pool.map(solve, indexed))

        results.sort(key=lambda r: r.item_idx)  # back into CORE item order
        run = RunResult(run=run_idx + 1, results=results, wall_clock_s=time.time() - run_started)
        run_results.append(run)

        say(
            f"  [{task_name}] run {run.run}/{runs}: acc={run.accuracy:.3f} "
            f"({run.correct}/{run.total})  trunc={run.truncated}  "
            f"empty={run.empty_parses}  no-reasoning={run.no_reasoning}  "
            f"errors={run.errors}  reused={run.reused}  {run.wall_clock_s / 60:.1f}m",
            force=True,
        )
        if run.total and run.no_reasoning == run.total:
            say(
                f"  !! [{task_name}] run {run.run}: EVERY response had empty "
                "reasoning_content. The scored text is reasoning+final, so the "
                "chain of thought is missing from every graded string. Fix the "
                "server (--reasoning-parser openai_gptoss) before reading any "
                "number below.",
                force=True,
            )

    result = TaskBaselineResult(
        task_name=task_name,
        core_task_name=None,
        n_problems=len(problems),
        runs=run_results,
        eval_ids_sha256=eval_ids_sha256,
        config={
            **cfg.to_dict(),
            "runs": runs,
            "concurrency": concurrency,
            "limit": limit,
            "core_eval_seed_offset": CORE_EVAL_SEED_OFFSET,
            "guidelines": None,
            "max_attempts": max_attempts,
        },
        wall_clock_s=time.time() - started,
        budget_report=budget.report(),
    )

    try:
        result.task_manifest = task.manifest()
        result.core_task_name = result.task_manifest.get("core_task_name")
    except Exception as exc:  # a broken manifest must not lose the measurement
        result.task_manifest = {"error": repr(exc)}

    comparison = compare_to_paper(result, tolerance_sems=tolerance_sems)
    _write_json_atomic(
        run_dir_root / "summary.json", {**result.to_dict(), "gate": comparison.to_dict()}
    )
    _emit_logs(
        result,
        comparison,
        output_dir=run_dir_root,
        logger=logger,
        log_core_validation=log_core_validation,
        write_manifest=write_manifest,
        client=client,
        task_seed=task_seed,
        provider_info=provider_info,
        say=say,
    )
    return result


def _prepare_problems(task: Task, *, limit: int | None, task_name: str) -> list[Problem]:
    """Fetch the eval split and check every parity precondition before spending money."""
    problems = list(task.eval_problems())
    if not problems:
        raise ValueError(f"task={task_name!r} produced no eval problems")

    bad_split = [p.id for p in problems if p.split != "eval"]
    if bad_split:
        raise ValueError(
            f"task={task_name!r}: non-eval problems in the eval split: {bad_split[:5]}"
        )

    if limit is None:
        if len(problems) != EVAL_ITEMS:
            raise ValueError(
                f"task={task_name!r}: expected {EVAL_ITEMS} eval problems, got "
                f"{len(problems)}. Pass limit=... to run a deliberate subset."
            )
    else:
        if limit < 1 or limit > len(problems):
            raise ValueError(f"limit must be in 1..{len(problems)}, got {limit}")
        problems = problems[:limit]  # CORE order, so item_idx and seeds still line up

    for problem in problems:
        rebuilt = problem.build_prompt(None)
        if rebuilt != problem.prompt:
            offset = next(
                (i for i, (a, b) in enumerate(zip(rebuilt, problem.prompt)) if a != b),
                min(len(rebuilt), len(problem.prompt)),
            )
            raise AssertionError(
                f"task={task_name!r} problem={problem.id!r}: build_prompt(None) is not "
                f"byte-identical to the recorded baseline prompt (first difference at "
                f"offset {offset}). The no-learning baseline would not be CORE's."
            )
    return problems


def _result_path(root: Path, run_idx: int, problem_id: str) -> Path:
    return root / f"run{run_idx + 1}" / f"{_slug(problem_id)}.json"


def _read_cached(
    path: Path, *, expected_fingerprint: str, say: _Reporter, warn: bool = True
) -> ProblemResult | None:
    """Replay a previous result, or return ``None`` if it must be re-generated.

    Follows ``skillrl/runner/run_skill_agent.py::run_batch``'s skip-if-exists idiom
    (a corrupt file falls through and re-runs), plus two checks that idiom does not
    need and this one does: a result carrying an ``error`` is retried, and a result
    whose fingerprint no longer matches the generation config is discarded rather
    than silently poisoning the gate.
    """
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        cached = ProblemResult.from_json(payload)
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
        return None
    if cached.error:
        return None
    if cached.fingerprint != expected_fingerprint:
        if warn:
            say(
                f"  ~ {cached.problem_id}: cached result was produced under a "
                f"different generation config; re-generating ({path.name})",
                force=True,
            )
        return None
    return cached


def _solve_one(
    problem: Problem,
    *,
    client: Any,
    run_idx: int,
    item_idx: int,
    cfg: GenerationConfig,
    eval_ids_sha256: str,
    path: Path,
    budget: RolloutBudget,
    lock: threading.Lock,
    resume: bool,
    max_attempts: int,
    say: _Reporter,
    n_total: int,
    task_name: str,
) -> ProblemResult:
    seed = core_generation_seed(cfg.llm_seed, CORE_EVAL_SEED_OFFSET, run_idx, item_idx)
    fingerprint = cfg.fingerprint(
        prompt=problem.prompt, seed=seed, eval_ids_sha256=eval_ids_sha256
    )

    if resume:
        cached = _read_cached(path, expected_fingerprint=fingerprint, say=say)
        if cached is not None:
            say(
                f"  [{task_name} run {run_idx + 1}] {item_idx + 1}/{n_total} "
                f"{problem.id}: reused {'OK' if cached.is_correct else 'X'}"
            )
            return cached

    started = time.time()
    last_error: BaseException | None = None
    gen: Any = None
    attempts = 0

    # One rollout per problem, charged once, before the first attempt. Retries are
    # not extra rollouts -- CORE's counters advance per item, not per HTTP request,
    # and charging per attempt would make `max_eval = runs * n` trip on the first
    # 503. The client's own retry loop is invisible to us for the same reason.
    with lock:
        budget.charge_eval(1)

    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        try:
            gen = client.generate_structured(
                prompt=problem.prompt,
                max_new_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                system_prompt=cfg.system_prompt,
                seed=seed,
                reasoning_effort=cfg.reasoning_effort,
            )
            if gen is None:
                # `VLLMClient.generate_structured` returns None when its own retries
                # are exhausted (and distinguishes that from an empty answer, which
                # comes back as Generation(raw_text="")). Treat it as this attempt
                # failing, never as an empty answer.
                raise RuntimeError("client returned None: generation retries exhausted")
            last_error = None
            break
        except Exception as exc:  # transport/server errors: retry, then record
            last_error = exc
            gen = None
            if attempt < max_attempts:
                time.sleep(min(2.0**attempt, 30.0))

    if gen is None:
        say(
            f"  [{task_name} run {run_idx + 1}] {item_idx + 1}/{n_total} "
            f"{problem.id}: ERROR after {attempts} attempt(s): {last_error!r}",
            force=True,
        )
        result = ProblemResult(
            problem_id=problem.id,
            run=run_idx + 1,
            item_idx=item_idx,
            seed=seed,
            is_correct=False,
            verifier_score=0.0,
            prediction="",
            finish_reason=None,
            raw_chars=0,
            reasoning_chars=0,
            final_chars=0,
            elapsed_s=time.time() - started,
            usage=None,
            error=repr(last_error),
            attempts=attempts,
            fingerprint=fingerprint,
        )
        _write_json_atomic(path, result.to_json())  # keeps the failure auditable
        return result

    raw_text, reasoning_text, final_text = _unpack_generation(gen, problem_id=problem.id)

    # `parse_answer` then `verify_parsed` is exactly `problem.verify(raw_text)`,
    # split only so the prediction itself can be logged (CORE logs it too).
    # Correctness and the verifier score come from CORE's two separate functions,
    # mirroring `_score_task_prediction` (run_core.py:1782-1789) rather than
    # deriving one from the other -- see `Problem.score`.
    prediction = problem.parse_answer(raw_text)
    score = problem.verify_parsed(prediction)
    verifier_score = problem.score(prediction)

    result = ProblemResult(
        problem_id=problem.id,
        run=run_idx + 1,
        item_idx=item_idx,
        seed=seed,
        is_correct=bool(score),
        verifier_score=verifier_score,
        prediction=prediction,
        finish_reason=_as_text(getattr(gen, "finish_reason", None)),
        raw_chars=len(raw_text),
        reasoning_chars=len(reasoning_text),
        final_chars=len(final_text),
        elapsed_s=time.time() - started,
        usage=_as_usage(getattr(gen, "usage", None)),
        attempts=attempts,
        fingerprint=fingerprint,
    )
    _write_json_atomic(path, result.to_json())
    say(
        f"  [{task_name} run {run_idx + 1}] {item_idx + 1}/{n_total} {problem.id}: "
        f"{'OK' if result.is_correct else 'X '} "
        f"({result.elapsed_s:.1f}s, {result.reasoning_chars}+{result.final_chars} chars"
        f"{', TRUNCATED' if result.truncated else ''})"
    )
    return result


def _unpack_generation(gen: Any, *, problem_id: str) -> tuple[str, str, str]:
    """Pull out the three texts and enforce CORE's concatenation identity.

    ``NvidiaClient.generate_structured`` returns ``raw_text = reasoning_text +
    final_text`` with no separator (``src/model/nvidia_client.py:141-146``) and CORE
    scores ``raw_text``. Checking it here turns "the server forgot
    ``--reasoning-parser openai_gptoss``" into a first-call failure rather than a
    900-call run with an unexplained accuracy gap.
    """
    for attr in ("raw_text", "reasoning_text", "final_text"):
        if not hasattr(gen, attr):
            raise TypeError(
                f"generation for problem {problem_id!r} has no {attr!r}; the client must "
                "return CORE's NvidiaGeneration shape (raw_text / reasoning_text / "
                "final_text / finish_reason / usage)"
            )
    raw_text = gen.raw_text or ""
    reasoning_text = gen.reasoning_text or ""
    final_text = gen.final_text or ""
    if raw_text != reasoning_text + final_text:
        raise AssertionError(
            f"problem {problem_id!r}: raw_text is not reasoning_text + final_text "
            f"({len(raw_text)} vs {len(reasoning_text)}+{len(final_text)} chars). CORE "
            "scores the concatenation, so this run would not be comparable."
        )
    return raw_text, reasoning_text, final_text


def _as_text(value: Any) -> str | None:
    return None if value is None else str(value)


def _as_usage(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    for attr in ("model_dump", "to_dict", "dict"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                dumped = method()
            except Exception:
                continue
            if isinstance(dumped, Mapping):
                return dict(dumped)
    return {"repr": repr(value)}


def _emit_logs(
    result: TaskBaselineResult,
    comparison: GateComparison,
    *,
    output_dir: Path,
    logger: Any | None,
    log_core_validation: bool,
    write_manifest: bool,
    client: Any,
    task_seed: int,
    provider_info: Mapping[str, Any] | None,
    say: _Reporter,
) -> None:
    """CORE-shaped ``validation.jsonl`` plus a :class:`RunManifest`.

    Neither write is allowed to lose a measurement: ``summary.json`` is already on
    disk by the time this runs, so a logging failure warns and returns.
    """
    note = (
        f"{RUN_DESCRIPTION}; task={result.task_name}; n={result.n_problems}; "
        f"{PROVIDER_CAVEAT}"
    )

    if log_core_validation:
        validation_logger = logger if logger is not None else ValidationLogger(output_dir)
        try:
            validation_logger.log(
                episode=0,
                rollout=0,
                val_acc=result.mean_accuracy,
                # CORE's key carries CORE's statistic: population std over run
                # accuracies (run_core.py:1601), which is NOT the SEM. The SEM lives
                # in summary.json under a name that cannot be confused with it.
                val_std=result.population_std,
                val_mean_score=result.mean_verifier_score,
                val_score_std=result.verifier_score_std,
                val_runs=result.num_runs,
                detailed_results=[r.to_core_run_dict() for r in result.runs],
                val_run_scores=[r.mean_verifier_score for r in result.runs],
                memory_size=0,
                note=note,
            )
        except Exception as exc:
            say(f"  !! CORE validation log failed: {exc!r}", force=True)

    if not write_manifest:
        return

    task_manifest = result.task_manifest
    eval_ids = list(task_manifest.get("eval_ids") or ())
    notes = [RUN_DESCRIPTION, PROVIDER_CAVEAT]
    if eval_ids and len(eval_ids) != result.n_problems:
        # `task.eval_ids` describes the task; `metrics.n_problems` describes what was
        # measured. Under --smoke they differ, and the manifest must not read as if
        # the whole held-out set had been evaluated.
        notes.append(
            f"SUBSET RUN: only the first {result.n_problems} of {len(eval_ids)} eval "
            "problems were evaluated (--smoke). Not a gate result."
        )
    manifest = RunManifest(
        task_name=result.task_name,
        num_train_items=int(task_manifest.get("num_train_items") or 0),
        seed=task_seed,
        train_ids=list(task_manifest.get("train_ids") or ()),
        eval_ids=eval_ids,
        core_task_name=result.core_task_name,
        task_manifest=task_manifest,
        llm=client,
        budget=result.budget_report,
        out_dir=output_dir,
        notes=notes,
        extra={
            "phase": "phase2-repro-gate",
            "kind": "baseline_no_learning",
            "argv": list(sys.argv),
            "generation_config": result.config,
            "provider": dict(provider_info or {}),
            "metrics": result.to_dict(),
            "gate": comparison.to_dict(),
        },
    )
    try:
        manifest.write(output_dir / "manifest.json")
    except Exception as exc:
        say(f"  !! RunManifest.write failed: {exc!r}", force=True)


def run_baseline_evals(
    tasks: Sequence[Task], client: Any, **kwargs: Any
) -> list[TaskBaselineResult]:
    """Convenience sequential sweep. Kept trivial: the CLI owns error handling."""
    return [run_baseline_eval(task, client, **kwargs) for task in tasks]
