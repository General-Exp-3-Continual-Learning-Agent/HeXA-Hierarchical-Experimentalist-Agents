"""CORE-compatible run logs: ``validation.jsonl``, ``validation_summary.jsonl``,
plus a HExA-side ``manifest.json`` and the mean-±-SEM helper every reported cell
needs.

The validation records here are byte-for-byte schema-compatible with
``core-reasoning/scripts/run_core.py`` (``_log_validation_results``, line 1923),
so CORE's own plotting and analysis code can be pointed at our run directories
unchanged. Every aggregate is a line-for-line port of the corresponding CORE
helper -- including its empty-input behaviour -- because a summary column that
means something slightly different to theirs is worse than no column at all.

Two standard-deviation conventions coexist in this module *by design*; mixing
them up is exactly the error that would make our table silently disagree with
CORE's:

* ``val_std_with_mem`` / ``val_verifier_score_std`` inside a validation record
  are CORE's ``np.std`` -- **population** std (ddof=0) -- over the ``val_runs``
  validation passes of a single run. :func:`population_std` reproduces it.
* :func:`mean_sem` (the reporting helper) defaults to **sample** std (ddof=1)
  over independent runs/seeds, which is the convention behind the paper cells we
  are reproducing (``0.681 ± 0.007``). ``ddof`` is a parameter and is carried in
  :meth:`MeanSEM.as_dict`, so the choice is never invisible in a results file.

``manifest.json`` is deliberately *not* named ``config.json``: CORE's
``--load-from-past-run`` resume path owns that filename inside a run directory
and validates ``algorithm_version`` in it.

Leaf-level on purpose -- stdlib plus :mod:`hexa_core.core_bridge` and
:mod:`hexa_core.tokens` only. Tasks, budgets and LLM clients are consumed
duck-typed so that anything in the package can import this module.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from . import core_bridge, tokens

VALIDATION_FILENAME = "validation.jsonl"
VALIDATION_SUMMARY_FILENAME = "validation_summary.jsonl"
MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1

#: Keys of a ``validation.jsonl`` record, in CORE's insertion order
#: (``run_core.py:1937``). ``note`` is appended only when it is not None.
VALIDATION_RECORD_KEYS: tuple[str, ...] = (
    "episode",
    "rollout",
    "val_acc_with_mem",
    "val_std_with_mem",
    "val_mean_verifier_score",
    "val_verifier_score_std",
    "val_runs",
    "individual_run_accuracies",
    "individual_run_mean_verifier_scores",
    "detailed_results",
    "timestamp",
    "memory_size",
)

#: Keys of a ``validation_summary.jsonl`` record, in CORE's insertion order
#: (``run_core.py:1898``). It is a six-key subset of the validation record, plus
#: ``rollout``/``val_runs``/``note`` when present, plus the six diagnostics keys
#: merged in from :func:`summarize_validation_attempt_diagnostics`. It is *not*
#: "the validation record minus detailed_results".
VALIDATION_SUMMARY_RECORD_KEYS: tuple[str, ...] = (
    "episode",
    "val_acc_with_mem",
    "val_std_with_mem",
    "individual_run_accuracies",
    "timestamp",
    "memory_size",
    "rollout",
    "val_runs",
    "total_attempt_count",
    "primary_finish_reason_counts",
    "prediction_source_counts",
    "primary_length_count",
    "missing_usable_answer_count",
    "missing_usable_answer_rate",
)


# --------------------------------------------------------------------------- #
# JSONL I/O
# --------------------------------------------------------------------------- #


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one JSON object as a line, matching ``_append_jsonl_record``.

    Same serialisation as CORE (``run_core.py:1853``): ``ensure_ascii=False``, no
    ``sort_keys``, no ``indent``. Unlike CORE it creates the parent directory,
    because our drivers build the run dir lazily.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank lines. Missing file -> ``[]``.

    A truncated final line raises: a preempted job leaves a half-written record
    and silently dropping it would understate a run.
    """
    target = Path(path)
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(target) as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# --------------------------------------------------------------------------- #
# Aggregates (ports of CORE helpers; empty-input behaviour matches)
# --------------------------------------------------------------------------- #


def mean(values: Sequence[float]) -> float:
    return float(sum(float(v) for v in values) / len(values)) if values else 0.0


def population_std(values: Sequence[float]) -> float:
    """``np.std`` with ddof=0 -- what CORE writes into every ``*_std`` field."""
    if not values:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((float(v) - mu) ** 2 for v in values) / len(values))


def mean_verifier_score_of_run(run_result: Mapping[str, Any]) -> float:
    """Port of ``_run_mean_verifier_score`` (``run_core.py:1837``)."""
    details = run_result.get("details", []) or []
    if not details:
        return 0.0
    return float(
        sum(float(item.get("verifier_score", 0.0)) for item in details) / len(details)
    )


def summarize_run_accuracies(
    run_results: Sequence[Mapping[str, Any]],
) -> tuple[float, float, list[float]]:
    """``(mean, population std, per-run accuracies)`` -- ``run_core.py:1600``.

    Indexes ``result["accuracy"]`` strictly, like CORE, so a malformed run dict
    fails loudly instead of logging a silent zero.
    """
    accuracies = [float(result["accuracy"]) for result in run_results]
    if not accuracies:
        return 0.0, 0.0, []
    return mean(accuracies), population_std(accuracies), accuracies


def summarize_mean_verifier_scores(
    run_results: Sequence[Mapping[str, Any]],
) -> tuple[float, float, list[float]]:
    """Port of ``_summarize_mean_verifier_scores`` (``run_core.py:1846``)."""
    mean_scores = [mean_verifier_score_of_run(result) for result in run_results]
    if not mean_scores:
        return 0.0, 0.0, []
    return mean(mean_scores), population_std(mean_scores), mean_scores


def _summary_label(value: Any) -> str:
    """Port of ``_validation_summary_label`` (``run_core.py:1859``)."""
    if value is None:
        return "null"
    text = str(value).strip()
    return text if text else "empty"


def summarize_validation_attempt_diagnostics(
    run_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Port of ``_summarize_validation_attempt_diagnostics`` (``run_core.py:1866``).

    ``primary_length_count`` is the truncation diagnostic: attempts whose
    ``primary_finish_reason`` was ``"length"``, i.e. the 32768-token cap was hit.
    ``"null"`` is a normal finish-reason label -- CORE only populates
    ``primary_finish_reason`` for structured-generation models.
    """
    primary_finish_reason_counts: Counter[str] = Counter()
    prediction_source_counts: Counter[str] = Counter()
    total_attempt_count = 0
    missing_usable_answer_count = 0

    for run_result in run_results:
        details = run_result.get("details", []) or []
        for detail in details:
            total_attempt_count += 1
            primary_finish_reason_counts[
                _summary_label(detail.get("primary_finish_reason"))
            ] += 1
            prediction_source_counts[_summary_label(detail.get("prediction_source"))] += 1
            if not str(detail.get("prediction") or "").strip():
                missing_usable_answer_count += 1

    return {
        "total_attempt_count": total_attempt_count,
        "primary_finish_reason_counts": dict(primary_finish_reason_counts),
        "prediction_source_counts": dict(prediction_source_counts),
        "primary_length_count": int(primary_finish_reason_counts.get("length", 0)),
        "missing_usable_answer_count": missing_usable_answer_count,
        "missing_usable_answer_rate": (
            float(missing_usable_answer_count) / float(total_attempt_count)
            if total_attempt_count > 0
            else None
        ),
    }


def build_run_result(
    run_index: int, details: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """One CORE run-level dict from its attempt list (``run_core.py:1584``).

    ``run_index`` is 0-based; the emitted ``run`` field is 1-based, as CORE's is.
    """
    total = len(details)
    correct = sum(1 for detail in details if bool(detail.get("is_correct")))
    return {
        "run": run_index + 1,
        "accuracy": correct / total if total else 0.0,
        "mean_verifier_score": mean_verifier_score_of_run({"details": details}),
        "correct": correct,
        "total": total,
        "details": list(details),
    }


# --------------------------------------------------------------------------- #
# Record builders
# --------------------------------------------------------------------------- #


def build_validation_record(
    *,
    episode: int,
    rollout: int | None,
    val_acc: float,
    val_std: float,
    val_mean_score: float,
    val_score_std: float,
    val_runs: int,
    detailed_results: Sequence[Mapping[str, Any]],
    val_run_scores: Sequence[float],
    memory_size: int,
    note: str | None = None,
    timestamp: float | None = None,
) -> dict[str, Any]:
    """Build a ``validation.jsonl`` record with CORE's exact keys and order."""
    record: dict[str, Any] = {
        "episode": episode,
        "rollout": rollout,
        "val_acc_with_mem": val_acc,
        "val_std_with_mem": val_std,
        "val_mean_verifier_score": val_mean_score,
        "val_verifier_score_std": val_score_std,
        "val_runs": val_runs,
        "individual_run_accuracies": [result["accuracy"] for result in detailed_results],
        "individual_run_mean_verifier_scores": list(val_run_scores),
        "detailed_results": list(detailed_results),
        "timestamp": time.time() if timestamp is None else timestamp,
        "memory_size": memory_size,
    }
    if note is not None:
        record["note"] = note
    return record


def build_validation_summary_record(
    validation_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Port of ``_build_validation_summary_record`` (``run_core.py:1898``)."""
    summary_record: dict[str, Any] = {
        "episode": validation_record["episode"],
        "val_acc_with_mem": validation_record["val_acc_with_mem"],
        "val_std_with_mem": validation_record["val_std_with_mem"],
        "individual_run_accuracies": list(
            validation_record.get("individual_run_accuracies", [])
        ),
        "timestamp": validation_record["timestamp"],
        "memory_size": validation_record["memory_size"],
    }
    if "rollout" in validation_record:
        summary_record["rollout"] = validation_record["rollout"]
    if "val_runs" in validation_record:
        summary_record["val_runs"] = validation_record["val_runs"]
    if "note" in validation_record:
        summary_record["note"] = validation_record["note"]
    summary_record.update(
        summarize_validation_attempt_diagnostics(
            validation_record.get("detailed_results", [])
        )
    )
    return summary_record


class ValidationLogger:
    """Writes CORE's ``validation.jsonl`` + ``validation_summary.jsonl`` pair.

    One instance per run directory. Both files are append-only, so a resumed or
    preempted run extends them exactly as CORE's would.
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        validation_filename: str = VALIDATION_FILENAME,
        summary_filename: str = VALIDATION_SUMMARY_FILENAME,
        create_dir: bool = True,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.validation_file = self.out_dir / validation_filename
        self.validation_summary_file = self.out_dir / summary_filename
        if create_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        episode: int,
        detailed_results: Sequence[Mapping[str, Any]],
        memory_size: int,
        rollout: int | None = None,
        val_runs: int | None = None,
        note: str | None = None,
        val_acc: float | None = None,
        val_std: float | None = None,
        val_mean_score: float | None = None,
        val_score_std: float | None = None,
        val_run_scores: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """Derive CORE's aggregates from the run-level list and log both records.

        ``detailed_results`` is a list of run-level dicts (see
        :func:`build_run_result`), one per validation pass. Explicitly passed
        aggregates win over the derived ones, which is how a replay of an
        existing CORE record keeps that record's own numbers.

        Returns the validation record that was written, so a caller can stash it
        in a checkpoint without re-reading the file.
        """
        if not detailed_results and val_acc is None:
            # CORE would produce nan here via np.mean([]), and nan serialises as
            # bare NaN -- invalid JSON that poisons their plotting loader. An
            # empty validation pass is a bug, so refuse rather than log it.
            raise ValueError(
                "detailed_results is empty and no explicit val_acc was given: "
                "refusing to log a validation record with no runs"
            )

        derived_acc, derived_std, _ = summarize_run_accuracies(detailed_results)
        derived_score, derived_score_std, derived_run_scores = (
            summarize_mean_verifier_scores(detailed_results)
        )

        record = build_validation_record(
            episode=episode,
            rollout=rollout,
            val_acc=derived_acc if val_acc is None else val_acc,
            val_std=derived_std if val_std is None else val_std,
            val_mean_score=derived_score if val_mean_score is None else val_mean_score,
            val_score_std=(
                derived_score_std if val_score_std is None else val_score_std
            ),
            val_runs=len(detailed_results) if val_runs is None else val_runs,
            detailed_results=detailed_results,
            val_run_scores=(
                derived_run_scores if val_run_scores is None else val_run_scores
            ),
            memory_size=memory_size,
            note=note,
        )
        return self.log_record(record)

    def log_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Write a caller-built validation record plus its derived summary."""
        append_jsonl(self.validation_file, record)
        append_jsonl(
            self.validation_summary_file, build_validation_summary_record(record)
        )
        return dict(record)

    def records(self) -> list[dict[str, Any]]:
        return read_jsonl(self.validation_file)

    def summary_records(self) -> list[dict[str, Any]]:
        return read_jsonl(self.validation_summary_file)


# --------------------------------------------------------------------------- #
# Run manifest
# --------------------------------------------------------------------------- #

_HEXA_REPO_ROOT = Path(__file__).resolve().parent.parent


def ids_digest(ids: Iterable[str]) -> str:
    """sha256 over ``"\\n".join(ids)`` -- same convention as ``adapter.eval_ids_digest``.

    Recorded for both splits so "eval ids are byte-identical across the 5/10/100
    regimes" and "train and eval did not overlap" are one-line comparisons
    between two manifests, with no task rebuild.
    """
    joined = "\n".join(ids)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _git_revision(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _git_dirty(repo: Path) -> bool | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(out.stdout.strip()) if out.returncode == 0 else None


def _core_provenance() -> tuple[str | None, str | None, bool | None]:
    """``(core_repo_root, core_git_revision, core_git_dirty)``, degrading to None.

    Guarded because ``core_bridge.core_git_revision()`` resolves the repo first
    and raises :class:`~hexa_core.core_bridge.CoreRepoNotFound` when CORE is not
    checked out. Provenance must never be the reason a run dies.

    The dirty flag is not decoration: every number produced against CORE is only
    meaningful if CORE was unmodified, and a revision on its own cannot say so.
    ``None`` means "could not be determined", which is not the same as clean and
    must not be read as such.
    """
    try:
        root = core_bridge.core_repo_root()
    except (core_bridge.CoreRepoNotFound, OSError):
        return None, None, None
    try:
        return str(root), core_bridge.core_git_revision(), core_bridge.core_git_dirty()
    except (core_bridge.CoreRepoNotFound, OSError):
        return str(root), None, None


def env_compute_snapshot() -> dict[str, Any]:
    """What served *this process*, read from the environment.

    Deliberately no ``nvidia-smi`` and no torch import: the vLLM server usually
    runs on a different node from the driver, so the driver's local GPU list
    would misreport what actually served the model. The authoritative server-side
    GPU/partition is passed explicitly via ``RunManifest.compute`` (or carried
    inside the LLM client manifest); this snapshot documents the client side.
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_nodelist": os.environ.get("SLURM_NODELIST"),
        "slurm_nodename": os.environ.get("SLURMD_NODENAME"),
        "slurm_gpus": os.environ.get("SLURM_GPUS_ON_NODE") or os.environ.get("SLURM_GPUS"),
        "cuda_visible_devices": cuda_visible,
        "num_visible_gpus": (
            len([d for d in cuda_visible.split(",") if d.strip()])
            if cuda_visible
            else None
        ),
        "hostname": socket.gethostname(),
    }


def _as_dict(obj: Any) -> Any:
    """Normalise a client/budget object to something JSON-serialisable.

    Duck-typed on purpose: it is how the LLM client manifest and the
    :class:`~hexa_core.budget.RolloutBudget` report get into the manifest without
    this module importing either of them.
    """
    if obj is None:
        return None
    manifest = getattr(obj, "manifest", None)
    if callable(manifest):
        return manifest()
    report = getattr(obj, "report", None)
    if callable(report):
        return report()
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Mapping):
        return dict(obj)
    if isinstance(obj, (str, int, float, bool)):
        return obj
    return {"repr": repr(obj)}


@dataclass
class RunManifest:
    """Everything needed to reconstruct a run, written to ``manifest.json``.

    Mutable by design: ``extra`` and ``notes`` accrete while a run executes, and
    :meth:`write` can be called again to refresh the file (it overwrites).
    """

    task_name: str
    num_train_items: int
    seed: int
    train_ids: Sequence[str] = ()
    eval_ids: Sequence[str] = ()
    core_task_name: str | None = None
    task_manifest: Mapping[str, Any] | None = None
    llm: Any = None
    budget: Any = None
    compute: Mapping[str, Any] | None = None
    run_id: str | None = None
    out_dir: str | Path | None = None
    notes: Sequence[str] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def default_run_id(self, created_at: datetime.datetime) -> str:
        return (
            f"{self.task_name}-n{self.num_train_items}-seed{self.seed}"
            f"-{created_at.strftime('%Y%m%d_%H%M%S')}"
        )

    def to_dict(self) -> dict[str, Any]:
        created_at = datetime.datetime.now()
        train_ids = list(self.train_ids)
        eval_ids = list(self.eval_ids)
        core_root, core_rev, core_dirty = _core_provenance()
        counter = tokens.counter_name()

        compute: dict[str, Any] = env_compute_snapshot()
        if self.compute:
            compute.update(self.compute)

        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "created_at": created_at.strftime("%Y-%m-%dT%H:%M:%S"),
            "timestamp": created_at.timestamp(),
            "run_id": self.run_id or self.default_run_id(created_at),
            "out_dir": str(self.out_dir) if self.out_dir is not None else None,
            "task": {
                "name": self.task_name,
                # CORE's own task name differs from our registry key for Hanoi
                # ("tower-of-hanoi-state"); anything joining our logs to CORE's
                # has to translate, so both names are recorded.
                "core_task_name": self.core_task_name,
                "num_train_items": self.num_train_items,
                "seed": self.seed,
                "num_eval_items": len(eval_ids),
                "train_ids": train_ids,
                "eval_ids": eval_ids,
                "train_ids_sha256": ids_digest(train_ids),
                "eval_ids_sha256": ids_digest(eval_ids),
                "manifest": dict(self.task_manifest) if self.task_manifest else None,
            },
            "code": {
                "core_repo_root": core_root,
                "core_git_revision": core_rev,
                # True here invalidates the run: CORE is a read-only dependency and
                # the comparison assumes it was executed unmodified. `None` means
                # undetermined, not clean.
                "core_git_dirty": core_dirty,
                "hexa_repo_root": str(_HEXA_REPO_ROOT),
                "hexa_git_revision": _git_revision(_HEXA_REPO_ROOT),
                "hexa_git_dirty": _git_dirty(_HEXA_REPO_ROOT),
                "python_version": sys.version.split()[0],
                # There are three interpreters on this machine with different
                # third-party packages (system 3.12, phyreagent, vllm-gptoss), and
                # which one ran the driver decides whether tiktoken/openai were
                # available. The version string alone does not distinguish them.
                "python_executable": sys.executable,
                "platform": platform.platform(),
            },
            "llm": _as_dict(self.llm),
            "token_counter": {
                "name": counter,
                # Honest flag: the heuristic fallback is fine for enforcing an
                # injection budget but its added-token figure is not quotable
                # against CORE's reported 0.92k.
                "citable": tokens.is_exact(),
                "note": tokens.TOKENIZER_NOTE,
            },
            "budget": _as_dict(self.budget),
            "compute": compute,
            "notes": list(self.notes),
            "extra": dict(self.extra),
        }

    def write(self, path: str | Path | None = None) -> Path:
        """Write ``manifest.json`` (pretty-printed for human diffing)."""
        if path is None:
            if self.out_dir is None:
                raise ValueError("RunManifest.write needs either path or out_dir")
            target = Path(self.out_dir) / MANIFEST_FILENAME
        else:
            target = Path(path)
            if target.is_dir():
                target = target / MANIFEST_FILENAME
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> dict[str, Any]:
        """Read a written manifest back as a plain dict (not a RunManifest).

        Deliberately lossy in the other direction: a manifest records resolved
        provenance (git SHAs, hostnames) that must not be rehydrated into an
        object and then silently rewritten with today's values.
        """
        target = Path(path)
        if target.is_dir():
            target = target / MANIFEST_FILENAME
        return json.loads(target.read_text())


# --------------------------------------------------------------------------- #
# mean +/- SEM over independent runs (every reported cell needs this)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MeanSEM:
    """A reportable cell: ``mean ± sem`` with ``sem = std / sqrt(n)``."""

    mean: float
    sem: float
    std: float
    n: int
    values: tuple[float, ...]
    ddof: int

    def format(self, digits: int = 3) -> str:
        return f"{self.mean:.{digits}f} ± {self.sem:.{digits}f}"

    def __str__(self) -> str:
        return self.format()

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "sem": self.sem,
            "std": self.std,
            "n": self.n,
            "ddof": self.ddof,
            "values": list(self.values),
        }


def mean_sem(values: Iterable[float], *, ddof: int = 1) -> MeanSEM:
    """Mean and standard error over ``n`` independent runs.

    ``ddof=1`` (sample std) is the default because reported error bars over
    independent seeds are conventionally sample-based -- this is *not* the
    ddof=0 convention CORE uses inside a validation record; see the module
    docstring. ``n <= 1`` yields ``std = sem = 0.0`` rather than ``nan``, so a
    single-seed smoke run still serialises to valid JSON.
    """
    data = [float(v) for v in values]
    n = len(data)
    if n == 0:
        return MeanSEM(0.0, 0.0, 0.0, 0, (), ddof)
    mu = mean(data)
    if n - ddof <= 0:
        std = 0.0
    else:
        std = math.sqrt(sum((v - mu) ** 2 for v in data) / (n - ddof))
    return MeanSEM(mu, std / math.sqrt(n), std, n, tuple(data), ddof)


def format_mean_sem(values: Iterable[float], *, digits: int = 3, ddof: int = 1) -> str:
    return mean_sem(values, ddof=ddof).format(digits)


def mean_sem_over_runs(
    records: Iterable[Mapping[str, Any]],
    *,
    key: str = "val_acc_with_mem",
    ddof: int = 1,
) -> MeanSEM:
    """``mean ± SEM`` across runs, one record per run.

    Feed it the record you consider each run's result -- typically the last line
    of that run's ``validation_summary.jsonl``. ``n`` is the number of runs, so
    the SEM is over independent runs and never over the ``val_runs`` passes
    within one.
    """
    return mean_sem([record[key] for record in records], ddof=ddof)


def final_summary_record(path: str | Path) -> dict[str, Any] | None:
    """Last record of a ``validation_summary.jsonl`` (``None`` if empty/missing).

    The per-run input to :func:`mean_sem_over_runs`.
    """
    records = read_jsonl(path)
    return records[-1] if records else None
