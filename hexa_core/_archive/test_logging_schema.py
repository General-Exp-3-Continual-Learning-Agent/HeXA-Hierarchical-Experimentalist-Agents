"""Log-schema checks: CORE record parity, manifest reconstructibility, mean +/- SEM.

Run with::

    PYTHONDONTWRITEBYTECODE=1 python -m hexa_core.tests.test_logging_schema

pytest is not installed, so this is a plain script: bare ``assert``, printed
progress, non-zero exit on the first failure. No network, no GPU.

The strongest check here compares our record builders against CORE's own
functions rather than against my reading of them. ``scripts/run_core.py``
``os.chdir(ROOT)`` at import time and pulls in ``sentence_transformers`` etc., so
:func:`_import_run_core` stubs the missing third-party packages and restores the
cwd. If that fails on a given machine the parity check is *skipped and said so* --
it never silently passes.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import math
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

from hexa_core import core_bridge, logging_schema as ls
from hexa_core.budget import RolloutBudget

# Packages run_core imports transitively that are not installed here. Stubbed so
# the pure helpers we care about become reachable.
_STUBBED_TOP_LEVEL = (
    "sentence_transformers",
    "torch",
    "faiss",
    "openai",
    "transformers",
    "vllm",
)


def _run_results() -> list[dict[str, Any]]:
    """Two validation passes covering every diagnostics branch."""
    details_a = [
        {"prediction": "x", "verifier_score": 1.0, "is_correct": True,
         "primary_finish_reason": "stop", "prediction_source": "primary"},
        # truncation at max_tokens -- the Hanoi n=6 failure mode
        {"prediction": "  ", "verifier_score": 0.0, "is_correct": False,
         "primary_finish_reason": "length", "prediction_source": "primary"},
        # unstructured model: finish reason absent -> "null" is a normal label
        {"prediction": "y", "verifier_score": 0.25, "is_correct": False,
         "primary_finish_reason": None, "prediction_source": "primary"},
    ]
    details_b = [
        {"prediction": "x", "verifier_score": 1.0, "is_correct": True,
         "primary_finish_reason": "stop", "prediction_source": "primary"},
        {"prediction": None, "verifier_score": 0.0, "is_correct": False,
         "primary_finish_reason": "", "prediction_source": None},
    ]
    return [ls.build_run_result(0, details_a), ls.build_run_result(1, details_b)]


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in _STUBBED_TOP_LEVEL and fullname not in sys.modules:
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []
        module.__getattr__ = lambda name: type(name, (), {})
        return module

    def exec_module(self, module) -> None:
        return None


def _import_run_core():
    """Import CORE's ``run_core`` for parity comparison, or return None."""
    finder = _StubFinder()
    cwd = os.getcwd()
    sys.meta_path.insert(0, finder)
    try:
        root = core_bridge.install_core_on_path()
        scripts = str(root / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        import run_core  # noqa: PLC0415

        return run_core
    except Exception as exc:  # noqa: BLE001 -- any import failure means "skip"
        print(f"[skip] run_core parity: {type(exc).__name__}: {exc}")
        return None
    finally:
        sys.meta_path.remove(finder)
        os.chdir(cwd)  # run_core.py chdirs to the CORE root at import time


def check_run_result_arithmetic() -> None:
    runs = _run_results()
    assert [r["run"] for r in runs] == [1, 2], "run index must be 1-based, as CORE's is"
    assert runs[0] == {
        "run": 1, "accuracy": 1 / 3, "mean_verifier_score": 1.25 / 3,
        "correct": 1, "total": 3, "details": runs[0]["details"],
    }
    empty = ls.build_run_result(0, [])
    assert empty["accuracy"] == 0.0 and empty["mean_verifier_score"] == 0.0
    assert ls.summarize_run_accuracies([]) == (0.0, 0.0, [])
    assert ls.summarize_mean_verifier_scores([]) == (0.0, 0.0, [])
    assert abs(ls.population_std([1 / 3, 2 / 3]) - 1 / 6) < 1e-12, "std must be ddof=0"
    print("[ok] run-level arithmetic and empty-input returns match CORE")


def check_record_keys(tmp: Path) -> dict[str, Any]:
    runs = _run_results()
    logger = ls.ValidationLogger(tmp / "run")
    record = logger.log(episode=0, detailed_results=runs, memory_size=7)

    assert list(record) == list(ls.VALIDATION_RECORD_KEYS), list(record)
    assert "rollout" in record and record["rollout"] is None, "rollout key must always exist"
    assert record["val_runs"] == 2
    assert record["individual_run_accuracies"] == [1 / 3, 0.5]
    assert record["val_acc_with_mem"] == ls.mean([1 / 3, 0.5])
    assert record["val_std_with_mem"] == ls.population_std([1 / 3, 0.5])

    summary = logger.summary_records()[0]
    assert list(summary) == list(ls.VALIDATION_SUMMARY_RECORD_KEYS), list(summary)
    assert summary["total_attempt_count"] == 5
    assert summary["primary_length_count"] == 1, "truncation diagnostic lost"
    assert summary["primary_finish_reason_counts"] == {
        "stop": 2, "length": 1, "null": 1, "empty": 1
    }
    assert summary["prediction_source_counts"] == {"primary": 4, "null": 1}
    assert summary["missing_usable_answer_count"] == 2
    assert abs(summary["missing_usable_answer_rate"] - 2 / 5) < 1e-12
    assert ls.summarize_validation_attempt_diagnostics([])["missing_usable_answer_rate"] is None

    noted = ls.ValidationLogger(tmp / "noted").log(
        episode=1, detailed_results=runs, memory_size=0, rollout=2000, note="calibration"
    )
    assert list(noted) == list(ls.VALIDATION_RECORD_KEYS) + ["note"]

    # JSON round-trip: the file must reproduce the returned record exactly, and
    # must never contain a bare NaN (invalid JSON, would break CORE's loader).
    assert logger.records()[0] == json.loads(json.dumps(record))
    assert "NaN" not in logger.validation_file.read_text()
    assert ls.read_jsonl(tmp / "absent.jsonl") == []
    assert ls.final_summary_record(logger.validation_summary_file) == summary

    try:
        logger.log(episode=2, detailed_results=[], memory_size=0)
    except ValueError:
        pass
    else:
        raise AssertionError("an empty validation pass must raise, not log nan")
    print("[ok] validation + summary records carry CORE's exact keys, in order")
    return record


def check_core_parity(record: dict[str, Any], tmp: Path) -> None:
    run_core = _import_run_core()
    if run_core is None:
        return
    runs = _run_results()
    assert run_core._run_mean_verifier_score(runs[0]) == ls.mean_verifier_score_of_run(runs[0])
    assert run_core._summarize_mean_verifier_scores(runs) == ls.summarize_mean_verifier_scores(runs)
    assert run_core._summarize_validation_attempt_diagnostics(runs) == \
        ls.summarize_validation_attempt_diagnostics(runs)
    for value in (None, "", "   ", "stop", 0, 5):
        assert run_core._validation_summary_label(value) == ls._summary_label(value)

    for note in (None, "pre-training"):
        rec = ls.build_validation_record(
            episode=3, rollout=None if note else 2000, val_acc=0.5, val_std=0.1,
            val_mean_score=0.4, val_score_std=0.05, val_runs=2, detailed_results=runs,
            val_run_scores=[0.41, 0.39], memory_size=12, note=note, timestamp=1.5,
        )
        theirs = run_core._build_validation_summary_record(dict(rec))
        mine = ls.build_validation_summary_record(rec)
        assert mine == theirs and list(mine) == list(theirs)

    # The written line, byte for byte.
    ours, core = tmp / "ours.jsonl", tmp / "core.jsonl"
    ls.append_jsonl(ours, record)
    run_core._append_jsonl_record(core, dict(record))
    assert ours.read_bytes() == core.read_bytes()
    print("[ok] parity EXACT against run_core's own summary/diagnostics/jsonl functions")


def check_manifest(tmp: Path) -> None:
    train_ids = [f"t{i}" for i in range(5)]
    eval_ids = [f"e{i}" for i in range(100)]

    class _Client:
        def manifest(self) -> dict[str, Any]:
            return {"model": "gpt-oss-120b", "reasoning_effort": "high",
                    "max_tokens": 32768, "temperature": 0.6}

    budget = RolloutBudget(max_train=2000, max_eval=100)
    budget.charge_baseline(50)
    manifest = ls.RunManifest(
        task_name="tower-of-hanoi", num_train_items=5, seed=42,
        train_ids=train_ids, eval_ids=eval_ids,
        core_task_name="tower-of-hanoi-state",
        task_manifest={"core_task_class": "TowerOfHanoiTask"},
        llm=_Client(), budget=budget,
        compute={"server_partition": "superpod-a100", "server_gpus": "4xA100-80GB"},
        out_dir=tmp / "run", notes=["phase-2 calibration"],
    )
    path = manifest.write()
    assert path.name == ls.MANIFEST_FILENAME
    assert path.name != "config.json", "must not collide with CORE's resume file"
    loaded = ls.RunManifest.load(tmp / "run")

    assert loaded["task"]["train_ids"] == train_ids, "full id lists are the point"
    assert loaded["task"]["eval_ids"] == eval_ids
    assert loaded["task"]["num_eval_items"] == 100
    assert loaded["task"]["core_task_name"] == "tower-of-hanoi-state"
    assert loaded["task"]["eval_ids_sha256"] == ls.ids_digest(eval_ids)
    assert loaded["llm"]["model"] == "gpt-oss-120b"
    assert loaded["budget"]["baseline"] == 50, "RolloutBudget.report() not captured"
    assert loaded["token_counter"]["name"] and "citable" in loaded["token_counter"]
    assert loaded["compute"]["server_partition"] == "superpod-a100"
    assert "slurm_partition" in loaded["compute"], "env snapshot missing"
    assert loaded["run_id"].startswith("tower-of-hanoi-n5-seed42-")
    assert loaded["schema_version"] == ls.MANIFEST_SCHEMA_VERSION

    if loaded["code"]["core_repo_root"] is not None:
        assert loaded["code"]["core_git_revision"], "CORE is present but SHA was not recorded"

    # A missing CORE checkout must degrade to null, never kill a run. $CORE_REPO
    # cannot simulate it (core_repo_root also probes the sibling directory), so
    # make the resolver itself raise.
    real = ls.core_bridge.core_repo_root

    def _raise() -> Path:
        raise core_bridge.CoreRepoNotFound("simulated missing CORE")

    ls.core_bridge.core_repo_root = _raise
    try:
        degraded = ls.RunManifest(task_name="matchstick", num_train_items=10, seed=1).to_dict()
    finally:
        ls.core_bridge.core_repo_root = real
    assert degraded["code"]["core_repo_root"] is None
    assert degraded["code"]["core_git_revision"] is None
    json.dumps(degraded)
    print("[ok] manifest is reconstructible and survives a missing CORE checkout")


def check_mean_sem() -> None:
    cell = ls.mean_sem([0.68, 0.67, 0.69])
    assert abs(cell.mean - 0.68) < 1e-12
    assert abs(cell.std - 0.01) < 1e-12, "default must be sample std (ddof=1)"
    assert abs(cell.sem - 0.01 / math.sqrt(3)) < 1e-12, "sem = std / sqrt(n_runs)"
    assert cell.n == 3 and cell.ddof == 1
    assert cell.format(3) == f"0.680 ± {0.01 / math.sqrt(3):.3f}"

    single = ls.mean_sem([0.5])
    assert single.std == 0.0 and single.sem == 0.0, "n=1 must not yield nan"
    empty = ls.mean_sem([])
    assert (empty.mean, empty.sem, empty.n) == (0.0, 0.0, 0)
    json.dumps(empty.as_dict())
    assert abs(ls.mean_sem([1 / 3, 2 / 3], ddof=0).std - 1 / 6) < 1e-12

    runs = [{"val_acc_with_mem": v} for v in (0.681, 0.674, 0.688)]
    over = ls.mean_sem_over_runs(runs)
    assert over.n == 3 and over.as_dict()["ddof"] == 1
    print(f"[ok] mean +/- SEM over runs: {over.format(3)} (n={over.n}, ddof={over.ddof})")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="hexa-logs-") as raw:
        tmp = Path(raw)
        check_run_result_arithmetic()
        record = check_record_keys(tmp)
        check_core_parity(record, tmp)
        check_manifest(tmp)
        check_mean_sem()
    print("\nall logging-schema checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
