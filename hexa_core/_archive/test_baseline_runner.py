"""End-to-end checks for the Phase 2 gate against a fake client. No GPU, no network.

Run with::

    PYTHONDONTWRITEBYTECODE=1 python -m hexa_core.tests.test_baseline_runner

The fake client returns CORE's reference answer for a chosen fraction of problems,
so accuracy is known in advance and the arithmetic can be checked rather than
eyeballed. It builds ``raw_text`` the way CORE does (``reasoning + final``), which
is what lets the concatenation guard be tested from both sides.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from hexa_core.adapter import Problem
from hexa_core.baseline_runner import (
    CORE_EVAL_SEED_OFFSET,
    PAPER_TARGETS,
    GenerationConfig,
    PaperTarget,
    ProblemResult,
    RunResult,
    TaskBaselineResult,
    compare_to_paper,
    core_generation_seed,
    run_baseline_eval,
)
from hexa_core.budget import BudgetExhausted, RolloutBudget
from hexa_core.tasks import make_task


class FakeGeneration:
    """CORE's ``NvidiaGeneration`` shape, built by CORE's own concatenation rule."""

    def __init__(
        self,
        reasoning_text: str,
        final_text: str,
        finish_reason: str | None = "stop",
        *,
        drop_reasoning_from_raw: bool = False,
    ) -> None:
        self.reasoning_text = reasoning_text
        self.final_text = final_text
        self.raw_text = final_text if drop_reasoning_from_raw else reasoning_text + final_text
        self.finish_reason = finish_reason
        self.usage = {"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33}


def reference_answer(problem: Problem) -> str:
    """Text that CORE's verifier scores 1 for, per task.

    Built from ``all_valid_outputs`` and, for Hanoi, from CORE's own formatter --
    the shapes differ per task (Hanoi's payload is a dict, not a list), so this is
    the one place the tests touch task specifics.
    """
    payload = problem.all_valid_outputs
    if problem.task_name == "matchstick":
        return "Correct version of the equation: " + payload[0]
    if problem.task_name == "zebralogic":
        return "So, this is my final answer: " + payload[0]
    return problem.build_prompt_fn.__self__.format_reference_answer_for_prompt(payload)


class FakeClient:
    """Deterministic stand-in for ``VLLMClient``. Counts calls; makes no I/O."""

    model = "fake/gpt-oss-120b"

    def __init__(
        self,
        problems: list[Problem],
        *,
        correct_every: int = 2,
        truncate_every: int | None = None,
        empty_every: int | None = None,
        drop_reasoning_from_raw: bool = False,
        fail_on: set[str] | None = None,
        return_none_on: set[str] | None = None,
    ) -> None:
        self.by_prompt = {p.prompt: p for p in problems}
        self.order = {p.prompt: i for i, p in enumerate(problems)}
        self.correct_every = correct_every
        self.truncate_every = truncate_every
        self.empty_every = empty_every
        self.drop_reasoning_from_raw = drop_reasoning_from_raw
        self.fail_on = fail_on or set()
        self.return_none_on = return_none_on or set()
        self.calls = 0
        self.seeds: list[int] = []

    def generate_structured(self, *, prompt: str, seed: int | None = None, **_: Any) -> Any:
        self.calls += 1
        self.seeds.append(seed if seed is not None else -1)
        problem = self.by_prompt[prompt]
        idx = self.order[prompt]
        if problem.id in self.fail_on:
            raise RuntimeError(f"synthetic transport failure for {problem.id}")
        if problem.id in self.return_none_on:
            return None
        drop = self.drop_reasoning_from_raw
        if self.empty_every and idx % self.empty_every == 0:
            return FakeGeneration(
                "thinking, but nothing came out", "", "stop", drop_reasoning_from_raw=drop
            )
        if self.truncate_every and idx % self.truncate_every == 0:
            return FakeGeneration(
                "thinking and never stopping", "", "length", drop_reasoning_from_raw=drop
            )
        body = (
            reference_answer(problem)
            if idx % self.correct_every == 0
            else "I have no idea what the answer is."
        )
        return FakeGeneration(
            f"<reasoning for {problem.id}>", body, drop_reasoning_from_raw=drop
        )

    def probe(self) -> dict[str, Any]:
        return {"fake": True, "reasoning_parser": True}


def check_seed_transcription() -> None:
    """The seed LCG must reproduce CORE's, which is why it was transcribed."""
    expected = 42
    for component in (CORE_EVAL_SEED_OFFSET, 0, 7):
        expected = (expected * 6364136223846793005 + component + 1442695040888963407) % (
            2**31 - 1
        )
    assert core_generation_seed(42, CORE_EVAL_SEED_OFFSET, 0, 7) == expected
    assert core_generation_seed(42, CORE_EVAL_SEED_OFFSET, 0, 0) != core_generation_seed(
        42, CORE_EVAL_SEED_OFFSET, 1, 0
    ), "run index does not change the seed"
    print("[ok] core_generation_seed matches CORE's LCG")


def check_arithmetic() -> None:
    """Accuracy / SEM / population std / binomial SE, on hand-computed inputs."""

    def run(run_idx: int, correct: int, total: int) -> RunResult:
        results = [
            ProblemResult(
                problem_id=f"p{i}",
                run=run_idx,
                item_idx=i,
                seed=i,
                is_correct=i < correct,
                verifier_score=float(i < correct),
                prediction="x" if i < correct else "",
                finish_reason="length" if i == 0 else "stop",
                raw_chars=10,
                reasoning_chars=0 if i == 1 else 5,
                final_chars=5,
                elapsed_s=0.1,
            )
            for i in range(total)
        ]
        return RunResult(run=run_idx, results=results, wall_clock_s=1.0)

    result = TaskBaselineResult(
        task_name="matchstick",
        core_task_name="matchstick",
        n_problems=10,
        runs=[run(1, 6, 10), run(2, 7, 10), run(3, 8, 10)],
        eval_ids_sha256="deadbeef",
        config={},
        wall_clock_s=3.0,
    )
    assert result.accuracies == [0.6, 0.7, 0.8]
    assert math.isclose(result.mean_accuracy, 0.7)
    # population std (CORE's val_std_with_mem) vs sample-std-based SEM: different
    # numbers, and conflating them is the easy mistake this asserts against.
    assert math.isclose(result.population_std, math.sqrt(0.02 / 3), rel_tol=1e-12)
    assert math.isclose(result.sem, 0.1 / math.sqrt(3), rel_tol=1e-12)
    assert not math.isclose(result.population_std, result.sem)
    assert math.isclose(result.binomial_se, math.sqrt(0.7 * 0.3 / 30), rel_tol=1e-12)
    # each of 3 runs has one finish_reason="length" and one zero-reasoning entry
    assert result.truncated == 3 and result.no_reasoning == 3
    assert result.empty_parses == (10 - 6) + (10 - 7) + (10 - 8)  # incorrect => ""
    assert result.attempts == 30

    single = TaskBaselineResult(
        task_name="matchstick",
        core_task_name="matchstick",
        n_problems=10,
        runs=[run(1, 7, 10)],
        eval_ids_sha256="deadbeef",
        config={},
        wall_clock_s=1.0,
    )
    assert math.isnan(single.sem), "a single run cannot have an across-run SEM"
    assert compare_to_paper(single).sem_basis.startswith("binomial")
    print("[ok] accuracy / SEM / population std / binomial SE arithmetic")


def check_verdicts() -> None:
    """PASS* / MISS* / ERROR, and that ERROR wins over the arithmetic."""

    def synthetic(accuracy: float, *, runs: int = 3, errors: int = 0) -> TaskBaselineResult:
        run_list = []
        for r in range(1, runs + 1):
            n = 100
            correct = round(accuracy * n)
            results = [
                ProblemResult(
                    problem_id=f"p{i}",
                    run=r,
                    item_idx=i,
                    seed=i,
                    is_correct=i < correct,
                    verifier_score=float(i < correct),
                    prediction="x",
                    finish_reason="stop",
                    raw_chars=1,
                    reasoning_chars=1,
                    final_chars=1,
                    elapsed_s=0.0,
                    error="boom" if (r == 1 and i < errors) else None,
                )
                for i in range(n)
            ]
            run_list.append(RunResult(run=r, results=results, wall_clock_s=0.0))
        return TaskBaselineResult(
            task_name="matchstick",
            core_task_name="matchstick",
            n_problems=100,
            runs=run_list,
            eval_ids_sha256="x",
            config={},
            wall_clock_s=0.0,
        )

    target = PAPER_TARGETS["matchstick"]
    on_target = compare_to_paper(synthetic(target.accuracy))
    assert on_target.verdict == "PASS*", on_target
    assert abs(on_target.delta) < 0.01
    assert "*" in on_target.verdict, "a bare PASS would imply provider comparability"

    way_off = compare_to_paper(synthetic(0.30))
    assert way_off.verdict == "MISS*", way_off
    assert way_off.delta < 0 and way_off.delta_in_sems < -2

    errored = compare_to_paper(synthetic(target.accuracy, errors=3))
    assert errored.verdict == "ERROR", "failed API calls must void the verdict"

    unknown = TaskBaselineResult(
        task_name="not-a-paper-task",
        core_task_name=None,
        n_problems=1,
        runs=[
            RunResult(
                run=1,
                results=[
                    ProblemResult(
                        problem_id="p",
                        run=1,
                        item_idx=0,
                        seed=1,
                        is_correct=True,
                        verifier_score=1.0,
                        prediction="x",
                        finish_reason="stop",
                        raw_chars=1,
                        reasoning_chars=1,
                        final_chars=1,
                        elapsed_s=0.0,
                    )
                ],
                wall_clock_s=0.0,
            )
        ],
        eval_ids_sha256="x",
        config={},
        wall_clock_s=0.0,
    )
    assert compare_to_paper(unknown).verdict == "NO-TARGET"
    assert PaperTarget(0.5, 0.01).provider == "NVIDIA NIM"
    print("[ok] gate verdicts (PASS* / MISS* / ERROR / NO-TARGET)")


def check_core_detail_shape() -> None:
    """CORE's own summarizer must work on our details, unmodified."""
    from hexa_core.logging_schema import summarize_validation_attempt_diagnostics

    results = [
        ProblemResult(
            problem_id="a",
            run=1,
            item_idx=0,
            seed=1,
            is_correct=False,
            verifier_score=0.0,
            prediction="",
            finish_reason="length",
            raw_chars=9,
            reasoning_chars=9,
            final_chars=0,
            elapsed_s=0.0,
        ),
        ProblemResult(
            problem_id="b",
            run=1,
            item_idx=1,
            seed=2,
            is_correct=True,
            verifier_score=1.0,
            prediction="ok",
            finish_reason="stop",
            raw_chars=4,
            reasoning_chars=2,
            final_chars=2,
            elapsed_s=0.0,
        ),
    ]
    run = RunResult(run=1, results=results, wall_clock_s=0.0)
    summary = summarize_validation_attempt_diagnostics([run.to_core_run_dict()])
    assert summary["total_attempt_count"] == 2
    assert summary["primary_length_count"] == run.truncated == 1
    assert summary["missing_usable_answer_count"] == run.empty_parses == 1
    assert summary["prediction_source_counts"] == {"primary": 2}
    print("[ok] our details feed CORE's diagnostics summarizer unchanged")


def check_concatenation_guard(problems: list[Problem], tmp: Path) -> None:
    """A client that drops the chain of thought from raw_text must not be tolerated."""
    task = _StubTask("matchstick", problems)
    client = FakeClient(problems, drop_reasoning_from_raw=True)
    try:
        run_baseline_eval(
            task,
            client,
            output_dir=tmp / "guard",
            runs=1,
            limit=len(problems),
            concurrency=1,
            progress=False,
            write_manifest=False,
            log_core_validation=False,
        )
    except AssertionError as exc:
        assert "reasoning_text + final_text" in str(exc), exc
        print("[ok] raw_text != reasoning+final is caught on the first call")
        return
    raise AssertionError("a client dropping reasoning_content from raw_text was accepted")


def check_prompt_identity_guard(problems: list[Problem], tmp: Path) -> None:
    """A problem whose recorded prompt is not build_prompt(None) must abort the run."""
    tampered = list(problems)
    tampered[1] = Problem(
        id=problems[1].id,
        prompt=problems[1].prompt + "\n\nPS: think harder.",
        problem_text=problems[1].problem_text,
        format_instructions=problems[1].format_instructions,
        task_name=problems[1].task_name,
        split="eval",
        metadata={},
        tools=None,
        build_prompt_fn=problems[1].build_prompt_fn,
        parse_answer_fn=problems[1].parse_answer_fn,
        evaluate_fn=problems[1].evaluate_fn,
        all_valid_outputs=problems[1].all_valid_outputs,
    )
    client = FakeClient(tampered)
    try:
        run_baseline_eval(
            _StubTask("matchstick", tampered),
            client,
            output_dir=tmp / "prompt-guard",
            runs=1,
            limit=len(tampered),
            progress=False,
            write_manifest=False,
            log_core_validation=False,
        )
    except AssertionError as exc:
        assert "byte-identical" in str(exc), exc
        assert client.calls == 0, "the guard must run before any API call"
        print("[ok] a non-baseline prompt aborts before spending anything")
        return
    raise AssertionError("a tampered prompt was accepted as CORE's baseline")


class _StubTask:
    """The `Task` protocol over a fixed problem list, so tests control `n`."""

    def __init__(self, name: str, problems: list[Problem]) -> None:
        self.name = name
        self._problems = problems

    def train_problems(self, n: int | None = None, seed: int | None = None) -> list[Problem]:
        return []

    def eval_problems(self) -> list[Problem]:
        return list(self._problems)

    def manifest(self) -> dict[str, Any]:
        return {
            "task_name": self.name,
            "core_task_name": self.name,
            "num_train_items": 5,
            "train_ids": [],
            "eval_ids": [p.id for p in self._problems],
        }


def check_end_to_end(task_name: str, problems: list[Problem], tmp: Path) -> None:
    """Counts, budget, resume, and the on-disk artefacts, on real CORE prompts."""
    out = tmp / "e2e" / task_name
    task = _StubTask(task_name, problems)
    n = len(problems)
    runs = 2

    client = FakeClient(problems, correct_every=2, truncate_every=5, empty_every=7)
    budget = RolloutBudget(max_train=0, max_eval=runs * n)
    result = run_baseline_eval(
        task,
        client,
        output_dir=out,
        runs=runs,
        limit=n,
        concurrency=4,
        progress=False,
        budget=budget,
        task_seed=42,
    )

    assert client.calls == runs * n, f"{client.calls} calls for {runs * n} problems"
    assert budget.eval == runs * n, budget.report()
    assert budget.training_rollouts == 0, "an eval must never charge training rollouts"
    assert result.num_runs == runs and result.n_problems == n
    assert [r.total for r in result.runs] == [n] * runs
    assert result.errors == 0

    # seeds are CORE's, and every (run, item) pair gets a distinct one
    expected_seeds = {
        core_generation_seed(42, CORE_EVAL_SEED_OFFSET, r, i)
        for r in range(runs)
        for i in range(n)
    }
    assert set(client.seeds) == expected_seeds, "seeds are not CORE's per-(run,item) seeds"

    # The fake's failure modes land in the columns the gate table prints. The
    # empty-response branch is checked first in FakeClient, so it wins on i % 35 == 0.
    truncating = [i for i in range(n) if i % 5 == 0 and i % 7 != 0]
    emptying = [i for i in range(n) if i % 7 == 0]
    assert result.truncated == runs * len(truncating), result.truncated
    assert result.empty_parses >= runs * len(emptying), result.empty_parses
    assert [r.item_idx for r in result.runs[0].results] == list(range(n)), "not in CORE order"

    summary = json.loads((out / task_name / f"n{n}" / "summary.json").read_text())
    assert summary["task_name"] == task_name
    assert summary["gate"]["verdict"] in ("PASS*", "MISS*", "ERROR")
    assert "CALIBRATION" in summary["provider_caveat"]
    assert summary["val_std_with_mem"] == result.population_std
    manifest = json.loads((out / task_name / f"n{n}" / "manifest.json").read_text())
    assert manifest["task"]["name"] == task_name
    assert manifest["extra"]["kind"] == "baseline_no_learning"

    validation = [
        json.loads(line)
        for line in (out / task_name / f"n{n}" / "validation.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(validation) == 1
    record = validation[0]
    assert record["val_runs"] == runs
    assert record["individual_run_accuracies"] == result.accuracies
    assert record["memory_size"] == 0
    assert "CALIBRATION" in record["note"]
    assert len(record["detailed_results"]) == runs

    # -- resume: a second identical invocation must issue zero calls -----------
    client2 = FakeClient(problems, correct_every=2, truncate_every=5, empty_every=7)
    budget2 = RolloutBudget(max_train=0, max_eval=runs * n)
    again = run_baseline_eval(
        task,
        client2,
        output_dir=out,
        runs=runs,
        limit=n,
        concurrency=4,
        progress=False,
        budget=budget2,
    )
    assert client2.calls == 0, f"resume still issued {client2.calls} calls"
    assert budget2.eval == 0, "a replayed result is not a rollout"
    assert again.reused == runs * n
    assert again.accuracies == result.accuracies, "resume changed the measurement"

    # -- a config change must invalidate the cache ------------------------------
    client3 = FakeClient(problems, correct_every=2, truncate_every=5, empty_every=7)
    run_baseline_eval(
        task,
        client3,
        output_dir=out,
        runs=1,
        limit=n,
        concurrency=1,
        progress=False,
        temperature=0.9,
        write_manifest=False,
        log_core_validation=False,
    )
    assert client3.calls == n, "a temperature change reused stale results"
    print(f"[ok] end-to-end on {task_name}: {runs}x{n} calls, resume, cache invalidation")


def check_errors_and_none(problems: list[Problem], tmp: Path) -> None:
    """A transport failure and a ``None`` return both land as recorded errors."""
    task = _StubTask("matchstick", problems)
    bad = {problems[0].id}
    none = {problems[1].id}
    client = FakeClient(problems, fail_on=bad, return_none_on=none)
    result = run_baseline_eval(
        task,
        client,
        output_dir=tmp / "errors",
        runs=1,
        limit=len(problems),
        concurrency=2,
        progress=False,
        max_attempts=2,
        write_manifest=False,
        log_core_validation=False,
    )
    assert result.errors == 2, result.errors
    assert compare_to_paper(result).verdict == "ERROR"
    failed = [r for r in result.runs[0].results if r.error]
    assert {r.problem_id for r in failed} == bad | none
    assert any("None" in (r.error or "") for r in failed), "None return was not labelled"

    # an errored result must be retried, not replayed
    retry = FakeClient(problems)
    result2 = run_baseline_eval(
        task,
        retry,
        output_dir=tmp / "errors",
        runs=1,
        limit=len(problems),
        concurrency=2,
        progress=False,
        write_manifest=False,
        log_core_validation=False,
    )
    assert retry.calls == 2, f"cached errors were not retried ({retry.calls} calls)"
    assert result2.errors == 0
    print("[ok] transport failures and None returns are recorded, then retried on resume")


def check_budget_guard(problems: list[Problem], tmp: Path) -> None:
    """An eval cap smaller than the workload must raise, not overspend."""
    task = _StubTask("matchstick", problems)
    client = FakeClient(problems)
    try:
        run_baseline_eval(
            task,
            client,
            output_dir=tmp / "budget",
            runs=1,
            limit=len(problems),
            concurrency=1,
            progress=False,
            budget=RolloutBudget(max_train=0, max_eval=2),
            write_manifest=False,
            log_core_validation=False,
        )
    except BudgetExhausted:
        assert client.calls == 2, f"spent {client.calls} calls under a cap of 2"
        print("[ok] the eval cap stops the run instead of burning GPU-days")
        return
    raise AssertionError("the eval budget cap did not fire")


def check_fingerprint() -> None:
    cfg = GenerationConfig(model="m")
    base = cfg.fingerprint(prompt="p", seed=1, eval_ids_sha256="e")
    assert base == cfg.fingerprint(prompt="p", seed=1, eval_ids_sha256="e")
    assert base != cfg.fingerprint(prompt="p!", seed=1, eval_ids_sha256="e")
    assert base != cfg.fingerprint(prompt="p", seed=2, eval_ids_sha256="e")
    assert base != cfg.fingerprint(prompt="p", seed=1, eval_ids_sha256="e2")
    for changed in (
        GenerationConfig(model="other"),
        GenerationConfig(model="m", temperature=0.7),
        GenerationConfig(model="m", max_tokens=1024),
        GenerationConfig(model="m", reasoning_effort="low"),
    ):
        assert base != changed.fingerprint(prompt="p", seed=1, eval_ids_sha256="e")
    print("[ok] the resume fingerprint covers every generation input")


def check_cli_rendering() -> None:
    """The CLI's table and checklist must render, and must never print a bare PASS."""
    import contextlib
    import importlib.util
    import io

    path = Path(__file__).resolve().parents[2] / "scripts" / "run_repro_gate.py"
    spec = importlib.util.spec_from_file_location("_repro_gate_cli", path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    def synthetic(task_name: str, accuracy: float, *, truncated: int = 0) -> TaskBaselineResult:
        run_list = []
        for r in range(1, 4):
            correct = round(accuracy * 100) + (r - 2)
            results = [
                ProblemResult(
                    problem_id=f"p{i}",
                    run=r,
                    item_idx=i,
                    seed=i,
                    is_correct=i < correct,
                    verifier_score=float(i < correct),
                    prediction="x" if i < correct else "",
                    finish_reason="length" if i < truncated else "stop",
                    raw_chars=1,
                    reasoning_chars=1,
                    final_chars=1,
                    elapsed_s=0.5,
                )
                for i in range(100)
            ]
            run_list.append(RunResult(run=r, results=results, wall_clock_s=1200.0))
        return TaskBaselineResult(
            task_name=task_name,
            core_task_name=task_name,
            n_problems=100,
            runs=run_list,
            eval_ids_sha256="x",
            config={
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 32768,
                "llm_seed": 42,
                "reasoning_effort": "high",
                "core_eval_seed_offset": CORE_EVAL_SEED_OFFSET,
            },
            wall_clock_s=3600.0,
        )

    results = [
        synthetic("matchstick", PAPER_TARGETS["matchstick"].accuracy),
        synthetic("zebralogic", 0.40, truncated=4),
        synthetic("tower-of-hanoi", PAPER_TARGETS["tower-of-hanoi"].accuracy),
    ]
    comparisons = [compare_to_paper(r) for r in results]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        cli.print_header(
            base_url="http://node:8000/v1",
            model="openai/gpt-oss-120b",
            cfg=results[0].config,
            runs=3,
            n=None,
        )
        cli.print_gate_table(results, comparisons, 2.0)
        cli.print_diagnostics(results, comparisons, {"reasoning_effort_mechanism": "top_level"})
    text = buffer.getvalue()

    assert "CALIBRATION" in text and "NOT a reproduction" in text
    assert "PASS*" in text and "MISS*" in text
    for line in text.splitlines():
        # Nothing may print a bare verdict: the asterisk is what carries "different
        # provider, so this is not a reproduction claim".
        stripped = line.replace("PASS*", "")
        assert "PASS" not in stripped, f"bare PASS leaked into the output: {line!r}"
        assert len(line) < 200, f"unwrapped line in the report: {line[:80]!r}..."
    assert "--reasoning-parser openai_gptoss" in text, "checklist lost the top suspect"
    assert "finish_reason == 'length'" in text
    assert cli.EXIT_OK == 0 and cli.EXIT_MISS == 1 and cli.EXIT_SETUP == 2
    print("[ok] CLI table + diagnostics render, and never print a bare PASS")


def main() -> int:
    check_seed_transcription()
    check_fingerprint()
    check_arithmetic()
    check_verdicts()
    check_core_detail_shape()
    check_cli_rendering()

    tmp = Path(tempfile.mkdtemp(prefix="hexa-baseline-"))
    try:
        # Real CORE tasks, so the prompt-identity guard and the verifiers are the
        # real ones; only the model is fake. 6 problems keeps it a few seconds.
        for task_name in ("matchstick", "zebralogic", "tower-of-hanoi"):
            problems = make_task(task_name, 5).eval_problems()[:6]
            check_end_to_end(task_name, problems, tmp)

        problems = make_task("matchstick", 5).eval_problems()[:4]
        check_concatenation_guard(problems, tmp)
        check_prompt_identity_guard(problems, tmp)
        check_errors_and_none(problems, tmp)
        check_budget_guard(problems, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nall baseline-runner checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
