"""The online build/eval loop for Episodic RAG and MemRL on a CORE task.

CORE's tasks are single-turn (`hexa_core/adapter.py`): one prompt, one scored
completion, no steps, no simulator calls. `baselines/memory/rag.py` and
`baselines/memory/memrl.py` are written generically enough that they never need
to change for this -- `add_episode` and its downstream `trajectory_fmt`
helpers are dict-first (see `trajectory_fmt.py`'s module docstring), reading
`simulator_calls` / `reward` / `verifier_feedback` / `instance_id` straight off
the result dict when present rather than deriving them from `trajectory`
steps. So every CORE attempt here is represented as a **one-step trajectory**
(`[{"action": "solve", "observation": raw_text}]`) with those four fields
stamped explicitly, which is enough for both baselines' write-back and for
`format_rag_block` / `format_memrl_block` to render correctly with no changes
to either.

Budget design (matched to HeXA's own accounting in `hexa_core/CORE_ENV_RESULTS.md`
§3/§6): HeXA's total per-task budget is 900 rollouts (600 training + 300 no-memory
`b_q` baseline calls, over 60 train items). Neither RAG nor MemRL has a `b_q`
analogue -- that quantity is HeXA-specific (M1 utility). The matched-compute
choice here is to give the *build* phase HeXA's full 900-rollout total, spent
entirely playing the same 60 train items round-robin: 900 / 60 = 15 visits per
item. That is `CoreStreamConfig.visits_per_problem`.

Round-robin, not a single item exhausted at a time: within one visit, every
item's retrieval reads the memory as it stood at the *start* of that visit (the
memory is frozen for the visit, exactly as HeXA's bank is frozen for a training
round), so all 60 generations in a visit are mutually independent and safe to
run concurrently; write-back happens after the whole visit's batch returns, in a
fixed item order, so the next visit's retrieval is deterministic regardless of
which thread finished first.

Eval is frozen (no write-back, matching CORE's own
`evaluate_memory_on(update_utilities=False, mark_use=False)`) and reseeded per
`hexa_core.seeds.core_generation_seed(42, CORE_EVAL_SEED_OFFSET, run_index,
item_index)` -- **the identical derivation `passive_loop._frozen_eval` uses**.
That is deliberate: it means a with-memory eval item here draws the *same*
generation seed as the paired no-memory item in an existing HeXA run
(`results_core/*/run*/eval_no_memory.jsonl`), so that file can be reused as this
arm's no-memory control without re-running it, exactly as HeXA's own with-memory
and no-memory arms are paired against each other.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from baselines.core.intent import retrieval_text
from baselines.memory.blocks import count_tokens, format_memrl_block, format_rag_block
from baselines.memory.embedder import Embedder
from baselines.memory.memrl import MemRL
from baselines.memory.rag import EpisodicRAG
from baselines.memory.store import MemoryStore
from hexa_core.adapter import Problem, Task
from hexa_core.seeds import (
    CORE_EVAL_SEED_OFFSET,
    CORE_TRAIN_SOLVE_SEED_OFFSET,
    core_generation_seed,
)

#: Matches HeXA's own default (`run_pilot.py --llm-seed` is not exposed; CORE's
#: own `--llm-seed` default is 42, transcribed in `hexa_core/train_baseline.py`).
LLM_SEED = 42

_FORMATTERS: dict[str, Callable[[Sequence[Any]], str]] = {
    "episodic_rag": format_rag_block,
    "memrl": format_memrl_block,
}


@dataclass
class CoreStreamConfig:
    task_name: str
    method: str  # "episodic_rag" | "memrl"
    visits_per_problem: int
    output_dir: Path
    concurrency: int = 16
    verbose: bool = False


def _block_for(method: str, records: Sequence[Any]) -> str:
    return _FORMATTERS[method](records)


def _generate_concurrent(
    generate: Callable[..., str | None],
    jobs: Sequence[tuple[str, int]],
    concurrency: int,
) -> list[str]:
    """jobs: (prompt, seed) pairs. Returns raw text, preserving input order."""
    if concurrency <= 1:
        return [generate(p, seed=s) or "" for p, s in jobs]

    def one(job: tuple[str, int]) -> str:
        try:
            return generate(job[0], seed=job[1]) or ""
        except Exception:
            return ""

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(one, jobs))


def _core_result(problem: Problem, raw: str, seed: int) -> dict[str, Any]:
    """One CORE attempt, shaped as the one-step trajectory dict-first callers expect.

    See the module docstring: this is what lets `EpisodicRAG.add_episode` /
    `MemRL.add_episode` -- and the `trajectory_fmt` helpers they call -- run
    completely unmodified against a single-turn task.
    """
    correct = bool(raw and problem.verify(raw))
    parsed = problem.parse_answer(raw) if raw else ""
    return {
        "success": correct,
        "instance_id": f"{problem.task_name}:{problem.id}",
        "trajectory": [{"action": "solve", "observation": raw}],
        "iterations": 1,
        "simulator_calls": 0,
        "reward": 1.0 if correct else -1.0,
        "verifier_feedback": (
            f"Verifier: {'CORRECT' if correct else 'INCORRECT'}\n"
            f"Parsed answer: {parsed}"
        ),
        "seed": seed,
    }


def _prefetch_memrl_summaries(
    memory,
    problems: Sequence[Problem],
    results: Sequence[dict[str, Any]],
    level: str,
    concurrency: int,
) -> Callable[[], None]:
    """Fire this visit's MemRL script/reflection LLM calls concurrently.

    A no-op for `EpisodicRAG` (no `_experience_text`) or when the run is
    smoke-tested with no summarizer configured -- both leave `add_episode`
    untouched, calling the same code path it always has. For MemRL with a
    real summarizer, returns a restore callback the caller must invoke
    (via `finally`) once the sequential write-back pass that consumes the
    cache has run.
    """
    if getattr(memory, "_experience_text", None) is None or memory.summarizer is None:
        return lambda: None

    def one(args: tuple[Problem, dict[str, Any]]) -> tuple[str, str]:
        problem, result = args
        return memory._experience_text(problem.problem_text, result, level, problem.id)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        precomputed = list(pool.map(one, zip(problems, results)))

    original = memory._experience_text
    queue = iter(precomputed)
    # `add_episode` calls `_experience_text(intent, result, level, seed)` with
    # the same four positional args every time; the replacement only needs to
    # accept and ignore them, since the answer for call N is already known.
    memory._experience_text = lambda *_args, **_kwargs: next(queue)

    def restore() -> None:
        memory._experience_text = original

    return restore


def build_memory(
    task: Task,
    memory,  # EpisodicRAG | MemRL
    store: MemoryStore,
    embedder: Embedder,
    generate: Callable[..., str | None],
    config: CoreStreamConfig,
) -> dict[str, Any]:
    """Play the train split `visits_per_problem` times each, online, in rounds.

    Returns a summary dict; per-rollout detail is written to
    ``<output_dir>/build_log.jsonl`` as it happens, so an interrupted build can
    at least be audited (this is NOT resumable -- a partial visit's retrieval
    state cannot be reconstructed from the log alone; re-run from scratch on
    interruption, as `MemoryStore` would otherwise silently mix visit orders).
    """
    problems = task.train_problems()
    embeddings = {
        p.id: embedder.embed_one(retrieval_text(config.task_name, p.problem_text))
        for p in problems
    }

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_log_path = out_dir / "build_log.jsonl"

    n_rollouts = 0
    n_correct = 0
    started = time.time()
    for visit in range(config.visits_per_problem):
        # ---- retrieval, frozen for the whole visit ----
        retrieved_by_id = {}
        jobs: list[tuple[str, int]] = []
        for item_index, problem in enumerate(problems):
            seed = core_generation_seed(
                LLM_SEED, CORE_TRAIN_SOLVE_SEED_OFFSET, visit, item_index
            )
            retrieved = memory.retrieve(
                embeddings[problem.id],
                level=config.task_name,
                # A problem excludes its OWN past memories from its own retrieval,
                # matching InterPhyre's `exclude_own_seed` -- so a later visit
                # learns from *other* items, not by recognising it solved this
                # exact one before.
                exclude_seeds=[problem.id],
            )
            retrieved_by_id[problem.id] = retrieved
            block = _block_for(config.method, retrieved.records)
            jobs.append((problem.build_prompt(block or None), seed))

        raws = _generate_concurrent(generate, jobs, config.concurrency)
        results = [_core_result(p, raws[i], jobs[i][1]) for i, p in enumerate(problems)]

        # MemRL.add_episode() calls its summarizer (one script/reflection LLM
        # call) unconditionally, every time, inside the write-back loop below --
        # which would otherwise fire 60 sequential single-in-flight requests
        # right after the 64-wide actor batch above, at 1/64th of server
        # capacity. Confirmed live on job 64619343: `Running: 1 req` for over
        # two minutes straight with a fresh request every ~10s during exactly
        # this phase. Fire them all concurrently first, then have
        # `_experience_text` replay the cache during the sequential pass
        # instead of recomputing it. `add_episode` calls `_experience_text`
        # exactly once per invocation (never conditionally), so a FIFO replay
        # in the same iteration order is exact, not an approximation.
        restore_experience_text = _prefetch_memrl_summaries(
            memory, problems, results, config.task_name, config.concurrency
        )

        # ---- write-back, sequential and in a fixed order ----
        try:
            for item_index, problem in enumerate(problems):
                result = results[item_index]
                seed = jobs[item_index][1]
                retrieved = retrieved_by_id[problem.id]
                n_rollouts += 1
                n_correct += int(result["success"])

                memory.update_utilities(retrieved, result)
                memory.add_episode(
                    result=result,
                    level=config.task_name,
                    seed=problem.id,
                    intent=problem.problem_text,
                    embedding=embeddings[problem.id],
                    step_index=n_rollouts,
                )
                with build_log_path.open("a") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "visit": visit,
                                "problem_id": problem.id,
                                "seed": seed,
                                "correct": result["success"],
                                "retrieved_memory_ids": [r.memory_id for r in retrieved.records],
                                "added_tokens": count_tokens(
                                    _block_for(config.method, retrieved.records)
                                ),
                            }
                        )
                        + "\n"
                    )
        finally:
            restore_experience_text()
        store.save()
        print(
            f"  [build {config.method}/{config.task_name}] visit "
            f"{visit + 1}/{config.visits_per_problem}: "
            f"{n_correct}/{n_rollouts} correct so far  "
            f"memory={len(store.records)}  ({time.time() - started:.0f}s elapsed)"
        )

    return {
        "n_rollouts": n_rollouts,
        "n_correct": n_correct,
        "build_accuracy": n_correct / n_rollouts if n_rollouts else 0.0,
        "memory_stats": store.stats(),
        "elapsed_s": time.time() - started,
    }


def frozen_eval(
    task: Task,
    memory,  # EpisodicRAG | MemRL
    embedder: Embedder,
    generate: Callable[..., str | None],
    config: CoreStreamConfig,
    *,
    run_index: int,
) -> dict[str, Any]:
    """One frozen-memory eval pass over the 100 held-out items.

    Seeded identically to `hexa_core.passive_loop._frozen_eval`'s with-memory
    arm -- see the module docstring for why that makes an existing HeXA
    no-memory eval file a valid paired control for this run, with no rerun.
    """
    problems = task.eval_problems()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, int]] = []
    retrieved_by_index = []
    for item_index, problem in enumerate(problems):
        seed = core_generation_seed(LLM_SEED, CORE_EVAL_SEED_OFFSET, run_index, item_index)
        retrieved = memory.retrieve(
            embedder.embed_one(retrieval_text(config.task_name, problem.problem_text)),
            level=config.task_name,
            exclude_seeds=(),  # eval items were never written to memory; nothing to exclude
        )
        retrieved_by_index.append(retrieved)
        block = _block_for(config.method, retrieved.records)
        jobs.append((problem.build_prompt(block or None), seed))

    raws = _generate_concurrent(generate, jobs, config.concurrency)

    records: list[dict[str, Any]] = []
    correct = 0
    n_empty = 0
    for item_index, problem in enumerate(problems):
        raw = raws[item_index]
        seed = jobs[item_index][1]
        retrieved = retrieved_by_index[item_index]
        if not raw:
            n_empty += 1
        is_correct = bool(raw and problem.verify(raw))
        correct += int(is_correct)
        block = _block_for(config.method, retrieved.records)
        records.append(
            {
                "problem_id": problem.id,
                "seed": seed,
                "run_index": run_index,
                "correct": is_correct,
                "empty": not raw,
                "added_tokens": count_tokens(block),
                "retrieved_memory_ids": [r.memory_id for r in retrieved.records],
                "parsed_answer": problem.parse_answer(raw) if raw else "",
            }
        )

    eval_path = out_dir / f"eval_run{run_index}.jsonl"
    eval_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    n = len(problems)
    added_tokens = [r["added_tokens"] for r in records if isinstance(r["added_tokens"], int)]
    return {
        "run_index": run_index,
        "accuracy": correct / n if n else 0.0,
        "n": n,
        "correct": correct,
        # A raw text of "" is scored incorrect the same as a genuine wrong
        # answer, but it can also mean every retry to the server failed (a
        # crashed/timed-out endpoint mid-eval) -- indistinguishable from "the
        # model got it wrong" without this count. Job 64581745's SLURM
        # walltime kill produced exactly this: an eval run that read 0.000,
        # not because the model failed 100/100 but because every request
        # errored. Callers must not average this run in without checking it.
        "n_empty": n_empty,
        "reliable": n_empty == 0,
        "mean_added_tokens": sum(added_tokens) / len(added_tokens) if added_tokens else None,
        "path": str(eval_path),
    }
