"""Eval driver for CORE Reflexion: K trials per item, concurrent across items.

No build phase (see `baselines/core/reflexion.py`'s module docstring) -- this
runs directly against the 100-item eval split, `--eval-runs` times (n=3,
matching Episodic RAG/MemRL's eval replicate count), each a fully independent
draw since a trial's reflection depends on that draw's own earlier trials.

Each item's K trials are inherently sequential (trial 2 needs trial 1's
outcome), but items are independent of each other, so concurrency is applied
*across* items: one worker thread runs one item's whole trial sequence
start-to-finish.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from baselines.core.reflexion import reflect_core, reflexion_block
from hexa_core.adapter import Problem, Task
from hexa_core.seeds import CORE_EVAL_SEED_OFFSET, core_generation_seed

#: Matches `baselines/core/runner.py`'s LLM_SEED (CORE's own default).
LLM_SEED = 42


def _run_one_item(
    problem: Problem,
    generate: Callable[..., str | None],
    generate_structured: Callable[..., Any],
    *,
    k_trials: int,
    run_index: int,
    item_index: int,
) -> dict[str, Any]:
    lessons: list[str] = []
    trials: list[dict[str, Any]] = []
    solved_at: int | None = None

    for trial in range(1, k_trials + 1):
        # Extends passive_loop's eval seed derivation with a trial component --
        # CORE has no native multi-trial concept, so this is our own namespace,
        # not a CORE-internal one. Still within CORE_EVAL_SEED_OFFSET, still
        # deterministic per (run_index, item, trial).
        seed = core_generation_seed(
            LLM_SEED, CORE_EVAL_SEED_OFFSET, run_index, item_index, trial
        )
        block = reflexion_block(lessons)
        raw = generate(problem.build_prompt(block), seed=seed) or ""
        correct = bool(raw and problem.verify(raw))
        parsed = problem.parse_answer(raw) if raw else ""
        trial_record: dict[str, Any] = {
            "trial": trial,
            "seed": seed,
            "correct": correct,
            "empty": not raw,
        }

        if correct:
            solved_at = trial
            trials.append(trial_record)
            break

        if trial < k_trials:
            reflection = reflect_core(
                generate_structured, problem.problem_text, raw, parsed, lessons
            )
            trial_record["reflection_empty"] = not reflection
            if reflection:
                lessons.append(reflection)
        trials.append(trial_record)

    return {
        "problem_id": problem.id,
        "run_index": run_index,
        "k_trials": k_trials,
        "solved_at_trial": solved_at,
        "trials": trials,
        "n_reflections": len(lessons),
        "n_empty_reflections": sum(1 for t in trials if t.get("reflection_empty")),
    }


def run_reflexion_eval(
    task: Task,
    generate: Callable[..., str | None],
    generate_structured: Callable[..., Any],
    *,
    k_trials: int,
    run_index: int,
    concurrency: int,
    output_dir: Path,
) -> dict[str, Any]:
    """One full Reflexion eval pass (K trials/item) over the 100 held-out items."""
    problems = task.eval_problems()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def one(args: tuple[int, Problem]) -> dict[str, Any]:
        item_index, problem = args
        return _run_one_item(
            problem, generate, generate_structured,
            k_trials=k_trials, run_index=run_index, item_index=item_index,
        )

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        records = list(pool.map(one, enumerate(problems)))

    path = output_dir / f"reflexion_run{run_index}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    n = len(records)

    def cumulative_acc(k: int) -> float:
        # Cumulative, not per-trial: a trial stops on success, so an item
        # solved at trial 1 was never re-attempted and counts toward every
        # later k too. See the module-level docstring in reflexion.py.
        if not n:
            return 0.0
        return sum(
            1 for r in records
            if r["solved_at_trial"] is not None and r["solved_at_trial"] <= k
        ) / n

    trial1_acc = sum(1 for r in records if r["trials"][0]["correct"]) / n if n else 0.0
    any_acc = sum(1 for r in records if r["solved_at_trial"] is not None) / n if n else 0.0
    n_empty_actor = sum(1 for r in records for t in r["trials"] if t["empty"])
    n_empty_reflections = sum(r["n_empty_reflections"] for r in records)
    mean_trials_used = sum(len(r["trials"]) for r in records) / n if n else 0.0

    return {
        "run_index": run_index,
        "n": n,
        "trial1_acc": trial1_acc,
        **{f"trial{k}_acc": cumulative_acc(k) for k in range(2, k_trials + 1)},
        "any_acc": any_acc,
        "mean_trials_used": mean_trials_used,
        # A trial's `raw == ""` can be a genuine no-answer or a dead endpoint;
        # same convention as `baselines/core/runner.py.frozen_eval`'s n_empty --
        # nonzero means this run's accuracy is not trustworthy as-is.
        "n_empty_actor_completions": n_empty_actor,
        "n_empty_reflections": n_empty_reflections,
        "reliable": n_empty_actor == 0,
        "path": str(path),
    }
