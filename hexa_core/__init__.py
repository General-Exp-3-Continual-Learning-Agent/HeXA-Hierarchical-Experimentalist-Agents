"""HExA on the CORE (`core-reasoning`) single-turn benchmarks.

This package is glue only. CORE's tasks, prompt templates, verifiers and data
splits are imported in place from the `core-reasoning` checkout and never forked,
copied or re-templated -- a verifier with a subtly different acceptance rule, or a
prompt that differs by a byte, invalidates every number in a HExA-vs-CORE
comparison.

Importing this package is deliberately cheap: it touches neither ``sys.path`` nor
the filesystem. The submodules that do (``adapter``, ``tasks``) are exposed lazily
via :pep:`562`, so ``import hexa_core`` stays free of CORE side effects and the
CORE checkout only has to exist for code that actually uses it.

Two modules are deliberately **not** re-exported here, so that the cheap-import
property above stays true. Import them by module path:

* ``hexa_core.hexa_bridge`` imports ``skillrl`` at module scope, so any
  ``hexa_core.<name>`` lookup would drag the whole HeXA package in.
* ``hexa_core.llm`` (``VLLMClient``, ``NimClient``, ``make_client``, ``Generation``)
  builds its transport lazily but is a subpackage with its own eager re-exports.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY: dict[str, str] = {
    "Problem": "adapter",
    "Task": "adapter",
    "Tool": "adapter",
    "CoreTask": "adapter",
    "EvalContamination": "adapter",
    "assert_no_eval_contamination": "adapter",
    "make_task": "tasks",
    "TaskSpec": "tasks",
    "TASK_SPECS": "tasks",
    "TASK_NAMES": "tasks",
    "REGIMES": "tasks",
    "RolloutBudget": "budget",
    "BudgetExhausted": "budget",
    "count_tokens": "tokens",
    "added_tokens": "tokens",
    "added_tokens_raw": "tokens",
    "counter_name": "tokens",
    "is_exact": "tokens",
    "core_repo_root": "core_bridge",
    "install_core_on_path": "core_bridge",
    "CoreRepoNotFound": "core_bridge",
    # logging_schema and trajectory are stdlib-only at module scope, so listing
    # them here does not cost the cheap-import property. They were left out while
    # stages were landing in parallel, which left `__all__` describing only a third
    # of the package -- `hexa_core.RolloutBudget` worked and
    # `hexa_core.ValidationLogger` raised AttributeError.
    "ValidationLogger": "logging_schema",
    "RunManifest": "logging_schema",
    "MeanSEM": "logging_schema",
    "mean_sem": "logging_schema",
    "mean_sem_over_runs": "logging_schema",
    "format_mean_sem": "logging_schema",
    "append_jsonl": "logging_schema",
    "read_jsonl": "logging_schema",
    "final_summary_record": "logging_schema",
    "Attempt": "trajectory",
    "attempt_to_trajectory": "trajectory",
    "trajectories_from_attempts": "trajectory",
    "read_attempts": "trajectory",
    "write_attempts": "trajectory",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    try:
        module_name = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return __all__
