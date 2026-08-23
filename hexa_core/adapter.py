"""Task-neutral view over CORE's single-turn tasks.

The rest of HExA talks to :class:`Problem` and :class:`Task` and never to a CORE
task object directly. Everything semantic -- prompt text, answer parsing, the
verifier's acceptance rule -- is delegated to bound methods of the *originating*
CORE task instance, so this module can never drift from CORE.

Two things here are load-bearing and easy to get wrong:

* **Per-instance binding.** ``ZebraLogicTask.build_solve_prompt`` looks the
  problem's output schema up in ``self._problem_output_schema_json``, a dict keyed
  by problem *text* and populated only from that instance's size-group-filtered
  pool, with a silent ``.get(problem, EXAMPLE_OUTPUT_SCHEMA_JSON)`` fallback.
  Calling it on a differently-configured instance yields a wrong prompt with no
  error. Every :class:`Problem` therefore carries bound methods of the one task
  instance it came from (which also keeps that instance alive).

* **What text gets scored.** CORE scores ``reasoning_text + final_text``
  concatenated with no separator (``src/model/nvidia_client.py:141``, consumed at
  ``scripts/run_core.py:433``). Callers of :meth:`Problem.verify` must pass that
  same concatenation. Tower of Hanoi's ``_extract_moves_from_text`` takes the
  *last* parseable move list in the string, so dropping the chain of thought
  quietly changes which candidate answer is graded.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

from .core_bridge import core_git_revision

EVAL_ITEMS = 100


@runtime_checkable
class Tool(Protocol):
    """An action a model may invoke mid-solve.

    Nothing consumes tools yet: all three CORE tasks in scope are single-turn and
    passive (Table A), so every :class:`Problem` currently has ``tools=None``.
    The Protocol exists to pin the shape an active-mode runner would need, and to
    give ``Problem.tools`` a type that is not ``Any``.
    """

    name: str
    description: str
    input_schema: dict[str, Any]

    def __call__(self, **kwargs: Any) -> str: ...


@dataclass(frozen=True)
class Problem:
    """One scoreable item, with CORE's prompt and verifier attached.

    Identity is the six string fields. ``metadata``, ``tools`` and the bound
    callables are excluded from equality and hashing: a frozen dataclass derives
    ``__hash__`` from every comparison field, and a ``dict``/``list`` field would
    make ``hash(problem)`` raise. Problems are used as dict keys (per-problem
    no-memory baselines ``b_q``, per-problem accuracy), so hashability is
    required, and two views of the same item must not compare unequal just because
    they were built from different task instances.
    """

    id: str
    prompt: str
    problem_text: str
    format_instructions: str
    task_name: str
    split: str
    metadata: dict[str, Any] = field(compare=False)
    tools: list[Tool] | None = field(compare=False)

    # Bound methods of the single CORE task instance this problem came from.
    build_prompt_fn: Callable[[str, str | None], str] = field(
        kw_only=True, compare=False, repr=False
    )
    parse_answer_fn: Callable[[str], str] = field(kw_only=True, compare=False, repr=False)
    evaluate_fn: Callable[[str, Any], bool] = field(kw_only=True, compare=False, repr=False)
    # CORE's *graded* scorer, separate from the boolean verifier. Optional so a
    # hand-built Problem in a test need not supply it.
    score_prediction_fn: Callable[[str, Any], float] | None = field(
        default=None, kw_only=True, compare=False, repr=False
    )
    # CORE's target payload (`TaskItem.all_valid_outputs`). Needed by
    # `verify_parsed`, and by distillation when it wants a reference answer.
    all_valid_outputs: Any = field(default=None, kw_only=True, compare=False, repr=False)

    def build_prompt(self, guidelines: str | None = None) -> str:
        """CORE's solve prompt, optionally with a guidelines block injected.

        ``build_prompt(None)`` reproduces :attr:`prompt` byte for byte. Injecting
        retrieved skills through this slot -- rather than through a separate system
        message -- is what keeps the added-token metric comparable to CORE's.
        """
        return self.build_prompt_fn(self.problem_text, guidelines)

    def parse_answer(self, raw_model_text: str) -> str:
        """CORE's parser. Pass ``reasoning_text + final_text``, not just the final text."""
        return self.parse_answer_fn(raw_model_text)

    def verify(self, raw_model_text: str) -> int:
        """Parse then evaluate. Returns 1/0 -- rewards are binary on all three tasks."""
        return self.verify_parsed(self.parse_answer(raw_model_text))

    def verify_parsed(self, parsed: str) -> int:
        return int(bool(self.evaluate_fn(parsed, self.all_valid_outputs)))

    def score(self, parsed: str) -> float:
        """CORE's graded verifier score for an already-parsed answer.

        CORE keeps two numbers per attempt and logs both: ``is_correct`` from
        ``task.evaluate`` and ``verifier_score`` from ``task.score_prediction``
        (``run_core.py:1787-1788``). On all three tasks in scope
        ``score_prediction`` is ``SingleTurnTask``'s base implementation, so the two
        agree exactly -- but they are *not the same function*, and hardcoding
        ``float(verify_parsed(...))`` would silently keep reporting a binary
        ``val_mean_verifier_score`` if CORE ever gave one of these tasks partial
        credit. Delegating means we inherit that change instead of missing it.
        """
        if self.score_prediction_fn is None:
            return float(self.verify_parsed(parsed))
        return float(self.score_prediction_fn(parsed, self.all_valid_outputs))


@runtime_checkable
class Task(Protocol):
    name: str

    def train_problems(self, n: int | None = None, seed: int | None = None) -> list[Problem]: ...

    def eval_problems(self) -> list[Problem]: ...

    def manifest(self) -> dict[str, Any]: ...


class EvalContamination(ValueError):
    """An eval item was attempted during training, evolution or admission."""


class CoreTask:
    """A CORE ``SingleTurnTask`` presented as a :class:`Task`.

    Splits are materialised once, in CORE's own order, and never resampled --
    train/eval membership is fixed by the CORE constructor, not by us.
    """

    def __init__(
        self,
        name: str,
        core_task: Any,
        *,
        tools: list[Tool] | None = None,
        expected_train_items: int | None = None,
        expected_eval_items: int | None = EVAL_ITEMS,
    ) -> None:
        self.name = name
        self.core_task = core_task
        self.tools = tools

        if expected_train_items is None:
            expected_train_items = getattr(core_task, "num_train_items", None)

        train_items = self._core_items(core_task, "train")
        eval_items = self._core_items(core_task, "eval")

        self._train: list[Problem] = [self._to_problem(item, "train") for item in train_items]
        self._eval: list[Problem] = [self._to_problem(item, "eval") for item in eval_items]

        self._assert_invariants(expected_train_items, expected_eval_items)

    # -- construction helpers -------------------------------------------------

    @staticmethod
    def _core_items(core_task: Any, split: str) -> list[Any]:
        # `SingleTurnTask.__init__` already caches both splits; prefer the cached
        # attribute over re-running the loader (matchstick would re-read its
        # JSONL, Hanoi would re-run its BFS sampler).
        cached = getattr(core_task, f"{split}_data", None)
        if cached:
            return list(cached)
        loader = getattr(core_task, f"load_{split}_data")
        return list(loader())

    def _to_problem(self, item: Any, split: str) -> Problem:
        core = self.core_task
        return Problem(
            id=str(item.task_item_id),
            prompt=core.build_solve_prompt(item.problem, None),
            problem_text=item.problem,
            format_instructions=core.output_instruction,
            task_name=self.name,
            split=split,
            metadata=dict(getattr(item, "metadata", None) or {}),
            tools=self.tools,
            build_prompt_fn=core.build_solve_prompt,
            parse_answer_fn=core.parse_answer,
            evaluate_fn=core.evaluate,
            score_prediction_fn=getattr(core, "score_prediction", None),
            all_valid_outputs=item.all_valid_outputs,
        )

    def _assert_invariants(
        self, expected_train_items: int | None, expected_eval_items: int | None
    ) -> None:
        where = f"task={self.name!r} (core={getattr(self.core_task, 'name', '?')!r})"

        if expected_eval_items is not None:
            assert len(self._eval) == expected_eval_items, (
                f"{where}: expected {expected_eval_items} eval items, got {len(self._eval)}"
            )

        for split, problems in (("train", self._train), ("eval", self._eval)):
            ids = [p.id for p in problems]
            if len(set(ids)) != len(ids):
                dupes = sorted({i for i in ids if ids.count(i) > 1})
                raise AssertionError(f"{where}: duplicate ids in {split} split: {dupes[:10]}")

        overlap = {p.id for p in self._train} & {p.id for p in self._eval}
        assert not overlap, f"{where}: train/eval id overlap: {sorted(overlap)[:10]}"

        # Catches CORE's soft-degrade path, which prints "Warning: Not enough ...
        # items" and silently returns a smaller split. Zebralogic at N=100
        # consumes its entire train pool, so it sits exactly on that boundary.
        if expected_train_items is not None:
            assert len(self._train) == expected_train_items, (
                f"{where}: requested {expected_train_items} train items but CORE "
                f"produced {len(self._train)} -- the split silently degraded"
            )

    # -- Task protocol --------------------------------------------------------

    def train_problems(self, n: int | None = None, seed: int | None = None) -> list[Problem]:
        """Training problems in CORE order.

        ``n`` is a sub-sample for smoke tests only -- the regime is already fixed
        by the CORE constructor. The sub-sample is drawn with ``seed`` (default 42)
        and then re-sorted back into CORE order, so ordering is never a hidden
        variable.
        """
        if n is None:
            return list(self._train)
        if n < 0 or n > len(self._train):
            raise ValueError(
                f"task={self.name!r}: cannot sample {n} of {len(self._train)} train problems"
            )
        rng = random.Random(42 if seed is None else seed)
        picked = sorted(rng.sample(range(len(self._train)), n))
        return [self._train[i] for i in picked]

    def eval_problems(self) -> list[Problem]:
        return list(self._eval)

    def manifest(self) -> dict[str, Any]:
        """Provenance for a run log.

        Records both names: ``task_name`` is our stable registry key and is what
        appears in ``Problem.task_name`` and ``Skill.source_level``, while
        ``core_task_name`` is CORE's own -- they differ for Tower of Hanoi, whose
        ``variant="state"`` instance calls itself ``tower-of-hanoi-state``.
        """
        eval_ids = [p.id for p in self._eval]
        manifest: dict[str, Any] = {
            "task_name": self.name,
            "core_task_name": getattr(self.core_task, "name", None),
            "core_task_class": type(self.core_task).__name__,
            "core_git_revision": core_git_revision(),
            "num_train_items": len(self._train),
            "num_eval_items": len(self._eval),
            "train_ids": [p.id for p in self._train],
            "eval_ids": eval_ids,
            "eval_ids_sha256": eval_ids_digest(eval_ids),
            "tools": None if self.tools is None else [t.name for t in self.tools],
        }
        summary = None
        try:
            summary = self.core_task.build_dataset_summary()
        except (AttributeError, TypeError):
            summary = None
        if summary is not None:
            manifest["dataset_summary"] = summary
        return manifest

    def __repr__(self) -> str:
        return (
            f"CoreTask(name={self.name!r}, train={len(self._train)}, eval={len(self._eval)})"
        )


def eval_ids_digest(eval_ids: Iterable[str]) -> str:
    """Stable digest of an eval id list, for cheap cross-regime parity checks in logs."""
    return hashlib.sha256("\n".join(eval_ids).encode("utf-8")).hexdigest()


def assert_no_eval_contamination(
    tasks: Task | Iterable[Task], attempted_ids: Iterable[str]
) -> None:
    """Raise if anything attempted during training touched an eval item.

    Cheap enough to call before every evaluation, which is the point: hard rule 4
    ("never evaluate on a problem seen during training/evolution/admission") is
    only worth anything if it is checked mechanically.
    """
    task_list = [tasks] if hasattr(tasks, "eval_problems") else list(tasks)  # type: ignore[arg-type]
    attempted = set(attempted_ids)
    if not attempted:
        return

    for task in task_list:
        leaked = {p.id for p in task.eval_problems()} & attempted
        if leaked:
            shown = sorted(leaked)
            raise EvalContamination(
                f"task={getattr(task, 'name', '?')!r}: {len(leaked)} eval item(s) were "
                f"attempted during training/evolution/admission: {shown[:10]}"
                + (" ..." if len(shown) > 10 else "")
            )
