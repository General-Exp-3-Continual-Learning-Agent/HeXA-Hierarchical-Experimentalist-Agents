"""One CORE solve as a record, and its translation into HeXA's trajectory dict.

A CORE single-turn attempt is a *one-step* trajectory: the model is handed one
prompt, emits one response, and a deterministic verifier accepts or rejects the
answer parsed out of it. HeXA's distillation code already consumes a trajectory
dict shaped like::

    {"seed": ..., "success": bool, "iterations": int,
     "trajectory": [{"thought": ..., "action": ..., "observation": ...}, ...],
     "action": <final action>}

so nothing in ``skillrl/`` needs to change: :func:`attempt_to_trajectory` emits
exactly that shape with one step, where *thought* is the model's reasoning text,
*action* is the parsed answer, and *observation* is the verifier's verdict. The
CORE ``task_item_id`` takes the place of HeXA's ``seed`` — it is the stable
identity of the problem, which is what every downstream lookup actually wants.

Two things worth knowing before using this module:

* ``Skill.source_seeds`` is annotated ``list[int]`` in ``skillrl.core.skill_bank``
  but is never coerced anywhere. CORE problem ids are *strings*. They round-trip
  through ``asdict``/JSON unharmed and ``distill._compute_initial_confidence``
  matches them by plain dict lookup, so string ids work as-is. The annotation is
  the thing that is wrong, not the data.
* The top-level ``action`` key is a *string* (the parsed answer). HeXA's stock
  ``format_trajectory_for_teacher`` renders it as ``x=<char0>, y=<char1>,
  r=<char2>`` — silent garbage, because it assumes InterPhyre's ``[x, y, r]``
  triple. :func:`hexa_core.hexa_bridge.patch` replaces that formatter, and must
  therefore be called before any distillation or evolution call.

This module imports nothing from ``skillrl`` or from the rest of ``hexa_core``,
so it stays cheap and cannot participate in an import cycle.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

VERDICT_CORRECT = "CORRECT — the verifier accepted the answer"
VERDICT_INCORRECT = "INCORRECT — the verifier rejected the answer"

#: Appended to the verdict when generation stopped at the token limit. Truncation
#: is a distinct failure mode from bad reasoning (the brief's diagnosis order puts
#: it fourth), so the teacher must be able to tell the two apart.
TRUNCATION_NOTE = " (generation hit the token limit; the answer may be truncated)"

#: Marker left in the rendered thought when a long chain of thought is clipped.
TRUNCATION_MARKER = "... [reasoning truncated]"


@dataclass(frozen=True)
class Attempt:
    """One model attempt at one CORE problem.

    ``reward`` is stored rather than derived from ``correct`` so that a run which
    ever flips CORE's ``uses_verifier_score_for_reward()`` records the real
    number. On the three tasks in scope it is always ``float(correct)``.

    ``baseline`` is the per-problem no-memory success rate b_q estimated from
    ``n_base=10`` samples (M1). It is ``None`` until those rollouts have been
    charged; :attr:`utility` then degrades to the raw reward, which is exactly
    HeXA's pre-M1 behaviour, and :attr:`has_baseline` lets a caller assert.
    """

    problem_id: str
    task_name: str
    prompt: str
    raw_text: str
    parsed_answer: str
    correct: bool
    reward: float
    baseline: float | None = None
    round: int = 0
    added_tokens: int = 0
    finish_reason: str = ""
    split: str = "train"

    @property
    def has_baseline(self) -> bool:
        return self.baseline is not None

    @property
    def utility(self) -> float:
        """Baseline-relative utility (M1), or the raw reward when b_q is unknown.

        ``U = float(correct) - b_q``, deliberately NOT ``reward - b_q``.

        M1 and M2 use different numeric conventions and subtracting across them
        overflows the domain everything downstream assumes. M2 sets ``reward`` to
        **±1** so it spans the same [-1,1] range as the shaped HeXA scale it replaces
        (`hexa_bridge.binary_trajectory_reward`). But ``b_q`` is a *rate* in [0,1], so
        ``reward - b_q`` lands in [-2, 1] -- e.g. a failure on a problem with
        ``b_q = 0.5`` gives ``-1 - 0.5 = -1.5``. `_compute_initial_confidence`
        (`distill.py:163`) maps [-1,1] onto [0.1,1.0], so anything past -1 silently
        clamps or extrapolates, and every failure on a mid-difficulty problem would be
        pushed to the floor regardless of how much worse than baseline it actually was.

        Using ``float(correct)`` in {0,1} -- CORE's own reward convention
        (``_reward_from_metrics`` returns ``float(bool(is_correct))``) -- keeps
        ``U`` in [-1,1] exactly. Both branches therefore land in the same range:
        ``reward`` is {-1,+1} and ``U`` is [-1,1].
        """
        if self.baseline is None:
            return self.reward
        return float(self.correct) - self.baseline

    @property
    def truncated(self) -> bool:
        """True when generation stopped at the token limit rather than finishing."""
        return self.finish_reason == "length"

    @property
    def verdict(self) -> str:
        """The observation text the teacher sees for this attempt."""
        verdict = VERDICT_CORRECT if self.correct else VERDICT_INCORRECT
        return verdict + TRUNCATION_NOTE if self.truncated else verdict

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Attempt:
        """Rebuild an attempt, ignoring keys added by later schema versions."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def attempt_to_trajectory(
    attempt: Attempt, *, thought_max_chars: int = 6000
) -> dict[str, Any]:
    """Render one :class:`Attempt` as a HeXA trajectory dict.

    The reasoning text is clipped at *thought_max_chars*: a Hanoi n=6 chain of
    thought can approach the 32768-token generation limit, and ``distill.py``
    packs up to five successes plus five failures into a single teacher prompt.

    Keys beyond HeXA's contract (``task_name``, ``round``, ``reward``,
    ``baseline``, ``utility``, ``added_tokens``, ``finish_reason``,
    ``truncated``) are additive: no existing consumer reads them, and the
    replacement reward and formatter installed by
    :func:`hexa_core.hexa_bridge.patch` do.
    """
    thought = attempt.raw_text
    if thought_max_chars > 0 and len(thought) > thought_max_chars:
        thought = thought[:thought_max_chars] + TRUNCATION_MARKER

    return {
        "seed": attempt.problem_id,
        "task_name": attempt.task_name,
        "success": bool(attempt.correct),
        "iterations": 1,
        "trajectory": [
            {
                "thought": thought,
                "action": attempt.parsed_answer,
                "observation": attempt.verdict,
            }
        ],
        "action": attempt.parsed_answer,
        "reward": attempt.reward,
        "baseline": attempt.baseline,
        "utility": attempt.utility,
        "round": attempt.round,
        "added_tokens": attempt.added_tokens,
        "finish_reason": attempt.finish_reason,
        "truncated": attempt.truncated,
    }


# ── Batch helpers (M3: a round is a batch of attempts) ──────────────────

def trajectories_from_attempts(
    attempts: Iterable[Attempt], *, thought_max_chars: int = 6000
) -> list[dict[str, Any]]:
    return [
        attempt_to_trajectory(a, thought_max_chars=thought_max_chars) for a in attempts
    ]


def split_successes_failures(
    attempts: Iterable[Attempt], *, thought_max_chars: int = 6000
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split into ``(successes, failures)`` — the contract every distil pass takes."""
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for attempt in attempts:
        traj = attempt_to_trajectory(attempt, thought_max_chars=thought_max_chars)
        (successes if traj["success"] else failures).append(traj)
    return successes, failures


def group_by_round(attempts: Iterable[Attempt]) -> dict[int, list[Attempt]]:
    """Group attempts by round, with rounds in ascending order."""
    grouped: dict[int, list[Attempt]] = {}
    for attempt in attempts:
        grouped.setdefault(attempt.round, []).append(attempt)
    return {r: grouped[r] for r in sorted(grouped)}


def chunk_attempts(
    attempts: Sequence[Attempt], batch_size: int
) -> list[list[Attempt]]:
    """Split a flat sequence into consecutive batches of at most *batch_size*."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    return [
        list(attempts[i : i + batch_size])
        for i in range(0, len(attempts), batch_size)
    ]


def assign_rounds(
    attempts: Sequence[Attempt], batch_size: int, *, start_round: int = 0
) -> list[Attempt]:
    """Stamp consecutive batches of *batch_size* with increasing round numbers."""
    out: list[Attempt] = []
    for offset, batch in enumerate(chunk_attempts(attempts, batch_size)):
        out.extend(replace(a, round=start_round + offset) for a in batch)
    return out


def round_summary(attempts: Sequence[Attempt]) -> dict[str, Any]:
    """Aggregate one round for logging. Empty input yields zeroed counters."""
    n = len(attempts)
    if n == 0:
        return {
            "n": 0,
            "n_correct": 0,
            "accuracy": 0.0,
            "mean_reward": 0.0,
            "mean_utility": 0.0,
            "mean_added_tokens": 0.0,
            "n_truncated": 0,
            "n_with_baseline": 0,
            "task_names": [],
            "rounds": [],
        }
    n_correct = sum(1 for a in attempts if a.correct)
    return {
        "n": n,
        "n_correct": n_correct,
        "accuracy": n_correct / n,
        "mean_reward": sum(a.reward for a in attempts) / n,
        "mean_utility": sum(a.utility for a in attempts) / n,
        "mean_added_tokens": sum(a.added_tokens for a in attempts) / n,
        "n_truncated": sum(1 for a in attempts if a.truncated),
        "n_with_baseline": sum(1 for a in attempts if a.has_baseline),
        "task_names": sorted({a.task_name for a in attempts}),
        "rounds": sorted({a.round for a in attempts}),
    }


# ── Persistence (JSON Lines, one attempt per line) ──────────────────────

def write_attempts(path: str | Path, attempts: Iterable[Attempt]) -> None:
    """Write attempts as JSON Lines, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for attempt in attempts:
            fh.write(json.dumps(attempt.to_dict(), ensure_ascii=False) + "\n")


def read_attempts(path: str | Path) -> list[Attempt]:
    """Read a JSON Lines attempt file, skipping blank lines."""
    out: list[Attempt] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Attempt.from_dict(json.loads(line)))
    return out


def load_attempt_trajectories(
    traj_dir: str | Path,
    task_name: str | None = None,
    *,
    thought_max_chars: int = 6000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load ``*.jsonl`` attempt files from a directory as HeXA trajectory dicts.

    Signature-compatible with ``skillrl.distillation.distill.load_trajectories``
    (which globs ``trajectory_seed*.json`` instead) so Phase 3 can rebind it
    through ``hexa_bridge.patch(load_trajectories_fn=...)`` rather than fork it.
    A missing directory warns and returns empty lists, matching the original.
    """
    traj_dir = Path(traj_dir)
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if not traj_dir.exists():
        print(f"[Warning] Attempt directory not found: {traj_dir}")
        return successes, failures

    for f in sorted(traj_dir.glob("*.jsonl")):
        for attempt in read_attempts(f):
            if task_name is not None and attempt.task_name != task_name:
                continue
            traj = attempt_to_trajectory(attempt, thought_max_chars=thought_max_chars)
            (successes if traj["success"] else failures).append(traj)

    return successes, failures
