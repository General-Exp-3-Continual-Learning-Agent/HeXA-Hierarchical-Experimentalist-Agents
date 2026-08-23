"""The Design B round loop: batch of instances -> evolve -> repeat -> frozen eval.

Structure and vocabulary follow ``hexa_core/passive_loop.py`` so the two arms of the
project are read the same way. What differs is the unit: a round here is a batch of
*instances*, each already containing its own three-episode loop, and the evolver
reflects on whole instances rather than single attempts.

Round *n* applies ``K_{n-1}`` -- evolved from batches ``1..n-1`` -- to a fresh batch
``n``, and training instances are consumed once and never revisited. So the bank can
never meet an instance it was distilled from, which means a parent-exclusion check
should never fire; :attr:`HexaRunResult.streaming_leak` exposes that as a property
expected to be ``False`` rather than as a safeguard doing work.

Persistence is per round, and ``bank.json`` is written **after** evolution: so
``roundN/bank.json`` is the bank round N *produced* and round N+1 consumed. Writing it
before would create an off-by-one invisible until someone plots bank size against
success rate. A round counts as complete only when its ``bank.json`` exists, which is
what makes :func:`resume_state` able to distinguish "died mid-evolution" (redo the
evolution from the runs on disk) from "died mid-batch" (redo the round).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from . import audit as audit_mod
from . import report as report_mod
from . import teacher as teacher_mod
from . import trajectory as trajectory_mod
from .actor import ActorConfig, GenerationFailed, InstanceRun, MazeActor
from .admission import accept_rate, admit_skills
from .adapter import MazeTask
from .pools import MazeInstanceSpec, leakage_split
from .reward import reward_bin

logger = logging.getLogger(__name__)

#: Bank capacity. ``skillrl.core.config.MAX_SKILLS_PER_LEVEL``'s value, restated so
#: the loop does not silently inherit a change to InterPhyre's default. Bounded
#: growth is load-bearing: forcing the teacher to choose is what removes redundancy.
MAX_SKILLS = 10
MAX_MISTAKES = 5

#: Token budget for the injected block, tracking CORE's reported 0.92k so the
#: added-token axis stays comparable across our own arms.
INJECTION_BUDGET_TOKENS = 1000

LEVEL = "maze"


# ── records ──────────────────────────────────────────────────────────────

@dataclass
class RoundRecord:
    """One round's outcome, with the diagnostics §12 requires."""

    round_index: int
    instance_ids: list[str]
    summary: dict[str, Any]
    reward_bins: dict[str, int]
    n_skills_before: int
    n_skills_after: int
    n_mistakes_after: int
    admission: dict[str, Any]
    audit: dict[str, Any]
    injection: dict[str, Any]
    confidence_stats: dict[str, int]
    n_errored: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_index,
            "instance_ids": self.instance_ids,
            "summary": self.summary,
            "reward_bins": self.reward_bins,
            "n_skills_before": self.n_skills_before,
            "n_skills_after": self.n_skills_after,
            "n_mistakes_after": self.n_mistakes_after,
            "admission": self.admission,
            "audit": self.audit,
            "injection": self.injection,
            "confidence_stats": self.confidence_stats,
            "n_errored": self.n_errored,
        }


@dataclass
class HexaRunResult:
    """Everything a run produced, plus the assertions that make it trustworthy."""

    task_name: str = LEVEL
    rounds: list[RoundRecord] = field(default_factory=list)
    consumed_ids: set[str] = field(default_factory=set)
    manifest: dict[str, Any] = field(default_factory=dict)
    eval_frozen: Optional[dict[str, Any]] = None
    eval_no_bank: Optional[dict[str, Any]] = None
    stopped_early: Optional[str] = None

    @property
    def streaming_leak(self) -> bool:
        """True if any instance appeared in two rounds -- under this loop, a bug.

        The curriculum guarantees each training instance is consumed once, so the
        bank never meets an instance it was distilled from. A duplicate means the
        batching broke.
        """
        seen: set[str] = set()
        for record in self.rounds:
            if seen & set(record.instance_ids):
                return True
            seen |= set(record.instance_ids)
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "manifest": self.manifest,
            "rounds": [r.to_dict() for r in self.rounds],
            "rounds_completed": len(self.rounds),
            "n_consumed_instances": len(self.consumed_ids),
            "streaming_leak": self.streaming_leak,
            "stopped_early": self.stopped_early,
            "eval_frozen": self.eval_frozen,
            "eval_no_bank": self.eval_no_bank,
        }


# ── retrieval and injection ──────────────────────────────────────────────

def render_block(bank: Any, *, budget_tokens: int = INJECTION_BUDGET_TOKENS) -> Any:
    """Retrieve a token-capped block from *bank* (M7). Returns a ``RetrievedBlock``."""
    from hexa_core.hexa_bridge import retrieve_within_token_budget  # noqa: PLC0415

    return retrieve_within_token_budget(
        bank,
        LEVEL,
        budget_tokens,
        max_general=0,       # single task family; cross-task transfer is Design C
        max_specific=MAX_SKILLS,
        max_mistakes=MAX_MISTAKES,
    )


def _injection_record(block: Any) -> dict[str, Any]:
    if block is None:
        return {"tokens": 0, "n_skills": 0, "n_mistakes": 0, "budget_bound": False}
    return {
        "tokens": block.tokens,
        "n_skills": len(block.skills),
        "n_mistakes": len(block.mistakes),
        "dropped_skills": block.dropped_skills,
        "dropped_mistakes": block.dropped_mistakes,
        "budget_bound": block.budget_bound,
        "block_text": block.block_text,
    }


# ── one batch of instances ───────────────────────────────────────────────

def play_batch(
    task: MazeTask,
    specs: Sequence[MazeInstanceSpec],
    actor_factory: Callable[..., MazeActor],
    guidelines: Optional[str],
    *,
    leaked_ids: Optional[set[str]] = None,
    budget: Any = None,
    on_instance: Optional[Callable[[int, InstanceRun], None]] = None,
) -> tuple[list[InstanceRun], list[InstanceRun]]:
    """Play a batch serially (D5). Returns ``(completed, errored)``.

    An instance whose generation fails is recorded with ``error`` set and excluded
    from everything downstream -- a transport failure is evidence of nothing, and
    folding it in as a failure would depress the round's reward distribution for a
    reason unrelated to the agent.
    """
    completed: list[InstanceRun] = []
    errored: list[InstanceRun] = []
    leaked_ids = leaked_ids or set()

    for position, spec in enumerate(specs, start=1):
        instance = task.instance(spec)
        # The instance is passed as well as the guidelines: a stub or oracle-driven
        # actor legitimately needs the live environment (it reads the map), and
        # letting it reach for one out of band led to `task.instance` being rebound
        # recursively. One explicit argument is cheaper than that.
        actor = actor_factory(guidelines, instance)
        try:
            run = actor.run_instance(instance, leaked=spec.instance_id in leaked_ids)
        except GenerationFailed as exc:
            run = InstanceRun(
                instance_id=spec.instance_id,
                dataset_seed=spec.dataset_seed,
                resolved_seed=spec.resolved_seed,
                shortest_path_len=spec.shortest_path_len,
                leaked=spec.instance_id in leaked_ids,
                steps=[], messages=[], metrics={},
                episode_successes=[], episode_lengths=[],
                success_count=0, added_tokens=0, token_counter="",
                error=f"{type(exc).__name__}: {exc}",
            )
            errored.append(run)
            logger.warning("instance %s errored: %s", spec.instance_id, exc)
        else:
            completed.append(run)
            if budget is not None:
                budget.charge_train_solve(len(run.steps))
        if on_instance is not None:
            on_instance(position, run)

    return completed, errored


# ── evolution ────────────────────────────────────────────────────────────

def evolve(
    round_dir: Path,
    prev_bank: Any,
    *,
    round_index: int,
    teacher_model: str,
) -> Any:
    """Round 1 distils from scratch; rounds 2+ evolve the previous bank.

    Round 1 has nothing to evolve *from*: HExA's headline InterPhyre config
    bootstraps from ``Initial_trajectories/``, and no ORBIT equivalent exists, so
    round 1 is the "pure online" configuration that spends itself collecting raw
    data (``docs/HEXA.md``).

    Both paths read the round's ``runs.jsonl`` through the rebound
    ``load_trajectories``, so writing that file is the entire integration.
    """
    from skillrl.core.skill_bank import SkillBank  # noqa: PLC0415
    from skillrl.distillation.distill import distill_contrastive, distill_mistakes  # noqa: PLC0415
    from skillrl.distillation.evolving_distill import evolve_skill_bank  # noqa: PLC0415

    output_path = round_dir / "bank.json"

    if round_index > 0 and prev_bank is not None:
        return evolve_skill_bank(
            LEVEL,
            prev_bank,
            round_dir,
            output_path,
            max_skills=MAX_SKILLS,
            max_mistakes=MAX_MISTAKES,
            teacher_model=teacher_model,
        )

    successes, failures = trajectory_mod.load_instance_trajectories(round_dir)
    bank = SkillBank()
    skills = distill_contrastive(LEVEL, successes, failures, teacher_model=teacher_model)
    mistakes, partial = distill_mistakes(LEVEL, failures, teacher_model=teacher_model)
    bank.level_skills[LEVEL] = (skills + partial)[:MAX_SKILLS]
    bank.level_mistakes[LEVEL] = mistakes[:MAX_MISTAKES]
    bank.save(output_path)
    return bank


# ── resumability ─────────────────────────────────────────────────────────

def resume_state(out_dir: Path) -> tuple[int, Any, set[str]]:
    """``(next_round_index, bank, consumed_instance_ids)`` from a partial run.

    A round is complete iff its ``bank.json`` exists. A ``runs.jsonl`` without one
    means the run died mid-evolution: that round's instances are already consumed,
    so they are counted, and the caller re-evolves from the runs rather than
    replaying ≤540 model calls.
    """
    from skillrl.core.skill_bank import SkillBank  # noqa: PLC0415

    out_dir = Path(out_dir)
    if not out_dir.exists():
        return 0, None, set()

    consumed: set[str] = set()
    bank = None
    next_round = 0
    for round_dir in sorted(
        (d for d in out_dir.glob("round*") if d.is_dir()),
        key=lambda d: int(d.name.removeprefix("round")),
    ):
        runs_path = round_dir / "runs.jsonl"
        if not runs_path.exists():
            continue
        consumed |= {r.instance_id for r in trajectory_mod.read_runs(runs_path)}
        bank_path = round_dir / "bank.json"
        if bank_path.exists():
            bank = SkillBank.load(bank_path)
            next_round = int(round_dir.name.removeprefix("round")) + 1
        else:
            logger.warning(
                "%s has runs but no bank.json -- died mid-evolution; re-evolving",
                round_dir,
            )
            next_round = int(round_dir.name.removeprefix("round"))
            break
    return next_round, bank, consumed


# ── the loop ─────────────────────────────────────────────────────────────

def run_hexa_loop(
    task: MazeTask,
    train_specs: Sequence[MazeInstanceSpec],
    actor_factory: Callable[..., MazeActor],
    out_dir: Path,
    *,
    rounds: int,
    batch_size: int,
    teacher_model: str,
    budget: Any = None,
    admission_enabled: bool = False,
    admission_trials: int = 3,
    injection_budget_tokens: int = INJECTION_BUDGET_TOKENS,
    resume: bool = True,
    manifest: Optional[dict[str, Any]] = None,
) -> HexaRunResult:
    """Run *rounds* rounds of *batch_size* instances each, then persist the bank."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher_mod.install()

    start_round, bank, consumed = (
        resume_state(out_dir) if resume else (0, None, set())
    )
    if consumed:
        logger.info("resuming at round %d; %d instances already consumed",
                    start_round, len(consumed))

    remaining = [s for s in train_specs if s.instance_id not in consumed]
    result = HexaRunResult(manifest=manifest or {}, consumed_ids=set(consumed))

    for round_index in range(start_round, rounds):
        batch = remaining[:batch_size]
        if not batch:
            result.stopped_early = "training pool exhausted"
            logger.warning("stopping at round %d: %s", round_index, result.stopped_early)
            break
        remaining = remaining[batch_size:]
        round_dir = out_dir / f"round{round_index}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # Retrieve once per round; the bank does not move during it.
        block = render_block(bank, budget_tokens=injection_budget_tokens) if bank else None
        guidelines = block.block_text if block else None

        n_before = len((getattr(bank, "level_skills", {}) or {}).get(LEVEL, [])) if bank else 0
        teacher_mod.reset_confidence_stats()

        completed, errored = play_batch(
            task, batch, actor_factory, guidelines, budget=budget,
            on_instance=lambda i, r: logger.info(
                "round %d [%d/%d] %s R=%d/3", round_index, i, len(batch),
                r.instance_id, r.success_count),
        )
        trajectory_mod.write_runs(round_dir / "runs.jsonl", completed + errored)
        result.consumed_ids |= {s.instance_id for s in batch}

        # Evolve. bank.json is written by `evolve`, AFTER the round's runs exist.
        new_bank = evolve(
            round_dir, bank, round_index=round_index, teacher_model=teacher_model
        )

        # Admission: trial candidates on training instances outside this round.
        candidates = [
            s for s in (getattr(new_bank, "level_skills", {}) or {}).get(LEVEL, [])
            if getattr(s, "is_new", True)
        ] if admission_enabled else []
        admitted, admission_results = admit_skills(
            candidates,
            remaining[:admission_trials],
            task,
            actor_factory,
            render_block=lambda b: render_block(b, budget_tokens=injection_budget_tokens).block_text,
            trials=admission_trials,
            enabled=admission_enabled,
            budget=budget,
        )
        if admission_enabled:
            kept = {getattr(s, "skill_id", None) for s in admitted}
            surviving = [
                s for s in (getattr(new_bank, "level_skills", {}) or {}).get(LEVEL, [])
                if not getattr(s, "is_new", True) or getattr(s, "skill_id", None) in kept
            ]
            new_bank.level_skills[LEVEL] = surviving
            new_bank.save(round_dir / "bank.json")

        trajectories = [trajectory_mod.instance_trajectory(r) for r in completed]
        bins: dict[str, int] = {}
        for traj in trajectories:
            label = reward_bin(traj)
            bins[label] = bins.get(label, 0) + 1

        round_audit = audit_mod.audit_bank(
            new_bank,
            level_name=LEVEL,
            round_index=round_index,
            known_ids=[t["seed"] for t in trajectories],
        )
        (round_dir / "audit.json").write_text(json.dumps(round_audit, indent=2))
        (round_dir / "admission.json").write_text(
            json.dumps([r.to_dict() for r in admission_results], indent=2)
        )
        injection = _injection_record(block)
        (round_dir / "injection.json").write_text(json.dumps(injection, indent=2))

        result.rounds.append(RoundRecord(
            round_index=round_index,
            instance_ids=[s.instance_id for s in batch],
            summary=report_mod.arm_summary(completed),
            reward_bins=bins,
            n_skills_before=n_before,
            n_skills_after=len((getattr(new_bank, "level_skills", {}) or {}).get(LEVEL, [])),
            n_mistakes_after=len((getattr(new_bank, "level_mistakes", {}) or {}).get(LEVEL, [])),
            admission=accept_rate(admission_results),
            audit=round_audit,
            injection={k: v for k, v in injection.items() if k != "block_text"},
            confidence_stats=teacher_mod.confidence_stats(),
            n_errored=len(errored),
        ))
        bank = new_bank
        logger.info(audit_mod.format_audit(round_audit))

    (out_dir / "run.json").write_text(json.dumps(result.to_dict(), indent=2))
    if result.streaming_leak:  # pragma: no cover - a bug if it ever fires
        raise AssertionError(
            "streaming leak: an instance appeared in more than one round, so the "
            "bank met an instance it was distilled from"
        )
    return result


# ── frozen eval ──────────────────────────────────────────────────────────

def frozen_eval(
    task: MazeTask,
    eval_specs: Sequence[MazeInstanceSpec],
    actor_factory: Callable[..., MazeActor],
    bank: Any,
    out_dir: Path,
    *,
    train_specs: Sequence[MazeInstanceSpec] = (),
    injection_budget_tokens: int = INJECTION_BUDGET_TOKENS,
    budget: Any = None,
) -> dict[str, Any]:
    """Evaluate on held-out instances with the bank frozen -- no writes, ever.

    ``bank=None`` is the no-bank arm, so the two arms differ only in the injected
    block. Nothing here calls ``evolve`` or ``bank.save``, and the runs are written
    to their own directory so they cannot be picked up by a later
    ``load_instance_trajectories`` on a round directory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    block = render_block(bank, budget_tokens=injection_budget_tokens) if bank else None
    guidelines = block.block_text if block else None
    leaked_ids = {
        s.instance_id
        for s in leakage_split(eval_specs, list(train_specs))["leaked"]
    } if train_specs else set()

    completed, errored = play_batch(
        task, eval_specs, actor_factory, guidelines,
        leaked_ids=leaked_ids, budget=budget,
        on_instance=lambda i, r: logger.info(
            "eval [%d/%d] %s R=%d/3%s", i, len(eval_specs), r.instance_id,
            r.success_count, " LEAKED" if r.leaked else ""),
    )
    trajectory_mod.write_runs(out_dir / "runs.jsonl", completed + errored)

    summary = {
        "arm": "frozen_bank" if bank else "no_bank",
        "report": report_mod.aggregate(completed, errored=errored),
        "injection": _injection_record(block),
        "n_eval": len(eval_specs),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
