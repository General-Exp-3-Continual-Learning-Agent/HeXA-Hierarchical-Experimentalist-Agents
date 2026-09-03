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

import copy
import json
import re
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from . import audit as audit_mod
from . import report as report_mod
from . import teacher as teacher_mod
from . import trajectory as trajectory_mod
from .actor import ActorConfig, GenerationFailed, InstanceRun, MazeActor
from . import admission as admission_mod
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

#: Task family the bank is keyed under. Mutable because the loop is otherwise
#: task-agnostic -- it drives whatever `task` object it is handed -- and the only
#: maze-specific things left are this string, the type annotations (structural,
#: not enforced) and the leakage split. Wrapped in a list for the same reason
#: `reward.GRADED_FAILURES` is: a module-level rebind would not reach callers that
#: already did `from .loop import LEVEL`.
_LEVEL = ["maze"]


def level() -> str:
    """The task family currently being run."""
    return _LEVEL[0]


def set_level(name: str) -> None:
    """Point the loop at a different task family (e.g. ``"mastermind"``)."""
    _LEVEL[0] = str(name)


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
    #: True when evolution returned a bank identical to the one it started from.
    #: `evolve_skill_bank` handles an unparseable teacher response by printing a
    #: warning and **returning the previous bank**, so a round that taught the
    #: bank nothing is otherwise indistinguishable from one where the teacher had
    #: nothing to add. That matters much more with a small teacher: a 7B model
    #: emitting strict JSON for a whole bank is exactly where this fails.
    bank_unchanged: bool = False

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
            "bank_unchanged": self.bank_unchanged,
        }


@dataclass
class HexaRunResult:
    """Everything a run produced, plus the assertions that make it trustworthy."""

    task_name: str = field(default_factory=level)
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
            "rounds_bank_unchanged": sum(1 for r in self.rounds if r.bank_unchanged),
            "n_consumed_instances": len(self.consumed_ids),
            "streaming_leak": self.streaming_leak,
            "stopped_early": self.stopped_early,
            "eval_frozen": self.eval_frozen,
            "eval_no_bank": self.eval_no_bank,
        }


# ── retrieval and injection ──────────────────────────────────────────────

def render_block(
    bank: Any,
    *,
    budget_tokens: int = INJECTION_BUDGET_TOKENS,
    top_k: Optional[int] = None,
) -> Any:
    """Retrieve a token-capped block from *bank* (M7). Returns a ``RetrievedBlock``.

    ``top_k`` caps how many skills are injected, highest-confidence first, instead
    of the full :data:`MAX_SKILLS`. It exists to separate two things the design
    conflates: whether the bank *contains* useful skills, and whether injecting all
    of them helps. On gpt-oss-120b the distilled bank was good -- procedures with
    explicit triggers, including the cross-episode divergence rule that the
    trajectory analysis independently identifies as the driver -- and injecting all
    seven of them cost 15.2 points of episode-3 accuracy against no bank at all
    (p=0.0002), with the bankless round of that same run scoring 0.750 against a
    banked mean near 0.35. Mistakes are dropped with ``top_k`` too: the point is to
    test a minimal injection, not a slightly smaller one.
    """
    from hexa_core.hexa_bridge import retrieve_within_token_budget  # noqa: PLC0415

    return retrieve_within_token_budget(
        bank,
        level(),
        budget_tokens,
        max_general=0,       # single task family; cross-task transfer is Design C
        max_specific=MAX_SKILLS if top_k is None else int(top_k),
        max_mistakes=MAX_MISTAKES if top_k is None else 0,
    )


#: A skill that only makes sense once an earlier episode exists. Matched on the
#: rendered text because that is what the agent reads; the teacher is not asked to
#: label this, so the classifier has to work on banks already on disk.
_NEEDS_PRIOR_EPISODE = re.compile(
    r"episode\s*1|earlier episode|prior episode|previous episode|previously "
    r"(?:visited|explored)|prior exploration|later episode|subsequent episode|"
    r"across episodes|from the (?:first|last) episode",
    re.IGNORECASE,
)


def requires_prior_episode(item: Any) -> bool:
    """Whether *item* (a Skill or Mistake) is unusable in episode 1."""
    parts = [
        str(getattr(item, field, "") or "")
        for field in (
            "principle", "when_to_apply", "example",
            "description", "why_it_happens", "how_to_avoid",
        )
    ]
    return bool(_NEEDS_PRIOR_EPISODE.search(" ".join(parts)))


def _filtered_bank(bank: Any, level_name: Optional[str] = None) -> Any:
    """A shallow copy of *bank* holding only what episode 1 can act on."""
    level_name = level_name or level()
    clone = copy.copy(bank)
    clone.level_skills = {
        lvl: [s for s in skills if not requires_prior_episode(s)]
        for lvl, skills in (getattr(bank, "level_skills", {}) or {}).items()
    }
    clone.level_mistakes = {
        lvl: [m for m in mistakes if not requires_prior_episode(m)]
        for lvl, mistakes in (getattr(bank, "level_mistakes", {}) or {}).items()
    }
    clone.general_skills = [
        s for s in (getattr(bank, "general_skills", []) or [])
        if not requires_prior_episode(s)
    ]
    clone.common_mistakes = [
        m for m in (getattr(bank, "common_mistakes", []) or [])
        if not requires_prior_episode(m)
    ]
    return clone


def render_blocks_by_episode(
    bank: Any, *, budget_tokens: int = INJECTION_BUDGET_TOKENS
) -> dict[int, str]:
    """``{0: episode-1 block, 1: full block}`` for per-episode injection.

    **Why the bank is split rather than injected whole.** A block is paid for on
    every turn of every episode, but much of it is inert in episode 1: in v3's
    final bank, 3 of 5 skills and 1 of 2 mistakes are phrased "based on prior
    exploration data", so ~60% of ~700 tokens described data the agent did not yet
    have. Across v1/v2/v3 episode-1 accuracy came in at -4.1, -1.6 and -2.9 points
    against the no-bank baseline while multi-episode success rose to +9.5 -- the
    shape you get when a block helps later episodes and taxes the first.

    Episodes 2+ get the whole bank, including the reuse skills, which is where
    those skills are worth their tokens. Returns an empty dict for a bank with
    nothing to say, so the caller injects nothing rather than an empty heading.
    """
    if bank is None:
        return {}
    first = render_block(_filtered_bank(bank), budget_tokens=budget_tokens)
    rest = render_block(bank, budget_tokens=budget_tokens)
    out: dict[int, str] = {}
    def _has_content(block: Any) -> bool:
        # `RetrievedBlock` exposes `skills`/`mistakes` lists, not counts -- reading
        # `n_skills` off it silently yields the getattr default and drops the block.
        if block is None:
            return False
        return bool(getattr(block, "skills", []) or getattr(block, "mistakes", []))

    if _has_content(first):
        out[0] = first.block_text
    if _has_content(rest):
        out[1] = rest.block_text
    return out


def bank_fingerprint(bank: Any, level_name: Optional[str] = None) -> frozenset[tuple[str, str]]:
    """Content identity of a bank: the (title, principle) pairs it holds.

    Compared across a round to detect an evolution that changed nothing. Ignores
    ids, confidence and generation on purpose -- those move even when the advice
    does not, so including them would mask exactly the case this exists to catch.
    """
    level_name = level_name or level()
    if bank is None:
        return frozenset()
    skills = list((getattr(bank, "level_skills", {}) or {}).get(level_name, []))
    return frozenset(
        (getattr(s, "title", ""), getattr(s, "principle", "")) for s in skills
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
    concurrency: int = 1,
    partial_path: Optional[Path] = None,
    carry_over: str = "",
) -> tuple[list[InstanceRun], list[InstanceRun]]:
    """Play a batch. Returns ``(completed, errored)``, in input order.

    ``concurrency`` runs that many instances at once. Reproducibility survives it
    because :data:`~hexa_orbit.adapter._RESET_LOCK` serialises the only part of the
    environment that touches the global RNG (maze generation); everything a
    concurrent thread does afterwards -- stepping, and the model call that dominates
    the wall clock -- is instance-local. See D5 for why this is a refinement of
    "run serially" rather than an abandonment of it.

    Results come back in **input order** regardless of completion order, so
    ``runs.jsonl`` is byte-comparable between a serial and a concurrent run of the
    same batch.

    ``partial_path`` appends each run **as it finishes**, which is crash safety for
    long runs: a 256-instance arm is ~2 hours at ``concurrency=8``, and a walltime
    kill or a server death at 90 minutes must not lose the work. The partial file is
    in *completion* order and is therefore not the artifact of record -- the caller
    writes the ordered ``runs.jsonl`` from the return value once the batch is done.
    Keeping the two separate is what lets crash safety and order stability coexist.

    An instance whose generation fails is recorded with ``error`` set and excluded
    from everything downstream -- a transport failure is evidence of nothing, and
    folding it in as a failure would depress the round's reward distribution for a
    reason unrelated to the agent.
    """
    leaked_ids = leaked_ids or set()
    partial_lock = threading.Lock()
    done_count = [0]
    if partial_path is not None:
        Path(partial_path).parent.mkdir(parents=True, exist_ok=True)
        # `carry_over` preserves rows from an earlier, interrupted pass; empty
        # truncates, which is what a fresh run wants.
        Path(partial_path).write_text(carry_over)

    def record_partial(run: InstanceRun) -> None:
        if partial_path is None:
            return
        # Serialised and flushed per run: an unflushed buffer is not crash safety.
        with partial_lock, Path(partial_path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(run.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()

    def report_progress(run: InstanceRun) -> None:
        """Call ``on_instance`` as each run lands, not after the whole batch.

        ``pool.map`` does not yield until every task is done, so reporting from the
        collection loop would print nothing for the hours a 256-instance arm takes
        and then everything at once. The counter is completion order, which is what
        a progress line wants; ordered results still come from the return value.
        """
        if on_instance is None:
            return
        with partial_lock:
            done_count[0] += 1
            position = done_count[0]
        on_instance(position, run)

    def play(spec: MazeInstanceSpec) -> InstanceRun:
        instance = task.instance(spec)
        # The instance is passed as well as the guidelines: a stub or oracle-driven
        # actor legitimately needs the live environment (it reads the map), and
        # letting it reach for one out of band led to `task.instance` being rebound
        # recursively. One explicit argument is cheaper than that.
        actor = actor_factory(guidelines, instance)
        try:
            run = actor.run_instance(instance, leaked=spec.instance_id in leaked_ids)
        except GenerationFailed as exc:
            logger.warning("instance %s errored: %s", spec.instance_id, exc)
            failed = InstanceRun(
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
            record_partial(failed)
            report_progress(failed)
            return failed
        record_partial(run)
        report_progress(run)
        return run

    specs = list(specs)
    if concurrency > 1 and len(specs) > 1:
        with ThreadPoolExecutor(max_workers=min(concurrency, len(specs))) as pool:
            runs = list(pool.map(play, specs))
    else:
        runs = [play(spec) for spec in specs]

    completed: list[InstanceRun] = []
    errored: list[InstanceRun] = []
    for run in runs:
        if run.error is not None:
            errored.append(run)
        else:
            completed.append(run)
            if budget is not None:
                budget.charge_train_solve(len(run.steps))

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
            level(),
            prev_bank,
            round_dir,
            output_path,
            max_skills=MAX_SKILLS,
            max_mistakes=MAX_MISTAKES,
            teacher_model=teacher_model,
        )

    # Whatever the task installed, not Maze's: `hexa_bridge.patch` rebinds
    # `load_trajectories` in both distillation consumers, and reading through that
    # binding is what keeps the seeding path on the same trajectories the evolver
    # will see. Hardcoding `trajectory_mod` here fed Mastermind runs through Maze's
    # position parser, which yields a turn table of "?" for every guess.
    from skillrl.distillation.distill import load_trajectories  # noqa: PLC0415

    successes, failures = load_trajectories(round_dir, level())
    bank = SkillBank()
    skills = distill_contrastive(level(), successes, failures, teacher_model=teacher_model)
    mistakes, partial = distill_mistakes(level(), failures, teacher_model=teacher_model)
    bank.level_skills[level()] = (skills + partial)[:MAX_SKILLS]
    bank.level_mistakes[level()] = mistakes[:MAX_MISTAKES]
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
    concurrency: int = 1,
    evolver_v2: bool = False,
    teacher_install: Optional[Callable[[], None]] = None,
    per_episode_injection: bool = False,
    inject_top_k: Optional[int] = None,
    resume: bool = True,
    manifest: Optional[dict[str, Any]] = None,
) -> HexaRunResult:
    """Run *rounds* rounds of *batch_size* instances each, then persist the bank."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Which task's prompts, formatter and trajectory loader the shared distillation
    # code runs with. Defaults to Maze so every existing caller is unchanged; a
    # different task passes its own. This was hardcoded, and because
    # `hexa_bridge.patch` rebinds `load_trajectories` in both consumers, a
    # Mastermind run that had already installed its own loader had it silently
    # replaced here -- the evolver then read the round with Maze's loader and
    # reported "0 successes, 0 failures" on a directory holding six runs.
    if teacher_install is not None:
        teacher_install()
    else:
        teacher_mod.install(reward_bin_selection=evolver_v2)

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
        block = render_block(
            bank, budget_tokens=injection_budget_tokens, top_k=inject_top_k
        ) if bank else None
        # A dict here is an episode-indexed injection (see
        # `render_blocks_by_episode`); `MazeActor` accepts either shape, and a
        # plain string reproduces the behaviour every completed run used.
        guidelines: Any
        if per_episode_injection and bank is not None:
            guidelines = render_blocks_by_episode(
                bank, budget_tokens=injection_budget_tokens
            ) or None
        else:
            guidelines = block.block_text if block else None

        n_before = len((getattr(bank, "level_skills", {}) or {}).get(level(), [])) if bank else 0
        teacher_mod.reset_confidence_stats()

        completed, errored = play_batch(
            task, batch, actor_factory, guidelines, budget=budget,
            concurrency=concurrency, partial_path=round_dir / "runs.partial.jsonl",
            on_instance=lambda i, r: logger.info(
                "round %d [%d/%d] %s R=%d/3", round_index, i, len(batch),
                r.instance_id, r.success_count),
        )
        trajectory_mod.write_runs(round_dir / "runs.jsonl", completed + errored)
        result.consumed_ids |= {s.instance_id for s in batch}

        # Evolve. bank.json is written by `evolve`, AFTER the round's runs exist.
        before_fingerprint = bank_fingerprint(bank)
        new_bank = evolve(
            round_dir, bank, round_index=round_index, teacher_model=teacher_model
        )
        bank_unchanged = (
            round_index > 0 and bank_fingerprint(new_bank) == before_fingerprint
        )
        if bank_unchanged:
            logger.warning(
                "round %d: evolution returned an IDENTICAL bank. Either the teacher "
                "had nothing to add, or its response failed to parse and "
                "evolve_skill_bank fell back to the previous bank. Check the round "
                "log for a teacher warning before reading this as convergence.",
                round_index,
            )

        # Admission: trial candidates on training instances outside this round.
        candidates = [
            s for s in (getattr(new_bank, "level_skills", {}) or {}).get(level(), [])
            if getattr(s, "is_new", True)
        ] if admission_enabled else []
        # The no-bank control on the same trial instances, computed once and
        # shared by every candidate this round. Without it the gate reduces to
        # "did three trials contain a success", which at this base rate admits
        # about two thirds of candidates on luck. Skipped entirely when the gate
        # is off, so an ungated run costs exactly what it did before.
        control_mean = None
        if admission_enabled and candidates:
            control_mean = admission_mod.control_mean_score(
                remaining[:admission_trials], task, actor_factory,
                trials=admission_trials, budget=budget, concurrency=concurrency,
            )
            logger.info("admission control (no bank): mean score %.3f", control_mean)

        admitted, admission_results = admit_skills(
            candidates,
            remaining[:admission_trials],
            task,
            actor_factory,
            render_block=lambda b: render_block(b, budget_tokens=injection_budget_tokens).block_text,
            trials=admission_trials,
            enabled=admission_enabled,
            budget=budget,
            control_mean=control_mean,
            # Whole-instance trials: a single-episode trial cannot express a
            # cross-episode skill, and episode 3 is the endpoint.
            whole_instance=True,
            concurrency=concurrency,
        )
        if admission_enabled:
            kept = {getattr(s, "skill_id", None) for s in admitted}
            surviving = [
                s for s in (getattr(new_bank, "level_skills", {}) or {}).get(level(), [])
                if not getattr(s, "is_new", True) or getattr(s, "skill_id", None) in kept
            ]
            new_bank.level_skills[level()] = surviving
            new_bank.save(round_dir / "bank.json")

        trajectories = [trajectory_mod.instance_trajectory(r) for r in completed]
        bins: dict[str, int] = {}
        for traj in trajectories:
            label = reward_bin(traj)
            bins[label] = bins.get(label, 0) + 1

        round_audit = audit_mod.audit_bank(
            new_bank,
            level_name=level(),
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
            n_skills_after=len((getattr(new_bank, "level_skills", {}) or {}).get(level(), [])),
            n_mistakes_after=len((getattr(new_bank, "level_mistakes", {}) or {}).get(level(), [])),
            admission=accept_rate(admission_results),
            audit=round_audit,
            injection={k: v for k, v in injection.items() if k != "block_text"},
            confidence_stats=teacher_mod.confidence_stats(),
            n_errored=len(errored),
            bank_unchanged=bank_unchanged,
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
    concurrency: int = 1,
    per_episode_injection: bool = False,
    inject_top_k: Optional[int] = None,
) -> dict[str, Any]:
    """Evaluate on held-out instances with the bank frozen -- no writes, ever.

    ``bank=None`` is the no-bank arm, so the two arms differ only in the injected
    block. Nothing here calls ``evolve`` or ``bank.save``, and the runs are written
    to their own directory so they cannot be picked up by a later
    ``load_instance_trajectories`` on a round directory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    block = render_block(
        bank, budget_tokens=injection_budget_tokens, top_k=inject_top_k
    ) if bank else None
    # A dict here is an episode-indexed injection (see `render_blocks_by_episode`);
    # `MazeActor` accepts either shape, and a plain string reproduces the behaviour
    # every completed run used.
    guidelines: Any
    if per_episode_injection and bank is not None:
        guidelines = render_blocks_by_episode(
            bank, budget_tokens=injection_budget_tokens
        ) or None
    else:
        guidelines = block.block_text if block else None
    # Maze-only: 19% of Maze eval layouts also appear in the train pool, which is
    # tolerated and reported as a split. Mastermind pools are disjoint by
    # construction -- the code *is* the instance, so a shared code would be the
    # answer, and `mastermind.assert_disjoint` refuses it outright. Guarded on the
    # field rather than on a task name so a new task gets the safe default.
    leaked_ids: set[str] = set()
    if train_specs and hasattr(eval_specs[0], "maze_hash"):
        leaked_ids = {
            s.instance_id
            for s in leakage_split(eval_specs, list(train_specs))["leaked"]
        }

    completed, errored = play_batch(
        task, eval_specs, actor_factory, guidelines,
        leaked_ids=leaked_ids, budget=budget, concurrency=concurrency,
        partial_path=out_dir / "runs.partial.jsonl",
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
