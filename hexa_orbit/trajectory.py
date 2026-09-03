"""One Maze instance as the trajectory dict ``skillrl`` consumes.

``skillrl``'s distil/evolve path takes trajectory dicts of a fixed shape
(``seed`` / ``success`` / ``iterations`` / ``trajectory`` / ``action``).
:func:`instance_trajectory` emits exactly that for a whole three-episode
instance, which is Design B's unit of reflection: a skill like "if episode 1
showed the left corridor dead-ends, don't re-walk it in episode 2" is only
expressible when the evolver sees all three episodes at once.

Two contract details, both inherited rather than rediscovered:

* ``seed`` carries a **string** instance id. ``Skill.source_seeds`` is annotated
  ``list[int]`` but never coerced anywhere, and ``_compute_initial_confidence``
  matches by plain dict lookup, so strings round-trip -- ``hexa_core`` relies on
  the same thing for CORE's problem ids. The annotation is what is wrong, not the
  data.
* ``iterations`` is the total step count across all three episodes. It exists so
  that ``skillrl``'s stock ``compute_trajectory_reward``, if it ever runs
  unpatched, lands somewhere sane instead of raising -- its thresholds are still
  wrong for this env, which is why :mod:`hexa_orbit.reward` replaces it.

The per-step ``observation`` is the *environment's* reply, and ``action`` is the
parsed direction. ``thought`` carries the model's reasoning only when
``include_reasoning`` is set: a 27-turn maze instance with full chain-of-thought
swamps the teacher's context and buries the coordinate sequence, which is where a
maze trajectory's information actually lives (see
:mod:`hexa_orbit.teacher`).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .actor import InstanceRun
from .adapter import EPISODES_PER_INSTANCE

#: Pulls the agent's position out of an observation. Both the opening
#: instruction and every move reply contain exactly one of these
#: (``envs/maze_env_adapter.py:_generate_observation``), and on an episode
#: boundary the concatenated observation contains the next episode's.
POSITION_RE = re.compile(r"you are at position \((\d+),\s*(\d+)\)", re.IGNORECASE)

#: The four fixed outcome strings the maze env can reply with, mapped to a short
#: label. Order matters: "arrived at the goal" must be tested before the generic
#: "You move to", since the goal reply contains neither.
_OUTCOMES: tuple[tuple[str, str], ...] = (
    ("arrived at the goal", "GOAL"),
    ("hit the wall", "wall"),
    ("Invalid action", "invalid"),
    ("maximum number of turns", "budget exhausted"),
    ("You move to", "moved"),
)


def position_in(text: str) -> Optional[tuple[int, int]]:
    """First ``(x, y)`` in *text*, or ``None``."""
    match = POSITION_RE.search(text)
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)))


def outcome_of(observation: str) -> str:
    """Short label for what the env did with an action."""
    for needle, label in _OUTCOMES:
        if needle in observation:
            return label
    return "?"


def first_success_episode(episode_successes: Sequence[bool]) -> Optional[int]:
    """1-based index of the first solved episode, or ``None``.

    The quantity the reward is keyed on (:mod:`hexa_orbit.reward`): Design B's
    claim is about a head start at episode 1, so *when* the instance was first
    solved carries the signal that a success count does not.
    """
    for index, success in enumerate(episode_successes, start=1):
        if success:
            return index
    return None


def step_entries(run: InstanceRun, *, include_reasoning: bool = False) -> list[dict[str, Any]]:
    """The ``trajectory`` list: one entry per turn, in order.

    ``position`` is the cell the agent stood in *before* acting, read from the
    observation it was answering -- so ``(position, action, outcome)`` reads as a
    move, which is what a maze skill is made of.
    """
    entries: list[dict[str, Any]] = []
    carried: Optional[tuple[int, int]] = None
    for step in run.steps:
        # A wall bump and an invalid action both reply without any coordinates
        # ("You hit the wall. Please choose another direction.",
        # `envs/maze_env_adapter.py:_execute_action`), and neither moves the agent
        # -- `current_position` is left untouched on both paths. So the last known
        # cell is still the true cell, and carrying it forward is a fact about the
        # env rather than an inference. Without this, a failing episode renders as
        # a column of "?" exactly where the teacher most needs to see where the
        # agent was stuck.
        stated = position_in(step.observation)
        if stated is not None:
            carried = stated
        entries.append(
            {
                "thought": step.reasoning_text if include_reasoning else "",
                "action": step.parsed_action,
                "observation": step.observation,
                # additive, consumed by the formatter
                "episode": step.episode_index + 1,
                "turn": step.episode_step,
                "position": stated if stated is not None else carried,
                "position_carried": stated is None and carried is not None,
                "outcome": outcome_of(step.observation),
                "truncated": step.truncated,
            }
        )
    # Each step's *result* is the next step's observation; the final step's result
    # is the terminal observation, which the actor keeps as the trailing message.
    for entry, following in zip(entries, entries[1:]):
        entry["result"] = outcome_of(following["observation"])
        entry["result_position"] = following["position"]
    if entries:
        terminal = run.messages[-1]["content"] if run.messages else ""
        entries[-1]["result"] = outcome_of(terminal)
        entries[-1]["result_position"] = position_in(terminal)
    return entries


def instance_trajectory(
    run: InstanceRun, *, include_reasoning: bool = False
) -> dict[str, Any]:
    """One :class:`~hexa_orbit.actor.InstanceRun` as a ``skillrl`` trajectory dict.

    ``success`` is "solved at all" (``success_count > 0``), matching ORBIT's own
    ``is_correct``. That is the binary ``distill_contrastive`` splits on, and
    solved-vs-never-solved is the informative contrast on this env; the finer
    fast-vs-slow distinction rides in ``reward`` and is visible in the rendered
    transcript, so nothing is lost by not encoding it here.
    """
    from .reward import instance_reward  # local: avoids an import cycle

    entries = step_entries(run, include_reasoning=include_reasoning)
    return {
        # --- skillrl's contract ---
        "seed": run.instance_id,
        "success": run.success_count > 0,
        "iterations": len(run.steps),
        "trajectory": entries,
        "action": entries[-1]["action"] if entries else "",
        # --- additive: our reward, formatter and audit read these ---
        "reward": instance_reward(run),
        "task_name": "maze",
        "episode_successes": list(run.episode_successes),
        "episode_lengths": list(run.episode_lengths),
        "success_count": run.success_count,
        "first_success_episode": first_success_episode(run.episode_successes),
        "adapted_across_episodes": run.adapted_across_episodes,
        "best_progress": run.best_progress,
        "episode_progress": list(run.episode_progress),
        "shortest_path_len": run.shortest_path_len,
        "num_episodes": EPISODES_PER_INSTANCE,
        "leaked": run.leaked,
        "n_unparsed": run.n_unparsed,
        "n_truncated": run.n_truncated,
        "added_tokens": run.added_tokens,
    }


def split_successes_failures(
    runs: Iterable[InstanceRun], *, include_reasoning: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(successes, failures)`` -- the contract every distil pass takes."""
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for run in runs:
        if run.error is not None:
            continue  # errored instances are evidence of nothing
        trajectory = instance_trajectory(run, include_reasoning=include_reasoning)
        (successes if trajectory["success"] else failures).append(trajectory)
    return successes, failures


# ── persistence: the whole `evolve_skill_bank` integration ───────────────

def write_runs(path: str | Path, runs: Iterable[InstanceRun]) -> Path:
    """Write runs as JSON Lines, one per instance, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(json.dumps(run.to_dict(), ensure_ascii=False) + "\n")
    return path


def read_runs(path: str | Path) -> list[InstanceRun]:
    """Read a ``runs.jsonl`` back into :class:`InstanceRun` objects."""
    out: list[InstanceRun] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(InstanceRun.from_dict(json.loads(line)))
    return out


def load_instance_trajectories(
    traj_dir: str | Path,
    task_name: str | None = None,
    *,
    include_reasoning: bool = False,
    thought_max_chars: int = 6000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load a round's ``runs.jsonl`` as ``(successes, failures)`` trajectory dicts.

    **This function is the entire evolver integration.** ``evolve_skill_bank`` takes
    a *directory* and calls ``load_trajectories`` on it; binding this in through
    ``hexa_bridge.patch(load_trajectories_fn=...)`` means writing a round's runs to
    disk is all that is needed -- no format conversion and no fork of
    ``evolving_distill``. ``hexa_core.trajectory.load_attempt_trajectories`` does
    the same job for CORE, and this mirrors its signature so the two are
    interchangeable at that seam.

    Signature-compatible with ``distill.load_trajectories``, which globs
    ``trajectory_seed*.json``; this globs ``*.jsonl``. A missing directory warns and
    returns empty lists, matching the original rather than raising -- a round that
    produced no runs should skip evolution, not crash the loop.

    ``thought_max_chars`` is accepted for signature compatibility and clips the
    rendered reasoning when ``include_reasoning`` is set. It has no effect by
    default, because reasoning is elided (see :mod:`hexa_orbit.teacher`).
    """
    traj_dir = Path(traj_dir)
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if not traj_dir.exists():
        print(f"[Warning] Run directory not found: {traj_dir}")
        return successes, failures

    # `runs.jsonl` only, NOT `*.jsonl`. A round directory also holds
    # `runs.partial.jsonl` -- the crash-safety file written per instance as it lands
    # -- which contains the same runs. Globbing `*.jsonl` loaded all 20 instances
    # TWICE (measured: 16 successes + 24 failures from 20 instances), so the
    # evolver's 6 slots could be filled with duplicate transcripts and the reported
    # "of N available" counts were doubled. Confidences were unaffected, since
    # `_compute_initial_confidence` dedups by seed.
    candidates = sorted(traj_dir.glob("runs.jsonl"))
    if not candidates:
        # A round interrupted before its ordered file was written still has its
        # partial rows; better to evolve from those than from nothing.
        candidates = sorted(traj_dir.glob("runs.partial.jsonl"))
    for file in candidates:
        for run in read_runs(file):
            if run.error is not None:
                continue
            traj = instance_trajectory(run, include_reasoning=include_reasoning)
            if include_reasoning and thought_max_chars > 0:
                for entry in traj["trajectory"]:
                    if len(entry["thought"]) > thought_max_chars:
                        entry["thought"] = entry["thought"][:thought_max_chars] + " ..."
            if task_name is not None and traj.get("task_name") != task_name:
                continue
            (successes if traj["success"] else failures).append(traj)

    return successes, failures

