"""Minesweeper and Wordle runs as ``skillrl`` trajectory dicts, for the evolver.

Field-for-field identical to :mod:`hexa_orbit.mastermind_trajectory` so the shared
distillation code needs no branch; what differs is what a turn *contains* and how
its result reads back out of the next observation.

- A Minesweeper turn is ``(cell, what it revealed)`` -- a count, a cascade, or a
  mine. The count is the entire deductive signal, and it is stated in the board
  render rather than in prose, so the result is recovered from the rendered board
  rather than from a sentence.
- A Wordle turn is ``(word, G/Y/X mask)``. The mask is stated verbatim in the
  next observation, so it needs no reconstruction -- the closest of the three
  tasks to Mastermind's peg line.

Both are registered by level so ``hexa_bridge.patch(load_trajectories_fn=...)``
can be pointed at one loader that serves either.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence

from .actor import InstanceRun
from .reward import reward_from_parts
from .trajectory import first_success_episode

#: ``"At turn 3, you successfully revealed cell (1, 4)."``
_MS_REVEALED_RE = re.compile(
    r"At turn (\d+), you successfully revealed cell \((\d+), (\d+)\)", re.I
)
_MS_MINE_RE = re.compile(r"you hit a mine at \((\d+), (\d+)\)", re.I)
_MS_BOARD_RE = re.compile(r"Here is the current board layout:\n(.+?)\nEnter your guess", re.S)

#: ``"At turn 2, you guessed CRANE\nFeedback:\nCRANE\nXYGXX"``
_WD_FEEDBACK_RE = re.compile(
    r"At turn (\d+), you guessed ([A-Z]+)\s*\nFeedback:\s*\n[A-Z]+\s*\n([GYX]+)", re.I
)


def _board_after(observation: str) -> Optional[list[str]]:
    """The rendered board carried in every Minesweeper observation's suffix."""
    match = _MS_BOARD_RE.search(str(observation or ""))
    if not match:
        return None
    return [row for row in match.group(1).splitlines() if row.strip()]


def _hidden_count(observation: str) -> Optional[int]:
    """Cells still showing ``.`` -- how much board the agent has left to clear.

    The one number that summarises a Minesweeper turn's *value*: a reveal that
    cascades drops it by many, a reveal at a wall of numbers drops it by one.
    Without it every successful reveal renders identically and the teacher cannot
    tell a productive probe from a wasted one.
    """
    board = _board_after(observation)
    if board is None:
        return None
    return sum(row.count(".") for row in board)


def minesweeper_outcome(observation: str, turn: Optional[int] = None,
                        previous: Optional[str] = None) -> str:
    text = str(observation or "")
    low = text.lower()
    if "successfully cleared the minesweeper board" in low:
        return "CLEARED"
    if _MS_MINE_RE.search(text):
        return "HIT A MINE"
    if "which has already been revealed" in low:
        return "wasted (already revealed)"
    if "outside the bounds" in low:
        return "wasted (off board)"
    if "did not provide a valid guess" in low:
        return "unparsed -- episode ended"
    if "invalid action" in low:
        return "wasted (not reveal/flag)"
    if "maximum number of turns" in low:
        return "out of turns"
    if _MS_REVEALED_RE.search(text):
        before, after = _hidden_count(previous or ""), _hidden_count(text)
        if before is not None and after is not None and after <= before:
            opened = before - after
            return f"safe, opened {opened} cell(s), {after} still hidden"
        return "safe"
    return "?"


def wordle_outcome(observation: str, turn: Optional[int] = None,
                   previous: Optional[str] = None) -> str:
    text = str(observation or "")
    low = text.lower()
    if "you guessed the secret word" in low:
        return "SOLVED"
    if "has been already guessed before" in low:
        return "wasted (repeat of an earlier guess)"
    if "letters but the secret word has" in low:
        return "wasted (wrong length)"
    if "did not provide a valid guess" in low:
        return "unparsed -- episode ended"
    if "maximum number of turns" in low:
        return "out of turns"
    # Every feedback line names its own turn; at an episode boundary the
    # observation holds the finished episode's last feedback *and* the next
    # episode's opening text, so matching the first line blind attributes the
    # wrong turn. Mastermind hit this exact bug.
    if turn is not None:
        anchored = re.search(
            rf"At turn {int(turn)}, you guessed ([A-Z]+)\s*\nFeedback:\s*\n[A-Z]+\s*\n([GYX]+)",
            text, re.I,
        )
        if anchored:
            return anchored.group(2)
    match = _WD_FEEDBACK_RE.search(text)
    if match:
        return match.group(3)
    return "?"


_OUTCOMES = {"minesweeper": minesweeper_outcome, "wordle": wordle_outcome}

#: Minesweeper reveals a cell; Wordle names a word. The column header the teacher
#: reads should say which.
_ACTION_LABEL = {"minesweeper": "revealed", "wordle": "guessed"}


def _render_action(action: Any) -> str:
    text = str(action or "").strip()
    return text.replace("\\boxed{", "").rstrip("}") or "(none)"


def step_entries(run: InstanceRun, level: str, *, include_reasoning: bool = False
                 ) -> list[dict[str, Any]]:
    """One entry per agent turn, with the outcome it drew.

    A turn's result arrives in the *next* observation, as it does for Maze and
    Mastermind, so the entries are zipped against their successors and the last
    turn takes the trailing message.
    """
    outcome_of = _OUTCOMES[level]
    entries: list[dict[str, Any]] = []
    for step in run.steps:
        entries.append({
            "thought": step.reasoning_text if include_reasoning else "",
            "action": step.parsed_action,
            "observation": step.observation,
            "episode": step.episode_index + 1,
            "turn": step.episode_step,
            "truncated": step.truncated,
        })
    for entry, following in zip(entries, entries[1:]):
        entry["result"] = outcome_of(following["observation"], entry["turn"],
                                     entry["observation"])
    if entries:
        terminal = run.messages[-1]["content"] if run.messages else ""
        entries[-1]["result"] = outcome_of(terminal, entries[-1]["turn"],
                                           entries[-1]["observation"])
    return entries


def instance_reward(run: InstanceRun) -> float:
    """Reward on the shared Maze scale, so outcome bins mean the same thing.

    ``shortest_path_len`` is the spec's ``min_turns`` -- safe cells left after the
    pinned opening for Minesweeper, 1 for Wordle -- playing the role Maze's
    shortest route and Mastermind's code length play: the least the agent could
    have got away with.
    """
    progress = max(run.episode_progress, default=0.0)
    return reward_from_parts(
        first_success_episode=first_success_episode(run.episode_successes),
        episode_1_steps=(list(run.episode_lengths) or [None])[0],
        shortest_path_len=int(run.shortest_path_len or 1),
        adapted=run.adapted_across_episodes,
        best_progress=progress if not any(run.episode_successes) else None,
    )


def instance_trajectory(run: InstanceRun, level: str, *,
                        include_reasoning: bool = False) -> dict[str, Any]:
    entries = step_entries(run, level, include_reasoning=include_reasoning)
    return {
        "seed": run.instance_id,
        "success": run.success_count > 0,
        "iterations": len(run.steps),
        "trajectory": entries,
        "action": entries[-1]["action"] if entries else "",
        "reward": instance_reward(run),
        "task_name": level,
        "episode_successes": list(run.episode_successes),
        "episode_lengths": list(run.episode_lengths),
        "success_count": run.success_count,
        "first_success_episode": first_success_episode(run.episode_successes),
        "adapted_across_episodes": run.adapted_across_episodes,
        "best_progress": max(run.episode_progress, default=0.0),
        "episode_progress": list(run.episode_progress),
        "shortest_path_len": run.shortest_path_len,
        "leaked": False,
        "n_truncated": sum(1 for s in run.steps if s.truncated),
        "n_unparsed": sum(1 for s in run.steps if not s.parsed_action),
        "num_episodes": len(run.episode_successes),
    }


def split_successes_failures(
    trajectories: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    successes = [t for t in trajectories if t.get("success")]
    return successes, [t for t in trajectories if not t.get("success")]


def loader_for(level: str):
    """A ``load_instance_trajectories`` bound to *level*, for ``hexa_bridge.patch``.

    ``runs.jsonl`` only, never a glob: a round directory also holds
    ``runs.partial.jsonl`` with the same runs, and globbing both doubled every
    trajectory in an earlier Maze run.
    """

    def load_instance_trajectories(
        traj_dir: str | Path,
        task_name: Optional[str] = None,
        *,
        include_reasoning: bool = False,
        thought_max_chars: int = 6000,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        path = Path(traj_dir) / "runs.jsonl"
        if not path.exists():
            print(f"[Warning] {path} not found")
            return [], []
        runs = [InstanceRun.from_dict(json.loads(line))
                for line in path.read_text().splitlines() if line.strip()]
        trajectories = [instance_trajectory(r, level, include_reasoning=include_reasoning)
                        for r in runs]
        return split_successes_failures(trajectories)

    return load_instance_trajectories
