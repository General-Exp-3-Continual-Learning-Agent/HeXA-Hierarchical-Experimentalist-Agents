"""Mastermind runs as ``skillrl`` trajectory dicts, for the evolver.

The Maze analogue is :mod:`hexa_orbit.trajectory`. The shapes match field for
field so the shared distillation code needs no branch; what differs is what a
"turn" contains. A maze turn is ``(position, action, outcome)`` -- where the agent
stood, where it moved, what happened. A Mastermind turn is ``(guess, black,
white)`` -- what it named and how much of it was right. Those pegs are the whole
observable signal, and unlike maze positions they are stated in the feedback text
rather than needing to be carried forward, so there is no equivalent of
``position_carried`` here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .actor import InstanceRun
from .mastermind import adapted_across_episodes, episode_progress
from .reward import reward_from_parts
from .trajectory import first_success_episode

#: ``"At turn 1, you guessed [1, 2, 3, 4]. This guess receives 0 black peg(s) and
#: 2 white peg(s)."`` -- the only feedback the env gives, so the only thing a
#: transferable skill can key on.
_FEEDBACK_RE = re.compile(
    r"you guessed \[([^\]]*)\].*?receives\s+(\d+)\s+black peg\(s\)\s+and\s+"
    r"(\d+)\s+white peg\(s\)",
    re.IGNORECASE | re.DOTALL,
)

#: A guess the env rejected -- wrong length, unparseable, or a repeat. Worth a
#: distinct label: re-naming a code already excluded by its own feedback is the
#: exact failure the consistency rule addresses, and the teacher cannot see it if
#: every non-solve renders the same way.
_REJECTED_RE = re.compile(r"which has \d+ entries but the code has length", re.I)


def pegs_in(observation: str, turn: Optional[int] = None) -> Optional[tuple[str, int, int]]:
    """``(guess, black, white)`` from a feedback line, or ``None``.

    ``turn`` selects the line for that specific guess. It matters at an episode
    boundary, where the observation is the finished episode's last feedback
    concatenated with the next episode's opening text: matching the first peg line
    there attributes the wrong turn's result, and matching the last one attributes
    nothing, since the opening text has no pegs at all. Every feedback line names
    its own turn, so ask for it.
    """
    text = str(observation or "")
    if turn is not None:
        anchored = re.search(
            rf"At turn {int(turn)}, you guessed \[([^\]]*)\].*?receives\s+(\d+)\s+"
            rf"black peg\(s\)\s+and\s+(\d+)\s+white peg\(s\)",
            text, re.IGNORECASE | re.DOTALL,
        )
        if anchored:
            return anchored.group(1).strip(), int(anchored.group(2)), int(anchored.group(3))
    match = _FEEDBACK_RE.search(text)
    if not match:
        return None
    return match.group(1).strip(), int(match.group(2)), int(match.group(3))


def outcome_of(observation: str, turn: Optional[int] = None) -> str:
    """Short label for what a turn produced."""
    text = str(observation or "")
    if "you guessed the correct code" in text.lower() or "congratulations" in text.lower():
        return "SOLVED"
    if _REJECTED_RE.search(text):
        return "rejected"
    # The env answers an exhausted episode with this and no pegs, so the final
    # guess of a failed episode would otherwise render as "?" -- losing the one
    # turn that most shows how close the agent got before the budget ran out.
    if "maximum number of turns" in text.lower():
        return "out of guesses"
    parsed = pegs_in(text, turn)
    if parsed is not None:
        _guess, black, white = parsed
        return f"{black}B/{white}W"
    return "?"


def step_entries(
    run: InstanceRun, *, include_reasoning: bool = False
) -> list[dict[str, Any]]:
    """One entry per guess, in order, with the feedback it drew."""
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
    # A guess's feedback arrives in the *next* observation, exactly as a maze move's
    # result does; the last guess's feedback is the trailing message.
    for entry, following in zip(entries, entries[1:]):
        entry["result"] = outcome_of(following["observation"], entry["turn"])
    if entries:
        terminal = run.messages[-1]["content"] if run.messages else ""
        entries[-1]["result"] = outcome_of(terminal, entries[-1]["turn"])
    return entries


def instance_reward(run: InstanceRun) -> float:
    """Reward for one instance, on the shared Maze scale.

    Reuses :func:`hexa_orbit.reward.reward_from_parts` so the outcome bins mean the
    same thing across tasks: solved-on-episode-1 at the top, never-solved-and-never
    -varied at the floor. ``shortest_path_len`` maps to the code length -- both are
    "the least the agent could have got away with".
    """
    progress = max(run.episode_progress, default=0.0)
    return reward_from_parts(
        first_success_episode=first_success_episode(run.episode_successes),
        episode_1_steps=(list(run.episode_lengths) or [None])[0],
        shortest_path_len=int(run.shortest_path_len or 4),
        adapted=run.adapted_across_episodes,
        best_progress=progress if not any(run.episode_successes) else None,
    )


def instance_trajectory(
    run: InstanceRun, *, include_reasoning: bool = False
) -> dict[str, Any]:
    entries = step_entries(run, include_reasoning=include_reasoning)
    return {
        "seed": run.instance_id,
        "success": run.success_count > 0,
        "iterations": len(run.steps),
        "trajectory": entries,
        "action": entries[-1]["action"] if entries else "",
        "reward": instance_reward(run),
        "task_name": "mastermind",
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


def load_instance_trajectories(
    traj_dir: str | Path,
    task_name: Optional[str] = None,
    *,
    include_reasoning: bool = False,
    thought_max_chars: int = 6000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A round's ``runs.jsonl`` as ``(successes, failures)``.

    Signature-compatible with :func:`hexa_orbit.trajectory.load_instance_trajectories`
    so ``hexa_bridge.patch(load_trajectories_fn=...)`` can swap one for the other.
    ``runs.jsonl`` only, never ``*.jsonl`` -- a round directory also holds
    ``runs.partial.jsonl`` with the same runs, and globbing both doubled every
    trajectory in an earlier Maze run.
    """
    path = Path(traj_dir) / "runs.jsonl"
    if not path.exists():
        print(f"[Warning] {path} not found")
        return [], []
    runs = [InstanceRun.from_dict(json.loads(line)) for line in
            path.read_text().splitlines() if line.strip()]
    trajectories = [instance_trajectory(r, include_reasoning=include_reasoning)
                    for r in runs]
    return split_successes_failures(trajectories)
