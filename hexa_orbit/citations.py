"""Check that a skill's cited evidence actually exists.

Rule 6 of the maze evolution prompt requires every skill to name the instance and
turn at which it would have changed the agent's action. The intent was to make a
skill provably grounded in a transcript rather than plausible-sounding. Measured
across every run to date, **14 of 24 such citations (58%) were fabricated** -- the
named turn either did not exist or the agent chose a different action there. One
example: "Instance 49289a818872a592, turn 1: agent chose 'right'", where the agent
actually chose `up`.

So the gate was an honour system, and the majority of skills passed it by
confabulation. This module makes it checkable: the transcripts are already in
memory when the evolver runs, and verifying a claimed ``(instance, turn, action)``
is a dictionary lookup.

**Conservative by design.** Only a citation that is *demonstrably false* rejects
its skill. A missing or unparseable citation is reported and kept, because the
alternative -- rejecting everything we cannot read -- would silently empty the bank
on a formatting change in the teacher's output, and would remove skills that may
be perfectly good. This direction can only ever delete a skill whose stated
evidence contradicts the recorded run.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

#: A 16-hex-character instance id, as `runs.jsonl` and the teacher block use.
_INSTANCE_RE = re.compile(r"\b([0-9a-f]{16})\b")

#: "turn 4", "turn: 4", "at turn 4".
_TURN_RE = re.compile(r"\bturn\s*:?\s*(\d+)", re.IGNORECASE)

#: The action the teacher claims the agent took. Taken as the FIRST quoted move
#: word: the field's shape is "agent chose 'left' vs skill would choose 'down'",
#: so the first is the claim about the transcript and the second is the
#: counterfactual, which by definition is not in the transcript and must not be
#: checked against it.
_MOVES = ("up", "down", "left", "right")
_QUOTED_MOVE_RE = re.compile(
    r"['\"`]\s*(" + "|".join(_MOVES) + r")\s*['\"`]", re.IGNORECASE
)


@dataclass(frozen=True)
class Citation:
    """What a skill claimed about a transcript."""

    instance_id: Optional[str]
    turn: Optional[int]
    action: Optional[str]

    @property
    def complete(self) -> bool:
        return None not in (self.instance_id, self.turn, self.action)


def parse_citation(text: Any) -> Citation:
    """Pull ``(instance, turn, action)`` out of a ``changes_action`` string."""
    if not text:
        return Citation(None, None, None)
    body = str(text)
    instance = _INSTANCE_RE.search(body)
    turn = _TURN_RE.search(body)
    action = _QUOTED_MOVE_RE.search(body)
    return Citation(
        instance.group(1) if instance else None,
        int(turn.group(1)) if turn else None,
        action.group(1).lower() if action else None,
    )


def _actions_by_turn(trajectory: Mapping[str, Any]) -> dict[int, str]:
    """Episode-1 actions keyed by turn.

    Episode 1 only: the teacher is asked for the turn an agent "would have chosen
    differently", and every citation observed names a first-episode turn. Checking
    against all episodes would let a turn number match the wrong episode and pass a
    citation that is wrong about the one it meant.
    """
    out: dict[int, str] = {}
    for entry in trajectory.get("trajectory") or []:
        if int(entry.get("episode", 1)) != 1:
            continue
        action = entry.get("action")
        if action is None:
            continue
        out[int(entry.get("turn", 0))] = str(action).strip().lower()
    return out


def verify(
    text: Any, trajectories: Sequence[Mapping[str, Any]]
) -> tuple[bool, str]:
    """``(ok, reason)`` for one citation against the round's trajectories.

    ``ok`` is False only when the claim contradicts a transcript we hold. Anything
    we cannot check -- no citation, unparseable, an instance not in this round --
    returns True with a reason saying so.
    """
    citation = parse_citation(text)
    if not citation.complete:
        return True, "unverifiable: citation incomplete"

    by_id = {str(t.get("seed", "")): t for t in trajectories}
    trajectory = by_id.get(citation.instance_id)
    if trajectory is None:
        return True, f"unverifiable: instance {citation.instance_id} not in this round"

    actions = _actions_by_turn(trajectory)
    if not actions:
        return True, "unverifiable: no per-turn record for that instance"
    actual = actions.get(citation.turn)
    if actual is None:
        return False, (
            f"FALSE: {citation.instance_id} episode 1 has no turn {citation.turn} "
            f"(it ran {len(actions)} turns)"
        )
    if actual != citation.action:
        return False, (
            f"FALSE: {citation.instance_id} turn {citation.turn} was "
            f"{actual!r}, not {citation.action!r}"
        )
    return True, f"verified: {citation.instance_id} turn {citation.turn} = {actual!r}"


def filter_fabricated(
    raw_skills: Sequence[Mapping[str, Any]],
    trajectories: Sequence[Mapping[str, Any]],
    *,
    field: str = "changes_action",
) -> tuple[list[Mapping[str, Any]], list[tuple[str, str]]]:
    """``(kept, [(title, reason), ...])``. Drops only demonstrably false claims."""
    kept: list[Mapping[str, Any]] = []
    notes: list[tuple[str, str]] = []
    for raw in raw_skills:
        ok, reason = verify(raw.get(field), trajectories)
        notes.append((str(raw.get("title", "?")), reason))
        if ok:
            kept.append(raw)
    return kept, notes
