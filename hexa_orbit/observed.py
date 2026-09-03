"""Cross-instance aggregates over what the *agent* observed.

Why this module exists. :func:`hexa_orbit.teacher.format_trajectory_for_teacher`
renders each instance as an independent turn table, so the evolver is asked for
cross-instance generalisations while being shown only within-instance case
studies. An invariant of the generator -- anything true of *every* maze rather
than of one -- is not visible in six separate transcripts, and the measured
consequence is a bank of five paraphrases of the one sentence the system prompt
already contains (v2 rounds 3-5: 0/4 of ``audit.STRUCTURAL_REGULARITIES`` present,
5/5 skills restating "use earlier episodes").

**The privilege boundary, which is the whole point of the module.** Every
quantity here is computed from ``steps[i]["observation"]`` and the agent's own
actions -- text the agent read and choices it made. Nothing reads the generator,
the config, or instance metadata the agent never saw. Concretely:

* ALLOWED: positions the agent stood on, what it was told lay in each direction,
  which action it chose, whether its position changed, how many turns it used,
  whether an episode ended solved.
* FORBIDDEN, and deliberately absent: ``shortest_path_len`` (the config's
  path-length filter), grid dimensions, and anything derived from
  ``envs/maze_env_adapter.py``. A skill founded on those is ours, not learned,
  and its lift would be manufactured.

Aggregating the agent's own observations across instances is *not* privilege --
it is induction from experience, which is what a teacher is for. Legitimacy comes
from train/eval instance disjointness (``pools.assert_disjoint``), not from the
teacher being kept ignorant of its own transcripts.

**Why the statistic list is mechanical rather than curated.** Choosing *which*
aggregate to compute is itself a leak channel: nobody measures "displacement from
start to goal" unless they already know the generator places the goal at the
farthest reachable cell, and a skill derived from a statistic chosen that way is
undetectably ours -- it would cite real ``source_seeds`` and look earned. So the
quantities below are enumerated from the *record schema*: every observable field,
summarised the obvious way for its type, with no quantity promoted or omitted on
the grounds that it happens to be the interesting one. :data:`QUANTITIES`
documents that list so a reviewer can check it against the schema rather than
against intuition.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

#: The four move words this env accepts, from the observation's own prompt line
#: ("Output your next move from up/down/left/right").
DIRECTIONS = ("up", "down", "left", "right")

#: ``"Now you are at position (5, 3) in the maze."``
_POS_RE = re.compile(r"position \((\d+)\s*,\s*(\d+)\)")

#: ``"up leads to wall, down leads to wall, left leads to path, ..."`` -- the
#: local structure the agent is handed each turn. The vocabulary is not assumed:
#: whatever word follows "leads to" is counted under that word.
_DIR_RE = re.compile(
    r"\b(" + "|".join(DIRECTIONS) + r")\b leads to (\w+)", re.IGNORECASE
)

#: The quantities computed, each named with the schema field it comes from. Kept
#: as data so a reviewer can audit the list against the record schema -- see the
#: module docstring on why the *choice* of statistic is a leak channel.
QUANTITIES: tuple[tuple[str, str], ...] = (
    ("turns_per_episode", "len(steps) grouped by episode_index"),
    ("solved_per_episode", "steps[-1].reward > 0 within each episode_index"),
    ("action_counts", "parsed_action"),
    ("action_blocked_rate", "parsed_action vs whether the parsed position changed"),
    ("cells_visited", "distinct positions parsed from observation"),
    ("revisit_rate", "repeated positions within one episode"),
    ("surroundings_seen", "the word after 'leads to', per direction"),
    ("start_cell", "position at the first step of each episode"),
    ("final_cell", "position at the last step of each episode"),
    ("net_displacement", "|final - start| per episode, the obvious two-point summary"),
    ("coord_extent", "min/max of every parsed coordinate"),
    ("goal_sighting_turn", "first turn whose observation reports a direction as 'goal'"),
    ("unparsed_turns", "parsed_action is None"),
    ("first_divergence_turn", "first turn where two episodes chose differently"),
    ("new_cells_vs_first_episode", "positions in a later episode absent from episode 1"),
)


def _entries(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The per-turn entries, preferring the normalised form.

    Two shapes carry the same turns. ``record["trajectory"]`` is what
    ``load_instance_trajectories`` produces and what the teacher is shown, with
    ``position`` already parsed and ``result`` already labelled
    (``moved``/``invalid``/``GOAL``); ``record["steps"]`` is the raw env log kept
    in ``runs.jsonl``, where the same facts must be re-parsed out of observation
    text and ``parsed_action`` sometimes holds a whole paragraph of reasoning
    instead of a move. Prefer the normalised form -- computing from it means these
    aggregates describe exactly the transcripts the teacher reads.
    """
    entries = record.get("trajectory")
    if entries:
        return list(entries)
    out = []
    for step in record.get("steps") or []:
        match = _POS_RE.search(str(step.get("observation") or ""))
        raw = str(step.get("parsed_action") or "").strip().lower()
        out.append({
            "episode": int(step.get("episode_index", 0)) + 1,
            "turn": int(step.get("episode_step", 0)),
            "action": raw if raw in DIRECTIONS else None,
            "position": (int(match.group(1)), int(match.group(2))) if match else None,
            "result": "GOAL" if (step.get("reward") or 0) > 0 else "moved",
            "observation": step.get("observation") or "",
        })
    return out


def instance_stats(record: Mapping[str, Any]) -> dict[str, Any]:
    """Observable quantities for one instance. Never touches ``shortest_path_len``."""
    by_episode: dict[int, list[Mapping[str, Any]]] = {}
    for entry in _entries(record):
        by_episode.setdefault(int(entry.get("episode", 1)), []).append(entry)
    for turns in by_episode.values():
        turns.sort(key=lambda e: int(e.get("turn", 0)))

    stats: dict[str, Any] = {
        "n_episodes": len(by_episode),
        "turns_per_episode": [],
        "solved_per_episode": [],
        "actions": Counter(),
        "results": Counter(),
        "blocked": 0,
        "moved": 0,
        "revisits": 0,
        "cells_visited": 0,
        "surroundings": Counter(),
        "net_displacement": [],
        "goal_sighting_turn": [],
        "new_cells_vs_first_episode": [],
        "unparsed_turns": 0,
        "coords": [],
    }
    action_seq: dict[int, list[str]] = {}

    for index in sorted(by_episode):
        turns = by_episode[index]
        stats["turns_per_episode"].append(len(turns))
        stats["solved_per_episode"].append(
            any(str(e.get("result")).upper() == "GOAL" for e in turns)
        )

        seen: set[tuple[int, int]] = set()
        sighting: int | None = None
        for entry in turns:
            action = entry.get("action")
            action = str(action).strip().lower() if action is not None else None
            if action in DIRECTIONS:
                stats["actions"][action] += 1
            else:
                stats["unparsed_turns"] += 1
            result = str(entry.get("result") or "")
            stats["results"][result] += 1
            # `invalid` is the env's own label for a move that did not happen --
            # authoritative, and cheaper than inferring it from position equality.
            if result == "invalid":
                stats["blocked"] += 1
            elif result:
                stats["moved"] += 1
            here = entry.get("position")
            if isinstance(here, (list, tuple)) and len(here) == 2:
                here = (int(here[0]), int(here[1]))
                stats["coords"].append(here)
                if here in seen:
                    stats["revisits"] += 1
                seen.add(here)
            for direction, what in _DIR_RE.findall(
                str(entry.get("observation") or "")
            ):
                stats["surroundings"][what.lower()] += 1
                if what.lower() == "goal" and sighting is None:
                    sighting = int(entry.get("turn", 0))
        stats["goal_sighting_turn"].append(sighting)
        stats["cells_visited"] += len(seen)

        cells = [
            (int(e["position"][0]), int(e["position"][1]))
            for e in turns
            if isinstance(e.get("position"), (list, tuple))
        ]
        if len(cells) >= 2:
            stats["net_displacement"].append(
                abs(cells[-1][0] - cells[0][0]) + abs(cells[-1][1] - cells[0][1])
            )
        action_seq[index] = [str(e.get("action")) for e in turns]

    # First turn at which two episodes of the same instance diverged. This is the
    # within-instance contrast the evolver log always reported as "0 same-problem
    # pair(s)": each instance appears once, but its own episodes are the pair.
    divergence: int | None = None
    keys = sorted(action_seq)
    for i in range(len(keys) - 1):
        a, b = action_seq[keys[i]], action_seq[keys[i + 1]]
        for turn, (x, y) in enumerate(zip(a, b), start=1):
            if x != y:
                divergence = turn if divergence is None else min(divergence, turn)
                break
    stats["first_divergence_turn"] = divergence

    # Cells a later episode reached that episode 1 never did. The same uniform
    # per-episode split already applied to every other quantity -- `cells_visited`
    # counted per episode -- not a statistic promoted for being interesting.
    keys_sorted = sorted(by_episode)
    if keys_sorted:
        first_cells = {
            (int(e["position"][0]), int(e["position"][1]))
            for e in by_episode[keys_sorted[0]]
            if isinstance(e.get("position"), (list, tuple))
        }
        for index in keys_sorted[1:]:
            later = {
                (int(e["position"][0]), int(e["position"][1]))
                for e in by_episode[index]
                if isinstance(e.get("position"), (list, tuple))
            }
            stats["new_cells_vs_first_episode"].append(len(later - first_cells))
    return stats


def aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pool :func:`instance_stats` over instances."""
    per = [instance_stats(r) for r in records]
    if not per:
        return {"n_instances": 0}

    def flat(key: str) -> list[Any]:
        return [v for s in per for v in s[key]]

    actions: Counter = Counter()
    surroundings: Counter = Counter()
    for s in per:
        actions.update(s["actions"])
        surroundings.update(s["surroundings"])
    coords = [c for s in per for c in s["coords"]]
    blocked, moved = sum(s["blocked"] for s in per), sum(s["moved"] for s in per)
    solved_turns = [
        t
        for s in per
        for t, ok in zip(s["turns_per_episode"], s["solved_per_episode"])
        if ok
    ]
    sightings = [t for s in per for t in s["goal_sighting_turn"] if t is not None]
    divergences = [s["first_divergence_turn"] for s in per
                   if s["first_divergence_turn"] is not None]
    return {
        "n_instances": len(per),
        "turns_per_episode": Counter(flat("turns_per_episode")),
        "turns_on_solved_episodes": Counter(solved_turns),
        "episodes_solved": sum(1 for v in flat("solved_per_episode") if v),
        "episodes_total": len(flat("solved_per_episode")),
        "action_counts": dict(actions),
        "blocked_moves": blocked,
        "successful_moves": moved,
        "blocked_rate": blocked / (blocked + moved) if blocked + moved else 0.0,
        "revisits": sum(s["revisits"] for s in per),
        "cells_visited": sum(s["cells_visited"] for s in per),
        "surroundings_seen": dict(surroundings),
        "net_displacement": Counter(flat("net_displacement")),
        "coord_extent": {
            "x": [min(c[0] for c in coords), max(c[0] for c in coords)],
            "y": [min(c[1] for c in coords), max(c[1] for c in coords)],
        } if coords else None,
        "goal_sighting_turn": Counter(sightings),
        "unparsed_turns": sum(s["unparsed_turns"] for s in per),
        "first_divergence_turn": Counter(divergences),
        "new_cells_vs_first_episode": Counter(flat("new_cells_vs_first_episode")),
    }


def _hist(counter: Counter) -> str:
    return ", ".join(f"{k}:{v}" for k, v in sorted(counter.items()))


def format_for_teacher(records: Sequence[Mapping[str, Any]]) -> str:
    """The aggregate block, as prose for the evolution prompt.

    Rendered as counts rather than means: a mean hides that a distribution is
    tight, and tightness is exactly what distinguishes an invariant of the
    generator from a habit of the agent.
    """
    a = aggregate(records)
    if not a.get("n_instances"):
        return "(no observed data)"
    lines = [
        f"=== WHAT THE AGENT OBSERVED ACROSS {a['n_instances']} INSTANCES ===",
        "Counts pooled from the agents' own observations and actions. Nothing here",
        "comes from the maze generator: these are facts the agent could have noticed.",
        "",
        f"Episodes solved: {a['episodes_solved']}/{a['episodes_total']}",
        f"Turns used, all episodes:        {_hist(a['turns_per_episode'])}",
        f"Turns used, episodes that solved: {_hist(a['turns_on_solved_episodes'])}",
        f"Moves chosen: {a['action_counts']}",
        f"Moves that did not change position: {a['blocked_moves']} of "
        f"{a['blocked_moves'] + a['successful_moves']} "
        f"({a['blocked_rate']:.0%})",
        f"Cells entered more than once in one episode: {a['revisits']} "
        f"(of {a['cells_visited']} cell-entries)",
        f"What the agent was told lay adjacent: {a['surroundings_seen']}",
        f"Positions ever reported: x in {a['coord_extent']['x']}, "
        f"y in {a['coord_extent']['y']}" if a["coord_extent"] else "",
        f"Straight-line distance from first to last cell of an episode: "
        f"{_hist(a['net_displacement'])}",
        f"Turn at which the goal first became visible: "
        f"{_hist(a['goal_sighting_turn'])}",
        f"Turn at which two episodes of one instance first chose differently: "
        f"{_hist(a['first_divergence_turn'])}",
        f"Turns that produced no parseable move: {a['unparsed_turns']}",
    ]
    return "\n".join(x for x in lines if x)


# ── contrastive view ─────────────────────────────────────────────────────

#: Outcome classes, in the order the comparison table prints them. Same partition
#: the trajectory selector uses (`hexa_orbit.reward.reward_bin`), restated here so
#: this module stays importable without the reward machinery.
CLASSES = ("fast success", "slow success", "failure")


def outcome_class(record: Mapping[str, Any]) -> str:
    """Which class *record* belongs to, from its episode outcomes alone."""
    successes = record.get("episode_successes") or []
    if successes and successes[0]:
        return "fast success"
    if any(successes):
        return "slow success"
    return "failure"


#: Quantities compared across classes, as ``(label, key, per_episode)``. Every one
#: is drawn from :data:`QUANTITIES` -- nothing is promoted here for being
#: interesting, which is the point: choosing *which* statistic to surface is
#: itself a way to leak an answer, so the comparison applies the same uniform
#: transformation (rate per episode) to the whole enumerated list.
_COMPARED: tuple[tuple[str, str, bool], ...] = (
    ("turns used per episode", "turns_per_episode", True),
    ("cells re-entered per episode", "revisits", True),
    ("turns with no parseable move", "unparsed_turns", True),
    ("moves that changed nothing", "blocked_moves", True),
    ("distinct cells entered per episode", "cells_visited", True),
    ("distance from first to last cell", "net_displacement", True),
    ("new cells a later episode reached", "new_cells_vs_first_episode", True),
)


def aggregate_by_class(
    records: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """:func:`aggregate`, computed separately for each outcome class."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(outcome_class(record), []).append(record)
    return {name: aggregate(rows) for name, rows in grouped.items()}


def _rate(agg: Mapping[str, Any], key: str) -> float | None:
    """A per-episode rate for *key*, or ``None`` when the class is empty."""
    episodes = agg.get("episodes_total") or 0
    if not episodes:
        return None
    value = agg.get(key)
    if isinstance(value, Counter):
        total = sum(k * v for k, v in value.items())
        count = sum(value.values())
        return total / count if count else None
    if value is None:
        return None
    return float(value) / episodes


def format_contrastive_for_teacher(records: Sequence[Mapping[str, Any]]) -> str:
    """The aggregate split by outcome class, as a comparison table.

    **Why split rather than pool.** The pooled block answers "what happened", and
    the evolution prompt asks "what separated the solves from the failures" -- a
    question no pooled number can answer. Measured consequence of pooling: the
    teacher was shown "84 of 416 cell-entries were repeats" and "31 turns produced
    no parseable move" every round, and banked neither, writing instead about cells
    with three walls -- re-entered 5 times in 2605 turns. Meanwhile episode-1
    success is nearly determined by wasted turns (79% at none, 48% at one, 1% at
    two), and the classes differ on exactly those rows.

    Rates are per episode because the classes have different sizes, and raw counts
    would make the largest class look like the worst offender by construction.
    """
    by_class = aggregate_by_class(records)
    present = [c for c in CLASSES if by_class.get(c, {}).get("n_instances")]
    if len(present) < 2:
        return format_for_teacher(records)

    width = max(len(label) for label, _, _ in _COMPARED) + 2
    header = "".join(f"{c:>20}" for c in present)
    lines = [
        "=== HOW THE OUTCOME CLASSES DIFFER ===",
        "Per-episode rates from the agents' own observations and actions, split by",
        "how the instance ended. Nothing here comes from the maze generator. A row",
        "where the classes differ is a candidate explanation; a row where they do",
        "not is not worth banking a skill about.",
        "",
        f"{'':{width}}{header}",
    ]
    for label, key, _ in _COMPARED:
        cells = []
        for name in present:
            value = _rate(by_class[name], key)
            cells.append(f"{value:>20.2f}" if value is not None else f"{'-':>20}")
        lines.append(f"{label:{width}}" + "".join(cells))
    counts = "".join(f"{by_class[c]['n_instances']:>20d}" for c in present)
    lines.append(f"{'instances in class':{width}}" + counts)
    lines.append("")
    lines.append(format_for_teacher(records))
    return "\n".join(lines)
