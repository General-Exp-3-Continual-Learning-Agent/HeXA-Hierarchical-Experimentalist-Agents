"""Maze shortest-path oracle -- free, because the generator already computes it.

RECON §6 found ORBIT ships no oracle code, but the Maze one costs nothing: the
generator places the goal at the BFS-farthest reachable cell from the start and
returns that distance as ``shortest_path_len``
(``envs/maze_env_adapter.py:_find_farthest_position``, :269-297). So the optimal
step count for a *known* map is already on every
:class:`~hexa_orbit.pools.MazeInstanceSpec`, and all that is missing is the
action sequence realising it -- which :func:`optimal_actions` recovers with a
plain BFS over the same grid.

**What this oracle is, precisely.** It is the optimum for an agent that *knows
the map*. It is not the optimum for an agent playing Maze, which observes only
its four neighbours (RECON §3) and cannot know where the goal is on episode 1.
No policy can achieve this bound on a first encounter, so regret against it is
not an achievability gap -- it is a measure of how much of the map an agent has
inferred. Read episode-1 regret as "exploration cost" and episode-3 regret as
"how well the route was retained", and never quote the bound as something a good
agent should reach.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Optional

import numpy as np

#: ORBIT's action-to-delta map (``envs/maze_env_adapter.py:DELTA``), inverted at
#: use so a step becomes the action naming it.
_DELTA: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _passable(maze: np.ndarray, x: int, y: int) -> bool:
    """Path or goal. Mirrors ``_get_cell_type``: out of bounds counts as wall."""
    if x < 0 or x >= maze.shape[0] or y < 0 or y >= maze.shape[1]:
        return False
    return maze[x, y] != 1


def goal_position(maze: np.ndarray) -> tuple[int, int]:
    """The cell the generator marked ``-1``."""
    found = np.argwhere(maze == -1)
    if len(found) != 1:
        raise ValueError(f"expected exactly one goal cell, found {len(found)}")
    return (int(found[0][0]), int(found[0][1]))


def optimal_actions(maze: np.ndarray, start: tuple[int, int]) -> list[str]:
    """A shortest action sequence from *start* to the goal.

    Ties are broken by :data:`_DELTA` order (up, down, left, right), so the
    sequence is deterministic -- required, since tests compare against it.
    """
    goal = goal_position(maze)
    start = (int(start[0]), int(start[1]))
    if start == goal:
        return []

    previous: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
    seen = {start}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for action, (dx, dy) in _DELTA.items():
            nxt = (current[0] + dx, current[1] + dy)
            if nxt in seen or not _passable(maze, *nxt):
                continue
            seen.add(nxt)
            previous[nxt] = (current, action)
            if nxt == goal:
                actions: list[str] = []
                node = goal
                while node != start:
                    node, action_taken = previous[node]
                    actions.append(action_taken)
                return list(reversed(actions))
            queue.append(nxt)
    raise ValueError(f"goal {goal} unreachable from {start}")


def optimal_steps(maze: np.ndarray, start: tuple[int, int]) -> int:
    """Length of the shortest path. Should equal ``spec.shortest_path_len``."""
    return len(optimal_actions(maze, start))


def boxed(action: str) -> str:
    """Wrap an action the way the actor must emit it."""
    return f"\\boxed{{{action}}}"


def instance_optimal_actions(instance: Any) -> list[str]:
    """Optimal action sequence for a live :class:`~hexa_orbit.adapter.MazeInstance`.

    Reads the map off the inner env, so the instance must have been reset.
    """
    env = instance._env
    if env.map is None or env.init_position is None:
        raise RuntimeError("instance has not been reset")
    return optimal_actions(env.map, env.init_position)


def regret(instance: Any, spec: Optional[Any] = None) -> dict[str, Any]:
    """Per-episode excess steps over the known-map optimum.

    A failed episode is charged its full step budget, which is a floor on its
    true cost rather than the cost itself -- the agent did not reach the goal at
    all, so no finite step count is the honest number. Reported separately from
    successes for that reason.
    """
    optimum = (spec or instance.spec).shortest_path_len
    per_episode = [
        {
            "episode": record.index + 1,
            "success": record.success,
            "steps": record.steps,
            "optimum": optimum,
            "excess": record.steps - optimum if record.success else None,
        }
        for record in instance.episodes
    ]
    solved = [e["excess"] for e in per_episode if e["excess"] is not None]
    return {
        "optimum": optimum,
        "per_episode": per_episode,
        "mean_excess_on_success": sum(solved) / len(solved) if solved else None,
        "n_success": len(solved),
        "n_failed": len(per_episode) - len(solved),
    }
