"""Maze and Mastermind behind one interface.

The two ORBIT tasks are already near-identical in shape -- `MastermindTask`
mirrors `MazeTask` method for method, and the two `instance_trajectory`
functions emit the same dict field for field -- but they differ in three places
that a memory baseline has to care about, and each one is a correctness issue
rather than a style one:

* **How the pools are drawn.**  Maze seeds come from a ~1e6 space, so train and
  eval are independent draws and a repeated *layout* is tolerated and reported.
  Mastermind's instance identity IS the secret code and only `perm(6,3) = 120`
  exist, so the pools partition one stream and any overlap is the answer itself.
* **Whether leakage is a thing.**  Only Maze has `maze_hash`, so only Maze gets
  a leaked/clean split.  Reported for Mastermind as all-clean, which is true by
  construction.
* **How a turn reads.**  A maze turn is (position, action, outcome); a
  Mastermind turn is (guess, black, white).  Two `instance_trajectory`
  implementations already exist for this, and are bound in rather than
  reimplemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

TASKS = ("maze", "mastermind")


@dataclass
class OrbitTask:
    """One ORBIT task, with everything the memory stream needs off it."""

    name: str
    task: Any                                        # MazeTask | MastermindTask
    #: `InstanceRun -> skillrl-shaped trajectory dict`, already carrying `reward`.
    to_trajectory: Callable[..., dict]
    #: Per-episode turn cap, for the added-context budget note in the manifest.
    max_turns_per_episode: int

    def train_specs(self, n: int) -> list:
        return list(self.task.train_specs(n))

    def eval_specs(self, n: int) -> list:
        return list(self.task.eval_specs(n))

    def instance(self, spec) -> Any:
        return self.task.instance(spec)

    def manifest(self) -> dict:
        return dict(self.task.manifest())

    # ── pool hygiene ──────────────────────────────────────────────────────

    def assert_disjoint(self, train: Sequence, eval_: Sequence) -> None:
        """Refuse pools that share an instance. Fatal on both tasks, for
        different reasons: a shared maze *seed* breaks instance identity, and a
        shared *code* is the answer."""
        if self.name == "mastermind":
            from hexa_orbit.mastermind import assert_disjoint  # noqa: PLC0415
        else:
            from hexa_orbit.pools import assert_disjoint  # noqa: PLC0415
        assert_disjoint(list(train), list(eval_))

    def leaked_ids(self, eval_: Sequence, train: Sequence) -> set[str]:
        """Eval instances whose *content* also occurs in the train pool.

        Maze only: 13 of 256 eval seeds land on a layout another seed already
        produced, and ~19% of eval layouts also appear in a 512-instance train
        pool.  Those instances are still played, but every headline metric is
        reported over the leaked and clean partitions separately, because a
        memory built on the train pool could carry a memorised route for them.

        Mastermind returns the empty set, and that is a fact rather than a
        default: `assert_disjoint` above has already refused any shared code.
        """
        if self.name != "maze" or not eval_ or not train:
            return set()
        from hexa_orbit.pools import leakage_split  # noqa: PLC0415

        return {s.instance_id for s in leakage_split(list(eval_), list(train))["leaked"]}


def build_task(
    name: str,
    *,
    num_episodes: Optional[int] = None,
    global_seed: Optional[int] = None,
) -> OrbitTask:
    """Construct one of :data:`TASKS`.

    Defaults are each task's own published configuration -- Maze's global seed
    and Mastermind's pinned 3-of-6 -- so a caller that passes nothing reproduces
    the pools the existing arms ran on.
    """
    if name == "maze":
        from hexa_orbit.adapter import MazeTask  # noqa: PLC0415
        from hexa_orbit.trajectory import instance_trajectory  # noqa: PLC0415

        kwargs: dict[str, Any] = {}
        if num_episodes is not None:
            kwargs["num_episodes"] = num_episodes
        if global_seed is not None:
            kwargs["global_seed"] = global_seed
        task = MazeTask(**kwargs)
        return OrbitTask(
            name="maze",
            task=task,
            to_trajectory=instance_trajectory,
            max_turns_per_episode=int(task.task_cfg["max_turns_per_episode"]),
        )

    if name == "mastermind":
        from hexa_orbit.mastermind import MastermindTask  # noqa: PLC0415
        from hexa_orbit.mastermind_trajectory import (  # noqa: PLC0415
            instance_trajectory,
        )

        kwargs = {}
        if num_episodes is not None:
            kwargs["num_episodes"] = num_episodes
        if global_seed is not None:
            kwargs["global_seed"] = global_seed
        task = MastermindTask(**kwargs)
        return OrbitTask(
            name="mastermind",
            task=task,
            to_trajectory=instance_trajectory,
            max_turns_per_episode=int(task.task_cfg["max_turns"]),
        )

    raise ValueError(f"Unknown ORBIT task {name!r}; expected one of {TASKS}")
