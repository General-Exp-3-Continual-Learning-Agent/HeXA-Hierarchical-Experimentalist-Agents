"""The instance reward, replacing ``skillrl``'s single-episode binning.

``skillrl``'s stock ``compute_trajectory_reward`` bins on ``(success,
iterations)`` with thresholds tuned to InterPhyre's 25-iteration budget: success
at 1-3 / 4-7 / 8-15 / 16+ iterations, failure split at 10. Every ORBIT Maze
instance runs 3-27 steps, so essentially all of them land in its top success
bucket or its ``iterations < 10`` failure bucket. It discriminates nothing here.

:func:`instance_reward` replaces it through the existing seam,
``hexa_bridge.patch(reward_fn=...)``, which rebinds the name in *both*
``teacher_prompts`` and ``distill`` because it is read in three places.

**Keyed on the first solved episode, not the success count.** The metric Design B
lives or dies on is episode-1 success on held-out instances, so episode-1 success
sits at the top of the scale and the evolver's incentive points at the claim.
That is training on the objective, not leaking the test: legitimacy comes from
train/eval instance disjointness -- asserted by ``pools.assert_disjoint`` -- not
from the reward being ignorant of the metric. ORBIT's own
``R(M;tau) = success_count in {0,1,2,3}`` is preserved unchanged on the
trajectory dict and stays reportable.

**Range is a hard constraint.** ``distill._compute_initial_confidence`` maps mean
reward from ``[-1,1]`` onto ``[0.1,1.0]``; a value outside that range clamps
silently, so every branch here is checked by :func:`instance_reward` before it
returns.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .actor import InstanceRun

#: Reward for a first success at episode 1, 2, 3. The episode-1 value is a
#: ceiling that the efficiency term erodes; the others are flat, because with a
#: 1-2 move slack budget an episode-2 success has almost no room to vary.
FIRST_SUCCESS_REWARD: Mapping[int, float] = {1: 1.0, 2: 0.6, 3: 0.3}

#: Per excess step over the known-map optimum on an episode-1 success.
EXCESS_STEP_PENALTY = 0.15

#: An episode-1 success never scores below this, however inefficient. Keeps it
#: strictly above an episode-2 success: solving immediately is the behaviour under
#: test, and a wasteful first-episode solve still demonstrates it.
FAST_SUCCESS_FLOOR = 0.7

#: Never solved, but later episodes tried something different.
EXPLORATORY_FAILURE_REWARD = -0.5

#: Never solved and every episode repeated the same action sequence -- the failure
#: mode ORBIT's Table 2 shows their single-episode RL baseline falling into on
#: Mastermind. Isolated at the floor so the evolver is pushed hardest against it.
NO_ADAPTATION_REWARD = -1.0


def reward_from_parts(
    *,
    first_success_episode: Optional[int],
    episode_1_steps: Optional[int],
    shortest_path_len: int,
    adapted: bool,
) -> float:
    """The reward as a pure function of its four inputs. See the module docstring."""
    if first_success_episode is None:
        return EXPLORATORY_FAILURE_REWARD if adapted else NO_ADAPTATION_REWARD

    if first_success_episode != 1:
        try:
            return FIRST_SUCCESS_REWARD[first_success_episode]
        except KeyError:
            # More episodes than the hard cap should allow (D2). Fail loudly:
            # silently reusing episode 3's value would hide a broken loop.
            raise ValueError(
                f"first_success_episode={first_success_episode} exceeds the "
                f"{max(FIRST_SUCCESS_REWARD)}-episode cap"
            ) from None

    excess = max(0, (episode_1_steps or shortest_path_len) - shortest_path_len)
    return max(FAST_SUCCESS_FLOOR, FIRST_SUCCESS_REWARD[1] - EXCESS_STEP_PENALTY * excess)


def instance_reward(run: InstanceRun) -> float:
    """Reward for one instance run, guaranteed inside ``[-1, 1]``."""
    from .trajectory import first_success_episode  # local: avoids a cycle

    first = first_success_episode(run.episode_successes)
    value = reward_from_parts(
        first_success_episode=first,
        episode_1_steps=run.episode_lengths[0] if run.episode_lengths else None,
        shortest_path_len=run.shortest_path_len,
        adapted=run.adapted_across_episodes,
    )
    if not -1.0 <= value <= 1.0:  # pragma: no cover - guards the constants above
        raise AssertionError(
            f"reward {value} outside [-1,1]; _compute_initial_confidence would clamp it"
        )
    return value


def trajectory_reward(trajectory: Mapping[str, Any]) -> float:
    """``compute_trajectory_reward`` replacement, reading a trajectory dict.

    This is what ``hexa_bridge.patch(reward_fn=...)`` installs. It reads the
    ``reward`` key :func:`~hexa_orbit.trajectory.instance_trajectory` already
    stamped, and recomputes from parts only if it is absent -- so the number the
    teacher reads is always the number the run recorded.
    """
    if "reward" in trajectory:
        return float(trajectory["reward"])
    return reward_from_parts(
        first_success_episode=trajectory.get("first_success_episode"),
        episode_1_steps=(trajectory.get("episode_lengths") or [None])[0],
        shortest_path_len=int(trajectory.get("shortest_path_len", 8)),
        adapted=bool(trajectory.get("adapted_across_episodes", True)),
    )


def reward_bin(trajectory: Mapping[str, Any]) -> str:
    """Human-readable bin name, for logs and the round record."""
    first = trajectory.get("first_success_episode")
    if first == 1:
        return "fast success"
    if first is not None:
        return "slow success"
    return (
        "exploratory failure"
        if trajectory.get("adapted_across_episodes")
        else "no adaptation"
    )
