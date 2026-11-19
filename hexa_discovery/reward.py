"""Graded reward for DiscoverPhysics, replacing HeXA's iteration-based reward.

HeXA's `compute_trajectory_reward` is binary success x iteration count — right
for InterPhyre (touch / no-touch) but wrong here, where the eval is graded
(continuous trajectory error + a 0..1 explanation-judge score).

We use a blend, sign-anchored so benchmark passes come out positive:

    norm_mse = mean_pos_error / Var(GT_world)          # 0 = perfect, 0.1 = pass line
    mse_term = clip(1 - norm_mse / PASS_ERR_THRESHOLD, -1, 1)
    reward   = 0.5 * expl_score + 0.5 * mse_term        # in [-0.5, 1.0]

`patch()` installs this into the two places HeXA reads the reward:
  - teacher_prompts.compute_trajectory_reward  (used by the trajectory formatter
    and by format_trajectories_block's sort order)
  - distill.compute_trajectory_reward          (used by _compute_initial_confidence
    and the contrastive-distillation logging)
Both are module-global name lookups resolved at call time, so overwriting the
bindings is sufficient — no edits to HeXA core.
"""

from __future__ import annotations

from typing import Optional

from hexa_discovery.config import PASS_ERR_THRESHOLD


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def blended_reward(expl_score: Optional[float], norm_mse: Optional[float]) -> float:
    """Blend explanation score (0..1) and normalized MSE into a reward in [-0.5, 1.0].

    Missing components degrade gracefully: a missing expl_score counts as 0,
    a missing/non-finite norm_mse counts as a full miss (mse_term = -1).
    """
    e = 0.0 if expl_score is None else float(expl_score)
    e = _clip(e, 0.0, 1.0)

    if norm_mse is None or norm_mse != norm_mse or norm_mse == float("inf"):
        mse_term = -1.0
    else:
        mse_term = _clip(1.0 - float(norm_mse) / PASS_ERR_THRESHOLD, -1.0, 1.0)

    return 0.5 * e + 0.5 * mse_term


def compute_trajectory_reward(trajectory: dict) -> float:
    """Drop-in replacement for HeXA's reward, reading DiscoverPhysics fields.

    Preference order:
      1. a precomputed `reward` stored by the adapter,
      2. recompute from stored `expl_score` / `norm_mse`,
      3. fall back to a coarse success-based value if neither is present.
    """
    if "reward" in trajectory and trajectory["reward"] is not None:
        return float(trajectory["reward"])
    if "expl_score" in trajectory or "norm_mse" in trajectory:
        return blended_reward(
            trajectory.get("expl_score"), trajectory.get("norm_mse")
        )
    # Last-resort fallback (should not happen for adapter-produced trajectories).
    return 1.0 if trajectory.get("success") else -0.5


def patch() -> None:
    """Install the blended reward into HeXA's distill + teacher_prompts modules."""
    import skillrl.distillation.teacher_prompts as _tp
    import skillrl.distillation.distill as _distill

    _tp.compute_trajectory_reward = compute_trajectory_reward
    _distill.compute_trajectory_reward = compute_trajectory_reward
