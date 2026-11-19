"""Confidence calibration for the DiscoverPhysics reward range.

HeXA's `distill._compute_initial_confidence` maps a source set's mean reward
onto [0.1, 1.0] with

    confidence = clip((mean_reward + 1) / 2, 0.1, 1.0)

which assumes reward spans **[-1, 1]** — true for InterPhyre (success x
iterations) and for CORE's binary +/-1. Our blended reward spans
**[-0.5, 1.0]** (`reward.blended_reward`), so that map is wrong here in two
compounding ways:

1. **The bottom of the scale is unreachable.** The worst possible episode — no
   law submitted, no explanation — scores -0.5 and lands at confidence 0.25,
   not 0.1. Every skill the bank can hold is crowded into the top 75% of the
   range, which is also the range retrieval ranks on.
2. **Confidence pruning is effectively dead.** The evolution prompt asks the
   teacher to drop skills below 0.3; under the stock map that needs
   reward < -0.4, i.e. a law reproducing nothing AND a ~zero explanation
   score. Measured on the 2-world smoke bank: every contrastive skill sat at
   0.9864 and nothing was ever prunable.

This module rescales from the actual range instead, so a floor episode really
does score 0.1 and the 0.3 threshold falls at reward ~= -0.17 (a law clearly
below the pass line) — a band real failures occupy.

It also fixes a second bug the stock function has on our data: it keeps **one
trajectory per seed** (`seed_to_traj` records the first occurrence). Under
InterPhyre seed -> trajectory is 1:1, but our seed numbers restart at
`--start-seed` for **every world**, so in `pooled` mode — where offline
trajectories from many worlds are distilled together — a seed collides across
worlds and all but the first are silently discarded. Pooling every trajectory
that matches a cited seed keeps the evidence and makes the mean meaningful.

`patch()` installs it into both modules that hold a binding: `distill` imports
it as a module global, and `evolving_distill` imported it **by value**
(`from skillrl.distillation.distill import _compute_initial_confidence`), so
rebinding only one leaves the other on the stock map. Mirrors what
`hexa_core.hexa_bridge` and `hexa_orbit.teacher` do for the same function.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

# The reward range this calibration is built for; `reward.blended_reward`'s
# codomain. Keep in sync with reward.py — `patch()` asserts it.
REWARD_MIN = -0.5
REWARD_MAX = 1.0

CONFIDENCE_MIN = 0.1
CONFIDENCE_MAX = 1.0


def confidence_from_reward(mean_reward: float) -> float:
    """Map a mean reward in [REWARD_MIN, REWARD_MAX] onto [0.1, 1.0], linearly.

    Out-of-range input clamps rather than raising: a stored `reward` from an
    older run, or a trajectory whose reward was computed under a different
    blend, should degrade to an endpoint instead of killing a distillation.
    """
    span = REWARD_MAX - REWARD_MIN
    frac = (float(mean_reward) - REWARD_MIN) / span
    conf = CONFIDENCE_MIN + frac * (CONFIDENCE_MAX - CONFIDENCE_MIN)
    return max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, conf))


def compute_initial_confidence(
    source_seeds: list,
    all_trajs: list[dict],
    fallback_confidence: float = 0.5,
    *,
    reward_fn: Optional[Callable[[dict], float]] = None,
) -> float:
    """Drop-in replacement for `distill._compute_initial_confidence`.

    Same signature and same fallback behaviour as the stock function — an
    unmatched citation falls back to the level-wide average rather than to a
    constant — but pools *every* trajectory matching a cited seed and rescales
    from the DiscoverPhysics reward range.
    """
    if not source_seeds or not all_trajs:
        return fallback_confidence

    if reward_fn is None:
        # Resolved at call time, so whatever reward.patch() installed is what
        # we read — not a copy captured at import.
        import skillrl.distillation.teacher_prompts as _tp
        reward_fn = _tp.compute_trajectory_reward

    by_seed: dict[Any, list[dict]] = {}
    for traj in all_trajs:
        by_seed.setdefault(traj.get("seed", -1), []).append(traj)

    rewards = [
        reward_fn(traj)
        for seed in source_seeds
        for traj in by_seed.get(seed, ())
    ]
    if not rewards:
        rewards = [reward_fn(t) for t in all_trajs]

    return confidence_from_reward(sum(rewards) / len(rewards))


def patch() -> None:
    """Install the rescaled confidence into distill + evolving_distill.

    Call AFTER reward.patch(), so the reward this reads is the blended one.
    """
    from hexa_discovery.reward import blended_reward

    # The map is only correct if the reward really spans [REWARD_MIN, REWARD_MAX];
    # a change to the blend that moves either endpoint must move these too.
    assert abs(blended_reward(0.0, float("inf")) - REWARD_MIN) < 1e-9, (
        "reward floor moved; update confidence.REWARD_MIN"
    )
    assert abs(blended_reward(1.0, 0.0) - REWARD_MAX) < 1e-9, (
        "reward ceiling moved; update confidence.REWARD_MAX"
    )

    import skillrl.distillation.distill as _distill
    import skillrl.distillation.evolving_distill as _evolving_distill

    _distill._compute_initial_confidence = compute_initial_confidence
    # Imported by value at module load, so this binding is separate.
    _evolving_distill._compute_initial_confidence = compute_initial_confidence
