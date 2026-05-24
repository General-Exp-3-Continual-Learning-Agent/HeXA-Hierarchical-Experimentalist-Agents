"""Targeted oracle for mind_the_gap.

Causal chain: green_ball at (0, 3.5) falls freely. A blocking_ball sits near
one edge of the hole between left_platform and right_platform. The oracle places
red_ball on the FAR SIDE of green_ball (opposite to the hole) at a near-horizontal
contact angle so the impact deflects green_ball TOWARD the hole. As green_ball
falls, it contacts blocking_ball, displacing it and clearing the hole, then falls
through to purple_ground.

The x_offset must be strictly less than sum_of_radii to guarantee contact:
placing red_ball outside contact range causes both balls to free-fall in parallel
with no collision, so green_ball receives no lateral impulse.

Two sampling zones (cycled 1:1):

Zone A (50%): tangent push near green_ball.y — near-horizontal placement just
  above green_ball, x on the far side of the hole. High per-trial success rate for
  seeds where the hole is reachable from the top.

Zone B (50%): x near hole center (±1.5), y in [−3.0, green_ball.y − 0.5].
  Covers a second causal path where red_ball intercepts green_ball after it has
  fallen further, using a lower-angle deflection geometry. y-min at -3.0: sweep
  found all valid solutions at y ≥ -2.821; -3.0 gives margin without sampling
  the dead zone below. A Zone C targeting blocking_ball directly was evaluated
  but reduced success rate by competing with A/B for attempts.
"""

from __future__ import annotations

import numpy as np

from interphyre.validation.oracles import (
    _run_attempt,
    register_defaults,
    register_oracle,
    register_solver,
)


@register_solver("mind_the_gap")
def solver(
    level, config, n_attempts, oracle_steps, rng
) -> list[tuple[float, float, float]] | None:
    from interphyre.environment import InterphyreEnv  # lazy: avoid circular import

    green_ball = level.objects["green_ball"]
    left_platform = level.objects["left_platform"]
    right_platform = level.objects["right_platform"]
    red_ball = level.objects["red_ball"]
    radius = red_ball.radius
    sum_radii = green_ball.radius + radius

    # Hole center: used by both zones.
    hole_cx = (left_platform.right + right_platform.left) / 2
    push_direction = 1.0 if hole_cx > green_ball.x else -1.0

    # Zone B bounds: x near hole (±1.5), y below green_ball.
    x_b_min = float(np.clip(hole_cx - 1.5, -4.5, 4.5))
    x_b_max = float(np.clip(hole_cx + 1.5, -4.5, 4.5))
    y_b_max = float(np.clip(green_ball.y - 0.5, -4.5, 4.5))

    env = InterphyreEnv(level, config=config)
    try:
        for i in range(n_attempts):
            if i % 2 == 0:
                # Zone A (50%): tangent push near green_ball — original mechanism.
                x_offset = rng.uniform(sum_radii * 0.6, sum_radii * 0.99)
                x = float(np.clip(green_ball.x - push_direction * x_offset, -4.5, 4.5))
                y_clearance = np.sqrt(max(0.0, sum_radii**2 - x_offset**2))
                y = rng.uniform(
                    green_ball.y + y_clearance + 0.02, green_ball.y + y_clearance + 0.5
                )
            else:
                # Zone B (50%): x near hole center, y below green_ball.
                # Sweep confirmed valid placements at y ∈ [−1.3, 2.5] for ~42% of
                # previously-impossible seeds; all have x ∈ [−1.3, 1.3].
                x = rng.uniform(x_b_min, x_b_max)
                y = rng.uniform(-3.0, y_b_max)

            if _run_attempt(env, [(x, y, radius)]):
                return [(x, y, radius)]
        return None
    finally:
        env.close()


@register_oracle("mind_the_gap")
def oracle(level, config, n_attempts, oracle_steps, rng) -> bool:
    return solver(level, config, n_attempts, oracle_steps, rng) is not None


# Geometric-decay model: p=0.398 per variant, model(k=20)=0.4 impossible.
# k=20 reduces expected impossible from 62 (k=10) to <1 per 10001 seeds.
# n_attempts raised 200→300 after audit: seed 6719 had 2 solvable variants out of 30
# with narrow solution region (~0.1 sq units in Zone B); 300 attempts raises per-variant
# P(find) from ~43% to ~59%, reducing P(miss both solvable variants) from 0.25 to 0.17.
# Combined with hole_width 1.05→1.15 (level edit), expect 0 impossible seeds.
register_defaults("mind_the_gap", max_variants=20, n_attempts=300)
