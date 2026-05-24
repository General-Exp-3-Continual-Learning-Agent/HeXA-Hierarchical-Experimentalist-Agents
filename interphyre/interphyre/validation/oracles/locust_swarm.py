"""Targeted oracle for locust_swarm.

Causal chain: green_ball starts near the top (green_ball.y = 4.0 constant). It must
reach the purple_floor at the bottom, navigating through two star chains. The
red ball must be placed to start the green ball moving downward.

Prior oracle had 64% false-negative rate. Two compounding bugs:

1. Zone A y-range collapse. Prior Zone A: y in [green_ball.y + 0.2, green_ball.y + 2.0] =
   [4.2, 4.5] -- only 0.3 units wide at the board ceiling. Zero valid
   solutions in the sweep fall in this range. 75% of oracle attempts were
   directed at a dead zone.

2. x-range too narrow. Prior +/-2.5 from green_ball.x misses 44% of solvable seeds,
   which require placements up to +/-5.89 units from the green_ball.

Two zones, x and y anchored to solution cluster.

Solution geometry (10001 valid seeds):
- 94.4% of solutions fall within +/-1.5 units of green_ball.x.
- Solution x offset distribution: mean=-0.02, std=0.71 (Gaussian-like).
  85.1% within +/-1.0; 49.5% within +/-0.5.
- Solution y distribution: mean=2.29, std=0.74. 99.1% in [0.5, 3.5].
  Strong peak in [1.5, 3.0]: 91.4% of solutions there vs only 50% of
  uniform [0.5, 3.5] samples. The lower half [0.5, 1.5] gets 33% of uniform
  Zone A samples but holds only ~5.3% of solutions -- a 6x density mismatch.

Zone A (80% of attempts): Gaussian x (sigma=0.75, clipped to +/-1.5) and Gaussian y
  (mu = green_ball.y - 1.71 = 2.29, sigma=0.74, clipped to [0.5, 3.5]).
  Y center anchored to green_ball.y - 1.71 (green_ball.y = 4.0 constant).
  Gaussian y concentrates 68% of Zone A y-samples in [1.55, 3.03] where 84.1%
  of solutions fall, vs uniform's 50% in the same interval.

Zone B (20% of attempts): full-board x [-4.5, 4.5], y in [-4.5, 4.5].
  Fallback for the 5.6% of solutions outside +/-1.5 from green_ball.x and the
  0.9% of solutions below y=0.5 or above y=3.5.
"""

from __future__ import annotations

import numpy as np

from interphyre.validation.oracles import (
    _run_attempt,
    register_defaults,
    register_oracle,
    register_solver,
)


@register_solver("locust_swarm")
def solver(
    level, config, n_attempts, oracle_steps, rng
) -> list[tuple[float, float, float]] | None:
    from interphyre.environment import InterphyreEnv  # lazy: avoid circular import

    red_ball = level.objects["red_ball"]
    radius = red_ball.radius

    green_ball = level.objects["green_ball"]
    green_ball_x = float(green_ball.x)
    green_ball_y = float(green_ball.y)
    # Zone A bounds: x +/-1.5 around green_ball.x, y anchored to solution cluster center.
    x_min_a = float(np.clip(green_ball_x - 1.5, -4.5, 4.5))
    x_max_a = float(np.clip(green_ball_x + 1.5, -4.5, 4.5))
    # Solution y mean = green_ball.y - 1.71 (4.0 - 1.71 = 2.29 across all seeds).
    y_center_a = green_ball_y - 1.71

    env = InterphyreEnv(level, config=config)
    try:
        for i in range(n_attempts):
            if i % 5 < 4:
                # Zone A (80%): Gaussian x (sigma=0.75) and Gaussian y (sigma=0.74, mu=y_center_a).
                # x offsets are Gaussian (std=0.71, 85% within +/-1.0).
                # y solutions peak at [1.5, 3.0]: Gaussian concentrates 68% of samples in
                # [y_center-0.74, y_center+0.74] = [1.55, 3.03] where 84.1% of solutions fall.
                x = float(np.clip(rng.normal(green_ball_x, 0.75), x_min_a, x_max_a))
                y = float(np.clip(rng.normal(y_center_a, 0.74), 0.5, 3.5))
            else:
                # Zone B (20%): full-board fallback for outlier solutions.
                x = rng.uniform(-4.5, 4.5)
                y = rng.uniform(-4.5, 4.5)

            if _run_attempt(env, [(x, y, radius)]):
                return [(x, y, radius)]
        return None
    finally:
        env.close()


@register_oracle("locust_swarm")
def oracle(level, config, n_attempts, oracle_steps, rng) -> bool:
    return solver(level, config, n_attempts, oracle_steps, rng) is not None


# Gaussian x+y sampling: combines Gaussian x (sigma=0.75) and
# Gaussian y (sigma=0.74, mu=2.29) for Zone A. Solution cluster at y in [1.5, 3.0]
# was undersampled by uniform [0.5, 3.5] (33% of samples, 5% of solutions in [0.5, 1.5]).
# max_variants=50: k=50 with improved p gives model(k=50) << 1 impossible per 10001 seeds.
register_defaults("locust_swarm", max_variants=50, n_attempts=500)
