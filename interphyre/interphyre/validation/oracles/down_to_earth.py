"""Targeted oracle for down_to_earth.

Causal chain: green_ball starts above a platform. It falls and lands on the
platform. Dropping red_ball beside the green_ball pushes it past the platform
edge so it continues to the purple_ground below.

Fix (this version): The original oracle sampled y ∈ [green_ball.y − 0.5,
green_ball.y + 2.0]. Since green_ball.y ≈ 4.0, this collapses to y ∈ [3.5,
4.5] — the top strip only. Full-board sweeps of all 214 impossible seeds
showed valid placements in y ∈ [platform.y, green_ball.y − 0.5], with zero
oracle-window hits in 4 of 5 tested seeds. The red ball can intercept the
green ball anywhere in the column between the platform and the green ball's
start. Fix: extend y_min down to platform.y − 1.0.
"""

from __future__ import annotations

import numpy as np

from interphyre.validation.oracles import (
    _run_attempt,
    register_oracle,
    register_solver,
)


@register_solver("down_to_earth")
def solver(
    level, config, n_attempts, oracle_steps, rng
) -> list[tuple[float, float, float]] | None:
    from interphyre.environment import InterphyreEnv  # lazy: avoid circular import

    green_ball = level.objects["green_ball"]
    platform = level.objects["platform"]
    red_ball = level.objects["red_ball"]
    radius = red_ball.radius

    # Target near platform edges — just inside or just outside either edge
    # pushes the green ball off the platform.
    x_min = np.clip(platform.left - 1.0, -4.5, 4.5)
    x_max = np.clip(platform.right + 1.0, -4.5, 4.5)
    # Valid placements span from just below the platform up to just above the
    # green ball — the red ball intercepts the falling green ball anywhere in
    # that column. The original oracle used y ∈ [gb.y-0.5, gb.y+2.0] which
    # collapsed to the top strip only and missed the entire working range.
    y_min = np.clip(platform.y - 1.0, -4.5, 4.5)
    y_max = np.clip(green_ball.y + 0.5, -4.5, 4.5)

    env = InterphyreEnv(level, config=config)
    try:
        for _ in range(n_attempts):
            x = rng.uniform(x_min, x_max)
            y = rng.uniform(y_min, y_max)
            if _run_attempt(env, [(x, y, radius)]):
                return [(x, y, radius)]
        return None
    finally:
        env.close()


@register_oracle("down_to_earth")
def oracle(level, config, n_attempts, oracle_steps, rng) -> bool:
    return solver(level, config, n_attempts, oracle_steps, rng) is not None
