"""Targeted oracle for two_body_problem.

Causal chain: green_ball and blue_ball are at the same height with a gap.
Dropping red_ball between them (or above green_ball toward blue_ball) causes a
collision. Direct placement between or above the green ball is optimal.
"""

from __future__ import annotations

import numpy as np

from interphyre.validation.oracles import (
    _run_attempt,
    register_oracle,
    register_solver,
)


@register_solver("two_body_problem")
def solver(
    level, config, n_attempts, oracle_steps, rng
) -> list[tuple[float, float, float]] | None:
    from interphyre.environment import InterphyreEnv  # lazy: avoid circular import

    green_ball = level.objects["green_ball"]
    blue_ball = level.objects["blue_ball"]
    red_ball = level.objects["red_ball"]
    radius = red_ball.radius

    # Cover the region from just right of green_ball to just left of blue_ball.
    x_min = np.clip(green_ball.x - 0.5, -4.5, 4.5)
    x_max = np.clip(blue_ball.x + 0.5, -4.5, 4.5)
    ball_y = green_ball.y
    y_min = np.clip(ball_y - 0.5, -4.5, 4.5)
    y_max = np.clip(ball_y + 2.0, -4.5, 4.5)

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


@register_oracle("two_body_problem")
def oracle(level, config, n_attempts, oracle_steps, rng) -> bool:
    return solver(level, config, n_attempts, oracle_steps, rng) is not None
