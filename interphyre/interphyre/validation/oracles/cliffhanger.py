"""Targeted oracle for cliffhanger.

Causal chain: green_bar stands vertically at the edge of a platform. Knocking it
sideways (or pushing it off the edge) causes it to fall to the purple_ground.
Drop red_ball near the top of the bar to apply maximum torque.
"""

from __future__ import annotations

import numpy as np

from interphyre.validation.oracles import (
    _run_attempt,
    register_oracle,
    register_solver,
)


@register_solver("cliffhanger")
def solver(
    level, config, n_attempts, oracle_steps, rng
) -> list[tuple[float, float, float]] | None:
    from interphyre.environment import InterphyreEnv  # lazy: avoid circular import

    green_bar = level.objects["green_bar"]
    red_ball = level.objects["red_ball"]
    radius = red_ball.radius

    bar_top = green_bar.y + green_bar.length / 2

    # Wide x band around bar top — either push off left or right edge works.
    x_min = np.clip(green_bar.x - 2.0, -4.5, 4.5)
    x_max = np.clip(green_bar.x + 2.0, -4.5, 4.5)
    y_min = np.clip(bar_top - 0.5, -4.5, 4.5)
    y_max = np.clip(bar_top + 2.5, -4.5, 4.5)

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


@register_oracle("cliffhanger")
def oracle(level, config, n_attempts, oracle_steps, rng) -> bool:
    return solver(level, config, n_attempts, oracle_steps, rng) is not None
