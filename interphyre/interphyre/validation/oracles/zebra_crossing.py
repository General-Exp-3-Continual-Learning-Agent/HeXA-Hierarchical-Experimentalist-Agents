"""Targeted oracle for zebra_crossing.

Causal chain: green_ball starts near the top (y ≈ 4.4) on the left side of a
vertical separator. Red ball placed below the green_ball (in the diagonal-bar
region) creates a chain reaction through the stacked bars and separator gap,
causing green_ball to fall to the right side and reach purple_ground.

Valid-placement geometry: green_ball.y ≈ 4.4 and red_ball.radius = 0.5, so
the minimum non-overlapping y above green_ball is 4.4 + 0.4 + 0.5 = 5.3,
which exceeds the world boundary (MAX_Y - radius = 4.5). Placing above the
green_ball is geometrically impossible. The effective zone is BELOW the
green_ball, in the y ∈ [green_ball.y - 2.0, green_ball.y - sum_radii - 0.01]
band, where red_ball interacts with the diagonal bars and separator.

Two-band sampling:

Band A (70%): x ∈ [green_ball.x ± 1.5], y ∈ [green_ball.y - 2.0, y_max]
    Narrow zone centered under the green_ball. Works for the majority of seeds
    where the bar configuration channels the ball through the separator from
    directly below.

Band B (30%): x ∈ [-4.4, 4.4], y ∈ [-4.3, y_max]
    Full-board wide sweep below green_ball. Required for seeds where the
    diagonal-bar configuration routes the solution through a region far from
    the green_ball's x position.

    Dense-sweep analysis of the 5 impossible seeds in the 10k bundle (3734,
    4570, 5193, 6797, 7333) confirmed all are solvable with placements at
    x ∈ [-3.1, +0.8] and y ∈ [-3.8, +3.5] — significantly wider than Band A.
    Hits per variant: 1–51 grid points out of 2500 (0.04%–2.0% hit density).
    Band B at 30 attempts × ~0.8% mean hit density → ~21% per variant; across
    8+ non-impossible variants → ~97% per-seed success rate.
"""

from __future__ import annotations

import numpy as np

from interphyre.validation.oracles import (
    _run_attempt,
    register_oracle,
    register_solver,
)


@register_solver("zebra_crossing")
def solver(
    level, config, n_attempts, oracle_steps, rng
) -> list[tuple[float, float, float]] | None:
    from interphyre.environment import InterphyreEnv  # lazy: avoid circular import

    green_ball = level.objects["green_ball"]
    red_ball = level.objects["red_ball"]
    radius = red_ball.radius
    sum_radii = green_ball.radius + radius

    # Effective ceiling: red_ball must not overlap green_ball.
    y_max = float(np.clip(green_ball.y - sum_radii - 0.01, -4.5, 4.5))

    # Band A: narrow zone directly below/around green_ball.
    x_min_a = float(np.clip(green_ball.x - 1.5, -4.5, 4.5))
    x_max_a = float(np.clip(green_ball.x + 1.5, -4.5, 4.5))
    y_min_a = float(np.clip(green_ball.y - 2.0, -4.5, 4.5))

    if x_min_a >= x_max_a or y_min_a >= y_max:
        return None

    env = InterphyreEnv(level, config=config)
    try:
        for i in range(n_attempts):
            if i % 10 < 7:
                # Band A (70%): narrow zone under green_ball.
                x = rng.uniform(x_min_a, x_max_a)
                y = rng.uniform(y_min_a, y_max)
            else:
                # Band B (30%): full-board sweep — covers seeds where the bar
                # geometry routes the solution far outside the ±1.5 x-window.
                x = rng.uniform(-4.4, 4.4)
                y = rng.uniform(-4.3, y_max)
            if _run_attempt(env, [(x, y, radius)]):
                return [(x, y, radius)]
        return None
    finally:
        env.close()


@register_oracle("zebra_crossing")
def oracle(level, config, n_attempts, oracle_steps, rng) -> bool:
    return solver(level, config, n_attempts, oracle_steps, rng) is not None
