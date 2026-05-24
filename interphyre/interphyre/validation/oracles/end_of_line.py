"""Targeted oracle for end_of_line.

Causal chain: green_ball sits on a shelf. It must be knocked off toward the
purple_wall (left or right side wall). Drop red_ball on the far side from the
wall — that is, between the green_ball and the shelf edge on the non-wall side —
to push the ball toward the wall.

Two-band y sampling:

Band A (70%): y ∈ [shelf_top, shelf_top + 2.5]
    Standard near-shelf drop. Works for the majority of seeds where the shelf is
    mid-board and a low-energy collision is sufficient.

Band B (30%): y ∈ [shelf_top + 2.5, 4.5]
    High-altitude drop. Required for seeds where the shelf is near the floor
    (shelf.y ≈ -4.2). In these geometries the standard zone caps y at -1.6,
    but the actual working placement is near the top of the board (y ≈ +2–4).
    The high drop delivers greater downward momentum that carries the red_ball
    past the low shelf to contact the green_ball at a steep angle.

    Empirically confirmed for seed 8067 (shelf.y=-4.2): full-board sweep finds
    all 10 grid hits in y ∈ [+2.3, +4.3], completely outside Band A's y range.
    Variants v=5 of seed 8067 is genuinely dynamically impossible (0 hits in
    40×40 grid sweep); v=0, v=6, v=8 are oracle-gap misses recovered by Band B.
"""

from __future__ import annotations

import numpy as np

from interphyre.validation.oracles import (
    _run_attempt,
    register_oracle,
    register_solver,
)


@register_solver("end_of_line")
def solver(
    level, config, n_attempts, oracle_steps, rng
) -> list[tuple[float, float, float]] | None:
    from interphyre.environment import InterphyreEnv  # lazy: avoid circular import

    green_ball = level.objects["green_ball"]
    purple_wall = level.objects["purple_wall"]
    red_ball = level.objects["red_ball"]
    radius = red_ball.radius

    wall_x = purple_wall.x
    ball_x = green_ball.x
    shelf_top = level.objects["shelf"].y + 0.1  # just above shelf

    # Push from the side OPPOSITE the wall — x range between ball and away from wall.
    if wall_x < 0:
        # Wall on left; push from right of ball
        x_min = np.clip(ball_x + 0.2, -4.5, 4.5)
        x_max = np.clip(ball_x + 2.5, -4.5, 4.5)
    else:
        # Wall on right; push from left of ball
        x_min = np.clip(ball_x - 2.5, -4.5, 4.5)
        x_max = np.clip(ball_x - 0.2, -4.5, 4.5)

    # Band A: near-shelf zone (standard drop height).
    y_min_a = float(np.clip(shelf_top, -4.5, 4.5))
    y_max_a = float(np.clip(shelf_top + 2.5, -4.5, 4.5))

    # Band B: high-altitude zone (covers deep-shelf seeds where Band A misses).
    y_min_b = y_max_a  # starts where Band A ends
    y_max_b = 4.4  # near world top boundary

    env = InterphyreEnv(level, config=config)
    try:
        for i in range(n_attempts):
            x = rng.uniform(x_min, x_max)
            if i % 10 < 7 and y_min_a < y_max_a:
                # Band A (70%): standard near-shelf drop.
                y = rng.uniform(y_min_a, y_max_a)
            else:
                # Band B (30%): high-altitude drop for deep-shelf seeds.
                if y_min_b >= y_max_b:
                    # Shelf is high — Band B collapses; fall back to Band A.
                    if y_min_a < y_max_a:
                        y = rng.uniform(y_min_a, y_max_a)
                    else:
                        continue
                else:
                    y = rng.uniform(y_min_b, y_max_b)
            if _run_attempt(env, [(x, y, radius)]):
                return [(x, y, radius)]
        return None
    finally:
        env.close()


@register_oracle("end_of_line")
def oracle(level, config, n_attempts, oracle_steps, rng) -> bool:
    return solver(level, config, n_attempts, oracle_steps, rng) is not None
