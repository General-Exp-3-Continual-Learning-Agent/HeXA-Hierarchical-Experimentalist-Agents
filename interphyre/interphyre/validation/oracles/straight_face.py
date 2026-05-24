"""Targeted oracle for straight_face.

Causal chain: green_ball (top, ball_x) must land on purple_pad (target_x, floor).
Gray_ball sits directly below at the same x. The red_ball acts as a deflector:
placed between ball_x and target_x, it redirects the falling stack horizontally
toward the pad.

Fix (v2): The original oracle used y_min = gray_ball.y − 1.5, which
for gray_ball.y ≈ 2.5 gave y_min = 1.0 — cutting off the lower two-thirds of
the board. Full-board sweeps confirmed valid placements reach down to y ≈ −3.5
(deflecting the ball after it passes the gray ball level). Fix: full-board y
from −4.5 to green_ball.y + 0.5, and full-board x. Note: ~40% of impossible
seeds are genuinely impossible due to extreme lateral separation between
green_ball and pad, and are unaffected by this fix.

Corridor sampling (v3): Valid deflector x positions are concentrated in the
lateral corridor between green_ball.x and purple_pad.x (±radius). Uniform
sampling over [−4.5, 4.5] wastes 55–78% of x draws outside this corridor when
ball_x ∈ [−4, 2] and target_x ∈ [−4, 4]. A 70/30 corridor/fallback split
focuses the majority of attempts on the high-probability corridor while keeping
30% of draws uniform as a fallback for seeds where pad and ball are very close
(narrow corridor). Expected p improvement: ~0.42 → ~0.60–0.65 per variant.
"""

from __future__ import annotations

import numpy as np

from interphyre.validation.oracles import (
    _run_attempt,
    register_defaults,
    register_oracle,
    register_solver,
)


@register_solver("straight_face")
def solver(
    level, config, n_attempts, oracle_steps, rng
) -> list[tuple[float, float, float]] | None:
    from interphyre.environment import InterphyreEnv  # lazy: avoid circular import

    green_ball = level.objects["green_ball"]
    purple_pad = level.objects["purple_pad"]
    red_ball = level.objects["red_ball"]
    radius = red_ball.radius

    # Full-board y: valid deflections exist anywhere below the green ball,
    # including well below the gray ball level (confirmed by full-board sweeps).
    y_min = -4.5
    y_max = float(np.clip(green_ball.y + 0.5, -4.5, 4.5))

    # Corridor bounds for x: the deflector must sit in the lateral corridor
    # between green_ball and purple_pad to redirect the falling stack onto the
    # pad. Compute once per solver call; clip to board limits.
    corridor_lo = float(np.clip(min(green_ball.x, purple_pad.x) - radius, -4.5, 4.5))
    corridor_hi = float(np.clip(max(green_ball.x, purple_pad.x) + radius, -4.5, 4.5))

    env = InterphyreEnv(level, config=config)
    try:
        for _ in range(n_attempts):
            # 70% corridor sampling, 30% full-board fallback. The fallback handles
            # seeds where pad and ball are nearly co-located (corridor < 0.1 units),
            # and preserves coverage for atypical deflection geometries.
            if rng.random() < 0.7 and corridor_hi > corridor_lo + 0.1:
                x = rng.uniform(corridor_lo, corridor_hi)
            else:
                x = rng.uniform(-4.5, 4.5)
            y = rng.uniform(y_min, y_max)
            if _run_attempt(env, [(x, y, radius)]):
                return [(x, y, radius)]
        return None
    finally:
        env.close()


@register_oracle("straight_face")
def oracle(level, config, n_attempts, oracle_steps, rng) -> bool:
    return solver(level, config, n_attempts, oracle_steps, rng) is not None


# Geometric-decay model: p=0.503 per variant, model(k=20)=0.0 impossible.
# k=20 reduces expected impossible from 9 (k=10) to <1 per 10001 seeds.
register_defaults("straight_face", max_variants=20, n_attempts=100)
