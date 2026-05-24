"""Targeted oracle for pass_the_parcel.

Causal chain: inverted top_basket sits on the platform; green_ball is next to it.
Pushing the top_basket off the platform causes green_ball to roll into the bottom
basket and contact the blue_ball. Drop above the top_basket.

Fix (this version): The original oracle used y ∈ [top_basket.y + 0.2,
top_basket.y + 3.5] — a 3.3-unit tall window. Fine-grid sweeps confirmed the
valid drop zone is only ~0.2 units above the basket rim (a low-energy graze);
higher drops are too fast and bounce without toppling. This made the valid
region ~1.3% of the sampling area, giving only ~48% success per variant at 50
attempts. Fix: tighten y to [+0.1, +1.5] (~10× density improvement) and
widen x rightward to +3.0 to capture ramp-assisted slides when top_basket is
in the right half of the board.
"""

from __future__ import annotations

import numpy as np

from interphyre.validation.oracles import (
    _run_attempt,
    register_oracle,
    register_solver,
)


@register_solver("pass_the_parcel")
def solver(
    level, config, n_attempts, oracle_steps, rng
) -> list[tuple[float, float, float]] | None:
    from interphyre.environment import InterphyreEnv  # lazy: avoid circular import

    top_basket = level.objects["top_basket"]
    red_ball = level.objects["red_ball"]
    radius = red_ball.radius

    x_min = np.clip(top_basket.x - 2.0, -4.5, 4.5)
    # Extend rightward to capture ramp-assisted slides when top_basket is in
    # the right half of the board (hits confirmed up to top_basket.x + 3.0).
    x_max = np.clip(top_basket.x + 3.0, -4.5, 4.5)
    # Tightened from +3.5 to +1.5: valid drops are a low-energy graze just
    # above the rim; higher drops bounce rather than topple the basket.
    y_min = np.clip(top_basket.y + 0.1, -4.5, 4.5)
    y_max = np.clip(top_basket.y + 1.5, -4.5, 4.5)

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


@register_oracle("pass_the_parcel")
def oracle(level, config, n_attempts, oracle_steps, rng) -> bool:
    return solver(level, config, n_attempts, oracle_steps, rng) is not None


# Seed 4846 found at 10k-attempt oracle sweep — was impossible at default 500 total
# attempts (max_variants=10, n_attempts=50). n_attempts=200, max_variants=20 gives
# 4000 total attempts → reliable coverage for edge-case seeds.
from interphyre.validation.oracles import register_defaults  # noqa: E402

register_defaults("pass_the_parcel", max_variants=20, n_attempts=200)
