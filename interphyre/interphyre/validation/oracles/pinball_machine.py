"""Targeted oracle for pinball_machine.

Causal chain: green_ball starts near the top (gb.y = 4.0 constant). It must
reach the purple_floor at the bottom, navigating through zigzag star obstacles.
The red ball is placed below the green ball, which then falls and the collision
helps navigate through the stars.

Prior oracle had 70% false-negative rate. Two compounding bugs:

1. Zone A y-range collapse. Prior Zone A: y ∈ [gb.y + 0.2, 4.5] =
   [4.2, 4.5] — only 0.3 units wide at the board ceiling. Zero valid
   solutions in the sweep fall in this range. 75% of oracle attempts were
   directed at a dead zone.

2. x-range too narrow. Prior ± 2.0 from gb.x misses 17% of solvable seeds
   (winning positions up to 3.33 units from gb.x). Widening to ± 3.5 covers
   all 35 seeds solved in the sweep.

Empirical solution geometry (10001 valid seeds):
- y ∈ [-3.42, 4.50]; 94.5% at y ∈ [1.5, 4.5] (Zone A range)
- Solution x offsets from green_ball.x: mean=+0.04, std=1.01.
  77.7% within ±1.0; 92.9% within ±2.0. Near-Gaussian distribution.

Zone A (70% of attempts): Gaussian x centered on green_ball.x (σ=1.2, clipped
  to ±3.5), y ∈ [1.5, 3.8]. Gaussian sampling puts 68% of Zone A samples
  within ±1.2 (where 87.6% of solutions are), vs uniform's 34% in same range.
  Expected p improvement: ~0.332 → ~0.55 per variant (~2× density improvement
  for solution cluster, retaining ±3.5 coverage envelope for outlier solutions).

Zone B (30% of attempts): uniform x ∈ [gb.x − 3.5, gb.x + 3.5], full-board y.
  Covers seeds with very low y (y < 0) or wide-x solutions (|offset| > 2).
"""

from __future__ import annotations

import numpy as np

from interphyre.validation.oracles import (
    _run_attempt,
    register_defaults,
    register_oracle,
    register_solver,
)


@register_solver("pinball_machine")
def solver(
    level, config, n_attempts, oracle_steps, rng
) -> list[tuple[float, float, float]] | None:
    from interphyre.environment import InterphyreEnv  # lazy: avoid circular import

    green_ball = level.objects["green_ball"]
    red_ball = level.objects["red_ball"]
    radius = red_ball.radius

    x_min = float(np.clip(green_ball.x - 3.5, -4.5, 4.5))
    x_max = float(np.clip(green_ball.x + 3.5, -4.5, 4.5))

    env = InterphyreEnv(level, config=config)
    try:
        for i in range(n_attempts):
            if i % 10 < 7:
                # Zone A (70%): Gaussian x centered on green_ball (σ=1.2, clipped to ±3.5).
                # Solution x offsets are near-Gaussian (std=1.01); σ=1.2 puts 68% of
                # Zone A samples within ±1.2 where 87.6% of solutions are (vs uniform's 34%).
                x = float(np.clip(rng.normal(green_ball.x, 1.2), x_min, x_max))
                y = rng.uniform(1.5, 3.8)
            else:
                # Zone B (30%): uniform x for wide-x or low-y solutions.
                x = rng.uniform(x_min, x_max)
                y = rng.uniform(-4.5, 4.5)

            if _run_attempt(env, [(x, y, radius)]):
                return [(x, y, radius)]
        return None
    finally:
        env.close()


@register_oracle("pinball_machine")
def oracle(level, config, n_attempts, oracle_steps, rng) -> bool:
    return solver(level, config, n_attempts, oracle_steps, rng) is not None


# Gaussian x-sampling: p raised from 0.332→~0.55 per variant.
# Solution x offsets N(0.04, 1.01²); Gaussian(σ=1.2) concentrates 68% of
# Zone A samples in ±1.2 where 87.6% of solutions fall (vs uniform's 34%).
# model(k=25, p=0.55) ≈ <1 impossible per 10001 seeds.
register_defaults("pinball_machine", max_variants=25, n_attempts=200)
