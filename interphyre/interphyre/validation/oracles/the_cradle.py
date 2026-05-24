"""Targeted oracle for the_cradle.

Causal chain: green_ball rests in a V-shaped cradle formed by two short bars
(left_holder at 175 deg, right_holder at 5 deg) meeting at the junction vertex below
the green_ball. The oracle must dislodge green_ball from the V so it falls to
the purple_floor below.

Prior oracle design (invalid): placed red_ball using a near-tangent lateral
approach from the side of green_ball (x_offset in [0.7, 0.99] * sum_r, y just
above tangent). It had 83% false-negative rate — the prior oracle never tried
placing the red ball ABOVE the cradle and dropping it from high on the board.

All 25 seeds solved by the full-board grid sweep have winning positions at
y in [2.59, 4.40] -- well above the cradle and the green_ball (which sits at
y in [-3, 0] depending on the seed). The mechanism is a top-down drop: the red
ball falls from high on the board, impacts the green_ball or the holder bars,
and dislodges the green_ball from the V so it falls to the purple_floor.

Dislodging mechanism: direct center hits settle the green_ball
deeper in the V; dislodging requires the red ball to hit laterally, within
sum_of_radii ~= 1.0 of gb.x horizontally. The original Zone A (+/-3.0 from gb.x,
6-unit wide) had very low solution density because the outer strips
[gb.x-3.0, gb.x-1.2] and [gb.x+1.2, gb.x+3.0] contribute almost no solutions.

Empirical solution geometry (10001 valid seeds):
- y: mean=3.85, std=0.48. 77.1% in [3.5, 4.5], 94.6% in [3.0, 4.5].
  The lower portion [2.5, 3.5] gets 50% of Zone A uniform y-samples
  but holds only 22.9% of solutions -- a 3.3x density mismatch.
- x: spread across board width, clustered within +/-1.5 of green_ball.x.

Zone A (75% of attempts): x in [gb.x +/- 1.5], y ~ Gaussian(mu=3.85, sigma=0.5,
  clipped to [2.5, 4.5]). Gaussian y concentrates 68% of Zone A y-samples in
  [3.35, 4.35] where 70.6% of solutions fall (vs uniform's 50% in same interval).
  Expected density improvement: ~40% more Zone A samples in the solution-rich zone.

Zone B (25% of attempts): x in [gb.x +/- 3.0], full-board y in [-4.5, 4.5].
  Covers seeds with shifted contact geometry and the rare 0.5% of solutions below y=2.5.
"""

from __future__ import annotations

import numpy as np

from interphyre.validation.oracles import (
    _run_attempt,
    register_defaults,
    register_oracle,
    register_solver,
)


@register_solver("the_cradle")
def solver(
    level, config, n_attempts, oracle_steps, rng
) -> list[tuple[float, float, float]] | None:
    from interphyre.environment import InterphyreEnv  # lazy: avoid circular import

    green_ball = level.objects["green_ball"]
    red_ball = level.objects["red_ball"]
    radius = red_ball.radius

    x_min_a = float(np.clip(green_ball.x - 1.5, -4.5, 4.5))
    x_max_a = float(np.clip(green_ball.x + 1.5, -4.5, 4.5))
    x_min_b = float(np.clip(green_ball.x - 3.0, -4.5, 4.5))
    x_max_b = float(np.clip(green_ball.x + 3.0, -4.5, 4.5))

    env = InterphyreEnv(level, config=config)
    try:
        for i in range(n_attempts):
            if i % 4 < 3:
                # Zone A (75%): narrow lateral-contact strip above the cradle.
                # x: +/-1.5 from gb.x keeps sampling in the dislodging contact zone.
                # y: Gaussian(3.85, 0.5) clipped to [2.5, 4.5]. Solutions cluster at
                # mean=3.85 (77.1% in [3.5, 4.5]); Gaussian concentrates 68% of samples
                # in [3.35, 4.35] where 70.6% of solutions are, vs uniform's 50%.
                x = rng.uniform(x_min_a, x_max_a)
                y = float(np.clip(rng.normal(3.85, 0.5), 2.5, 4.5))
            else:
                # Zone B (25%): wider fallback for seeds with shifted geometry;
                # full-board y covers the 5.4% of solutions below y=3.0.
                x = rng.uniform(x_min_b, x_max_b)
                y = rng.uniform(-4.5, 4.5)

            if _run_attempt(env, [(x, y, radius)]):
                return [(x, y, radius)]
        return None
    finally:
        env.close()


@register_oracle("the_cradle")
def oracle(level, config, n_attempts, oracle_steps, rng) -> bool:
    return solver(level, config, n_attempts, oracle_steps, rng) is not None


# Gaussian y sampling: solution y clusters at mean=3.85 (std=0.48). Gaussian(3.85, 0.5)
# concentrates 68% of Zone A y-samples in [3.35, 4.35] where 70.6% of solutions fall
# (vs uniform [2.5, 4.5]'s 50%). Expected p improvement: ~0.37 -> ~0.52+ per variant.
# k=20 variants with improved p gives well under 1 impossible per 10001 seeds.
register_defaults("the_cradle", max_variants=20, n_attempts=200)
