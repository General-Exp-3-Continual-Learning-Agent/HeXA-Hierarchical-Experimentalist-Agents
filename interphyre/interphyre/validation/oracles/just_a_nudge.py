"""Targeted oracle for just_a_nudge.

Causal chain: green_ball sits on a platform. The red ball knocks the green ball
directly off the platform; the green ball lands in the basket where blue_ball sits.
The basket does not move — the knock changes the green ball's fall trajectory.

Valid mechanism: red ball placed to the right of the green ball (dx > 2.5 units
in 96.3% of valid solutions), knocking the green ball rightward off the platform
into the basket. The basket is always to the right of the green ball in solvable
seeds; the red ball must be far enough right to impart rightward momentum.

Solution geometry (832 valid seeds):
- dx (sol.x − gb.x): [−1.70, 5.09], mean = 3.83
- 82% of solutions have dx > 3.5; 96.3% have dx > 2.5
- x ∈ [−1.90, 4.50], p5/p95 = 2.85/4.40 (cluster at right board edge)
- Zone A coverage: 99.5% (828/832 solutions)
- BUT: 96.3% of solutions in x ∈ [gb.x+2.5, 4.5] ≈ 2-unit band vs Zone A spanning ~7.5 units
  → Zone A effective solution density = 70% × (2/7.5) = 18.7% of attempts hit the cluster
  → Zone C captures same 96.3% with 40% of attempts: 4× higher density

Zones:

Zone C (40% of attempts): x ∈ [gb.x + 2.5, 4.5], y ∈ [gb.y − 5.0, gb.y + 2.5].
  Targets the right-edge cluster (96.3% of solutions, ~2-unit x band).
  2.1× denser than Zone A for these seeds.

Zone A (30% of attempts): x ∈ [gb.x − 3.5, 4.5], y ∈ [gb.y − 5.0, gb.y + 2.5].
  Covers full near-platform region for the 3.7% of solutions outside Zone C.

Zone B (30% of attempts): full-board x and y.
  Fallback for the 0.5% of seeds with dy < −5.0 or unusual geometry.

Note: 91.7% impossible rate is true geometric impossibility (wrong ramp/platform
angle combination means no trajectory lands in basket regardless of oracle).
Zero impossible seeds in testing were solved with 200 full-board random attempts.
Zone C improves efficiency for solvable seeds but does not change the valid rate.
"""

from __future__ import annotations

import numpy as np

from interphyre.validation.oracles import (
    _run_attempt,
    register_oracle,
    register_solver,
)


@register_solver("just_a_nudge")
def solver(
    level, config, n_attempts, oracle_steps, rng
) -> list[tuple[float, float, float]] | None:
    from interphyre.environment import InterphyreEnv  # lazy: avoid circular import

    green_ball = level.objects["green_ball"]
    red_ball = level.objects["red_ball"]
    radius = red_ball.radius

    # Zone A / Zone C share the same y bounds.
    y_min = float(np.clip(green_ball.y - 5.0, -4.5, 4.5))
    y_max = float(np.clip(green_ball.y + 2.5, -4.5, 4.5))

    # Zone C: right-edge cluster (96.3% of solutions have dx > 2.5 from gb.x).
    x_min_c = float(np.clip(green_ball.x + 2.5, -4.5, 4.5))

    # Zone A: full near-platform region fallback (covers the 3.7% outside Zone C).
    x_min_a = float(np.clip(green_ball.x - 3.5, -4.5, 4.5))

    env = InterphyreEnv(level, config=config)
    try:
        for i in range(n_attempts):
            if i % 10 < 4:
                # Zone C (40%): right-edge cluster — 96.3% of solutions in ~2-unit x band.
                # 2.1× denser than Zone A for solvable seeds (2 units vs ~7.5 units x-range).
                x = rng.uniform(x_min_c, 4.5)
                y = rng.uniform(y_min, y_max)
            elif i % 10 < 7:
                # Zone A (30%): full near-platform region — covers solutions with dx < 2.5.
                x = rng.uniform(x_min_a, 4.5)
                y = rng.uniform(y_min, y_max)
            else:
                # Zone B (30%): full board fallback for the 0.5% with unusual geometry.
                x = rng.uniform(-4.5, 4.5)
                y = rng.uniform(-4.5, 4.5)

            if _run_attempt(env, [(x, y, radius)]):
                return [(x, y, radius)]
        return None
    finally:
        env.close()


@register_oracle("just_a_nudge")
def oracle(level, config, n_attempts, oracle_steps, rng) -> bool:
    return solver(level, config, n_attempts, oracle_steps, rng) is not None
