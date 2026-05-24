"""Targeted oracle for the_funnel.

Causal chain: green_ball starts at the top (MAX_Y). The funnel channels it
toward the center gap, then it falls to the floor. The purple_target is on one
side (left or right). A blocker bar deflects the ball away from the non-target
side. Drop red_ball near green_ball on the target side so it enters the funnel
correctly.

Fix (v2 — x-range bug): Zone B in v1 still used the same target-biased
x ∈ [cx − 2.0, cx + 2.0] as Zone A. Full-board sweeps confirmed 8/20 tested
impossible seeds (40%) have ALL valid solutions outside this x-range (e.g.,
seeds 151, 190, 244, 376 solve at x ≈ 0.57, which lies outside
oracle_x = [−4.5, −0.65] for those seeds). The v1 docstring claimed
"full-board x" for Zone B but the code shared the same x sampler as Zone A.

Fix: Zone B now samples x uniformly from the full board [−4.5, 4.5].
Zone A retains the target-biased x (covers the main high-y mechanism).

Two sampling zones (cycled across attempts):

Zone A (60% of attempts): target-biased x in [cx−2.0, cx+2.0], y in [4.35, 4.5].
  Covers the majority of seeds where the green_ball enters the funnel from the
  very top — standard mechanism, preserving existing high success rate.

Zone B (40% of attempts): full-board x [-4.5, 4.5], full-board y [-4.5, 4.5].
  Covers seeds where valid placements are outside the target-biased x-range.
  Confirmed by sweep: 8/20 impossible seeds have solutions only at x values
  on the opposite side of the board from the oracle's cx ± 2.0 range.
"""

from __future__ import annotations

import numpy as np

from interphyre.validation.oracles import (
    _run_attempt,
    register_oracle,
    register_solver,
)


@register_solver("the_funnel")
def solver(
    level, config, n_attempts, oracle_steps, rng
) -> list[tuple[float, float, float]] | None:
    from interphyre.environment import InterphyreEnv  # lazy: avoid circular import

    green_ball = level.objects["green_ball"]
    purple_target = level.objects["purple_target"]
    red_ball = level.objects["red_ball"]
    radius = red_ball.radius

    # Target-biased x center: 70% weight toward target, 30% toward green_ball.
    # Stronger bias than the original 50/50 since the ball must enter the funnel
    # on the correct side regardless of where green_ball starts.
    target_cx = (purple_target.left + purple_target.right) / 2
    cx = 0.3 * green_ball.x + 0.7 * target_cx
    x_min_a = float(np.clip(cx - 2.0, -4.5, 4.5))
    x_max_a = float(np.clip(cx + 2.0, -4.5, 4.5))

    # Zone A: y near top of board — standard mechanism (green_ball near y=4.7).
    y_min_a = float(np.clip(green_ball.y - 0.5, -4.5, 4.5))

    env = InterphyreEnv(level, config=config)
    try:
        for i in range(n_attempts):
            if i % 5 < 3:
                # Zone A (60%): target-biased x, y near green_ball.y.
                # Standard funnel entry — preserves high success rate for majority of seeds.
                x = rng.uniform(x_min_a, x_max_a)
                y = rng.uniform(y_min_a, 4.5)
            else:
                # Zone B (40%): full-board x AND full-board y.
                # Covers seeds where valid placements are outside the target-biased x-range
                # (confirmed by sweep: 8/20 impossible seeds have solutions at x values
                # entirely outside [cx-2.0, cx+2.0]).
                x = rng.uniform(-4.5, 4.5)
                y = rng.uniform(-4.5, 4.5)

            if _run_attempt(env, [(x, y, radius)]):
                return [(x, y, radius)]
        return None
    finally:
        env.close()


@register_oracle("the_funnel")
def oracle(level, config, n_attempts, oracle_steps, rng) -> bool:
    return solver(level, config, n_attempts, oracle_steps, rng) is not None


# Seed 3324 found at 10k-attempt oracle sweep — was impossible at default 500 total
# attempts (max_variants=10, n_attempts=50). Zone B (40% full-board) needs more
# attempts to cover the wide x-y solution space. n_attempts=200, max_variants=20
# gives 4000 total attempts → reliable coverage.
from interphyre.validation.oracles import register_defaults  # noqa: E402

register_defaults("the_funnel", max_variants=20, n_attempts=200)
