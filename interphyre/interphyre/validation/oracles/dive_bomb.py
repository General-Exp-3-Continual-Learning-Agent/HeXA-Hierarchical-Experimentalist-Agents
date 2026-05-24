"""Targeted oracle for dive_bomb.

Causal chain: green_ball sits on the angled cannon. Dropping red_ball above the
green_ball pushes it into the cannon chute and out toward the purple_pad.
The cannon exit is to the right of center, so we bias x toward the green_ball.

The prior two-zone oracle had 100% false-negative rate on labeled-impossible seeds. Root cause:
the gray_ball acts as a causal intermediary for 38% of valid placements — a path
completely absent from the original oracle. Three zones at weights 50/20/30:

  Zone A (50%): above green_ball — standard chute-push mechanism.
  Zone B (20%): near cannon ramp exit — widened to ramp.x ± 3.0 and y-floor
    lowered to ramp.y − 3.5 to cover edge-case seeds identified in sweep.
  Zone C (30%): gray_ball intermediary region — new causal path confirmed by
    full-board sweep; 38% of valid placements cluster here.

Note: ~24% of seeds in this level are "trivial" (green_ball already rests on
purple_pad before any action). These are correctly handled separately by the
bundle generator and should be counted as solvable, not impossible.
"""

from __future__ import annotations

import numpy as np

from interphyre.validation.oracles import (
    _run_attempt,
    register_defaults,
    register_oracle,
    register_solver,
)


@register_solver("dive_bomb")
def solver(
    level, config, n_attempts, oracle_steps, rng
) -> list[tuple[float, float, float]] | None:
    from interphyre.environment import InterphyreEnv  # lazy: avoid circular import

    green_ball = level.objects["green_ball"]
    ramp = level.objects["ramp"]
    gray_ball = level.objects["gray_ball"]
    red_ball = level.objects["red_ball"]
    radius = red_ball.radius

    # Zone A: above green_ball — standard chute-push mechanism.
    # y_max_a fixed at board ceiling (4.5): seed 1223 grid search found solution at
    # y=4.25 but green_ball.y=-0.454 → old y_max_a=3.046 missed it by 1.2 units.
    # High drops give more kinetic energy for low-y green_balls in the cannon chute.
    x_min_a = float(np.clip(green_ball.x - 1.5, -4.5, 4.5))
    x_max_a = float(np.clip(green_ball.x + 1.5, -4.5, 4.5))
    y_min_a = float(np.clip(green_ball.y + 0.2, -4.5, 4.5))
    y_max_a = (
        4.5  # always extend to board ceiling: high drops needed for low-cannon seeds
    )

    # Zone B: near cannon ramp exit — widened to ramp.x ± 3.0 and y-floor lowered
    # to ramp.y − 3.5 based on sweep analysis of edge-case seeds.
    x_min_b = float(np.clip(ramp.x - 3.0, -4.5, 4.5))
    x_max_b = float(np.clip(ramp.x + 3.0, -4.5, 4.5))
    y_min_b = float(np.clip(ramp.y - 3.5, -4.5, 4.5))
    y_max_b = float(np.clip(ramp.y + 1.5, -4.5, 4.5))

    # Zone C: gray_ball region — sweep confirmed 38% of valid placements cluster
    # near the gray_ball intermediary, which the original oracle did not model.
    x_min_c = float(np.clip(gray_ball.x - 2.0, -4.5, 4.5))
    x_max_c = float(np.clip(gray_ball.x + 2.0, -4.5, 4.5))
    y_min_c = float(np.clip(gray_ball.y - 0.5, -4.5, 4.5))
    y_max_c = float(np.clip(gray_ball.y + 2.5, -4.5, 4.5))

    env = InterphyreEnv(level, config=config)
    try:
        for i in range(n_attempts):
            zone = i % 10
            if zone < 5:
                # Zone A (50%): above green_ball.
                x = rng.uniform(x_min_a, x_max_a)
                y = rng.uniform(y_min_a, y_max_a)
            elif zone < 7:
                # Zone B (20%): ramp-exit region.
                x = rng.uniform(x_min_b, x_max_b)
                y = rng.uniform(y_min_b, y_max_b)
            else:
                # Zone C (30%): gray_ball intermediary region — new causal path
                # identified by full-board sweep of all 629 labeled-impossible seeds.
                x = rng.uniform(x_min_c, x_max_c)
                y = rng.uniform(y_min_c, y_max_c)
            if _run_attempt(env, [(x, y, radius)]):
                return [(x, y, radius)]
        return None
    finally:
        env.close()


@register_oracle("dive_bomb")
def oracle(level, config, n_attempts, oracle_steps, rng) -> bool:
    return solver(level, config, n_attempts, oracle_steps, rng) is not None


# Geometric-decay analysis (2026-04-14): p=0.395 per variant, model(k=20)=0.4 impossible.
# k=20 reduces expected impossible from 65 (k=10) to <1 per 10001 seeds.
# n_attempts raised 200→500 after audit: seed 1223 has 3 solvable non-trivial variants
# each with ~70% per-trial success at n=200 (30% failure). P(all 3 fail) ≈ 2.7% per seed.
# At n=500, per-trial success rate rises to 95%, P(all 3 fail) drops to 0.015%.
register_defaults("dive_bomb", max_variants=20, n_attempts=500)
