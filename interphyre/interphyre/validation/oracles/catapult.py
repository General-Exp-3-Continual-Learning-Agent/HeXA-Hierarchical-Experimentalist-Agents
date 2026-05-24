"""Targeted oracle for catapult.

Causal chain: red ball dropped onto or near the catapult arm (right of pivot_ball)
adds torque that rotates the arm, launching green_ball from the left tip rightward
into the basket where blue_ball rests.

Three mechanisms (four zones):

Mechanism 1 — catapult throw (Zones A + C):
  Red ball drops onto the arm RIGHT of the pivot → torque rotates arm → green_ball
  launches from left tip into basket.
  Zone A (~54% of attempts): high drop y ∈ [arm_top + 1.0, 4.5] for max torque.
  Zone C (~15% of attempts): near-arm drop y ∈ [arm_top, arm_top + 1.0] for
    gentle push / roll-off mechanism. Covers the gap between arm surface and Zone A.
    Allows smaller radius (0.6–1.2) for precision placement on the arm.
    User-identified as missing: "create a bridge with the gray bar and roll towards
    the basket" — red ball lands on arm for gentle roll rather than full launch.

Mechanism 2 — basket destabilisation (Zone B, ~21% of attempts):
  Red ball placed near basket (x > 1.97) destabilises it or ejects blue_ball.

Mechanism 3 — indirect / bounce (Zone D, ~10% of attempts):
  Small-radius (0.6–0.9) placements anywhere in the upper half of the board.
  Covers wall-bounce trajectories, indirect arm hits, and other low-probability
  but non-zero solution paths. User-identified as missing: "sometimes you bounce
  off the wall."

oracle_steps audit (20-seed 15×15 grid sweep): 2/20 FNR = 10%. ~228/253 impossible seeds
in v5 bundle genuinely impossible (geometric impossibility, not oracle misses). Level
constraints applied in v5:
  - black_platform_x ∈ [−2.0, −1.5]: arm_right ≥ 0.725 (was −2.5; 75% of impossibility)
  - ledge_center_y ∈ [−4, −2.5]: eliminates high-basket geometry

Impossibility analysis (v5 bundle, 253 impossible seeds): No simple parameter bound
achieves 100% valid. Best single discriminator: arm_right (d=−0.47). Best combined:
arm_right − 0.5×basket_scale (d=−0.64, ratio=4.8 only at 6.7% coverage). Floating-basket
redesign (v6: ledge_center_x = arm_right + 2.5) was tested and showed no solvability
improvement (3.0% impossible vs 2.5% in v5). The impossibility arises from complex
non-linear trajectory physics — the feasibility boundary shifts when individual parameters
change. Zones C + D added to exhaust all strategy coverage before accepting remaining
seeds as genuinely impossible.
"""

from __future__ import annotations

import numpy as np

from interphyre.validation.oracles import (
    _run_attempt,
    register_defaults,
    register_oracle,
    register_solver,
)


@register_solver("catapult")
def solver(
    level, config, n_attempts, oracle_steps, rng
) -> list[tuple[float, float, float]] | None:
    from interphyre.environment import InterphyreEnv  # lazy: avoid circular import

    gray_platform = level.objects["gray_platform"]

    # Catapult throw + ballistic flight takes 8–17 simulated seconds.
    # Cap at config.max_steps: never certify solutions that exceed the user-visible
    # simulation window. Callers must pass oracle_steps = config.max_steps (1000) to
    # avoid missing solutions that complete in the 500–1000 step range.
    # oracle_steps=500 (8.3 s) truncates trajectories mid-flight;
    # full-board test at config.max_steps recovered 40% of false-negative impossible seeds.

    arm_right = gray_platform.x + gray_platform.length / 2
    arm_top = gray_platform.y + gray_platform.thickness / 2

    # pivot_x: the catapult arm rotates around gray_ball, which sits at arm center.
    # arm_right = gray_platform center + length/2; pivot = center = arm_right - length/2.
    # 100% of Zone A solutions have x right of pivot (x > pivot_x); sampling the
    # left half of Zone A (x < pivot_x) yields zero solutions and wastes 50% of attempts.
    pivot_x = arm_right - gray_platform.length / 2
    x_min_a = float(np.clip(pivot_x - 0.5, -4.5, 4.5))  # 0.5-unit margin left of pivot
    x_max_a = float(np.clip(arm_right + 1.0, -4.5, 4.5))
    y_min_a = float(np.clip(arm_top + 1.0, -4.5, 4.5))

    # Zone B: basket-destabilisation mechanism — covers ~6% of solutions.
    # All right-side solutions have x > 1.97 (near basket at x = 3.5).
    x_min_b = 2.0  # hardcoded: basket x = 3.5; all right-side solutions x > 1.97
    y_min_b = float(np.clip(arm_top + 0.5, -4.5, 4.5))

    # Zone C: near-arm placement — gentle push / bridge-and-roll mechanism.
    # Covers y ∈ [arm_top, arm_top + 1.0]: red ball grazes or lands lightly on arm,
    # giving green_ball a gentle push rather than a full catapult launch. Allows
    # smaller radius (0.6–1.2) for precision. Same x-range as Zone A (right of pivot).
    y_min_c = float(np.clip(arm_top, -4.5, 4.5))
    y_max_c = float(np.clip(arm_top + 1.0, -4.5, 4.5))

    # Zone D: small-radius indirect placements — wall bounce and other indirect paths.
    # radius ∈ [0.6, 0.9]: insufficient torque for full catapult throw but sufficient
    # for precise basket-destabilisation or indirect trajectory via wall reflection.
    # x covers the full board; y ∈ [arm_top, 4.5].
    y_min_d = float(np.clip(arm_top, -4.5, 4.5))

    env = InterphyreEnv(level, config=config)
    try:
        for i in range(n_attempts):
            zone = i % 100
            if zone < 54:
                # Zone A (54%): high drop right-of-pivot — primary catapult throw mechanism.
                # r ∈ [0.9, 1.2]: sufficient torque for reliable launch.
                radius = rng.uniform(0.9, 1.2)
                x = rng.uniform(x_min_a, x_max_a)
                y = rng.uniform(y_min_a, 4.5)
            elif zone < 75:
                # Zone B (21%): basket-destabilisation mechanism.
                # x ∈ [2.0, 4.5]; r ∈ [0.9, 1.2] for sufficient basket impact.
                radius = rng.uniform(0.9, 1.2)
                x = rng.uniform(x_min_b, 4.5)
                y = rng.uniform(y_min_b, 4.5)
            elif zone < 90:
                # Zone C (15%): near-arm placement — bridge/gentle-roll mechanism.
                # y just above arm surface; allows smaller radii for gentle push.
                radius = rng.uniform(0.6, 1.2)
                x = rng.uniform(x_min_a, x_max_a)
                y = rng.uniform(y_min_c, y_max_c)
            else:
                # Zone D (10%): small-radius indirect / wall-bounce placements.
                # Full-board x; r ∈ [0.6, 0.9] covers precision + indirect trajectories.
                radius = rng.uniform(0.6, 0.9)
                x = rng.uniform(-4.5, 4.5)
                y = rng.uniform(y_min_d, 4.5)

            if _run_attempt(env, [(x, y, radius)]):
                return [(x, y, radius)]
        return None
    finally:
        env.close()


@register_oracle("catapult")
def oracle(level, config, n_attempts, oracle_steps, rng) -> bool:
    return solver(level, config, n_attempts, oracle_steps, rng) is not None


# oracle_steps calibration: 500 steps (8.3 s simulated) truncates many trajectories.
# Full-board test with oracle_steps=1000 recovered 40% of false-negative impossible seeds —
# the catapult throw takes 8–17 s simulated; 500 steps is insufficient for many paths.
# oracle_steps must be 1000 in the bundle script (--oracle-steps 1000).
# Zones C + D added to cover bridge/roll and wall-bounce strategies identified in user audit.
register_defaults("catapult", max_variants=20, n_attempts=300)
# max_variants=20: retry pipeline confirmed seeds 6135 and 9920 required variants
# 11–19 to find a solution; max_variants=10 left them unfixable.
# n_attempts=300 per variant = 6000 total oracle calls/seed.
