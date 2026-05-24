"""Empirically designed oracle for wedge_issue.

Causal chain: green_ball starts near the ceiling (y≈4.5, always). Two angled
bars form a V-wedge: black_bar from the left wall, purple_bar from the right
wall. The ball must sustain contact with purple_bar for 3 seconds.

The spec B8 formula places the red ball "strictly above" the green ball, but
green_ball.y≈4.5 means y_min = green_ball.y + both_radii + 0.05 ≈ 5.4, which
clips to 4.5 = y_max — a degenerate range that gives 82% exhaustion.

Empirically identified mechanism: the red ball placed near the BLACK bar (left
side) falls onto it, deflects rightward, and lands on the left portion of the
purple bar. The green ball falls independently and slides left along the
tilted purple bar. Both balls converge at the same location and together
maintain the 3-second contact requirement.

Five complementary sampling regions (cycled across attempts) together cover
the valid placement zones observed across seeds 0–9999:

1. Near black bar (right half): x in [bb_right - 2.0, bb_right + 0.3],
   y in [bb.y, 4.5] — red ball falls onto bar and deflects rightward.
2. Near green ball (left side): x in [gb.x - 2.5, gb.x + 0.3],
   y in [gb.y - 1.0, gb.y + 0.1] — lateral contact or near-overlap push.
3. Wide left-half sweep: x in [-4.5, gb.x + 0.5], y in [bb.y - 1.0, 4.5]
   — covers geometry variants not captured by the narrower regions.
4. Precision left-half band: x in [-4.5, 0.0], y in [-4.0, 3.8].
   Dense sweep of impossible seeds confirmed valid placements cluster at
   x ∈ [−4.5, −0.44] and y ∈ [−1.5, 3.7]. Extended left bound from −4.3
   to −4.5 to cover seeds with valid zones at the world boundary (e.g. seed
   1172 with valid placements at x ∈ [−4.40, −4.10]).
5. Inter-bar bridge: x in [bb_right - 0.3, pb_left + 1.5], y in [pb.y, 4.5]
   — covers seeds where the purple bar starts far to the right of the black
   bar's right end (large length_left variance). Red ball placed in the gap
   or above the left end of the purple bar acts as a stopper when the green
   ball slides off the black bar and falls toward the purple bar.

Note: ~14% of seeds are geometrically impossible (3-second unbroken contact
cannot be achieved for any red ball placement). These seeds correctly exhaust.
"""

from __future__ import annotations

import math

import numpy as np

from interphyre.validation.oracles import (
    _run_attempt,
    register_defaults,
    register_oracle,
    register_solver,
)


@register_solver("wedge_issue")
def solver(
    level, config, n_attempts, oracle_steps, rng
) -> list[tuple[float, float, float]] | None:
    from interphyre.environment import InterphyreEnv  # lazy: avoid circular import

    green_ball = level.objects["green_ball"]
    black_bar = level.objects["black_bar"]
    purple_bar = level.objects["purple_bar"]
    red_ball = level.objects["red_ball"]
    radius = red_ball.radius

    # Black bar's right end: x-coordinate where the bar meets open air.
    # A ball placed above this region falls onto the bar and slides right.
    black_bar_angle_rad = math.radians(black_bar.angle)
    bb_right_x = np.clip(
        black_bar.x + black_bar.length / 2 * abs(math.cos(black_bar_angle_rad)),
        -4.5,
        4.5,
    )

    # Purple bar's left end: leftmost x-extent of the purple bar.
    # Used to define the inter-bar bridge region (region 4).
    purple_bar_angle_rad = math.radians(purple_bar.angle)
    pb_left_x = np.clip(
        purple_bar.x - purple_bar.length / 2 * abs(math.cos(purple_bar_angle_rad)),
        -4.5,
        4.5,
    )

    env = InterphyreEnv(level, config=config)
    try:
        for i in range(n_attempts):
            region = i % 5

            if region == 0:
                # Black bar vicinity: red ball falls onto bar, deflects right toward
                # purple bar, and acts as a stopper for the sliding green ball.
                x = rng.uniform(
                    np.clip(bb_right_x - 2.0, -4.5, 4.5),
                    np.clip(bb_right_x + 0.3, -4.5, 4.5),
                )
                y = rng.uniform(np.clip(black_bar.y, -4.5, 4.5), 4.5)

            elif region == 1:
                # Near green ball (left side): creates a lateral impulse that can
                # redirect the falling green ball toward the purple bar, or places
                # the red ball in close proximity for a stopper interaction.
                x = rng.uniform(
                    np.clip(green_ball.x - 2.5, -4.5, 4.5),
                    np.clip(green_ball.x + 0.3, -4.5, 4.5),
                )
                y = rng.uniform(
                    np.clip(green_ball.y - 1.0, -4.5, 4.5),
                    np.clip(green_ball.y + 0.1, -4.5, 4.5),
                )

            elif region == 2:
                # Wide left-half sweep: catches seeds where the valid region is at
                # an atypical position in the left half of the world.
                x = rng.uniform(-4.5, np.clip(green_ball.x + 0.5, -4.5, 4.5))
                y = rng.uniform(np.clip(black_bar.y - 1.0, -4.5, 4.5), 4.5)

            elif region == 3:
                # Precision left-half band: x ∈ [-4.5, 0.0], y ∈ [-4.0, 3.8].
                # Valid placements cluster at x ∈ [−4.5, −0.44] and y ∈ [−1.5, 3.7].
                # Left bound extended to −4.5 (from −4.3) to cover seeds whose valid
                # zone sits at the world boundary (e.g. seed 1172: x ∈ [−4.40, −4.10]).
                x = rng.uniform(-4.5, 0.0)
                y = rng.uniform(-4.0, 3.8)

            else:
                # Inter-bar bridge: covers seeds where the purple bar starts well to
                # the right of the black bar's right end (large length_left values).
                # Red ball placed in the gap or above the left end of the purple bar
                # acts as a stopper when the green ball slides off the black bar.
                x_lo = np.clip(bb_right_x - 0.3, -4.5, 4.5)
                x_hi = np.clip(pb_left_x + 1.5, -4.5, 4.5)
                x = rng.uniform(x_lo, max(x_lo + 0.1, x_hi))
                y = rng.uniform(np.clip(purple_bar.y, -4.5, 4.5), 4.5)

            if _run_attempt(env, [(x, y, radius)]):
                return [(x, y, radius)]
        return None
    finally:
        env.close()


@register_oracle("wedge_issue")
def oracle(level, config, n_attempts, oracle_steps, rng) -> bool:
    return solver(level, config, n_attempts, oracle_steps, rng) is not None


register_defaults("wedge_issue", max_variants=20, n_attempts=300)
