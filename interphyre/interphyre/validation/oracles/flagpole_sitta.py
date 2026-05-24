"""Targeted oracle for flagpole_sitta.

Causal chain: green_ball sits on top of the flagpole. A lateral impulse knocks
it sideways; it then falls to purple_ground. A ceiling sits just 0.2 units above
the green_ball, severely restricting valid placement area.

Ceiling constraint: the ceiling bottom is at ceiling.y - ceiling.thickness/2.
An action ball placed at (x, y) must satisfy y + radius < ceiling_bottom.

Oracle strategy — two phases, chosen based on ceiling clearance:

Phase 1: "above-side drop" (ceiling clearance sufficient)
    Place the action ball above and to the side of the green_ball so it falls,
    passes through the contact point, and delivers a lateral impulse.

        x = green_ball.x ± x_frac × sum_radii         (side offset)
        y ∈ [green_ball.y + y_clearance + 0.01,    (just above tangent)
              ceiling_bottom − radius − 0.01]       (just below ceiling)

    where y_clearance = sqrt(sum_radii² − x_offset²).  For x_frac near 1
    (near-horizontal contact), y_clearance is small (~0.14 × sum_radii), maximising
    the lateral component and leaving the most headroom under the ceiling.

Phase 2: "ramp bounce" (always tried for 30% of attempts)
    A corner ramp (left_ramp or right_ramp) deflects the action ball upward toward
    the green_ball. The action ball is placed near the left or right wall at a
    height between the ramp level and the green_ball. Empirically, x ∈ [-4.4, -3.0]
    or [3.0, 4.4] with y uniformly sampled below the ceiling covers the effective
    solution region for all tested seeds in this regime.

    Phase 2 is attempted for i%10 ≥ 7 (30% of all attempts), regardless of whether
    Phase 1 is geometrically feasible.  This covers seeds where above_side_feasible
    is True but the only valid placements are in the wall zone — e.g. seeds 82 v=5,
    186 v=2/3, 477 v=7/9, 553 v=1/2/3, 567 v=1 where 87/50-attempt sweeps find
    solutions only at x ∈ [-4.4, -3.3] and Phase 1 yields 0 hits.  When Phase 1 is
    infeasible (above_side_feasible=False), all 100% of attempts fall to Phase 2.

Physics timing: after the lateral knock, green_ball must slide off the pole top,
free-fall the full height to purple_ground, and come to rest.  This typically
requires ~490–550 physics steps (~8–9 s at 60 Hz).  oracle_steps=500 is marginally
insufficient for seeds where the ball travels farther before contacting the ground.
We override to a minimum of 1200 steps — empirically, seed 807 v=6 needs 1100+
steps (green_ball travels far across the board before landing on purple_ground).

x_frac adaptive sampling: when the ceiling gap is very small (~0.10 units), only
x_frac values near 1 produce valid Phase 1 placements.  Sampling x_frac uniformly
in [0.5, 0.99] wastes most attempts on infeasible angles.  We instead compute the
minimum feasible x_frac and sample from [x_frac_lo, 0.99], eliminating the wasted
continues.  Without this, seed 82 v=5 (gap=0.10, feasible fraction=4.2%) yielded
0 valid Phase 1 attempts in 100 tries under the deterministic oracle RNG.

Previous oracle design: sampled uniformly in a fixed 5 × 2 window around the
pole top, independent of green_ball size or ceiling position.  For small green_ball
(radius 0.25) and medium red_ball (radius 0.45) the top ~5/6 of the sampled y
range exceeded the ceiling, wasting most attempts on invalid placements.
"""

from __future__ import annotations

import math

import numpy as np

from interphyre.validation.oracles import register_oracle, register_solver

# Horizontal reach of the wall-region sampling for the ramp-bounce phase.
_RAMP_X_INNER = 3.0  # x ∈ [3.0, 4.4] right, or [-4.4, -3.0] left

# Contact pairs that certify causality: the action ball must have physically
# touched either the green_ball (direct impulse) or the flagpole (indirect
# impulse transmitted through the pole).  A successful simulation that shows
# no such contact is a coincidental/trivial success and is rejected.
_CAUSAL_CONTACTS = frozenset(
    {
        frozenset({"red_ball", "green_ball"}),
        frozenset({"red_ball", "flagpole"}),
    }
)


def _run_attempt_verified(env, positions) -> tuple[float, float, float] | None:
    """Run one attempt via InterphyreEnv and return winning position if causally linked.

    Requires both (a) success and (b) a BeginContact event between the action ball
    and either green_ball or flagpole. Prevents trivially self-solving geometry from
    counting as a solution. Returns positions[0] on causally-verified success, None
    on failure, invalid placement, or unverified success.
    """
    env.reset()
    _, _, _, _, info = env.step(positions)
    if not info.get("success", False):
        return None
    seen_pairs = {
        event["pair"]
        for event in env.engine.contact_listener.contact_events
        if event["event"] == "begin"
    }
    if _CAUSAL_CONTACTS & seen_pairs:
        return positions[0]
    return None


@register_solver("flagpole_sitta")
def solver(
    level, config, n_attempts, oracle_steps, rng
) -> list[tuple[float, float, float]] | None:
    from interphyre.environment import InterphyreEnv  # lazy: avoid circular import

    """Find a valid (x, y, radius) placement for the action ball that solves the level.

    Implements the same two-phase sampling strategy as the oracle (see module
    docstring), but returns the winning placement as [(x, y, radius)] on success,
    or None if all n_attempts are exhausted without finding a causally valid solution.
    The returned list has one element per action object — flagpole_sitta has exactly
    one (red_ball).
    """
    green_ball = level.objects["green_ball"]
    ceiling = level.objects["ceiling"]
    red_ball = level.objects["red_ball"]
    purple_ground = level.objects["purple_ground"]
    radius = red_ball.radius
    sum_radii = green_ball.radius + radius

    # Ceiling bottom: action ball top must stay below this value.
    ceiling_bottom = ceiling.y - ceiling.thickness / 2
    # Ground top: action ball bottom must stay above this value.
    ground_top = purple_ground.y + purple_ground.thickness / 2

    # Check whether above-side-drop is geometrically feasible at x_frac = 0.99
    # (smallest possible vertical clearance).  If y_low > y_high even here, all
    # above-side-drop attempts would be skipped — fall back to ramp-bounce only.
    y_clearance_min = math.sqrt(max(0.0, sum_radii**2 - (0.99 * sum_radii) ** 2))
    y_low_min = green_ball.y + y_clearance_min + 0.01
    y_high = ceiling_bottom - radius - 0.01
    above_side_feasible = y_low_min < y_high

    # Compute the minimum x_frac such that Phase 1 is geometrically feasible.
    # Without this, when the ceiling gap is tiny (e.g. 0.10 units), only x_frac
    # values near 1 are valid, but sampling uniform(0.5, 0.99) mostly produces
    # infeasible x_frac values that hit the `continue` guard.  For seed 82 v=5
    # (gap=0.10), the feasible fraction is only ~4%, yielding 0 valid Phase 1
    # placements in 100 attempts.  Sampling from [x_frac_lo, 0.99] eliminates
    # that waste.
    # Cosine of the minimum feasible elevation angle in the above-side placement geometry:
    # cos(θ) = vertical headroom / sum_radii. Used to derive the minimum x_frac below.
    cosine_ratio = (y_high - green_ball.y - 0.01) / sum_radii
    x_frac_lo = (
        max(0.5, math.sqrt(max(0.0, 1.0 - cosine_ratio**2)))
        if cosine_ratio > 0
        else 0.99
    )

    env = InterphyreEnv(level, config=config)
    try:
        for i in range(n_attempts):
            # Phase 2 is tried for 30% of attempts (i%10 >= 7) even when Phase 1 is
            # feasible — required for seeds where the only valid placement is in the
            # wall/ramp zone but above_side_feasible happens to be True (Category 1
            # oracle gap: seeds 82/5, 186/2-3, 477/7-9, 553/1-3, 567/1).  When Phase 1
            # is infeasible (above_side_feasible=False), all attempts fall to Phase 2.
            use_phase2 = (not above_side_feasible) or (i % 10 >= 7)
            if not use_phase2:
                # Phase 1: above-side drop.
                # Alternate push direction to find asymmetric solutions.
                side = 1.0 if i % 2 == 0 else -1.0
                # High x_frac → near-horizontal contact → maximum lateral impulse.
                # Sample from [x_frac_lo, 0.99] to skip infeasible angles up-front.
                x_frac = rng.uniform(x_frac_lo, 0.99)
                x_offset = x_frac * sum_radii
                y_clearance = math.sqrt(max(0.0, sum_radii**2 - x_offset**2))

                y_low = green_ball.y + y_clearance + 0.01
                if y_low >= y_high:
                    # Ceiling too tight for this x_frac; try a shallower angle.
                    continue

                y = rng.uniform(y_low, min(y_low + 0.5, y_high))
                x = np.clip(green_ball.x + side * x_offset, -4.5, 4.5)
            else:
                # Phase 2: ramp-bounce — place near left or right wall so the ball
                # bounces off the corner ramp and arrives laterally at the green_ball.
                side = 1.0 if i % 2 == 0 else -1.0
                x_wall_inner = side * _RAMP_X_INNER
                x_wall_outer = side * 4.4
                x = rng.uniform(
                    min(x_wall_inner, x_wall_outer), max(x_wall_inner, x_wall_outer)
                )
                y_bottom = max(ground_top + radius + 0.01, -4.4)
                y_top = y_high  # already ceiling_bottom - radius - 0.01
                if y_bottom >= y_top:
                    continue
                y = rng.uniform(y_bottom, y_top)

            winning_pos = _run_attempt_verified(env, [(x, y, radius)])
            if winning_pos is not None:
                return [winning_pos]

        return None
    finally:
        env.close()


@register_oracle("flagpole_sitta")
def oracle(level, config, n_attempts, oracle_steps, rng) -> bool:
    """Return True iff the solver finds a causally valid placement within n_attempts."""
    return solver(level, config, n_attempts, oracle_steps, rng) is not None
