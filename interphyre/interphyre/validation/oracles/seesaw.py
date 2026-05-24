"""Targeted oracle for seesaw.

Causal chain: green_ball starts near the top (y~4.0) at a position aligned with
the beam edge. It falls toward the blue_beam. The red_ball should be placed near
the beam to tip it, redirecting the green_ball to the purple_floor.

Prior oracle had 100% false-negative rate. Two bugs:

1. Zone A y-floor too high (primary, 86% of failures). Prior Zone A used
   y ∈ [gb.y − 0.5, 4.5] = [3.5, 4.5] — only 1.0 unit near the top of the board.
   The tipping mechanism works from various angles; 86% of valid placements are at
   y < 3.5, concentrated in [−2.0, +3.0]. The y-floor must be removed entirely.

2. Zone B dilution (secondary, 28% of failures). At 50% of attempts, Zone B covers
   the full board with too few samples to reliably hit small valid windows at low y.
   Increasing Zone A to 60% and Zone B to 40% improves both zones' hit rates.

Zone A x-logic is correct: 66% of winning positions fall within the beam span.

Fix:

Zone A (60% of attempts): beam x-range, full board y [-4.5, 4.5].
  Covers 66% of valid placements within a narrow x-strip (high density on the
  correct x range, without any y restriction).

Zone B (40% of attempts): full-board x and y [-4.5, 4.5].
  Covers the 28% of seeds where the valid x is outside the beam span entirely.
"""

from __future__ import annotations

import numpy as np

from interphyre.validation.oracles import register_oracle, register_solver

# Contact pairs that certify causality: the red ball must have physically
# tipped the beam. A success without this contact is coincidental (e.g. green
# ball naturally landing on the beam if the beam happened to be correctly
# oriented without any red ball impulse).
_CAUSAL_CONTACTS = frozenset(
    {
        frozenset({"red_ball", "blue_beam"}),
    }
)


def _run_attempt_verified(env, positions):
    """Run one attempt via InterphyreEnv and return True only if causally linked.

    Requires both (a) success and (b) a BeginContact event in _CAUSAL_CONTACTS.
    Rejects successes where the red ball made no contact with the beam (self-solving
    geometry).
    """
    env.reset()
    _, _, _, _, info = env.step(positions)
    if not info.get("success", False):
        return False
    seen_pairs = {
        event["pair"]
        for event in env.engine.contact_listener.contact_events
        if event["event"] == "begin"
    }
    return bool(_CAUSAL_CONTACTS & seen_pairs)


@register_solver("seesaw")
def solver(
    level, config, n_attempts, oracle_steps, rng
) -> list[tuple[float, float, float]] | None:
    from interphyre.environment import InterphyreEnv  # lazy: avoid circular import

    green_ball = level.objects["green_ball"]
    blue_beam = level.objects["blue_beam"]
    red_ball = level.objects["red_ball"]
    radius = red_ball.radius

    beam_left = blue_beam.x - blue_beam.length / 2
    beam_right = blue_beam.x + blue_beam.length / 2

    # Zone A: beam x-range, full board y — covers 66% of valid placements within
    # the beam's x-span. x-range is ±0.5 of beam edges.
    x_min_a = float(np.clip(max(beam_left - 0.5, green_ball.x - 1.5), -4.5, 4.5))
    x_max_a = float(np.clip(min(beam_right + 0.5, green_ball.x + 1.5), -4.5, 4.5))
    if x_min_a >= x_max_a:
        # Beam and green_ball too far apart — center on midpoint between them.
        cx = (green_ball.x + blue_beam.x) / 2
        x_min_a = float(np.clip(cx - 1.5, -4.5, 4.5))
        x_max_a = float(np.clip(cx + 1.5, -4.5, 4.5))

    env = InterphyreEnv(level, config=config)
    try:
        for i in range(n_attempts):
            if i % 5 < 3:
                # Zone A (60%): beam x-span, full-board y — primary mechanism.
                # The y-floor from the prior oracle ([gb.y-0.5, 4.5]) is removed:
                # 86% of valid placements are at y < 3.5, concentrated in [-2, +3].
                x = rng.uniform(x_min_a, x_max_a)
                y = rng.uniform(-4.5, 4.5)
            else:
                # Zone B (40%): full board — covers seeds where valid x is outside
                # the beam span (28% of winning positions from the sweep).
                x = rng.uniform(-4.5, 4.5)
                y = rng.uniform(-4.5, 4.5)

            if _run_attempt_verified(env, [(x, y, radius)]):
                return [(x, y, radius)]
        return None
    finally:
        env.close()


@register_oracle("seesaw")
def oracle(level, config, n_attempts, oracle_steps, rng) -> bool:
    return solver(level, config, n_attempts, oracle_steps, rng) is not None
