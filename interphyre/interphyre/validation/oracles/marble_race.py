"""Targeted oracle for marble_race.

Causal chain: left_beam (gray, dynamic) is a horizontal gate resting on two static
black balls. black_ball_2 sits near the LEFT end of the beam (acts as pivot).
black_ball_1 sits near the RIGHT end. Dropping red_ball on the outer right portion
of the beam — between black_ball_1 and the right edge — tips the beam clockwise
(right end down, left end up) around the left pivot. The raised left end opens the
path for green_ball (rolling down left_ramp_2) to pass along the beam and continue
via left_ramp_1 to the basket.

Empirical sweep (seeds 0-29, 50×12 grid) confirmed effective placement:
  x: [black_ball_1.x - 0.10, left_beam.right + 0.30]  (outer right arm past support)
  y: [left_beam.y + 0.15, min(left_beam.y + 2.5, ceiling_bottom - radius - 0.05)]

Physics timing: the full chain (tip → green ball traverse + ramp sequence + basket
contact) requires ≥1000 physics steps (~8 s at 60 Hz). oracle_steps=500 misses ~75%
of solvable seeds. We override to a minimum of 1500 steps so that seeds with small-
radius red balls (which tip more slowly) are correctly classified as solvable.
"""

from __future__ import annotations

import numpy as np

from interphyre.validation.oracles import register_oracle, register_solver

# Contact pairs that certify causality: the red ball must have physically
# tipped the left_beam gate. A success without this contact means the green
# ball traversed the path without the beam being tipped by the agent.
_CAUSAL_CONTACTS = frozenset(
    {
        frozenset({"red_ball", "left_beam"}),
    }
)


def _run_attempt_verified(env, positions):
    """Run one attempt via InterphyreEnv and return True only if causally linked.

    Requires both (a) success and (b) a BeginContact event in _CAUSAL_CONTACTS.
    Rejects successes where the red ball made no contact with the beam gate.
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


@register_solver("marble_race")
def solver(
    level, config, n_attempts, oracle_steps, rng
) -> list[tuple[float, float, float]] | None:
    from interphyre.environment import InterphyreEnv  # lazy: avoid circular import

    left_beam = level.objects["left_beam"]
    black_ball_1 = level.objects["black_ball_1"]  # right support
    ceiling = level.objects["ceiling"]
    red_ball = level.objects["red_ball"]

    radius = red_ball.radius

    # Right arm of left_beam: from just left of the right support to just past the
    # right edge. This is the only zone that generates clockwise tipping torque.
    x_min = np.clip(black_ball_1.x - 0.10, -4.5, 4.5)
    x_max = np.clip(left_beam.right + 0.30, -4.5, 4.5)

    # Drop height: above beam surface, but below the ceiling.
    ceiling_bottom = ceiling.y - ceiling.thickness / 2
    y_min = np.clip(left_beam.y + 0.15, -4.5, 4.5)
    y_max = float(
        np.clip(min(left_beam.y + 2.5, ceiling_bottom - radius - 0.05), -4.5, 4.5)
    )

    # Degenerate geometry: beam too close to ceiling — no valid drop zone exists.
    if y_max <= y_min:
        return None

    env = InterphyreEnv(level, config=config)
    try:
        for _ in range(n_attempts):
            x = rng.uniform(x_min, x_max)
            y = rng.uniform(y_min, y_max)
            if _run_attempt_verified(env, [(x, y, radius)]):
                return [(x, y, radius)]
        return None
    finally:
        env.close()


@register_oracle("marble_race")
def oracle(level, config, n_attempts, oracle_steps, rng) -> bool:
    return solver(level, config, n_attempts, oracle_steps, rng) is not None
