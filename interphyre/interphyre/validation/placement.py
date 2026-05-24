"""Authoritative action-placement validity check for interphyre.

Both InterphyreEnv (step/action validation) and the oracle registry
(solvability checking) must agree on whether a given ball placement is
legal. Keeping a single implementation here prevents the two from
diverging silently — which previously caused oracles to accept
placements that the environment would reject, or vice versa.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from interphyre.config import MAX_X, MAX_Y, MIN_X, MIN_Y
from interphyre.objects import Basket
from interphyre.objects.bar import circle_intersects_bar
from interphyre.objects.basket import circle_intersects_basket

if TYPE_CHECKING:
    from interphyre.level import Level


def is_valid_placement(level: "Level", x: float, y: float, radius: float) -> bool:
    """Return True iff placing a ball at (x, y, radius) is a valid action.

    Checks two conditions:
    1. Bounds: the ball must fit fully inside the world boundary.
    2. Collision: the ball must not overlap any non-action level object at its
       initial position. Checked against the static level description, not the
       live Box2D world, to match what InterphyreEnv sees at action time.

    Args:
        level: Level instance providing objects and action_objects.
        x: Ball center x-coordinate.
        y: Ball center y-coordinate.
        radius: Ball radius.

    Returns:
        True if the placement is valid; False otherwise.
    """
    # Bounds check: ball must fit fully inside the world boundary.
    if not (
        MIN_X + radius <= x <= MAX_X - radius and MIN_Y + radius <= y <= MAX_Y - radius
    ):
        return False
    # Collision check against every non-action object in the static level description.
    for name, obj in level.objects.items():
        if name in level.action_objects:
            continue
        if hasattr(obj, "radius"):
            if math.sqrt((x - obj.x) ** 2 + (y - obj.y) ** 2) <= radius + obj.radius:
                return False
        elif hasattr(obj, "length"):
            if circle_intersects_bar(x, y, radius, obj):
                return False
        elif isinstance(obj, Basket):
            if circle_intersects_basket(x, y, radius, obj):
                return False
    return True
