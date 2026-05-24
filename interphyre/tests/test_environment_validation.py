"""
Tests for action placement validity — shared logic used by both InterphyreEnv
and the oracle registry via interphyre.validation.placement.is_valid_placement.
"""

import pytest

from interphyre.level import Level
from interphyre.objects import Ball, Bar, Basket
from interphyre.validation.placement import is_valid_placement


def _make_validation_level():
    def success_condition(engine):
        return False

    objects = {
        "action_ball": Ball(x=-4.0, y=4.0, radius=0.5, color="red", dynamic=True),
        "static_ball": Ball(x=0.0, y=0.0, radius=0.5, color="green", dynamic=False),
        "bar": Bar(x=2.0, y=0.0, length=4.0, angle=0.0, thickness=0.2, dynamic=False),
        "basket": Basket(x=-2.0, y=0.0, bottom_width=2.0, enable_sensor=False),
    }
    return Level(
        name="validation_level",
        objects=objects,
        action_objects=["action_ball"],
        success_condition=success_condition,
        metadata={},
    )


@pytest.mark.fast
def test_is_within_bounds_edges():
    """Placement bounds should reject coordinates outside world limits."""
    level = _make_validation_level()
    # (4.0, -4.0) is far from all level objects — valid inside bounds.
    assert is_valid_placement(level, 4.0, -4.0, 0.1) is True
    # x=5.0, radius=0.6: right edge (5.0 + 0.6 > 5.0) — out of bounds.
    assert is_valid_placement(level, 5.0, 0.0, 0.6) is False
    # x=-5.0, radius=0.6: left edge — out of bounds.
    assert is_valid_placement(level, -5.0, 0.0, 0.6) is False


@pytest.mark.fast
def test_would_collide_with_ball():
    """Collision check should flag overlap with existing balls."""
    level = _make_validation_level()
    # (0.2, 0.0) with radius 0.5 overlaps static_ball at (0.0, 0.0, r=0.5).
    assert is_valid_placement(level, 0.2, 0.0, 0.5) is False
    # (3.5, 3.5) with radius 0.2 is far from all objects — valid.
    assert is_valid_placement(level, 3.5, 3.5, 0.2) is True


@pytest.mark.fast
def test_would_collide_with_bar():
    """Collision check should flag overlap with bars."""
    level = _make_validation_level()
    # (2.0, 0.0) is the center of the bar at x=2.0, y=0.0.
    assert is_valid_placement(level, 2.0, 0.0, 0.5) is False


@pytest.mark.fast
def test_would_collide_with_basket_wall():
    """Collision check should flag overlap with basket walls."""
    level = _make_validation_level()
    basket = level.objects["basket"]
    x = basket.x + basket.total_width / 2 - 0.05
    assert is_valid_placement(level, x, basket.y, 0.2) is False


@pytest.mark.fast
def test_would_collide_with_basket_wall_upper():
    """Collision check should catch overlaps with the upper half of basket walls.

    The basket anchor (y=0) is the floor bottom for anchor='bottom_center'.
    The full basket wall extends from y=0 to y=total_height.  A placement near
    the upper wall portion (above y=total_height/2) must still be caught.
    Previously, a bounding-box offset bug caused the upper half to be invisible.
    """
    level = _make_validation_level()
    basket = level.objects["basket"]
    # Point near the right wall at 75% of basket height — well above basket.y.
    x = basket.x + basket.total_width / 2 - 0.05
    y = basket.y + basket.total_height * 0.75
    assert is_valid_placement(level, x, y, 0.2) is False


@pytest.mark.fast
def test_action_objects_are_skipped():
    """Action objects should not count as obstacles for placement validation."""
    level = _make_validation_level()
    # action_ball is at (-4.0, 4.0, r=0.5) — placing there should be valid
    # because action objects are ignored during collision checking.
    assert is_valid_placement(level, -4.0, 4.0, 0.5) is True
