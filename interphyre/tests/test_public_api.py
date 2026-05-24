"""Tests for public API wrappers: validate_action, get_object_position, get_object_state."""

import pytest

from interphyre.environment import InterphyreEnv
from interphyre.level import Level
from interphyre.objects import Ball, Bar


def _make_level():
    objects = {
        "action_ball": Ball(x=-4.0, y=4.0, radius=0.5, color="red", dynamic=True),
        "green_ball": Ball(x=0.0, y=2.0, radius=0.5, color="green", dynamic=True),
        "bar": Bar(x=0.0, y=-2.0, length=4.0, angle=0.0, thickness=0.2, dynamic=False),
    }
    return Level(
        name="api_test_level",
        objects=objects,
        action_objects=["action_ball"],
        success_condition=lambda engine: False,
        metadata={},
    )


# ── validate_action ──


@pytest.mark.fast
def test_validate_action_valid():
    env = InterphyreEnv(_make_level())
    env.reset()
    result = env.validate_action([(3.0, 3.0, 0.5)])
    assert result["invalid"] is False
    assert result["action"] is not None
    assert result["error"] is None
    env.close()


@pytest.mark.fast
def test_validate_action_invalid_placement():
    env = InterphyreEnv(_make_level())
    env.reset()
    # Place directly on top of existing green_ball at (0, 2)
    result = env.validate_action([(0.0, 2.0, 0.5)])
    assert result["invalid"] is True
    assert result["error"] is not None
    env.close()


@pytest.mark.fast
def test_validate_action_bad_format():
    env = InterphyreEnv(_make_level())
    env.reset()
    result = env.validate_action("not_an_action")
    assert result["invalid"] is True
    assert result["error"] is not None
    env.close()


# ── get_object_position ──


@pytest.mark.fast
def test_get_object_position_after_reset():
    env = InterphyreEnv(_make_level())
    env.reset()
    pos = env.get_object_position("green_ball")
    assert isinstance(pos, tuple)
    assert len(pos) == 2
    # Should be near the construction-time position
    assert abs(pos[0] - 0.0) < 0.1
    assert abs(pos[1] - 2.0) < 0.1
    env.close()


@pytest.mark.fast
def test_get_object_position_unknown_object():
    env = InterphyreEnv(_make_level())
    env.reset()
    with pytest.raises(KeyError, match="no_such_object"):
        env.get_object_position("no_such_object")
    env.close()


@pytest.mark.fast
def test_get_object_position_matches_engine():
    """Public accessor returns the same values as direct engine access."""
    env = InterphyreEnv(_make_level())
    env.reset()
    env.step([(3.0, 3.0, 0.5)])  # run a step to move physics forward
    pos = env.get_object_position("green_ball")
    body = env.engine.bodies["green_ball"]
    assert pos == (float(body.position.x), float(body.position.y))
    env.close()


# ── get_object_state ──


@pytest.mark.fast
def test_get_object_state_keys():
    env = InterphyreEnv(_make_level())
    env.reset()
    state = env.get_object_state("green_ball")
    expected_keys = {"x", "y", "vx", "vy", "angle", "angular_velocity", "dynamic"}
    assert set(state.keys()) == expected_keys
    env.close()


@pytest.mark.fast
def test_get_object_state_dynamic_flag():
    env = InterphyreEnv(_make_level())
    env.reset()
    assert env.get_object_state("green_ball")["dynamic"] is True
    assert env.get_object_state("bar")["dynamic"] is False
    env.close()


@pytest.mark.fast
def test_get_object_state_unknown_object():
    env = InterphyreEnv(_make_level())
    env.reset()
    with pytest.raises(KeyError, match="missing"):
        env.get_object_state("missing")
    env.close()
