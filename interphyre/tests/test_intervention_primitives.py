"""Integration tests for core intervention primitives.

Covers: reset → place_action → step_physics → describe_scene → snapshot → restore,
plus add_object, remove_object, and step_until with boundary conditions.
"""

import pytest

from interphyre import InterphyreEnv, SimulationConfig
from interphyre.interventions.state import StateSnapshot
from interphyre.interventions.triggers import at_step, when
from interphyre.level import Level
from interphyre.objects import Ball, Bar


def _make_level(success_condition=None):
    """Minimal level for intervention testing."""
    objects = {
        "action_ball": Ball(x=-4.0, y=4.0, radius=0.5, color="red", dynamic=True),
        "target_ball": Ball(x=0.0, y=2.0, radius=0.5, color="green", dynamic=True),
        "platform": Bar(
            x=0.0, y=-2.0, length=6.0, angle=0.0, thickness=0.2, dynamic=False
        ),
    }
    return Level(
        name="intervention_test_level",
        objects=objects,
        action_objects=["action_ball"],
        success_condition=success_condition or (lambda engine: False),
        metadata={},
    )


def _make_env(success_condition=None):
    config = SimulationConfig(
        fps=60,
        time_step=1 / 60,
        enable_interventions=True,
        enable_profiling=False,
    )
    env = InterphyreEnv(_make_level(success_condition), config=config)
    return env


# ── step_physics ──


@pytest.mark.fast
def test_step_physics_advances_step_count():
    env = _make_env()
    env.reset()
    initial_count = env.step_count
    env.step_physics(10)
    assert env.step_count == initial_count + 10
    env.close()


@pytest.mark.fast
def test_step_physics_moves_dynamic_object():
    """A dynamic ball under gravity should change y-position after stepping."""
    env = _make_env()
    env.reset()
    scene_before = env.describe_scene()
    y_before = scene_before["objects"]["target_ball"]["y"]
    env.step_physics(30)
    scene_after = env.describe_scene()
    y_after = scene_after["objects"]["target_ball"]["y"]
    assert y_after != pytest.approx(y_before, abs=0.01)
    env.close()


# ── place_action ──


@pytest.mark.fast
def test_place_action_positions_object():
    env = _make_env()
    env.reset()
    env.place_action((3.0, 3.0, 0.5))
    scene = env.describe_scene()
    action_obj = scene["objects"]["action_ball"]
    assert abs(action_obj["x"] - 3.0) < 0.2
    assert abs(action_obj["y"] - 3.0) < 0.2
    env.close()


@pytest.mark.fast
def test_place_action_invalid_raises():
    """Placing on top of an existing object should raise ValueError."""
    env = _make_env()
    env.reset()
    with pytest.raises(ValueError, match="Invalid action"):
        env.place_action((0.0, 2.0, 0.5))
    env.close()


# ── describe_scene ──


@pytest.mark.fast
def test_describe_scene_keys():
    env = _make_env()
    env.reset()
    scene = env.describe_scene()
    assert "objects" in scene
    assert "contacts" in scene
    assert "step_count" in scene
    assert "success" in scene
    assert "target_ball" in scene["objects"]
    assert "platform" in scene["objects"]
    env.close()


@pytest.mark.fast
def test_describe_scene_object_fields():
    env = _make_env()
    env.reset()
    scene = env.describe_scene()
    ball = scene["objects"]["target_ball"]
    for key in ("type", "color", "x", "y", "vx", "vy", "angle", "dynamic", "size"):
        assert key in ball, f"Missing key: {key}"
    assert ball["type"] == "Ball"
    assert ball["color"] == "green"
    assert "radius" in ball["size"]
    env.close()


# ── snapshot and restore ──


@pytest.mark.fast
def test_snapshot_restore_preserves_position():
    """Capture state, step forward, restore, verify position matches snapshot."""
    env = _make_env()
    env.reset()
    env.place_action((3.0, 4.0, 0.5))
    env.step_physics(20)

    snapshot = StateSnapshot.capture(
        env.engine, metadata={"step_index": env.step_count}
    )
    scene_at_snapshot = env.describe_scene()

    # Evolve further
    env.step_physics(30)
    scene_after = env.describe_scene()
    assert scene_after["step_count"] != scene_at_snapshot["step_count"]

    # Restore
    env.restore(snapshot)
    scene_restored = env.describe_scene()
    for name in ("target_ball", "platform"):
        assert scene_restored["objects"][name]["x"] == pytest.approx(
            scene_at_snapshot["objects"][name]["x"], abs=1e-4
        )
        assert scene_restored["objects"][name]["y"] == pytest.approx(
            scene_at_snapshot["objects"][name]["y"], abs=1e-4
        )
    env.close()


@pytest.mark.fast
def test_snapshot_restore_resets_step_count():
    env = _make_env()
    env.reset()
    env.step_physics(15)
    snapshot = StateSnapshot.capture(
        env.engine, metadata={"step_index": env.step_count}
    )
    saved_count = env.step_count
    env.step_physics(20)
    assert env.step_count != saved_count
    env.restore(snapshot)
    assert env.step_count == saved_count
    env.close()


# ── full loop: reset → place → step → describe → snapshot → restore ──


@pytest.mark.fast
def test_full_intervention_loop():
    """End-to-end: reset, place action, step, snapshot, intervene, restore, continue."""
    env = _make_env()
    env.reset()

    # Place action and run a few steps
    env.place_action((3.0, 4.0, 0.5))
    env.step_physics(30)

    # Snapshot mid-simulation
    snapshot = StateSnapshot.capture(
        env.engine, metadata={"step_index": env.step_count}
    )
    scene_mid = env.describe_scene()

    # Run more physics
    env.step_physics(30)
    scene_later = env.describe_scene()
    assert scene_later["step_count"] > scene_mid["step_count"]

    # Restore to mid-point
    env.restore(snapshot)
    scene_restored = env.describe_scene()
    assert scene_restored["step_count"] == scene_mid["step_count"]

    # Continue from restored state — simulation should still work
    env.step_physics(10)
    assert env.step_count == scene_mid["step_count"] + 10
    env.close()


# ── add_object ──


@pytest.mark.fast
def test_add_object_appears_in_scene():
    env = _make_env()
    env.reset()
    new_ball = Ball(x=2.0, y=3.0, radius=0.3, color="blue", dynamic=True)
    env.add_object("injected_ball", new_ball)
    scene = env.describe_scene()
    assert "injected_ball" in scene["objects"]
    assert scene["objects"]["injected_ball"]["color"] == "blue"
    env.close()


@pytest.mark.fast
def test_add_object_duplicate_name_raises():
    env = _make_env()
    env.reset()
    duplicate = Ball(x=1.0, y=1.0, radius=0.3, color="blue", dynamic=True)
    with pytest.raises(ValueError, match="already exists"):
        env.add_object("target_ball", duplicate)
    env.close()


@pytest.mark.fast
def test_add_object_with_impulse():
    env = _make_env()
    env.reset()
    new_ball = Ball(x=2.0, y=3.0, radius=0.3, color="blue", dynamic=True)
    env.add_object("fast_ball", new_ball, impulse=(5.0, 0.0))
    # Check velocity immediately after impulse, before physics can dominate
    scene = env.describe_scene()
    assert scene["objects"]["fast_ball"]["vx"] != 0.0
    env.close()


# ── remove_object ──


@pytest.mark.fast
def test_remove_object_disappears_from_scene():
    env = _make_env()
    env.reset()
    assert "target_ball" in env.describe_scene()["objects"]
    env.remove_object("target_ball")
    assert "target_ball" not in env.describe_scene()["objects"]
    env.close()


@pytest.mark.fast
def test_remove_nonexistent_object_raises():
    env = _make_env()
    env.reset()
    with pytest.raises(ValueError, match="not found"):
        env.remove_object("no_such_object")
    env.close()


# ── step_until ──


@pytest.mark.fast
def test_step_until_immediately_true_predicate():
    """step_until with an always-true trigger should return within one step."""
    env = _make_env()
    env.reset()
    env.place_action((3.0, 4.0, 0.5))

    obs, reward, terminated, truncated, info = env.step_until(
        when(lambda engine: True), max_steps=100
    )
    # Should fire almost immediately — not run all 100 steps
    assert info["final_step"] <= 2
    env.close()


@pytest.mark.fast
def test_step_until_timeout():
    """step_until with a never-true trigger should exhaust max_steps."""
    env = _make_env()
    env.reset()
    env.place_action((3.0, 4.0, 0.5))

    obs, reward, terminated, truncated, info = env.step_until(
        when(lambda engine: False), max_steps=20
    )
    assert truncated is True
    assert info["final_step"] == 20
    env.close()


@pytest.mark.fast
def test_step_until_with_at_step_trigger():
    """step_until with at_step(n) should stop at exactly step n."""
    env = _make_env()
    env.reset()
    env.place_action((3.0, 4.0, 0.5))

    obs, reward, terminated, truncated, info = env.step_until(
        at_step(10), max_steps=100
    )
    assert info["final_step"] == 10
    env.close()
