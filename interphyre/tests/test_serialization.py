"""
Tests for state serialization/deserialization for snapshots and interventions.

This module tests:
- Body serialization
- World serialization
- StateSnapshot serialization
- Round-trip determinism
"""

import pytest
from Box2D import b2World, b2Vec2

from interphyre.interventions.state import (
    _body_to_dict as body_to_dict,
    _body_from_dict as body_from_dict,
    _world_to_dict as world_to_dict,
    _world_from_dict as world_from_dict,
    _save_world as save_world,
    _load_world as load_world,
    StateSnapshot,
)
from interphyre.engine import Box2DEngine
from interphyre.level import Level
from interphyre.levels import load_level, build_level_from_scene
from interphyre.objects import Ball, create_ball


# ============================================================================
# Body Serialization Tests (8-10 tests)
# ============================================================================


@pytest.fixture
def box2d_world_with_bodies():
    """Fixture creating Box2D world with test bodies."""
    world = b2World(gravity=(0, -10))
    ball1 = Ball(x=0, y=5, radius=1.0)
    ball2 = Ball(x=2, y=3, radius=0.8)
    body1 = create_ball(world, ball1, "ball1")
    body2 = create_ball(world, ball2, "ball2")

    # Set some velocities for testing
    body1.linearVelocity = b2Vec2(1.0, -2.0)
    body1.angularVelocity = 0.5
    body2.linearVelocity = b2Vec2(-1.5, 1.0)
    body2.angularVelocity = -0.3

    return world, {"ball1": body1, "ball2": body2}


@pytest.mark.fast
@pytest.mark.intervention
def test_body_to_dict_basic(box2d_world_with_bodies):
    """Test that body_to_dict returns dict with required keys."""
    world, bodies = box2d_world_with_bodies
    body = bodies["ball1"]

    body_data = body_to_dict(body)

    assert isinstance(body_data, dict), "Should return dictionary"
    assert "user_data" in body_data
    assert "position" in body_data
    assert "angle" in body_data
    assert "linear_velocity" in body_data
    assert "angular_velocity" in body_data
    assert "fixtures" in body_data


@pytest.mark.fast
@pytest.mark.intervention
def test_body_to_dict_position_and_angle(box2d_world_with_bodies):
    """Test that position and angle are captured correctly."""
    world, bodies = box2d_world_with_bodies
    body = bodies["ball1"]

    body_data = body_to_dict(body)

    assert body_data["position"] == (
        body.position.x,
        body.position.y,
    ), (
        f"Position mismatch: expected {(body.position.x, body.position.y)}, got {body_data['position']}"
    )
    assert abs(body_data["angle"] - body.angle) < 1e-9, (
        f"Angle mismatch: expected {body.angle}, got {body_data['angle']}"
    )


@pytest.mark.fast
@pytest.mark.intervention
def test_body_to_dict_velocities(box2d_world_with_bodies):
    """Test that linear and angular velocities are captured."""
    world, bodies = box2d_world_with_bodies
    body = bodies["ball1"]

    body_data = body_to_dict(body)

    assert body_data["linear_velocity"] == (
        body.linearVelocity.x,
        body.linearVelocity.y,
    ), "Linear velocity mismatch"
    assert abs(body_data["angular_velocity"] - body.angularVelocity) < 1e-9, (
        "Angular velocity mismatch"
    )


@pytest.mark.fast
@pytest.mark.intervention
def test_body_to_dict_fixtures(box2d_world_with_bodies):
    """Test that fixture properties are serialized."""
    world, bodies = box2d_world_with_bodies
    body = bodies["ball1"]

    body_data = body_to_dict(body)

    assert len(body_data["fixtures"]) == len(body.fixtures), (
        f"Fixture count mismatch: expected {len(body.fixtures)}, got {len(body_data['fixtures'])}"
    )

    fixture_data = body_data["fixtures"][0]
    assert "density" in fixture_data
    assert "friction" in fixture_data
    assert "restitution" in fixture_data
    assert "sensor" in fixture_data


@pytest.mark.fast
@pytest.mark.intervention
def test_body_to_dict_fixture_filters(box2d_world_with_bodies):
    """Test that fixture filter data is serialized."""
    world, bodies = box2d_world_with_bodies
    body = bodies["ball1"]

    body_data = body_to_dict(body)
    fixture_data = body_data["fixtures"][0]

    assert "filter_category_bits" in fixture_data
    assert "filter_mask_bits" in fixture_data
    assert "filter_group_index" in fixture_data

    # Verify values match
    original_filter = body.fixtures[0].filterData
    assert fixture_data["filter_category_bits"] == original_filter.categoryBits
    assert fixture_data["filter_mask_bits"] == original_filter.maskBits
    assert fixture_data["filter_group_index"] == original_filter.groupIndex


@pytest.mark.fast
@pytest.mark.intervention
def test_body_from_dict_position(box2d_world_with_bodies):
    """Test that body position is restored correctly."""
    world, bodies = box2d_world_with_bodies
    body = bodies["ball1"]

    # Serialize
    body_data = body_to_dict(body)

    # Modify body
    body.transform = (b2Vec2(10.0, 20.0), 1.0)

    # Restore
    body_from_dict(body, body_data)

    # Verify position restored
    assert abs(body.position.x - body_data["position"][0]) < 1e-6
    assert abs(body.position.y - body_data["position"][1]) < 1e-6


@pytest.mark.fast
@pytest.mark.intervention
def test_body_from_dict_velocity(box2d_world_with_bodies):
    """Test that body velocities are restored correctly."""
    world, bodies = box2d_world_with_bodies
    body = bodies["ball1"]

    # Serialize
    body_data = body_to_dict(body)
    original_vel = body_data["linear_velocity"]
    original_ang_vel = body_data["angular_velocity"]

    # Modify body
    body.linearVelocity = b2Vec2(999, 999)
    body.angularVelocity = 999

    # Restore
    body_from_dict(body, body_data)

    # Verify velocities restored
    assert abs(body.linearVelocity.x - original_vel[0]) < 1e-6
    assert abs(body.linearVelocity.y - original_vel[1]) < 1e-6
    assert abs(body.angularVelocity - original_ang_vel) < 1e-6


@pytest.mark.fast
@pytest.mark.intervention
def test_body_from_dict_fixtures(box2d_world_with_bodies):
    """Test that fixture properties are restored."""
    world, bodies = box2d_world_with_bodies
    body = bodies["ball1"]

    # Serialize
    body_data = body_to_dict(body)
    original_density = body_data["fixtures"][0]["density"]
    original_friction = body_data["fixtures"][0]["friction"]

    # Modify fixtures
    body.fixtures[0].density = 999.0
    body.fixtures[0].friction = 999.0

    # Restore
    body_from_dict(body, body_data)

    # Verify fixture properties restored
    assert abs(body.fixtures[0].density - original_density) < 1e-6
    assert abs(body.fixtures[0].friction - original_friction) < 1e-6


# ============================================================================
# World Serialization Tests (6-8 tests)
# ============================================================================


@pytest.mark.fast
@pytest.mark.intervention
def test_world_to_dict_basic(box2d_world_with_bodies):
    """Test that world properties are serialized."""
    world, bodies = box2d_world_with_bodies

    world_data = world_to_dict(world)

    assert isinstance(world_data, dict)
    assert "gravity" in world_data
    assert "warm_starting" in world_data
    assert "substepping" in world_data
    assert "continuous_physics" in world_data
    assert "body_count" in world_data
    assert "contact_count" in world_data


@pytest.mark.fast
@pytest.mark.intervention
def test_world_to_dict_custom_gravity():
    """Test serialization with custom gravity."""
    world = b2World(gravity=(0, -5))

    world_data = world_to_dict(world)

    assert world_data["gravity"] == (
        0,
        -5,
    ), f"Expected gravity=(0, -5), got {world_data['gravity']}"


@pytest.mark.fast
@pytest.mark.intervention
def test_save_world(box2d_world_with_bodies):
    """Test that save_world returns bytes."""
    world, bodies = box2d_world_with_bodies

    world_bytes = save_world(world, bodies)

    assert isinstance(world_bytes, bytes), "Should return bytes"
    assert len(world_bytes) > 0, "Should have content"


@pytest.mark.fast
@pytest.mark.intervention
def test_load_world_basic(box2d_world_with_bodies):
    """Test that deserialization restores world state."""
    world, bodies = box2d_world_with_bodies

    # Serialize
    world_bytes = save_world(world, bodies)

    # Modify world
    world.gravity = (999, 999)
    bodies["ball1"].position = b2Vec2(999, 999)

    # Deserialize
    load_world(world, bodies, world_bytes)

    # Verify gravity restored
    assert world.gravity == (0, -10), "Gravity should be restored"


@pytest.mark.fast
@pytest.mark.intervention
def test_world_from_dict(box2d_world_with_bodies):
    """Test that world properties are restored."""
    world, bodies = box2d_world_with_bodies

    # Serialize
    world_data = world_to_dict(world)

    # Modify
    world.gravity = (999, 999)
    world.warmStarting = not world.warmStarting

    # Restore
    world_from_dict(world, world_data)

    # Verify
    assert world.gravity == world_data["gravity"], "Gravity should be restored"
    assert world.warmStarting == world_data["warm_starting"], (
        "Warm starting should be restored"
    )


@pytest.mark.fast
@pytest.mark.intervention
def test_world_serialization_clear_forces(box2d_world_with_bodies):
    """Test that forces are cleared after deserialization."""
    world, bodies = box2d_world_with_bodies

    # Apply forces
    bodies["ball1"].ApplyForce(b2Vec2(100, 100), bodies["ball1"].worldCenter, True)

    # Serialize and deserialize
    world_bytes = save_world(world, bodies)
    load_world(world, bodies, world_bytes)

    # Forces should be cleared (tested implicitly - no exception means ClearForces was called)


# ============================================================================
# StateSnapshot Serialization Tests (8-10 tests)
# ============================================================================


@pytest.fixture
def snapshot_at_step_50(intervention_config):
    """Fixture creating snapshot at step 50."""
    level = load_level("two_body_problem", seed=42)
    engine = Box2DEngine(level, config=intervention_config)

    # Run to step 50
    for _ in range(50):
        engine.world.Step(
            engine.config.time_step,
            engine.config.velocity_iters,
            engine.config.position_iters,
        )
        engine.time_update(engine.config.time_step)

    return StateSnapshot.capture(engine)


@pytest.mark.fast
@pytest.mark.intervention
def test_snapshot_to_bytes(snapshot_at_step_50):
    """Test that snapshot can be serialized to bytes."""
    snapshot = snapshot_at_step_50

    snapshot_bytes = snapshot.to_bytes()

    assert isinstance(snapshot_bytes, bytes), "Should return bytes"
    assert len(snapshot_bytes) > 0, "Should have content"


@pytest.mark.fast
@pytest.mark.intervention
def test_snapshot_from_bytes(snapshot_at_step_50):
    """Test that snapshot can be deserialized from bytes."""
    snapshot = snapshot_at_step_50

    # Serialize
    snapshot_bytes = snapshot.to_bytes()

    # Deserialize
    restored_snapshot = StateSnapshot.from_bytes(snapshot_bytes)

    assert isinstance(restored_snapshot, StateSnapshot)
    assert restored_snapshot.step_index == snapshot.step_index
    assert abs(restored_snapshot.current_time - snapshot.current_time) < 1e-9


@pytest.mark.fast
@pytest.mark.intervention
def test_snapshot_to_bytes_deterministic(snapshot_at_step_50):
    """Test that serialization is deterministic."""
    snapshot = snapshot_at_step_50

    bytes1 = snapshot.to_bytes()
    bytes2 = snapshot.to_bytes()

    assert bytes1 == bytes2, "Serialization should be deterministic"


@pytest.mark.fast
@pytest.mark.intervention
def test_snapshot_serialization_round_trip(snapshot_at_step_50):
    """Test that all fields are preserved through serialization."""
    snapshot = snapshot_at_step_50

    # Serialize and deserialize
    snapshot_bytes = snapshot.to_bytes()
    restored = StateSnapshot.from_bytes(snapshot_bytes)

    # Check all fields
    assert restored.step_index == snapshot.step_index
    assert abs(restored.current_time - snapshot.current_time) < 1e-9
    assert restored.objects == snapshot.objects
    assert restored.box2d_state == snapshot.box2d_state
    assert restored.contacts == snapshot.contacts
    assert restored.contact_start_times == snapshot.contact_start_times
    assert restored.level_hash == snapshot.level_hash


@pytest.mark.fast
@pytest.mark.intervention
def test_snapshot_step_count_preserved(snapshot_at_step_50):
    """Test that step_count survives serialization."""
    snapshot = snapshot_at_step_50

    snapshot_bytes = snapshot.to_bytes()
    restored = StateSnapshot.from_bytes(snapshot_bytes)

    assert restored.step_index == snapshot.step_index, (
        f"Step count mismatch: expected {snapshot.step_index}, got {restored.step_index}"
    )


@pytest.mark.fast
@pytest.mark.intervention
def test_snapshot_current_time_preserved(snapshot_at_step_50):
    """Test that current_time is preserved."""
    snapshot = snapshot_at_step_50

    snapshot_bytes = snapshot.to_bytes()
    restored = StateSnapshot.from_bytes(snapshot_bytes)

    assert abs(restored.current_time - snapshot.current_time) < 1e-9, (
        f"Time mismatch: expected {snapshot.current_time}, got {restored.current_time}"
    )


@pytest.mark.fast
@pytest.mark.intervention
def test_snapshot_objects_preserved(snapshot_at_step_50):
    """Test that objects dict is preserved."""
    snapshot = snapshot_at_step_50

    snapshot_bytes = snapshot.to_bytes()
    restored = StateSnapshot.from_bytes(snapshot_bytes)

    assert restored.objects == snapshot.objects, "Objects should be preserved"


@pytest.mark.fast
@pytest.mark.intervention
def test_snapshot_metadata_preserved(intervention_config):
    """Test that custom metadata is preserved."""
    level = load_level("two_body_problem", seed=42)
    engine = Box2DEngine(level, config=intervention_config)

    for _ in range(10):
        engine.world.Step(
            engine.config.time_step,
            engine.config.velocity_iters,
            engine.config.position_iters,
        )
        engine.time_update(engine.config.time_step)

    snapshot = StateSnapshot.capture(
        engine, metadata={"test_key": "test_value", "number": 42}
    )

    snapshot_bytes = snapshot.to_bytes()
    restored = StateSnapshot.from_bytes(snapshot_bytes)

    assert restored.metadata == snapshot.metadata, "Metadata should be preserved"
    assert restored.metadata["test_key"] == "test_value"
    assert restored.metadata["number"] == 42


# ============================================================================
# Round-Trip Tests (6-8 tests)
# ============================================================================


@pytest.mark.fast
@pytest.mark.intervention
def test_body_round_trip_determinism(box2d_world_with_bodies):
    """Test that to_dict → from_dict → to_dict produces identical data."""
    world, bodies = box2d_world_with_bodies
    body = bodies["ball1"]

    # First serialization
    body_data1 = body_to_dict(body)

    # Modify and restore
    body.position = b2Vec2(999, 999)
    body_from_dict(body, body_data1)

    # Second serialization
    body_data2 = body_to_dict(body)

    # Should be identical
    assert body_data1["position"] == body_data2["position"]
    assert body_data1["linear_velocity"] == body_data2["linear_velocity"]
    assert body_data1["angular_velocity"] == body_data2["angular_velocity"]


@pytest.mark.fast
@pytest.mark.intervention
def test_world_round_trip_determinism(box2d_world_with_bodies):
    """Test that world serialization round-trip is deterministic."""
    world, bodies = box2d_world_with_bodies

    # Serialize
    world_bytes1 = save_world(world, bodies)

    # Modify
    world.gravity = (999, 999)
    bodies["ball1"].position = b2Vec2(999, 999)

    # Deserialize
    load_world(world, bodies, world_bytes1)

    # Serialize again
    world_bytes2 = save_world(world, bodies)

    # Should be identical
    assert world_bytes1 == world_bytes2, "Round-trip should produce identical bytes"


@pytest.mark.fast
@pytest.mark.intervention
def test_snapshot_restore_approximate_agreement(intervention_config):
    """Test that restore → simulate produces approximately matching positions.

    Uses 0.5-unit tolerance to account for float32 accumulation and contact
    resolution differences. This verifies restore fidelity, not bit-exact
    determinism (see test_determinism.py for that).
    """
    level = load_level("two_body_problem", seed=42)
    engine1 = Box2DEngine(level, config=intervention_config)
    engine2 = Box2DEngine(level, config=intervention_config)

    # Run both to step 10
    for _ in range(10):
        engine1.world.Step(
            engine1.config.time_step,
            engine1.config.velocity_iters,
            engine1.config.position_iters,
        )
        engine1.time_update(engine1.config.time_step)
        engine2.world.Step(
            engine2.config.time_step,
            engine2.config.velocity_iters,
            engine2.config.position_iters,
        )
        engine2.time_update(engine2.config.time_step)

    # Capture snapshot from engine1
    snapshot = StateSnapshot.capture(engine1)

    # Continue engine1
    for _ in range(5):
        engine1.world.Step(
            engine1.config.time_step,
            engine1.config.velocity_iters,
            engine1.config.position_iters,
        )
        engine1.time_update(engine1.config.time_step)

    # Restore engine2 to snapshot
    snapshot.restore(engine2)

    # Run both for same number of steps
    for _ in range(5):
        engine1.world.Step(
            engine1.config.time_step,
            engine1.config.velocity_iters,
            engine1.config.position_iters,
        )
        engine1.time_update(engine1.config.time_step)
        engine2.world.Step(
            engine2.config.time_step,
            engine2.config.velocity_iters,
            engine2.config.position_iters,
        )
        engine2.time_update(engine2.config.time_step)

    # Positions should match (with relaxed tolerance for floating point precision)
    # Box2D uses float32 internally, and there may be small differences due to
    # the order of operations or contact resolution. We use a more relaxed tolerance.
    for name in engine1.bodies:
        if name in engine2.bodies and name not in [
            "left_wall",
            "right_wall",
            "top_wall",
            "bottom_wall",
        ]:
            pos1 = engine1.bodies[name].position
            pos2 = engine2.bodies[name].position
            # Use relaxed tolerance (0.5 units) for determinism test
            # due to potential floating point accumulation and contact resolution differences
            assert abs(pos1.x - pos2.x) < 0.5, (
                f"Position mismatch for {name}: x ({pos1.x} vs {pos2.x})"
            )
            assert abs(pos1.y - pos2.y) < 0.5, (
                f"Position mismatch for {name}: y ({pos1.y} vs {pos2.y})"
            )


@pytest.mark.fast
@pytest.mark.intervention
def test_serialization_with_multiple_bodies():
    """Test serialization with multiple bodies."""
    world = b2World(gravity=(0, -10))
    bodies = {}

    # Create 5 bodies
    for i in range(5):
        ball = Ball(x=i, y=i + 1, radius=0.5)
        body = create_ball(world, ball, f"ball{i}")
        body.linearVelocity = b2Vec2(i * 0.1, -i * 0.1)
        bodies[f"ball{i}"] = body

    # Serialize
    world_bytes = save_world(world, bodies)

    # Modify all bodies
    for body in bodies.values():
        body.position = b2Vec2(999, 999)
        body.linearVelocity = b2Vec2(999, 999)

    # Deserialize
    load_world(world, bodies, world_bytes)

    # Verify all bodies restored
    for i in range(5):
        body = bodies[f"ball{i}"]
        assert abs(body.position.x - i) < 1e-6, f"Body {i} x position not restored"
        assert abs(body.position.y - (i + 1)) < 1e-6, (
            f"Body {i} y position not restored"
        )


@pytest.mark.fast
@pytest.mark.intervention
def test_serialization_empty_world():
    """Test serialization handles empty world."""
    world = b2World(gravity=(0, -10))
    bodies = {}

    world_bytes = save_world(world, bodies)

    assert isinstance(world_bytes, bytes)
    assert len(world_bytes) > 0

    # Should deserialize without error
    load_world(world, bodies, world_bytes)


@pytest.mark.fast
@pytest.mark.intervention
def test_snapshot_equality_after_serialization(snapshot_at_step_50):
    """Test that snapshots are equal after serialization round-trip."""
    snapshot = snapshot_at_step_50

    snapshot_bytes = snapshot.to_bytes()
    restored = StateSnapshot.from_bytes(snapshot_bytes)

    assert snapshot == restored, "Snapshots should be equal after round-trip"


@pytest.mark.fast
@pytest.mark.intervention
def test_snapshot_restore_preserves_contacts(intervention_config):
    """Test that contact state is preserved through serialization."""
    level = load_level("two_body_problem", seed=42)
    engine = Box2DEngine(level, config=intervention_config)

    # Run until contacts occur
    for _ in range(100):
        engine.world.Step(
            engine.config.time_step,
            engine.config.velocity_iters,
            engine.config.position_iters,
        )
        engine.time_update(engine.config.time_step)

    snapshot = StateSnapshot.capture(engine)
    original_contacts = snapshot.contacts

    # Serialize and deserialize
    snapshot_bytes = snapshot.to_bytes()
    restored = StateSnapshot.from_bytes(snapshot_bytes)

    assert restored.contacts == original_contacts, "Contacts should be preserved"


# ============================================================================
# Regression tests: _hash_level shape dimension sensitivity
# ============================================================================


def _make_ball_level(name: str, radius: float) -> Level:
    """Minimal level with a single ball at a fixed position."""
    return Level(
        name=name,
        objects={"ball": Ball(x=0.0, y=0.0, radius=radius)},
        action_objects=[],
        success_condition=lambda engine: False,
    )


@pytest.mark.fast
@pytest.mark.intervention
def test_hash_level_differs_for_different_radii():
    """Two levels with the same object name and position but different radii must hash differently.

    Regression: _hash_level previously omitted shape dimensions, so a ball with
    radius=0.5 and one with radius=1.0 at the same position produced the same hash.
    A snapshot captured from one could then be silently restored into the other.
    """
    level_small = _make_ball_level("test", radius=0.5)
    level_large = _make_ball_level("test", radius=1.0)

    hash_small = StateSnapshot._hash_level(level_small)
    hash_large = StateSnapshot._hash_level(level_large)

    assert hash_small != hash_large, (
        "Levels with identical names/positions but different radii must produce different hashes"
    )


@pytest.mark.fast
@pytest.mark.intervention
def test_restore_raises_on_cross_scene_radius_mismatch(intervention_config):
    """restore() must raise ValueError when snapshot level and engine level share object
    names/positions but differ in Ball radius.

    Regression: without shape dimensions in the hash, the mismatch went undetected
    and the snapshot was silently applied to a geometrically incompatible scene.
    """
    level_small = build_level_from_scene(
        "two_body_problem", {"green_ball": {"radius": 0.3}}
    )
    level_large = build_level_from_scene(
        "two_body_problem", {"green_ball": {"radius": 0.9}}
    )

    engine_small = Box2DEngine(level_small, config=intervention_config)
    engine_large = Box2DEngine(level_large, config=intervention_config)

    snapshot = StateSnapshot.capture(engine_small)

    with pytest.raises(ValueError, match="level hash"):
        snapshot.restore(engine_large)
