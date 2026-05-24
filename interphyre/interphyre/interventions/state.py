"""
State snapshot and restoration functionality.

This module provides the StateSnapshot class for capturing and restoring
complete simulation state, enabling deterministic replay and branching.
"""

import hashlib
import pickle
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from Box2D import b2World, b2Body, b2Vec2

if TYPE_CHECKING:
    from interphyre.engine import Box2DEngine


# Box2D Serialization Helpers


def _body_to_dict(body: b2Body) -> dict[str, Any]:
    """
    Serialize a single Box2D body to a dictionary.

    Args:
        body: The b2Body to serialize

    Returns:
        Dictionary containing all body state
    """
    fixtures = []
    for fixture in body.fixtures:
        fixture_data = {
            "density": fixture.density,
            "friction": fixture.friction,
            "restitution": fixture.restitution,
            "sensor": fixture.sensor,
            "filter_category_bits": fixture.filterData.categoryBits,
            "filter_mask_bits": fixture.filterData.maskBits,
            "filter_group_index": fixture.filterData.groupIndex,
        }
        fixtures.append(fixture_data)

    return {
        "user_data": body.userData,
        "position": (body.position.x, body.position.y),
        "angle": body.angle,
        "linear_velocity": (body.linearVelocity.x, body.linearVelocity.y),
        "angular_velocity": body.angularVelocity,
        "linear_damping": body.linearDamping,
        "angular_damping": body.angularDamping,
        "gravity_scale": body.gravityScale,
        "bullet": body.bullet,
        "awake": body.awake,
        "active": body.active,
        "fixed_rotation": body.fixedRotation,
        "type": body.type,  # 0=static, 1=kinematic, 2=dynamic
        "fixtures": fixtures,
    }


def _body_from_dict(body: b2Body, body_data: dict[str, Any]) -> None:
    """
    Restore a Box2D body's state from serialized data.

    This updates an existing body rather than creating a new one, since
    the body must already exist with the correct fixtures/shapes from
    the level definition.

    Args:
        body: The existing b2Body to update
        body_data: Dictionary containing serialized body state
    """
    # Wake the body first to ensure it can be modified
    was_awake = body_data["awake"]
    if not was_awake:
        body.awake = True

    # Restore transform (position and angle)
    body.transform = (b2Vec2(*body_data["position"]), body_data["angle"])

    # Restore velocities
    body.linearVelocity = b2Vec2(*body_data["linear_velocity"])
    body.angularVelocity = body_data["angular_velocity"]
    body.linearDamping = body_data["linear_damping"]
    body.angularDamping = body_data["angular_damping"]
    body.gravityScale = body_data["gravity_scale"]
    body.bullet = body_data["bullet"]
    body.active = body_data["active"]
    body.fixedRotation = body_data["fixed_rotation"]
    # Note: body.type cannot be changed after creation

    # Restore fixture properties
    for fixture, fixture_data in zip(body.fixtures, body_data["fixtures"]):
        fixture.density = fixture_data["density"]
        fixture.friction = fixture_data["friction"]
        fixture.restitution = fixture_data["restitution"]
        fixture.sensor = fixture_data["sensor"]
        # Filter data
        filter_data = fixture.filterData
        filter_data.categoryBits = fixture_data["filter_category_bits"]
        filter_data.maskBits = fixture_data["filter_mask_bits"]
        filter_data.groupIndex = fixture_data["filter_group_index"]
        fixture.filterData = filter_data

    # Reset mass data after modifying fixtures
    body.ResetMassData()

    # Restore awake state
    body.awake = was_awake


def _world_to_dict(world: b2World) -> dict[str, Any]:
    """
    Serialize Box2D world-level properties.

    Args:
        world: The b2World to serialize

    Returns:
        Dictionary containing world properties
    """
    return {
        "gravity": (world.gravity.x, world.gravity.y),
        "warm_starting": world.warmStarting,
        "substepping": world.subStepping,
        "continuous_physics": world.continuousPhysics,
        "body_count": world.bodyCount,
        "contact_count": world.contactCount,
    }


def _world_from_dict(world: b2World, world_data: dict[str, Any]) -> None:
    """
    Restore Box2D world-level properties.

    Args:
        world: The b2World to update
        world_data: Dictionary containing serialized world properties
    """
    world.gravity = tuple(world_data["gravity"])
    world.warmStarting = world_data["warm_starting"]
    world.subStepping = world_data["substepping"]
    world.continuousPhysics = world_data["continuous_physics"]


def _save_world(world: b2World, body_names: dict[str, b2Body]) -> bytes:
    """
    Serialize complete Box2D world state to bytes.

    Args:
        world: The b2World to serialize
        body_names: Mapping from object names to b2Body objects

    Returns:
        Pickled bytes containing complete world state
    """
    state = {
        "world_properties": _world_to_dict(world),
        "bodies": {
            name: _body_to_dict(body) for name, body in sorted(body_names.items())
        },
    }
    return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)


def _load_world(world: b2World, body_names: dict[str, b2Body], data: bytes) -> None:
    """
    Restore Box2D world state from serialized bytes.

    This updates the existing world and bodies rather than creating new ones,
    since the world structure (bodies, fixtures, shapes) must match the level.

    Args:
        world: The existing b2World to update
        body_names: Mapping from object names to existing b2Body objects
        data: Pickled bytes containing serialized world state
    """
    state = pickle.loads(data)

    # Restore world properties
    _world_from_dict(world, state["world_properties"])

    # Restore body states
    for name, body_data in state["bodies"].items():
        if name in body_names:
            _body_from_dict(body_names[name], body_data)

    # Clear forces after restoration
    world.ClearForces()


# StateSnapshot


@dataclass(frozen=True)
class StateSnapshot:
    """
    Immutable snapshot of complete simulation state.

    This captures everything needed to restore the simulation to an exact
    state, including Box2D physics state, contact tracking, and metadata.

    Attributes:
        step_index: Simulation step index when snapshot was taken
        current_time: Simulation time in seconds
        objects: PhyreObject state (position, velocity, etc.)
        box2d_state: Serialized Box2D world state
        contacts: Set of active contact pairs
        contact_start_times: Start time of each contact
        level_hash: Hash of level configuration for validation
        metadata: Optional user-provided metadata
    """

    step_index: int
    current_time: float
    objects: dict[str, dict[str, Any]]
    box2d_state: bytes
    contacts: frozenset[frozenset[str]]
    contact_start_times: dict[str, float]
    level_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def capture(
        cls, engine: "Box2DEngine", metadata: dict[str, Any] | None = None
    ) -> "StateSnapshot":
        """
        Capture complete engine state as an immutable snapshot.

        Args:
            engine: The Box2DEngine to snapshot
            metadata: Optional metadata to attach to snapshot

        Returns:
            Immutable StateSnapshot containing complete state
        """
        # Capture object states from Box2D bodies
        objects = {}
        if engine.level is None:
            raise ValueError(
                "Level is not set. Please call reset() with a valid level before capturing state."
            )
        for name in engine.level.objects.keys():
            if name in engine.bodies:
                body = engine.bodies[name]
                objects[name] = {
                    "position": (body.position.x, body.position.y),
                    "velocity": (body.linearVelocity.x, body.linearVelocity.y),
                    "angle": body.angle,
                    "angular_velocity": body.angularVelocity,
                    "type": type(engine.level.objects[name]).__name__,
                    "dynamic": body.type == 2,  # b2_dynamicBody
                }
            else:
                # Object not yet placed (e.g., action objects before placement)
                obj = engine.level.objects[name]
                objects[name] = {
                    "position": (obj.x, obj.y),
                    "velocity": (0.0, 0.0),
                    "angle": obj.angle,
                    "angular_velocity": 0.0,
                    "type": type(obj).__name__,
                    "dynamic": obj.dynamic,
                }

        # Serialize Box2D world state
        box2d_state = _save_world(engine.world, engine.bodies)

        # Capture contact tracking state
        contacts = frozenset(engine.contact_listener.contacts)

        # Serialize frozenset keys as "obj1|obj2" strings
        contact_start_times = {
            f"{sorted(pair)[0]}|{sorted(pair)[1]}": time
            for pair, time in engine.contact_listener.contact_start_time.items()
        }

        # Compute level hash for validation
        level_hash = cls._hash_level(engine.level)

        # Derive step index from contact_listener.current_time rather than maintaining
        # a separate counter, so snapshot step index is always consistent with the
        # time that contact durations are measured against.
        step_index = int(
            round(engine.contact_listener.current_time / engine.config.time_step)
        )

        return cls(
            step_index=step_index,
            current_time=engine.contact_listener.current_time,
            objects=objects,
            box2d_state=box2d_state,
            contacts=contacts,
            contact_start_times=contact_start_times,
            level_hash=level_hash,
            metadata=metadata or {},
        )

    def restore(self, engine: "Box2DEngine") -> None:
        """
        Restore engine to this snapshot state.

        This performs a complete state restoration, ensuring the engine
        returns to the exact state when the snapshot was captured.

        Args:
            engine: The Box2DEngine to restore

        Raises:
            ValueError: If snapshot level doesn't match engine level
        """
        # Validate level matches
        if self.level_hash != self._hash_level(engine.level):
            raise ValueError(
                "Cannot restore snapshot to different level. "
                "Snapshot level hash does not match current engine level."
            )

        # Restore Box2D world state (_load_world calls ClearForces internally)
        _load_world(engine.world, engine.bodies, self.box2d_state)

        # Restore contact listener state
        engine.contact_listener.contacts = set(self.contacts)

        # Restore contact start times: reconstruct frozenset keys directly from the
        # serialized "obj1|obj2" strings rather than searching self.contacts, which
        # would silently drop pairs absent from the set and adds an unnecessary inner loop.
        engine.contact_listener.contact_start_time = {
            frozenset(key.split("|", 1)): time
            for key, time in self.contact_start_times.items()
        }

        engine.contact_listener.current_time = self.current_time

    @staticmethod
    def _hash_level(level) -> str:
        """
        Compute deterministic hash of level configuration.

        Args:
            level: The Level object to hash

        Returns:
            Hash string identifying the level
        """
        # Create hashable representation of level.
        # Shape dimensions are included to distinguish levels that share object
        # names/positions but differ only in geometry (e.g. two balls at the same
        # position with different radii). Each attribute defaults to 0.0 for object
        # types that do not carry that dimension, so the tuple remains well-defined
        # across all PhyreObject subclasses.
        obj_data = tuple(
            sorted(
                [
                    (
                        name,
                        type(obj).__name__,
                        round(obj.x, 8),
                        round(obj.y, 8),
                        round(obj.angle, 8),
                        # Ball
                        round(getattr(obj, "radius", 0.0), 8),
                        # Bar
                        round(getattr(obj, "length", 0.0), 8),
                        round(getattr(obj, "thickness", 0.0), 8),
                        # Basket
                        round(getattr(obj, "bottom_width", 0.0), 8),
                        round(getattr(obj, "top_width", 0.0), 8),
                        round(getattr(obj, "height", 0.0), 8),
                    )
                    for name, obj in level.objects.items()
                ]
            )
        )

        hash_input = str(
            (
                level.name,
                obj_data,
                tuple(sorted(level.action_objects)),
            )
        )

        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def to_bytes(self) -> bytes:
        """
        Serialize snapshot to bytes for storage.

        Returns:
            Pickled bytes containing complete snapshot
        """
        return pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_bytes(cls, data: bytes) -> "StateSnapshot":
        """
        Deserialize snapshot from bytes.

        Args:
            data: Pickled bytes containing snapshot

        Returns:
            StateSnapshot instance
        """
        return pickle.loads(data)

    def __eq__(self, other: Any) -> bool:
        """
        Check equality with another snapshot.

        Two snapshots are equal if they represent the same simulation state,
        excluding metadata.

        Args:
            other: Object to compare with

        Returns:
            True if snapshots represent same state
        """
        if not isinstance(other, StateSnapshot):
            return False

        return (
            self.step_index == other.step_index
            and abs(self.current_time - other.current_time) < 1e-9
            and self.objects == other.objects
            and self.box2d_state == other.box2d_state
            and self.contacts == other.contacts
            and self.contact_start_times == other.contact_start_times
            and self.level_hash == other.level_hash
        )

    def __repr__(self) -> str:
        return (
            f"StateSnapshot(step={self.step_index}, "
            f"time={self.current_time:.3f}s, "
            f"objects={len(self.objects)}, "
            f"contacts={len(self.contacts)})"
        )
