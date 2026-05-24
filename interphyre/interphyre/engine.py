import math
from collections import deque
from typing import Any

from Box2D import b2Contact, b2ContactListener, b2World, b2_dynamicBody

from interphyre.config import (
    PRECISION,
    PerformanceProfiler,
    SimulationConfig,
)
from interphyre.level import Level
from interphyre.objects import (
    Ball,
    Bar,
    Basket,
    PhyreObject,
    create_ball,
    create_bar,
    create_basket,
    create_walls,
)

# Static wall body names — used in reset_attempt to skip positional restoration.
_WALL_NAMES = frozenset({"left_wall", "right_wall", "top_wall", "bottom_wall"})


class GoalContactListener(b2ContactListener):
    """Contact listener for tracking object collisions and success conditions.

    This listener monitors all contact events in the physics world and tracks
    which objects are in contact with each other. It supports both performance-
    optimized tracking (relevant contacts only) and comprehensive research logging.

    Attributes:
        track_all_contacts (bool): Whether to track all contact events
        track_relevant_only (bool): Whether to only track relevant contact pairs
        profiler (Optional[PerformanceProfiler]): Performance profiler for timing
        relevant_pairs (set): Set of contact pairs to track for performance
        contacts (set): Currently active contact pairs
        contact_start_time (dict): Start time of each contact
        current_time (float): Current simulation time
        contact_events (list): Detailed list of contact events
    """

    def __init__(
        self,
        track_all_contacts: bool = True,
        track_relevant_only: bool = False,
        profiler: PerformanceProfiler | None = None,
        relevant_pairs: set | None = None,
    ):
        """Initialize the contact listener.

        Args:
            track_all_contacts: Whether to track all contact events (default: True)
            track_relevant_only: Whether to only track relevant pairs for performance (default: False)
            profiler: Performance profiler for timing analysis (default: None)
            relevant_pairs: Set of contact pairs to track for performance (default: None)
        """
        super().__init__()
        self.track_all_contacts = track_all_contacts
        self.track_relevant_only = track_relevant_only
        self.profiler = profiler
        self.relevant_pairs = relevant_pairs or set()

        # Contact pairs stored as frozensets for order-independent identity.
        self.contacts = set()
        self.contact_start_time = {}
        self.current_time = 0

        # Logging
        self.contact_events = []

    def BeginContact(self, contact: b2Contact):
        name_a = contact.fixtureA.body.userData
        name_b = contact.fixtureB.body.userData
        if name_a and name_b:
            # Use frozenset for consistent contact pair representation
            contact_pair = frozenset((name_a, name_b))

            # Check if we should track this contact
            should_track = (
                self.track_all_contacts
                or not self.track_relevant_only
                or contact_pair in self.relevant_pairs
            )

            if should_track:
                self.contacts.add(contact_pair)
                self.contact_start_time[contact_pair] = self.current_time

            # Log contact event whenever all-contact tracking is enabled
            if self.track_all_contacts:
                self.contact_events.append(
                    {
                        "time": self.current_time,
                        "event": "begin",
                        "pair": contact_pair,
                        "objects": (name_a, name_b),
                    }
                )

    def EndContact(self, contact: b2Contact):
        name_a = contact.fixtureA.body.userData
        name_b = contact.fixtureB.body.userData
        if name_a and name_b:
            contact_pair = frozenset((name_a, name_b))

            # Check if we should track this contact
            should_track = (
                self.track_all_contacts
                or not self.track_relevant_only
                or contact_pair in self.relevant_pairs
            )

            if should_track:
                self.contacts.discard(contact_pair)
                # Reset the contact start time when contact ends
                if contact_pair in self.contact_start_time:
                    del self.contact_start_time[contact_pair]

            # Log contact event whenever all-contact tracking is enabled
            if self.track_all_contacts:
                self.contact_events.append(
                    {
                        "time": self.current_time,
                        "event": "end",
                        "pair": contact_pair,
                        "objects": (name_a, name_b),
                    }
                )

    def Update(self, dt):
        """Update the internal simulation time counter.

        Args:
            dt: Time delta in seconds to add to the current simulation time.
        """
        self.current_time += dt

    def GetContactDuration(self, a, b):
        """Get the current unbroken contact duration between two objects.

        Args:
            a: Name of the first object
            b: Name of the second object

        Returns:
            float: Duration in seconds that objects have been in continuous contact.
                  Returns 0 if objects are not currently in contact.
        """
        contact_pair = frozenset((a, b))
        if contact_pair in self.contacts and contact_pair in self.contact_start_time:
            return self.current_time - self.contact_start_time[contact_pair]
        return 0

    def IsInContactForDuration(self, a, b, required_duration):
        """Check if objects are currently in unbroken contact for at least the required duration.

        Args:
            a: Name of the first object
            b: Name of the second object
            required_duration: Minimum contact duration in seconds

        Returns:
            bool: True if objects are in contact and have been for at least required_duration seconds.
        """
        contact_pair = frozenset((a, b))

        # First check: Are they in the contact tracking set?
        if (
            contact_pair not in self.contacts
            or contact_pair not in self.contact_start_time
        ):
            return False

        # Check duration
        current_duration = self.current_time - self.contact_start_time[contact_pair]
        return current_duration >= required_duration

    def get_contact_log(self):
        """Get the full contact event log for research purposes."""
        return self.contact_events.copy()

    def get_contact_statistics(self):
        """Get statistics about all contacts for research purposes."""
        # Return empty stats if no events have been recorded
        if not self.contact_events:
            return {
                "total_events": 0,
                "unique_pairs": 0,
                "pair_counts": {},
                "current_contacts": len(self.contacts),
            }

        # Count contact events by object pairs
        pair_counts = {}
        for event in self.contact_events:
            pair = event["pair"]
            if pair not in pair_counts:
                pair_counts[pair] = {"begins": 0, "ends": 0, "invalidates": 0}
            event_type = event["event"]
            # Map event types to dictionary keys
            if event_type == "begin":
                pair_counts[pair]["begins"] += 1
            elif event_type == "end":
                pair_counts[pair]["ends"] += 1
            elif event_type == "invalidate":
                pair_counts[pair]["invalidates"] += 1

        return {
            "total_events": len(self.contact_events),
            "unique_pairs": len(pair_counts),
            "pair_counts": pair_counts,
            "current_contacts": len(self.contacts),
        }

    def ClearContacts(self):
        """Clear all contact tracking data and reset the simulation time.

        Removes all active contacts, contact start times, contact event log, and
        resets the internal time counter to zero. This method is intended for full
        simulation resets only (e.g., when resetting the simulation or loading a new
        level). It should not be called mid-simulation as it will incorrectly reset
        the time counter, potentially breaking contact duration tracking.
        """
        self.contacts = set()
        self.contact_start_time = {}
        self.contact_events = []
        self.current_time = 0.0

    def invalidate_contact(self, contact_pair):
        """Invalidate a tracked contact pair when external validation determines it is invalid.

        Centralizes contact invalidation to avoid direct state mutation.
        """
        # Remove from active contacts
        self.contacts.discard(contact_pair)
        # Remove any recorded start time
        if contact_pair in self.contact_start_time:
            del self.contact_start_time[contact_pair]
        # Log contact invalidation event whenever all-contact tracking is enabled
        if self.track_all_contacts:
            self.contact_events.append(
                {
                    "time": self.current_time,
                    "event": "invalidate",
                    "pair": contact_pair,
                    "objects": tuple(contact_pair),
                }
            )


class Box2DEngine:
    """Main physics engine for the Interphyre simulation.

    This engine manages the Box2D physics world, object creation, contact tracking,
    and simulation stepping. It provides the core physics simulation functionality
    for the Interphyre environment.

    Attributes:
        config (SimulationConfig): Configuration parameters for the simulation
        profiler (PerformanceProfiler): Performance profiler for timing analysis
        world (b2World): The Box2D physics world
        contact_listener (GoalContactListener): Contact listener for collision tracking
        level (Optional[Level]): Current level being simulated
        bodies (Dict[str, b2Body]): Dictionary mapping object names to Box2D bodies
    """

    def __init__(
        self, level: Level | None = None, config: SimulationConfig | None = None
    ):
        """Initialize the physics engine.

        Args:
            level: Initial level to load (default: None)
            config: Simulation configuration parameters (default: SimulationConfig())
        """
        self.config = config or SimulationConfig()
        self.profiler = PerformanceProfiler(self.config.enable_profiling)

        self.world = b2World(gravity=self.config.gravity, doSleep=self.config.do_sleep)
        self.world.warmStarting = self.config.warm_starting
        self.world.subStepping = self.config.substepping
        self.world.continuousPhysics = self.config.continuous_physics

        self.contact_listener = GoalContactListener(
            track_all_contacts=self.config.track_all_contacts,
            track_relevant_only=self.config.track_relevant_contacts_only,
            profiler=self.profiler,
        )
        self.world.contactListener = self.contact_listener

        # Velocity history for time-based stationary detection (bounded sliding window)
        self._velocity_history: deque[float] = deque(
            maxlen=self.config.stationary_check_frames
        )

        self.reset(level)

    def reset(self, level: Level | None = None):
        """Reset the engine with a new level.

        Clears the current physics world and loads a new level. This destroys
        all existing bodies and creates new ones based on the level definition.

        Args:
            level: New level to load (default: None, clears the world)
        """
        self.world.ClearForces()
        for body in list(self.world.bodies):
            self.world.DestroyBody(body)
        self.level = level
        self.contact_listener.ClearContacts()
        self.bodies = {}
        self._velocity_history = deque(maxlen=self.config.stationary_check_frames)
        if level is not None:
            self._create_world(level)
            # Update relevant contact pairs based on level
            self._update_relevant_contacts()

    def reset_attempt(self) -> None:
        """Reset engine state between oracle attempts without rebuilding the world.

        Cheaper than reset(level) for oracle hot loops: walls and static level bodies
        are left in place. Only action-object bodies are destroyed (they will be
        re-placed by place_action_objects). Dynamic non-action bodies are restored to
        their initial positions and zeroed velocities using the level's stored geometry.
        Contact state and velocity history are cleared.
        """
        if self.level is None:
            return

        # Destroy action-object bodies; place_action_objects will recreate them.
        for name in self.level.action_objects:
            if name in self.bodies:
                self.world.DestroyBody(self.bodies.pop(name))

        # Restore dynamic non-action level bodies to initial positions.
        # The level object stores the original geometry (x, y, angle in degrees).
        for name, body in self.bodies.items():
            if name in _WALL_NAMES or name not in self.level.objects:
                continue
            if body.type == b2_dynamicBody:
                obj = self.level.objects[name]
                body.position = (
                    round(float(obj.x), PRECISION),
                    round(float(obj.y), PRECISION),
                )
                body.angle = math.radians(float(obj.angle))
                body.linearVelocity = (0.0, 0.0)
                body.angularVelocity = 0.0
                body.awake = True

        self.contact_listener.ClearContacts()
        self._velocity_history = deque(maxlen=self.config.stationary_check_frames)

    def close(self) -> None:
        """Destroy the Box2D world and release native memory.

        Explicitly destroys all bodies before nulling the world reference so
        that box2d-py's SWIG layer does not hold dangling C++ pointers.  Call
        this when the engine will not be used again (e.g. from
        InterphyreEnv.close()) rather than relying on Python GC, which is
        non-deterministic in long-lived worker processes.
        """
        if self.world is None:
            return
        for body in list(self.world.bodies):
            self.world.DestroyBody(body)
        self.world = None

    def _create_world(self, level):
        # Create walls on the edges of the screen
        left_wall, right_wall, top_wall, bottom_wall = create_walls(
            self.world, 0.01, 10, 10
        )
        self.bodies["left_wall"] = left_wall
        self.bodies["right_wall"] = right_wall
        self.bodies["top_wall"] = top_wall
        self.bodies["bottom_wall"] = bottom_wall

        # Create objects in a deterministic order to ensure reproducibility
        for name in sorted(level.objects.keys()):
            obj = level.objects[name]
            # Skip placement of the action object
            if name in level.action_objects:
                continue
            if isinstance(obj, Ball):
                if self.world is None:
                    raise RuntimeError(
                        "World is not initialized. Call reset() before placing objects."
                    )
                body = create_ball(
                    self.world,
                    obj,
                    name,
                    use_ccd=self.config.continuous_collision_detection,
                )
            elif isinstance(obj, Bar):
                body = create_bar(
                    self.world,
                    obj,
                    name,
                    use_ccd=self.config.continuous_collision_detection,
                )
            elif isinstance(obj, Basket):
                body = create_basket(
                    self.world,
                    obj,
                    name,
                    use_ccd=self.config.continuous_collision_detection,
                )
            else:
                raise ValueError(f"Unknown object type for '{name}': {type(obj)}")
            self.bodies[name] = body

    def _update_relevant_contacts(self):
        """Update the list of relevant contact pairs based on the level's success condition."""
        if self.level is None:
            return

        # Track all contact pairs between all objects so the success condition
        # can evaluate the full contact graph regardless of object color or role.
        # The prior green-name heuristic silently dropped contacts for levels
        # whose success condition did not involve green-named objects.
        obj_names = list(self.level.objects.keys())
        relevant_pairs = {
            frozenset((a, b))
            for i, a in enumerate(obj_names)
            for b in obj_names[i + 1 :]
        }

        self.contact_listener.relevant_pairs = relevant_pairs

    def place_action_objects(
        self,
        positions: list[tuple[int | float, int | float, int | float]],
    ):
        """Place action objects at the start of the simulation.

        Args:
            positions: List of (x, y, size) tuples for each action object.
                For bars and baskets, size is ignored but must be provided.

        Note:
            All position and size values are rounded to the configured PRECISION
            (see interphyre.config.PRECISION) to ensure determinism.
        """
        if self.level is None:
            raise ValueError(
                "The level is not set. Please call reset() with a valid level before placing action objects."
            )
        if self.world is None:
            raise RuntimeError(
                "World is not initialized. Call reset() before placing objects."
            )

        for name, pos in zip(self.level.action_objects, positions):
            obj = self.level.objects[name]
            if isinstance(obj, Ball):
                x, y, size = pos
                obj.x = round(float(x), PRECISION)
                obj.y = round(float(y), PRECISION)
                obj.radius = round(float(size), PRECISION)
                body = create_ball(
                    self.world,
                    obj,
                    name,
                    use_ccd=self.config.continuous_collision_detection,
                )
            elif isinstance(obj, Bar):
                x, y, _ = pos
                obj.x = round(float(x), PRECISION)
                obj.y = round(float(y), PRECISION)
                body = create_bar(
                    self.world,
                    obj,
                    name,
                    use_ccd=self.config.continuous_collision_detection,
                )
            elif isinstance(obj, Basket):
                x, y, _ = pos
                obj.x = round(float(x), PRECISION)
                obj.y = round(float(y), PRECISION)
                body = create_basket(
                    self.world,
                    obj,
                    name,
                    use_ccd=self.config.continuous_collision_detection,
                )
            else:
                raise ValueError(f"Unknown object type for '{name}': {type(obj)}")
            self.bodies[name] = body

    def get_state(self) -> dict[str, Any]:
        """
        Return the current simulation state.

        Returns:
            Dictionary containing the current physics state including:
            - object positions, velocities, angles, and angular velocities
            - contact information
            - world properties
        """
        if self.world is None or self.level is None:
            return {}

        state = {
            "objects": {},
            "contacts": {},
            "world_properties": {
                "gravity": self.world.gravity,
                "body_count": self.world.bodyCount,
                "contact_count": self.world.contactCount,
            },
        }

        # Get object states
        for name, obj in self.level.objects.items():
            if name in self.bodies:
                body = self.bodies[name]
                state["objects"][name] = {
                    "position": (body.position.x, body.position.y),
                    "velocity": (body.linearVelocity.x, body.linearVelocity.y),
                    "angle": body.angle,
                    "angular_velocity": body.angularVelocity,
                    "type": type(obj).__name__,
                    "dynamic": body.type == b2_dynamicBody,
                }
            else:
                # Object not yet placed (e.g., action objects)
                state["objects"][name] = {
                    "position": (obj.x, obj.y),
                    "velocity": (0.0, 0.0),
                    "angle": obj.angle,
                    "angular_velocity": 0.0,
                    "type": type(obj).__name__,
                    "dynamic": obj.dynamic,
                }

        # Get contact information (sorted keys for deterministic ordering)
        for contact_pair in self.contact_listener.contacts:
            obj1, obj2 = sorted(contact_pair)
            state["contacts"][f"{obj1}_{obj2}"] = {
                "objects": (obj1, obj2),
                "duration": self.contact_listener.GetContactDuration(obj1, obj2),
            }

        return state

    def objects(self) -> dict[str, PhyreObject]:
        if self.level is None:
            raise ValueError(
                "The level is not set. Please call reset() with a valid level before accessing objects."
            )
        return self.level.objects

    def has_contact(self, name1: str, name2: str) -> bool:
        """
        Check if the two object names have come into contact.
        """
        contact_pair = frozenset((name1, name2))
        return contact_pair in self.contact_listener.contacts

    def world_is_stationary(self) -> bool:
        """Check if the world is stationary using time-based averaging.

        Uses a sliding window of recent frames to determine if all objects have been
        stationary for a sustained period. This prevents false positives from momentary
        oscillations or floating-point jitter.

        Returns:
            bool: True if all objects have been below stationary_tolerance for the
                  last stationary_check_frames frames, False otherwise.

        Raises:
            ValueError: If world or level is not initialized.
        """
        if self.world is None:
            raise ValueError(
                "World is not initialized. Call reset() before checking for stationary bodies."
            )
        if self.level is None:
            raise ValueError(
                "Level is not set. Please call reset() before checking for stationary bodies."
            )

        # Check current frame's maximum velocity
        max_velocity = 0.0
        for body in self.world.bodies:
            if body.userData in self.level.objects:
                linear_vel = body.linearVelocity.length
                angular_vel = abs(body.angularVelocity)
                max_velocity = max(max_velocity, linear_vel, angular_vel)

        # Add current frame to history; deque(maxlen=N) evicts oldest automatically.
        self._velocity_history.append(max_velocity)

        # Need a full window before declaring stationary.
        if len(self._velocity_history) < self.config.stationary_check_frames:
            return False

        # World is stationary if ALL frames in window are below tolerance
        return all(
            vel <= self.config.stationary_tolerance for vel in self._velocity_history
        )

    def _is_point_inside_polygon(
        self, x: float, y: float, polygon: list[tuple[float, float]]
    ) -> bool:
        n_vertices = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n_vertices + 1):
            p2x, p2y = polygon[i % n_vertices]
            if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def is_in_basket(self, basket_name: str, target_name: str) -> bool:
        """
        Check if a ball is inside a basket using the sensor fixture.

        Args:
            basket_name: Name of the basket object
            target_name: Name of the ball object

        Returns:
            bool: True if the ball is inside the basket, False otherwise
        """
        if self.level is None or self.world is None:
            raise ValueError("Level or world not initialized.")
        if (
            target_name not in self.level.objects
            or basket_name not in self.level.objects
        ):
            return False
        basket = self.level.objects[basket_name]
        if not isinstance(basket, Basket):
            raise ValueError(f"{basket_name} is not a basket.")
        target = self.level.objects[target_name]
        if not isinstance(target, Ball):
            raise ValueError(
                f"{target_name} is a {type(target)}, is_in_basket_sensor currently only works with Balls."
            )

        # Get the basket and target bodies from the world
        basket_body = None
        target_body = None
        for body in self.world.bodies:
            if body.userData == basket_name:
                basket_body = body
            elif body.userData == target_name:
                target_body = body

        if basket_body is None or target_body is None:
            return False

        # Check if the target is in contact with the basket's sensor fixture
        for contact in self.world.contacts:
            # Check if this contact involves our basket and target
            if (
                contact.fixtureA.body == basket_body
                and contact.fixtureB.body == target_body
            ) or (
                contact.fixtureA.body == target_body
                and contact.fixtureB.body == basket_body
            ):
                # Check if one of the fixtures is a sensor (our basket's interior)
                if contact.fixtureA.sensor or contact.fixtureB.sensor:
                    return True

        return False

    def is_in_contact_for_duration(self, a, b, success_time: float | None = None):
        """Check if objects are currently in unbroken contact for the required duration.

        Args:
            a: Name of the first object
            b: Name of the second object
            success_time: Required contact duration in seconds. If None, uses config.default_success_time.

        Returns:
            bool: True if objects are in contact and have been for at least success_time seconds.

        Raises:
            ValueError: If level is not set or objects are not in the level.
        """
        if self.level is None:
            raise ValueError(
                "Level is not set. Please call reset() with a valid level before checking for contact duration."
            )
        if a not in self.level.objects or b not in self.level.objects:
            return False
        if success_time is None:
            success_time = self.config.default_success_time

        return self.contact_listener.IsInContactForDuration(a, b, success_time)

    def time_update(self, dt):
        """Update the contact listener's internal time tracking.

        Args:
            dt: Time delta in seconds to add to the current simulation time.
        """
        self.contact_listener.Update(dt)

    def get_contact_duration(self, a, b):
        """Get the current unbroken contact duration between two objects.

        Args:
            a: Name of the first object
            b: Name of the second object

        Returns:
            float: Duration in seconds that objects have been in continuous contact.
                  Returns 0 if objects are not currently in contact.

        Raises:
            ValueError: If level is not set or objects are not in the level.
        """
        if self.level is None:
            raise ValueError(
                "Level is not set. Please call reset() with a valid level before checking for contact duration."
            )
        if a not in self.level.objects or b not in self.level.objects:
            return 0
        return self.contact_listener.GetContactDuration(a, b)

    def get_contact_log(self):
        """Get the full contact event log for research purposes."""
        return self.contact_listener.get_contact_log()

    def get_contact_statistics(self):
        """Get statistics about all contacts for research purposes."""
        return self.contact_listener.get_contact_statistics()
