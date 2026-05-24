from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any, Callable

import gymnasium as gym
import numpy as np

from interphyre.config import PRECISION, SimulationConfig
from interphyre.engine import Box2DEngine
from interphyre.level import Level
from interphyre.render import Renderer
from interphyre.validation.placement import is_valid_placement

if TYPE_CHECKING:
    from Box2D import b2Body

    from interphyre.interventions.state import StateSnapshot
    from interphyre.interventions.triggers import Trigger
    from interphyre.objects import PhyreObject
    from interphyre.validation.registry import SeedRegistry

logger = logging.getLogger(__name__)


class InterventionContext:
    """Context manager for scoped interventions.

    Provides batched modifications and optional auto-rollback on exception.
    Use for level-structural changes or when you need transactional semantics.

    Example:
        with env.intervention_context() as ctx:
            ctx.add_object("ball", Ball(x=0, y=0, radius=0.5))
            ctx.apply_impulse("ball", impulse=(5.0, 0.0))
            ctx.modify_success_condition(lambda engine: custom_check(engine))
    """

    def __init__(self, env: "InterphyreEnv", auto_rollback: bool = False):
        """Initialize intervention context.

        Args:
            env: The InterphyreEnv instance to operate on
            auto_rollback: If True, automatically restore state on exception
        """
        self._env = env
        self._auto_rollback = auto_rollback
        self._snapshot: StateSnapshot | None = None
        self._original_success_condition: Callable | None = None

    def __enter__(self) -> "InterventionContext":
        if self._auto_rollback:
            from interphyre.interventions.state import StateSnapshot

            self._snapshot = StateSnapshot.capture(self._env.engine)
            # Save Python-level success_condition before any mutation; StateSnapshot
            # restores Box2D body state only, not Level attributes.
            self._original_success_condition = self._env._level.success_condition
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None and self._auto_rollback and self._snapshot:
            self._snapshot.restore(self._env.engine)
            # Restore success_condition independently of StateSnapshot.
            if self._original_success_condition is not None:
                self._env._level.success_condition = self._original_success_condition
            # Suppress the exception: caller requested auto_rollback, which implies
            # the exception is expected and state has been cleanly restored.
            return True
        return False

    # === Object Management ===

    def add_object(
        self,
        name: str,
        obj: "PhyreObject",
        impulse: tuple[float, float] | None = None,
    ) -> None:
        """Add a new object to the simulation."""
        self._env.add_object(name, obj, impulse=impulse)

    def remove_object(self, name: str) -> None:
        """Remove an object from the simulation."""
        self._env.remove_object(name)

    def apply_impulse(
        self,
        name: str,
        impulse: tuple[float, float],
        point: tuple[float, float] | None = None,
    ) -> None:
        """Apply an impulse to an object."""
        self._env.apply_impulse(name, impulse, point=point)

    def apply_force(
        self,
        name: str,
        force: tuple[float, float],
        point: tuple[float, float] | None = None,
    ) -> None:
        """Apply a force to an object."""
        self._env.apply_force(name, force, point=point)

    def set_velocity(
        self,
        name: str,
        vx: float | None = None,
        vy: float | None = None,
    ) -> None:
        """Set object linear velocity."""
        self._env.set_velocity(name, vx=vx, vy=vy)

    def set_position(
        self,
        name: str,
        x: float | None = None,
        y: float | None = None,
    ) -> None:
        """Set object position."""
        self._env.set_position(name, x=x, y=y)

    def freeze(self, name: str) -> None:
        """Freeze object by zeroing all velocities."""
        self._env.freeze(name)

    # === Level-Structural Changes (only available in context) ===

    def modify_success_condition(
        self, condition: Callable[[Box2DEngine], bool]
    ) -> None:
        """Modify the level's success condition.

        Args:
            condition: New success condition function that takes engine and returns bool
        """
        self._env._level.success_condition = condition

    def modify_metadata(self, **kwargs) -> None:
        """Modify the level's metadata.

        Args:
            **kwargs: Key-value pairs to update in metadata
        """
        if self._env._level.metadata is None:
            self._env._level.metadata = {}
        self._env._level.metadata.update(kwargs)


class InterphyreEnv(gym.Env):
    """Gymnasium environment for physics-based puzzles.

    This environment simulates physics puzzles where agents place objects to achieve
    specific goals. The environment follows a one-shot paradigm: agents provide an
    action (object placement), then the full simulation runs to completion.

    Example (standard RL usage):
        env = InterphyreEnv("catapult", seed=42, render_mode="human")
        obs, info = env.reset()
        obs, reward, term, trunc, info = env.step([(0.5, 3.0, 0.6)])

    Example (intervention/replanning):
        env = InterphyreEnv("catapult", seed=42, enable_interventions=True)
        env.place_action((0.5, 3.0, 0.6))
        snapshot, step = env.run_until(on_contact("ball", "platform"))
        if snapshot:
            env.restore(snapshot)
            env.add_object("ball2", Ball(x=0, y=2, radius=0.3))
            obs, reward, term, trunc, info = env.step_until(on_success())
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 30,
        "name": "InterphyreEnv",
    }

    def __init__(
        self,
        level_name: str | "Level",
        seed: int | None = None,
        config: SimulationConfig | None = None,
        render_mode: str | None = None,
        observation_type: str = "physics_state",
        action_type: str = "continuous",
        image_size: tuple[int, int] = (600, 600),
        image_ppm: float = 60.0,
        discrete_colors: bool = False,
        enable_interventions: bool = False,
        validate: bool = True,
        registry: "SeedRegistry | None" = None,
    ):
        """Initialize the Phyre environment.

        Args:
            level_name: Level name string (loaded from registry) or a pre-built Level
                object (used directly; seed is ignored). Pass a Level object when you
                have a custom or pre-built level that is not in the registry.
            seed: Random seed for level variation (only used when level_name is a str)
            config: Optional simulation configuration (uses defaults if None)
            render_mode: Rendering mode - "human" for pygame, "rgb_array" for images, None for no rendering
            observation_type: Type of observation space ("physics_state", "image", "both")
            action_type: Type of action space ("continuous", "discrete")
            image_size: Size of rendered images (width, height) for image observations
            image_ppm: Pixels per Box2D unit for image rendering
            discrete_colors: If True, use single-channel discrete colors instead of RGB
            enable_interventions: If True, enable intervention scheduling in the engine
            validate: If True (default), ensures the level has a valid placement
                before running. Bundled seeds are served instantly from the bundle;
                unbundled seeds run the oracle live on first call and cache the result
                for subsequent calls. Trivial and impossible variants are retried
                automatically. For pre-built Level objects, a courtesy trivial check
                is performed but no oracle is run.
                Set to False to skip all validation: the raw level geometry at
                variant 0 is used directly, with no bundle lookup and no oracle.
                Only use False when developing oracles or inspecting raw geometry.
            registry: Optional SeedRegistry for bundled/SQLite lookup. When None,
                the module-level default registry is used.

        Raises:
            RuntimeError: When validate=True and no valid level can be found for the
                given level_name and seed after exhausting all variants. This occurs
                for levels with very low valid-seed rates or seeds that are impossible
                for the given level. Pass a bundled seed or extend the bundle to avoid
                live oracle cost.
        """
        super().__init__()

        # Dispatch on level type and validate flag.
        #
        # Path 1 — registered level by name, validate=True:
        #   Full pipeline: load_valid_level() handles trivial/impossible variants
        #   transparently using the variant system and oracle. Logs INFO if the
        #   oracle runs live (seed not in bundled data).
        #
        # Path 2 — pre-built Level object, validate=True:
        #   Courtesy is_trivial check only. No oracle, no registry, no variant
        #   system (geometry is already fixed). Logs WARNING if trivial but does
        #   not raise — the caller constructed this level explicitly.
        #
        # Path 3 — validate=False (any input type):
        #   Original behavior: load by name or use Level directly, no checks.
        if isinstance(level_name, Level):
            self._level = level_name
            self._level_name = level_name.name
            self._seed = None
            if validate:
                # Path 2: courtesy trivial check on a pre-built Level.
                from interphyre.validation.checks import extract_scene_dict, is_trivial

                if is_trivial(self._level):
                    logger.warning(
                        "InterphyreEnv: pre-built Level '%s' satisfies the success "
                        "condition at t=0 (trivial). Consider revising the level "
                        "geometry or using a registered level name with validate=True.",
                        self._level_name,
                    )
                self._variant: int = 0
                self._scene_dict: dict | None = extract_scene_dict(self._level)
            else:
                # Path 3: no validation for pre-built Level.
                self._variant = 0
                self._scene_dict = None
        elif validate:
            # Path 1: registered level by name with full validation pipeline.
            from interphyre.validation import _get_registry, load_valid_level

            reg = registry if registry is not None else _get_registry()
            # Emit INFO if the oracle will need to run live (seed absent from
            # bundled data), so the user is not surprised by latency.
            # Uses get_valid_entry (in-memory bundle only) to avoid opening
            # the SQLite connection for this advisory check.
            if reg.get_valid_entry(level_name, seed if seed is not None else 0) is None:
                logger.info(
                    "InterphyreEnv: running oracle live for '%s' seed=%s "
                    "(not in bundled data — result will be cached for future calls).",
                    level_name,
                    seed,
                )
            validated = load_valid_level(
                level_name, seed if seed is not None else 0, registry=reg
            )
            self._level = validated.level
            self._level_name = level_name
            self._seed = validated.seed
            self._variant = validated.variant
            self._scene_dict = validated.scene_dict
        else:
            # Path 3: validate=False, original load-by-name behavior.
            from interphyre.levels import load_level

            self._level = load_level(level_name, seed=seed)
            self._level_name = level_name
            self._seed = seed
            self._variant = 0
            self._scene_dict = None

        # Set up config with intervention flag
        self.config = config or SimulationConfig()
        if enable_interventions:
            self.config = dataclasses.replace(self.config, enable_interventions=True)

        # Set up renderer based on render_mode
        self.render_mode = render_mode
        self.renderer: Renderer | None = None
        if render_mode == "human":
            from interphyre.render.pygame import PygameRenderer

            width, height = image_size
            self.renderer = PygameRenderer(width=width, height=height, ppm=image_ppm)

        self.observation_type = observation_type
        self.action_type = action_type
        self.image_size = image_size
        self.image_ppm = image_ppm
        self.discrete_colors = discrete_colors

        # Initialize engine
        self.engine = Box2DEngine(config=self.config)
        self.action_placed = False
        self.current_obs = None
        self.current_state = None
        self.step_count = 0
        self.max_steps = self.config.max_steps
        self._rollout_complete = False

        # Cached image renderer — allocated once, reused every observation step.
        # Only instantiated when the observation type actually needs it.
        self._image_renderer = None

        # Set up action space
        self._setup_action_space()

        # Set up observation space
        self._setup_observation_space()

        # Initialize state
        self.reset()

    # === Properties ===

    @property
    def level(self) -> Level:
        """Get the current level (read-only)."""
        return self._level

    @property
    def objects(self) -> dict[str, Any]:
        """Get the level's objects dictionary (read-only view)."""
        return self._level.objects

    @property
    def success(self) -> bool:
        """Check if the current state satisfies the success condition."""
        return self._level.success_condition(self.engine)

    @property
    def variant(self) -> int:
        """Variant index used for the current level.

        0 for the canonical geometry (most seeds and all pre-built levels).
        Positive when validate=True advanced past a trivial/impossible variant=0.
        Experiment logs should record (level_name, seed, variant) as the
        short-form provenance triple.
        """
        return self._variant

    @property
    def scene_dict(self) -> dict | None:
        """Full geometry of the current level as a plain dict, or None.

        None when validate=False. Otherwise the JSON-serializable scene dict
        extracted from the validated level — use as the long-form reproducibility
        artifact alongside (level_name, seed, variant).
        """
        return self._scene_dict

    # === Intervention API ===

    def run_until(
        self,
        trigger: "Trigger",
        action: tuple[float, float, float]
        | list[tuple[float, float, float]]
        | None = None,
        max_steps: int = 240,
    ) -> tuple["StateSnapshot | None", int]:
        """Run simulation until trigger fires.

        Args:
            trigger: Trigger condition to wait for
            action: Optional action to place before running. Can be:
                - Single (x, y, radius) tuple for one action object
                - List of tuples for multiple action objects
                - None if action already placed or no action objects
            max_steps: Maximum steps to simulate

        Returns:
            (snapshot, step_index) if triggered, (None, final_step) if timeout

        Example:
            snapshot, step = env.run_until(
                on_contact("ball", "platform"),
                action=(0.5, 3.0, 0.6),
                max_steps=500
            )
        """
        from interphyre.interventions.state import StateSnapshot

        # Reset trigger state so once_only triggers can fire again across episodes
        trigger.reset()

        # Place action if provided and not already placed
        if action is not None and not self.action_placed:
            if isinstance(action, tuple) and len(action) == 3:
                action = [action]

            validation_result = self._validate_action_with_failure(action)
            if validation_result["invalid"]:
                raise ValueError(f"Invalid action: {validation_result['error']}")

            self._place_action_objects(validation_result["action"])
            self.action_placed = True

        start = self.step_count

        for step_index in range(start, start + max_steps):
            self._step_physics()
            self.render()

            if trigger.should_fire(step_index + 1, self.engine):
                snapshot = StateSnapshot.capture(
                    self.engine,
                    metadata={"step_index": step_index + 1, "trigger": str(trigger)},
                )
                return snapshot, step_index + 1

        return None, start + max_steps

    def restore(self, snapshot: "StateSnapshot") -> None:
        """Restore simulation to a previous state.

        Args:
            snapshot: StateSnapshot to restore
        """
        snapshot.restore(self.engine)
        if snapshot.metadata and "step_index" in snapshot.metadata:
            self.step_count = snapshot.metadata["step_index"]
        self._rollout_complete = False

    def step_until(
        self,
        trigger: "Trigger",
        max_steps: int = 240,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Continue simulation until trigger fires, returning Gym-style output.

        This is the intervention-aware equivalent of step() for continuing
        after a restore or intervention.

        Args:
            trigger: Trigger condition to wait for (e.g., on_success())
            max_steps: Maximum steps to simulate

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        snapshot, final_step = self.run_until(trigger, action=None, max_steps=max_steps)

        success = self._level.success_condition(self.engine)
        truncated = snapshot is None and not success

        obs = self._get_observation()
        reward = self._calculate_reward(success, truncated)
        info = self._get_info_dict(success, success, truncated)
        info["final_step"] = final_step

        return obs, reward, success, truncated, info

    def intervention_context(self, auto_rollback: bool = False) -> InterventionContext:
        """Create an intervention context for scoped modifications.

        Args:
            auto_rollback: If True, automatically restore state if exception occurs

        Returns:
            InterventionContext for use in a with statement
        """
        return InterventionContext(self, auto_rollback=auto_rollback)

    # === Object Management API ===

    def add_object(
        self,
        name: str,
        obj: "PhyreObject",
        impulse: tuple[float, float] | None = None,
    ) -> None:
        """Add a new object to the simulation.

        Args:
            name: Unique name for the object
            obj: PhyreObject instance (Ball, Bar, or Basket)
            impulse: Optional initial impulse (ix, iy)

        Raises:
            ValueError: If name already exists
        """
        if name in self.engine.bodies:
            raise ValueError(f"Object '{name}' already exists")

        from interphyre.objects import (
            Ball,
            Bar,
            Basket,
            create_ball,
            create_bar,
            create_basket,
        )

        # Add to level objects
        self._level.objects[name] = obj

        # Create physics body
        if isinstance(obj, Ball):
            body = create_ball(
                self.engine.world,
                obj,
                name,
                use_ccd=self.config.continuous_collision_detection,
            )
        elif isinstance(obj, Bar):
            body = create_bar(
                self.engine.world,
                obj,
                name,
                use_ccd=self.config.continuous_collision_detection,
            )
        elif isinstance(obj, Basket):
            body = create_basket(
                self.engine.world,
                obj,
                name,
                use_ccd=self.config.continuous_collision_detection,
            )
        else:
            raise TypeError(f"Unknown object type: {type(obj)}")

        self.engine.bodies[name] = body

        # Apply initial impulse if provided
        if impulse is not None:
            self.apply_impulse(name, impulse)

    def remove_object(self, name: str) -> None:
        """Remove an object from the simulation.

        Args:
            name: Name of object to remove

        Raises:
            ValueError: If object doesn't exist
        """
        if name not in self.engine.bodies:
            raise ValueError(f"Object '{name}' not found")

        # Destroy physics body
        self.engine.world.DestroyBody(self.engine.bodies[name])
        del self.engine.bodies[name]

        # Remove from level
        if name in self._level.objects:
            del self._level.objects[name]

    def apply_impulse(
        self,
        name: str,
        impulse: tuple[float, float],
        point: tuple[float, float] | None = None,
    ) -> None:
        """Apply an impulse to an object.

        Args:
            name: Object name
            impulse: (ix, iy) impulse vector
            point: Application point (default: center of mass)
        """
        body = self._get_body(name)
        from Box2D import b2Vec2

        ix, iy = impulse
        if point is None:
            point_vec = body.worldCenter
        else:
            point_vec = b2Vec2(point[0], point[1])

        body.ApplyLinearImpulse(b2Vec2(ix, iy), point_vec, True)

    def apply_force(
        self,
        name: str,
        force: tuple[float, float],
        point: tuple[float, float] | None = None,
    ) -> None:
        """Apply a force to an object.

        Args:
            name: Object name
            force: (fx, fy) force vector
            point: Application point (default: center of mass)
        """
        body = self._get_body(name)
        from Box2D import b2Vec2

        fx, fy = force
        if point is None:
            point_vec = body.worldCenter
        else:
            point_vec = b2Vec2(point[0], point[1])

        body.ApplyForce(b2Vec2(fx, fy), point_vec, True)

    def set_velocity(
        self,
        name: str,
        vx: float | None = None,
        vy: float | None = None,
    ) -> None:
        """Set object linear velocity.

        Args:
            name: Object name
            vx: X velocity (None to keep current)
            vy: Y velocity (None to keep current)
        """
        body = self._get_body(name)
        from Box2D import b2Vec2

        current = body.linearVelocity
        new_vx = vx if vx is not None else current.x
        new_vy = vy if vy is not None else current.y
        body.linearVelocity = b2Vec2(new_vx, new_vy)

    def set_position(
        self,
        name: str,
        x: float | None = None,
        y: float | None = None,
    ) -> None:
        """Set object position.

        Args:
            name: Object name
            x: X position (None to keep current)
            y: Y position (None to keep current)
        """
        body = self._get_body(name)
        from Box2D import b2Vec2

        current = body.position
        new_x = x if x is not None else current.x
        new_y = y if y is not None else current.y
        body.transform = (b2Vec2(new_x, new_y), body.angle)

    def freeze(self, name: str) -> None:
        """Freeze object by zeroing all velocities.

        Args:
            name: Object name
        """
        body = self._get_body(name)
        from Box2D import b2Vec2

        body.linearVelocity = b2Vec2(0, 0)
        body.angularVelocity = 0.0

    def _get_body(self, name: str) -> b2Body:
        """Get Box2D body by name."""
        body = self.engine.bodies.get(name)
        if body is None:
            raise ValueError(f"Object '{name}' not found")
        return body

    # === Standard Gym Methods ===

    def _setup_action_space(self) -> None:
        """Set up the action space based on action_type and level configuration."""
        if self.action_type == "continuous":
            if len(self._level.action_objects) == 0:
                self.action_space = gym.spaces.Box(
                    low=np.array([], dtype=np.float32),
                    high=np.array([], dtype=np.float32),
                    dtype=np.float32,
                )
            else:
                # Check for custom action bounds in level metadata
                if (
                    self._level.metadata is not None
                    and "action_bounds" in self._level.metadata
                ):
                    action_bounds = self._level.metadata["action_bounds"]
                else:
                    action_bounds = {
                        "x": (-5.0, 5.0),
                        "y": (-5.0, 5.0),
                        "r": (0.1, 1.5),
                    }
                x_low, x_high = action_bounds["x"]
                y_low, y_high = action_bounds["y"]
                r_low, r_high = action_bounds["r"]

                # Each action object gets (x, y, size)
                action_dim = len(self._level.action_objects) * 3
                lows = np.array(
                    [x_low, y_low, r_low] * len(self._level.action_objects),
                    dtype=np.float32,
                )
                highs = np.array(
                    [x_high, y_high, r_high] * len(self._level.action_objects),
                    dtype=np.float32,
                )
                self.action_space = gym.spaces.Box(
                    low=lows, high=highs, shape=(action_dim,), dtype=np.float32
                )
        elif self.action_type == "discrete":
            num_objects = len(self._level.action_objects)
            if num_objects == 0:
                self.action_space = gym.spaces.MultiDiscrete(
                    np.array([], dtype=np.int64)
                )
            else:
                # Read bounds from level metadata — same source as the continuous
                # action space — so discrete bin indices map to the same coordinate
                # range as continuous actions for the same level.
                if (
                    self._level.metadata is not None
                    and "action_bounds" in self._level.metadata
                ):
                    action_bounds = self._level.metadata["action_bounds"]
                else:
                    action_bounds = {
                        "x": (-5.0, 5.0),
                        "y": (-5.0, 5.0),
                        "r": (0.1, 1.5),
                    }

                x_low, x_high = action_bounds["x"]
                y_low, y_high = action_bounds["y"]
                r_low, r_high = action_bounds["r"]
                step = 0.1
                x_bins = int(round((x_high - x_low) / step)) + 1
                y_bins = int(round((y_high - y_low) / step)) + 1
                r_bins = int(round((r_high - r_low) / step)) + 1

                self._discrete_step = step
                self._discrete_bins = (x_bins, y_bins, r_bins)
                self._discrete_lows = (x_low, y_low, r_low)

                nvec = np.array(list(self._discrete_bins) * num_objects, dtype=np.int64)
                self.action_space = gym.spaces.MultiDiscrete(nvec)
        else:
            raise ValueError(f"Unknown action_type: {self.action_type}")

    def _build_physics_state_space(self) -> gym.spaces.Dict:
        """Build the physics-state observation space from the current level's objects."""
        n_objects = len(self._level.objects)
        return gym.spaces.Dict(
            {
                "objects": gym.spaces.Dict(
                    {
                        name: gym.spaces.Dict(
                            {
                                "position": gym.spaces.Box(
                                    low=-10, high=10, shape=(2,), dtype=np.float32
                                ),
                                # Ball falling from y=5 under gravity reaches ~13 m/s
                                # at ground impact; bound set to 50 m/s for headroom.
                                "velocity": gym.spaces.Box(
                                    low=-50, high=50, shape=(2,), dtype=np.float32
                                ),
                                "angle": gym.spaces.Box(
                                    low=-np.pi,
                                    high=np.pi,
                                    shape=(),
                                    dtype=np.float32,
                                ),
                                "angular_velocity": gym.spaces.Box(
                                    low=-10, high=10, shape=(), dtype=np.float32
                                ),
                                "type": gym.spaces.Text(max_length=20),
                            }
                        )
                        for name in self._level.objects.keys()
                    }
                ),
                "contacts": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(n_objects, n_objects),
                    dtype=np.bool_,
                ),
                "step_count": gym.spaces.Discrete(self.max_steps + 1),
            }
        )

    def _build_image_space(self) -> gym.spaces.Box:
        """Build the image observation space from current image settings."""
        width, height = self.image_size
        if self.discrete_colors:
            return gym.spaces.Box(low=0, high=7, shape=(height, width), dtype=np.uint8)
        return gym.spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)

    def _setup_observation_space(self) -> None:
        """Set up the observation space based on observation_type."""
        if self.observation_type == "physics_state":
            self.observation_space = self._build_physics_state_space()
        elif self.observation_type == "image":
            self.observation_space = self._build_image_space()
        elif self.observation_type == "both":
            self.observation_space = gym.spaces.Dict(
                {
                    "physics_state": self._build_physics_state_space(),
                    "image": self._build_image_space(),
                }
            )
        else:
            raise ValueError(f"Unknown observation_type: {self.observation_type}")

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility (optional)
            options: Additional options for reset (e.g., interventions)

        Returns:
            Tuple of (observation, info)
        """
        # gymnasium's super().reset(seed=seed) is the sole owner of self.np_random.
        # It seeds a PCG64 generator when seed is provided, or carries forward the
        # existing generator when seed is None — consistent behavior across all calls.
        super().reset(seed=seed)

        # Reset engine state.
        # First call (level not yet loaded): full world build via reset(level).
        # Subsequent calls: reset_attempt() restores body positions without
        # destroying and recreating Box2D bodies. This preserves warm-start data,
        # matching the oracle path used to validate bundle solutions and ensuring
        # that stored solutions reproduce identically through env.step().
        if self.engine.level is None:
            self.engine.reset(self._level)
        else:
            self.engine.reset_attempt()
        self.action_placed = False
        self.step_count = 0
        self._rollout_complete = False

        observation = self._get_observation()

        info = {
            "level_name": self._level.name,
            "action_objects": self._level.action_objects,
            "total_objects": len(self._level.objects),
            "step_count": self.step_count,
            "action_placed": self.action_placed,
            "success": False,
            "truncated": False,
            "terminated": False,
        }

        return observation, info

    def step(
        self, action: list[tuple[float, float, float]] | np.ndarray
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute one episode: place objects and run full simulation to completion.

        This is a one-shot environment - step() can only be called once per episode.
        After calling step(), you must call reset() to start a new episode.

        Args:
            action: Action to execute. For continuous actions, should be:
                - List of (x, y, size) tuples for each action object
                - Numpy array of shape (n_objects * 3,) with flattened coordinates

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self._rollout_complete:
            raise RuntimeError(
                "Episode already complete. Call reset() to start a new episode."
            )

        validation_result = self._validate_action_with_failure(action)
        if validation_result["invalid"]:
            obs = self._get_observation()
            info = {
                "level_name": self._level.name,
                "step_count": 0,
                "action_placed": False,
                "success": False,
                "terminated": True,
                "truncated": False,
                "world_stationary": False,
                "validation_error": validation_result["error"],
                "invalid_action": True,
            }
            self._rollout_complete = True
            return obs, -1.0, True, False, info

        self._place_action_objects(validation_result["action"])
        self.action_placed = True

        obs, reward, terminated, truncated, info = self._run_simulation_rollout()
        self._rollout_complete = True
        return obs, reward, terminated, truncated, info

    def step_physics(self, n: int = 1) -> None:
        """Advance simulation by n physics frames without serialization cost."""
        for _ in range(n):
            self._step_physics()

    def _step_physics(self) -> None:
        """Execute a single physics step (internal method)."""
        self.engine.world.Step(
            self.config.time_step,
            self.config.velocity_iters,
            self.config.position_iters,
        )

        self.engine.time_update(self.config.time_step)
        self.step_count += 1

    def _run_simulation_rollout(self) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Run physics simulation to completion."""
        for step_index in range(self.max_steps):
            self._step_physics()
            self.render()

            success = self._level.success_condition(self.engine)
            terminated = success
            truncated = step_index >= self.max_steps - 1

            if success or truncated:
                break

        # Capture stationary status here — calling world_is_stationary() inside
        # _get_info_dict would append a spurious extra frame to _velocity_history
        # post-loop, which persists into the next episode in simulate() calls.
        world_stationary = (
            self.engine.world_is_stationary() if self.engine.world else False
        )

        obs = self._get_observation()
        reward = self._calculate_reward(success, truncated)
        info = self._get_info_dict(
            success, terminated, truncated, world_stationary=world_stationary
        )

        return obs, reward, terminated, truncated, info

    def _validate_action(
        self, action: list[tuple[float, float, float]] | np.ndarray
    ) -> list[tuple[float, float, float]]:
        """Validate action format and convert to standard format."""
        if len(self._level.action_objects) == 0:
            if action != [] and not (
                isinstance(action, np.ndarray) and action.size == 0
            ):
                raise ValueError(
                    f"No action objects in level, but received action: {action}"
                )
            return []

        expected_dim = len(self._level.action_objects) * 3

        if self.action_type == "discrete":
            x_bins, y_bins, s_bins = self._discrete_bins
            x_low, y_low, s_low = self._discrete_lows
            step = self._discrete_step

            if isinstance(action, np.ndarray):
                if action.shape != (expected_dim,):
                    raise ValueError(
                        f"Expected action shape ({expected_dim},), got {action.shape}"
                    )
                indices = action.astype(np.int64).tolist()
            elif isinstance(action, list):
                if len(action) != len(self._level.action_objects):
                    raise ValueError(
                        f"Expected {len(self._level.action_objects)} action tuples, got {len(action)}"
                    )
                indices = []
                for i, pos in enumerate(action):
                    if not isinstance(pos, (tuple, list)) or len(pos) != 3:
                        raise ValueError(
                            f"Action {i} must be a tuple/list of length 3 (x, y, size), got {pos}"
                        )
                    if not all(isinstance(v, (int, np.integer)) for v in pos):
                        raise ValueError(
                            f"Action {i} must contain integer indices for discrete mode, got {pos}"
                        )
                    indices.extend([int(pos[0]), int(pos[1]), int(pos[2])])
            else:
                raise ValueError(
                    f"Action must be list of tuples or numpy array, got {type(action)}"
                )

            converted_action: list[tuple[float, float, float]] = []
            for i in range(0, expected_dim, 3):
                xi, yi, si = int(indices[i]), int(indices[i + 1]), int(indices[i + 2])
                if not (0 <= xi < x_bins and 0 <= yi < y_bins and 0 <= si < s_bins):
                    raise ValueError(
                        f"Discrete indices out of bounds at object {i // 3}: {(xi, yi, si)}"
                    )
                x = round(x_low + step * xi, PRECISION)
                y = round(y_low + step * yi, PRECISION)
                s = round(s_low + step * si, PRECISION)
                converted_action.append((float(x), float(y), float(s)))
        else:
            # Read radius bounds from the same source as the action space.
            if (
                self._level.metadata is not None
                and "action_bounds" in self._level.metadata
            ):
                r_low, r_high = self._level.metadata["action_bounds"]["r"]
            else:
                r_low, r_high = 0.1, 1.5

            if isinstance(action, np.ndarray):
                if action.shape != (expected_dim,):
                    raise ValueError(
                        f"Expected action shape ({expected_dim},), got {action.shape}"
                    )
                converted_action = [
                    (action[i], action[i + 1], np.clip(action[i + 2], r_low, r_high))
                    for i in range(0, len(action), 3)
                ]
            elif isinstance(action, list):
                if len(action) != len(self._level.action_objects):
                    raise ValueError(
                        f"Expected {len(self._level.action_objects)} action tuples, got {len(action)}"
                    )
                for i, pos in enumerate(action):
                    if not isinstance(pos, (tuple, list)) or len(pos) != 3:
                        raise ValueError(
                            f"Action {i} must be a tuple/list of length 3 (x, y, size), got {pos}"
                        )
                    if not all(isinstance(x, (int, float)) for x in pos):
                        raise ValueError(
                            f"Action {i} coordinates must be numbers, got {pos}"
                        )
                converted_action = [
                    (x, y, np.clip(s, r_low, r_high)) for (x, y, s) in action
                ]
            else:
                raise ValueError(
                    f"Action must be list of tuples or numpy array, got {type(action)}"
                )

        return converted_action

    def validate_action(
        self, action: list[tuple[float, float, float]] | np.ndarray
    ) -> dict[str, Any]:
        """Validate an action and return failure information instead of raising.

        Returns:
            dict with keys:
              - "invalid" (bool): True if the action cannot be placed.
              - "action" (list | None): Validated list of (x, y, radius) tuples,
                or None when invalid.
              - "error" (str | None): Human-readable reason for rejection,
                or None when valid.
        """
        return self._validate_action_with_failure(action)

    def _validate_action_with_failure(
        self, action: list[tuple[float, float, float]] | np.ndarray
    ) -> dict[str, Any]:
        """Validate action and return failure information instead of raising exceptions."""
        try:
            converted_action = self._validate_action(action)

            for i, (x, y, radius) in enumerate(converted_action):
                if not self._is_valid_placement(x, y, radius):
                    return {
                        "invalid": True,
                        "action": None,
                        "error": f"Action object {i} at ({x:.2f}, {y:.2f}) with radius {radius:.2f} is invalid",
                    }

            return {"invalid": False, "action": converted_action, "error": None}
        except ValueError as e:
            return {"invalid": True, "action": None, "error": str(e)}

    def _is_valid_placement(self, x: float, y: float, radius: float) -> bool:
        """Check if placing an object at (x, y) with given radius is valid."""
        return is_valid_placement(self._level, x, y, radius)

    def place_action(self, action) -> None:
        """Place action objects at the given position without running physics.

        Accepts the same formats as step(): a flat (x, y, r) tuple for single-object
        levels, or a list of (x, y, r) tuples for multi-object levels.
        """
        normalized: list[tuple[float, float, float]] = (
            [action] if isinstance(action, tuple) and len(action) == 3 else action  # type: ignore[assignment]
        )
        validation_result = self._validate_action_with_failure(normalized)
        if validation_result["invalid"]:
            raise ValueError(f"Invalid action: {validation_result['error']}")
        self._place_action_objects(validation_result["action"])
        self.action_placed = True

    def _place_action_objects(self, action: list[tuple[float, float, float]]) -> None:
        """Place action objects at the specified positions and sizes."""
        if len(action) != len(self._level.action_objects):
            raise ValueError(
                f"Expected {len(self._level.action_objects)} positions, got {len(action)}"
            )
        self.engine.place_action_objects(action)

    def _get_observation(self) -> Any:
        """Get the current observation based on observation_type."""
        if self.observation_type == "physics_state":
            return self._get_physics_state()
        elif self.observation_type == "image":
            return self._get_image_observation()
        elif self.observation_type == "both":
            return {
                "physics_state": self._get_physics_state(),
                "image": self._get_image_observation(),
            }
        else:
            raise ValueError(f"Unknown observation_type: {self.observation_type}")

    def _get_physics_state(self) -> dict[str, Any]:
        """Get the physics state observation."""
        if self.engine.world is None:
            return {}

        objects_state = {}
        object_names = list(self._level.objects.keys())

        for name in object_names:
            if name in self.engine.bodies:
                body = self.engine.bodies[name]
                objects_state[name] = {
                    "position": np.array(
                        [body.position.x, body.position.y], dtype=np.float32
                    ),
                    "velocity": np.array(
                        [body.linearVelocity.x, body.linearVelocity.y], dtype=np.float32
                    ),
                    "angle": float(body.angle),
                    "angular_velocity": float(body.angularVelocity),
                    "type": type(self._level.objects[name]).__name__,
                }
            else:
                obj = self._level.objects[name]
                objects_state[name] = {
                    "position": np.array([obj.x, obj.y], dtype=np.float32),
                    "velocity": np.array([0.0, 0.0], dtype=np.float32),
                    "angle": float(obj.angle),
                    "angular_velocity": 0.0,
                    "type": type(obj).__name__,
                }

        contact_matrix = np.zeros(
            (len(object_names), len(object_names)), dtype=np.bool_
        )
        for i, name1 in enumerate(object_names):
            for j, name2 in enumerate(object_names):
                if i != j and self.engine.has_contact(name1, name2):
                    contact_matrix[i, j] = True

        return {
            "objects": objects_state,
            "contacts": contact_matrix,
            "step_count": self.step_count,
        }

    def _get_image_observation(self) -> np.ndarray:
        """Get image observation by rendering current simulation state."""
        if self._image_renderer is None:
            from interphyre.render import OpenCVRenderer

            width, height = self.image_size
            world_size = 10.0
            target_ppm = min(width, height) / world_size
            ppm = min(target_ppm, self.image_ppm)
            self._image_renderer = OpenCVRenderer(width=width, height=height, ppm=ppm)

        if self.discrete_colors:
            return self._image_renderer.render_discrete(self.engine)
        return self._image_renderer.render(self.engine)

    def _calculate_reward(self, success: bool, truncated: bool) -> float:
        """Calculate the reward for the current state."""
        if success:
            return 1.0
        elif truncated:
            return -0.1
        else:
            return 0.0

    def _get_info_dict(
        self,
        success: bool,
        terminated: bool,
        truncated: bool,
        *,
        world_stationary: bool | None = None,
    ) -> dict[str, Any]:
        """Get the info dictionary for the current step.

        world_stationary can be pre-computed by the caller to avoid appending
        a spurious frame to the velocity history after the simulation loop ends.
        When None, it is computed here (appropriate for step_until callers).
        """
        if terminated and truncated:
            truncated = False

        if world_stationary is None:
            world_stationary = (
                self.engine.world_is_stationary() if self.engine.world else False
            )

        info = {
            "level_name": self._level.name,
            "step_count": self.step_count,
            "action_placed": self.action_placed,
            "success": success,
            "terminated": terminated,
            "truncated": truncated,
            "world_stationary": world_stationary,
        }

        if hasattr(self.engine, "get_contact_statistics"):
            contact_stats = self.engine.get_contact_statistics()
            info["contact_statistics"] = contact_stats

        if self.config.enable_profiling:
            perf_stats = self.engine.profiler.get_stats()
            info["performance_stats"] = perf_stats

        return info

    def simulate(
        self,
        steps: int | None = None,
        return_trace: bool = False,
        verbose: bool = False,
    ) -> list[tuple[Any, float, bool, bool, dict[str, Any]]] | None:
        """Public method for debugging/profiling: run simulation with custom parameters."""
        if steps is None:
            steps = self.config.max_steps

        if self.engine.world is None:
            raise ValueError(
                "World is not initialized. Call reset() before simulating."
            )

        if self._rollout_complete:
            raise RuntimeError(
                "Rollout is already complete. Call reset() before calling simulate() again."
            )

        trace = []
        status = "running"
        terminated = False

        if self.config.enable_profiling:
            self.engine.profiler.start_step_batch()

        for i in range(steps):
            self._step_physics()

            done = self._level.success_condition(self.engine)
            if done:
                status = "success"
            elif self.engine.world_is_stationary():
                status = "world_is_stationary"
            elif i == steps - 1:
                status = "timeout"
                terminated = True

            if return_trace:
                observation = self._get_observation()
                reward = self._calculate_reward(done, terminated)
                info = self._get_info_dict(done, done, terminated)
                trace.append((observation, reward, done, terminated, info))

            self.render()

            if verbose:
                print(f"Step {i + 1}/{steps}, status: {status}")
            if done or terminated:
                break

        if self.config.enable_profiling:
            self.engine.profiler.end_step_batch(steps)

        return trace if return_trace else None

    def render(self) -> None:
        """Render the current state."""
        if self.renderer:
            self.renderer.render(self.engine)

    def close(self) -> None:
        """Close the environment and release all resources including the Box2D world."""
        if self.renderer:
            self.renderer.close()
        if self._image_renderer is not None:
            self._image_renderer.close()
            self._image_renderer = None
        self.engine.close()

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics from the engine's profiler."""
        return self.engine.profiler.get_stats()

    def reset_profiler(self) -> None:
        """Reset the performance profiler."""
        self.engine.profiler.reset()

    def get_contact_log(self) -> list[dict[str, Any]]:
        """Get the full contact event log for research purposes."""
        return self.engine.get_contact_log()

    def get_contact_statistics(self) -> dict[str, Any]:
        """Get statistics about all contacts for research purposes."""
        return self.engine.get_contact_statistics()

    def get_level_info(self) -> dict[str, Any]:
        """Get information about the current level."""
        return {
            "name": self._level.name,
            "action_objects": self._level.action_objects,
            "total_objects": len(self._level.objects),
            "object_types": {
                name: type(obj).__name__ for name, obj in self._level.objects.items()
            },
            "metadata": self._level.metadata,
        }

    def describe_scene(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the current scene state.

        Includes live physics state (positions, velocities) from the Box2D engine,
        per-object metadata (color, size, dynamic flag), current contact pairs,
        step count, and whether the success condition is currently satisfied.

        Returns:
            dict with keys:
              - "objects": dict mapping name -> {type, color, x, y, vx, vy, angle,
                angular_velocity, dynamic, size: {radius} | {length, thickness} |
                {bottom_width, top_width, height}}
              - "contacts": list of [name_a, name_b] pairs currently in contact
              - "step_count": int
              - "success": bool
        """
        from interphyre.objects import Ball, Bar, Basket

        objects: dict[str, Any] = {}
        for name, obj in self._level.objects.items():
            # Read live kinematics from the physics body when available;
            # fall back to construction-time values for unplaced action objects.
            if self.engine.world is not None and name in self.engine.bodies:
                body = self.engine.bodies[name]
                x = float(body.position.x)
                y = float(body.position.y)
                vx = float(body.linearVelocity.x)
                vy = float(body.linearVelocity.y)
                angle = float(body.angle)
                angular_velocity = float(body.angularVelocity)
            else:
                x, y = float(obj.x), float(obj.y)
                vx, vy = 0.0, 0.0
                angle = float(obj.angle)
                angular_velocity = 0.0

            # Size fields depend on object geometry.
            if isinstance(obj, Ball):
                size: dict[str, float] = {"radius": float(obj.radius)}
            elif isinstance(obj, Bar):
                size = {"length": float(obj.length), "thickness": float(obj.thickness)}
            elif isinstance(obj, Basket):
                size = {
                    "bottom_width": float(obj.bottom_width),
                    "top_width": float(obj.top_width),
                    "height": float(obj.height),
                }
            else:
                size = {}

            objects[name] = {
                "type": type(obj).__name__,
                "color": obj.color,
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "angle": angle,
                "angular_velocity": angular_velocity,
                "dynamic": obj.dynamic,
                "size": size,
            }

        # Contact pairs come from the engine's contact listener (frozensets of names).
        contacts = [list(pair) for pair in self.engine.contact_listener.contacts]

        success = (
            self._level.success_condition(self.engine)
            if self.engine.world is not None
            else False
        )

        return {
            "objects": objects,
            "contacts": contacts,
            "step_count": self.step_count,
            "success": success,
        }

    def get_object_position(self, name: str) -> tuple[float, float]:
        """Return the current (x, y) position of a named object.

        Reads from the live Box2D body when the world is active, otherwise
        falls back to the construction-time position stored on the object.

        Raises:
            KeyError: If *name* is not a recognised object in the current level.
        """
        if name not in self._level.objects:
            raise KeyError(f"Unknown object: {name!r}")

        if self.engine.world is not None and name in self.engine.bodies:
            body = self.engine.bodies[name]
            return (float(body.position.x), float(body.position.y))

        obj = self._level.objects[name]
        return (float(obj.x), float(obj.y))

    def get_object_state(self, name: str) -> dict[str, Any]:
        """Return the full kinematic state of a named object.

        Returns:
            dict with keys: x, y, vx, vy, angle, angular_velocity, dynamic.

        Raises:
            KeyError: If *name* is not a recognised object in the current level.
        """
        if name not in self._level.objects:
            raise KeyError(f"Unknown object: {name!r}")

        obj = self._level.objects[name]

        if self.engine.world is not None and name in self.engine.bodies:
            body = self.engine.bodies[name]
            return {
                "x": float(body.position.x),
                "y": float(body.position.y),
                "vx": float(body.linearVelocity.x),
                "vy": float(body.linearVelocity.y),
                "angle": float(body.angle),
                "angular_velocity": float(body.angularVelocity),
                "dynamic": obj.dynamic,
            }

        return {
            "x": float(obj.x),
            "y": float(obj.y),
            "vx": 0.0,
            "vy": 0.0,
            "angle": float(obj.angle),
            "angular_velocity": 0.0,
            "dynamic": obj.dynamic,
        }
