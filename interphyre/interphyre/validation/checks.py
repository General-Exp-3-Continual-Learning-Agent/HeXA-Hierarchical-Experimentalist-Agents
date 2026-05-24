"""Primitive validity checks for procedurally generated levels.

These two functions are the foundation of the validation pipeline:

  - is_trivial: detects levels where the success condition is already met at t=0,
    before the agent places any action objects. Cost: one engine init, zero steps.

  - extract_scene_dict: serializes the full geometry of a Level to a plain dict
    suitable for JSON storage and for passing to build_level_from_scene. This is the
    reproducibility artifact — any level can be reconstructed bit-identically from
    its scene dict, independent of seeds or RNG state.

Both functions are re-exported from interphyre.validation.__init__ as public API.
"""

from __future__ import annotations

from interphyre.config import SimulationConfig
from interphyre.engine import Box2DEngine
from interphyre.level import Level
from interphyre.objects import Ball, Bar, Basket
from interphyre.validation.placement import is_valid_placement

# Corners for the trivial-check ball probe: placed at ±4.3 (inside the ±5 world
# boundary) so that even a min-radius ball (0.1) fits fully within bounds.
_TRIVIAL_CORNER_POSITIONS = [(-4.3, 4.3), (4.3, 4.3), (-4.3, -4.3), (4.3, -4.3)]
# Smallest ball radius that Box2D will simulate stably.
_TRIVIAL_MIN_RADIUS = 0.1


def is_trivial(
    level: Level,
    config: SimulationConfig | None = None,
    physics_steps: int = 1000,
) -> bool:
    """Return True if the success condition is met at t=0 or by physics alone.

    Runs physics_steps simulation steps with no agent action. If the success
    condition is ever satisfied, the level is trivial — the agent is not needed.

    physics_steps=1000 corresponds to ~5 seconds at the default 60Hz time step,
    which is sufficient for unstable dynamics (swaying poles, rolling balls,
    tipping objects) to play out.

    Design note: action objects are absent from the engine by construction
    (Box2DEngine._create_world skips them), so the simulation represents purely
    the scene's own dynamics.
    """
    cfg = config or SimulationConfig()
    engine = Box2DEngine(level=level, config=cfg)

    try:
        # t=0 check: success before any physics
        if level.success_condition(engine):
            return True

        # Post-physics check: simulate without agent action
        for _ in range(physics_steps):
            engine.world.Step(cfg.time_step, cfg.velocity_iters, cfg.position_iters)
            engine.time_update(cfg.time_step)
            if level.success_condition(engine):
                return True
    finally:
        engine.close()

    # Check 2: min-radius ball at a safe world corner.
    # Catches levels where any agent presence (even a tiny non-interacting ball)
    # changes Box2D dynamics enough to trigger success — a physics-level trivial
    # solution that the no-ball check cannot detect (different island structure,
    # different sleeping thresholds when a body is absent entirely).
    #
    # IMPORTANT: place_action_objects mutates level.objects[name].radius in-place
    # (engine.py sets obj.radius = size). Save and restore action object radii so
    # the caller's level object is not corrupted after this check returns.
    _saved_radii = {
        name: level.objects[name].radius
        for name in level.action_objects
        if hasattr(level.objects[name], "radius")
    }
    # Require all valid corners to trigger success independently.  A genuine
    # "any ball presence" trivial fires from every corner; a false positive from
    # Box2D island/sleeping perturbation is position-dependent and fires from
    # only some corners.  Requiring consensus eliminates those false positives.
    try:
        corner_results: list[bool] = []
        for cx, cy in _TRIVIAL_CORNER_POSITIONS:
            if not is_valid_placement(level, cx, cy, _TRIVIAL_MIN_RADIUS):
                continue
            engine2 = Box2DEngine(level=level, config=cfg)
            triggered = False
            try:
                engine2.place_action_objects([(cx, cy, _TRIVIAL_MIN_RADIUS)])
                if level.success_condition(engine2):
                    triggered = True
                else:
                    for _ in range(physics_steps):
                        engine2.world.Step(
                            cfg.time_step, cfg.velocity_iters, cfg.position_iters
                        )
                        engine2.time_update(cfg.time_step)
                        if level.success_condition(engine2):
                            triggered = True
                            break
            finally:
                engine2.close()
            corner_results.append(triggered)
        if corner_results and all(corner_results):
            return True
    finally:
        for name, r in _saved_radii.items():
            level.objects[name].radius = r
    return False


# --- Per-type attribute extraction ---
# Each helper returns a dict of all constructor parameters for that object type.
# Only concrete, serializable attributes are included — no Box2D internal state,
# no computed properties (e.g. Bar.x1/y1/x2/y2 are derived from x/y/length/angle).
# The dicts are compatible with build_level_from_scene: _apply_scene_overrides uses
# setattr, so all keys must correspond to settable attributes on the object.

_PHYRE_OBJECT_FIELDS = (
    "x",
    "y",
    "angle",
    "color",
    "dynamic",
    "restitution",
    "friction",
    "linear_damping",
    "angular_damping",
    "density",
)


def _extract_ball(obj: Ball) -> dict:
    attrs = {field: getattr(obj, field) for field in _PHYRE_OBJECT_FIELDS}
    attrs["radius"] = obj.radius
    return attrs


def _extract_bar(obj: Bar) -> dict:
    # Use center-based representation (x, y, length, angle, thickness).
    # Endpoint properties (x1/y1/x2/y2) are derived from these and are not stored —
    # storing them would create redundancy that could diverge after a setattr round-trip.
    attrs = {field: getattr(obj, field) for field in _PHYRE_OBJECT_FIELDS}
    attrs["length"] = obj.length
    attrs["thickness"] = obj.thickness
    return attrs


def _extract_basket(obj: Basket) -> dict:
    # Store the fully-resolved dimensions (not just scale) so reconstruction does not
    # depend on the scale → dimension formula remaining constant. scale is also stored
    # for reference, but bottom_width/top_width/height are the authoritative values.
    attrs = {field: getattr(obj, field) for field in _PHYRE_OBJECT_FIELDS}
    attrs["bottom_width"] = obj.bottom_width
    attrs["top_width"] = obj.top_width
    attrs["height"] = obj.height
    attrs["scale"] = obj.scale
    attrs["wall_thickness"] = obj.wall_thickness
    attrs["floor_thickness"] = obj.floor_thickness
    attrs["anchor"] = obj.anchor
    attrs["double_walls"] = obj.double_walls
    attrs["enable_sensor"] = obj.enable_sensor
    attrs["sensor_margin"] = obj.sensor_margin
    attrs["sensor_height_ratio"] = obj.sensor_height_ratio
    return attrs


def extract_scene_dict(level: Level) -> dict:
    """Return {object_name: {attr: value, ...}} for all objects in the level.

    Serializes all constructor parameters of each PhyreObject in the level. The
    result is a plain dict containing only JSON-serializable Python scalars
    (float, int, str, bool) — no Box2D internal state.

    The output is compatible with build_level_from_scene: passing this dict as the
    scene argument will reproduce the level geometry bit-identically, regardless of
    seed or RNG state.

    Raises ValueError if the level contains an object of an unrecognised type.
    """
    scene: dict = {}
    for name, obj in level.objects.items():
        if isinstance(obj, Ball):
            scene[name] = _extract_ball(obj)
        elif isinstance(obj, Bar):
            scene[name] = _extract_bar(obj)
        elif isinstance(obj, Basket):
            scene[name] = _extract_basket(obj)
        else:
            raise ValueError(
                f"extract_scene_dict: unrecognised object type '{type(obj).__name__}' "
                f"for object '{name}'. Add a handler in checks.py."
            )
    return scene
