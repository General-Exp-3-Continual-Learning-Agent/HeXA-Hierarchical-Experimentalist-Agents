import importlib
import logging
from collections.abc import Callable

from interphyre.level import Level

logger = logging.getLogger(__name__)

# Registry for level builders
_level_registry: dict[str, Callable] = {}


def _apply_scene_overrides(objects: dict, scene: dict | None) -> None:
    """Apply scene dict overrides to already-constructed objects.

    Each key in the scene dict must be an object name present in `objects`.
    Each value is a dict of attribute names to override values.
    Raises ValueError for unknown object names or attributes.
    """
    if not scene:
        return
    for name, overrides in scene.items():
        if name not in objects:
            raise ValueError(
                f"Scene specifies unknown object '{name}'. "
                f"Known objects: {sorted(objects.keys())}"
            )
        obj = objects[name]
        for attr, value in overrides.items():
            if not hasattr(obj, attr):
                raise ValueError(
                    f"Object '{name}' ({type(obj).__name__}) has no attribute '{attr}'"
                )
            setattr(obj, attr, value)


# Decorator to build and register a level without instantiating it at import time
def register_level(func: Callable):
    def wrapper(
        seed: int | None = None, variant: int = 0, scene: dict | None = None
    ) -> Level:
        level = func(seed, variant=variant, scene=scene)
        # Apply scene overrides to the constructed objects. Levels that handle
        # scene internally (e.g. two_body_problem) already have correct values;
        # re-applying identical values is a harmless no-op.
        _apply_scene_overrides(level.objects, scene)
        return level

    # Use the module name as the level name (matches filenames like "tipping_point")
    level_name = func.__module__.split(".")[-1]
    _level_registry[level_name] = wrapper

    return wrapper


def load_level(
    name: str, seed: int | None = None, variant: int = 0, scene: dict | None = None
) -> Level:
    if name not in _level_registry:
        # Try to dynamically import it
        importlib.import_module(f"interphyre.levels.{name}")
        if name not in _level_registry:
            raise ValueError(f"Level '{name}' could not be registered.")
    return _level_registry[name](seed, variant=variant, scene=scene)


def build_level_from_scene(level_name: str, scene: dict) -> Level:
    """Build a level directly from a fully-specified scene dict, bypassing RNG.

    The scene dict maps object names to their construction kwargs::

        {
            "green_ball": {"x": 1.0, "y": 0.5, "radius": 0.4},
            "blue_ball":  {"x": 3.0, "y": 0.5, "radius": 0.35},
            "red_ball":   {"radius": 0.5},
        }

    When the scene fully specifies all objects the result is bit-identical
    regardless of seed or RNG state. Partial scenes fall back to RNG for
    unspecified fields while preserving the draw order so that overriding one
    variable does not shift downstream draws.

    Args:
        level_name: Name of the registered level (e.g. "two_body_problem").
        scene: Mapping of object name → construction kwargs.

    Returns:
        A Level whose geometry matches the scene spec exactly.
    """
    return load_level(level_name, seed=None, scene=scene)


def list_levels() -> list[str]:
    """List all registered level names.

    Returns:
        List of level names sorted alphabetically

    Example:
        >>> from interphyre.levels import list_levels
        >>> levels = list_levels()
        >>> print(levels[:3])
        ['basket_case', 'catapult', 'dive_bomb']
    """
    return sorted(_level_registry.keys())


# Import all level modules to trigger @register_level decorators
_LEVEL_MODULES = [
    "basket_case",
    "catapult",
    "cliffhanger",
    "dive_bomb",
    "down_to_earth",
    "end_of_line",
    "falling_into_place",
    "flagpole_sitta",
    "just_a_nudge",
    "keyhole",
    "locust_swarm",
    "marble_race",
    "mind_the_gap",
    "off_the_rails",
    "pass_the_parcel",
    "pinball_machine",
    "seesaw",
    "staircase",
    "straight_face",
    "the_cradle",
    "the_funnel",
    "tipping_point",
    "two_body_problem",
    "wedge_issue",
    "zebra_crossing",
]

for _module in _LEVEL_MODULES:
    importlib.import_module(f"interphyre.levels.{_module}")
