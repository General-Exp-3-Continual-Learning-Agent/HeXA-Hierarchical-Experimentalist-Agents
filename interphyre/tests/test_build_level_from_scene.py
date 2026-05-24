"""Regression tests for build_level_from_scene across all 25 levels.

Validates FIX-BUILD-LEVEL-FROM-SCENE-ALL-LEVELS: every level's build_level()
accepts scene=None and the @register_level wrapper applies scene overrides
to constructed objects.
"""

import pytest
from interphyre.environment import InterphyreEnv
from interphyre.levels import build_level_from_scene, list_levels, load_level


ALL_LEVELS = list_levels()


@pytest.mark.parametrize("level_name", ALL_LEVELS)
def test_seed_determinism_unchanged(level_name):
    """build_level(seed=42) produces bit-identical output across two calls."""
    l1 = load_level(level_name, seed=42)
    l2 = load_level(level_name, seed=42)
    for obj_name in l1.objects:
        o1, o2 = l1.objects[obj_name], l2.objects[obj_name]
        assert o1.x == o2.x, f"{obj_name}.x mismatch"
        assert o1.y == o2.y, f"{obj_name}.y mismatch"


@pytest.mark.parametrize("level_name", ALL_LEVELS)
def test_scene_position_override(level_name):
    """build_level_from_scene with a position override produces correct position."""
    level = load_level(level_name, seed=42)
    # Pick the first object to override
    target = next(iter(level.objects))
    original_x = level.objects[target].x
    new_x = original_x + 0.5
    scene = {target: {"x": new_x}}
    rebuilt = build_level_from_scene(level_name, scene)
    assert abs(rebuilt.objects[target].x - new_x) < 1e-9


@pytest.mark.parametrize("level_name", ALL_LEVELS)
def test_round_trip_via_describe_scene(level_name):
    """env.reset() -> describe_scene() -> build_level_from_scene() -> positions match."""
    # validate=False: this test checks geometry round-trips only, not level validity.
    env = InterphyreEnv(level_name, seed=42, validate=False)
    env.reset()
    scene1 = env.describe_scene()

    # Extract construction-compatible scene dict
    scene_dict = {}
    for obj_name, obj_data in scene1["objects"].items():
        spec = {"x": obj_data["x"], "y": obj_data["y"]}
        size = obj_data.get("size", {})
        if "radius" in size:
            spec["radius"] = size["radius"]
        scene_dict[obj_name] = spec

    # Rebuild from scene (same seed to match dynamic object names)
    level2 = load_level(level_name, seed=42, scene=scene_dict)
    env2 = InterphyreEnv(level_name, seed=42, validate=False)
    env2._level = level2
    env2.reset()
    scene2 = env2.describe_scene()

    for obj_name in scene1["objects"]:
        if obj_name not in scene2["objects"]:
            continue
        s1 = scene1["objects"][obj_name]
        s2 = scene2["objects"][obj_name]
        assert abs(s1["x"] - s2["x"]) < 0.01, f"{obj_name}.x mismatch"
        assert abs(s1["y"] - s2["y"]) < 0.01, f"{obj_name}.y mismatch"

    env.close()
    env2.close()
