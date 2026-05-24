import json
import os
import sys

import gymnasium as gym
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from interphyre import list_levels
from interphyre.config import SimulationConfig
from interphyre.environment import InterphyreEnv
from interphyre.level import Level
from interphyre.levels import load_level
from interphyre.objects import Ball


def _make_multi_action_level() -> Level:
    def success_condition(engine):
        return False

    objects = {
        "red_ball1": Ball(x=-1.0, y=2.0, radius=0.5, color="red", dynamic=True),
        "red_ball2": Ball(x=1.0, y=2.0, radius=0.5, color="red", dynamic=True),
        "green_ball": Ball(x=0.0, y=-2.0, radius=0.5, color="green", dynamic=False),
    }
    return Level(
        name="multi_action_test",
        objects=objects,
        action_objects=["red_ball1", "red_ball2"],
        success_condition=success_condition,
        metadata={"description": "Synthetic multi-action level for API tests."},
    )


def test_environment_initialization():
    """Test environment initialization with different configurations."""
    level = load_level("two_body_problem", seed=42)

    # Test basic initialization
    env = InterphyreEnv(level)
    assert env.level == level
    assert env.config is not None
    assert env.engine is not None
    assert env.step_count == 0
    assert not env.action_placed

    # Test with custom config
    config = SimulationConfig(fps=60, enable_profiling=True)
    env = InterphyreEnv(level, config=config)
    assert env.config.fps == 60
    assert env.config.enable_profiling is True

    # Test with different observation types
    env_physics = InterphyreEnv(level, observation_type="physics_state")
    env_image = InterphyreEnv(level, observation_type="image")
    env_both = InterphyreEnv(level, observation_type="both")

    assert env_physics.observation_type == "physics_state"
    assert env_image.observation_type == "image"
    assert env_both.observation_type == "both"


def test_action_space_setup():
    """Test action space setup for different level configurations."""
    # Test level with action objects
    level = load_level("two_body_problem", seed=42)
    env = InterphyreEnv(level)

    expected_dim = len(level.action_objects) * 3
    assert env.action_space.shape == (expected_dim,)
    assert env.action_space.dtype == np.float32
    assert env.action_space.low[0] == -5.0
    assert env.action_space.high[0] == 5.0
    assert env.action_space.low[2] == pytest.approx(0.1)
    assert env.action_space.high[2] == pytest.approx(1.5)

    # Test level without action objects (should have empty action space)
    # We'd need a level without action objects to test this properly
    # For now, just verify the logic works for levels with action objects


def test_multi_object_action_space():
    """Test action space setup for levels with multiple action objects."""
    # Test level with multiple action objects
    level = _make_multi_action_level()
    env = InterphyreEnv(level)

    # Should have 2 action objects, so action space should be (6,)
    _ = len(level.action_objects) * 3  # 2 objects * 3 coordinates = 6
    assert env.action_space.shape == (6,)
    assert env.action_space.dtype == np.float32
    assert env.action_space.low[0] == -5.0
    assert env.action_space.high[0] == 5.0

    # Test that the action space matches the expected dimensions
    assert len(level.action_objects) == 2
    assert "red_ball1" in level.action_objects
    assert "red_ball2" in level.action_objects


def test_multi_object_step():
    """Test that the environment can handle actions for multiple objects."""
    level = _make_multi_action_level()
    env = InterphyreEnv(level)

    # Test with a 6D action (2 objects * 3 coordinates)
    action = np.array([1.0, 2.0, 0.5, 3.0, 4.0, 0.8], dtype=np.float32)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

    # Should not fail due to validation
    assert isinstance(terminated, bool)
    assert env.step_count >= 1
    assert env.action_placed is True

    # Test with list of tuples
    env.reset()
    action_list = [(1.0, 2.0, 0.5), (3.0, 4.0, 0.8)]
    obs, reward, terminated, truncated, info = env.step(action_list)

    assert isinstance(terminated, bool)
    assert env.step_count >= 1
    assert env.action_placed is True


def test_observation_space_setup():
    """Test observation space setup for different observation types."""
    level = load_level("two_body_problem", seed=42)

    # Test physics state observation
    env = InterphyreEnv(level, observation_type="physics_state")
    obs_space = env.observation_space

    assert isinstance(obs_space, gym.spaces.Dict)
    assert "objects" in obs_space.spaces
    assert "contacts" in obs_space.spaces
    assert "step_count" in obs_space.spaces

    # Test image observation
    env = InterphyreEnv(level, observation_type="image")
    obs_space = env.observation_space

    assert isinstance(obs_space, gym.spaces.Box)
    assert obs_space.shape == (600, 600, 3)
    assert obs_space.dtype == np.uint8

    # Test both observation types
    env = InterphyreEnv(level, observation_type="both")
    obs_space = env.observation_space

    assert isinstance(obs_space, gym.spaces.Dict)
    assert "physics_state" in obs_space.spaces
    assert "image" in obs_space.spaces


def test_action_validation():
    """Test action validation with various input formats."""
    level = load_level("two_body_problem", seed=42)
    env = InterphyreEnv(level)

    # Test valid numpy array action
    valid_action = np.array(
        [1.0, 2.0, 0.5], dtype=np.float32
    )  # Only 1 action object (red_ball) with size
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(valid_action)
    assert isinstance(terminated, bool)

    # Test valid list action
    env.reset()
    valid_list_action = [(1.0, 2.0, 0.5)]  # Only 1 action object with size
    obs, reward, terminated, truncated, info = env.step(valid_list_action)
    assert isinstance(terminated, bool)

    # Test invalid action shape
    env.reset()
    invalid_action = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32
    )  # Wrong shape (6 instead of 3)
    obs, reward, terminated, truncated, info = env.step(invalid_action)
    assert terminated is True
    assert info.get("invalid_action") is True

    # Test invalid action type
    env.reset()
    obs, reward, terminated, truncated, info = env.step("invalid_action")
    assert terminated is True
    assert info.get("invalid_action") is True


def test_reset_behavior():
    """Test reset behavior and return values."""
    level = load_level("two_body_problem", seed=42)
    env = InterphyreEnv(level)

    # Test initial reset
    obs, info = env.reset()

    assert env.step_count == 0
    assert not env.action_placed
    assert isinstance(obs, dict)
    assert isinstance(info, dict)

    # Check info dictionary contents
    assert "level_name" in info
    assert "action_objects" in info
    assert "total_objects" in info
    assert "step_count" in info
    assert "action_placed" in info
    assert "success" in info
    assert "truncated" in info

    assert info["level_name"] == level.name
    assert info["action_objects"] == level.action_objects
    assert info["total_objects"] == len(level.objects)
    assert info["step_count"] == 0
    assert info["action_placed"] is False
    assert info["success"] is False
    assert info["truncated"] is False


def test_step_behavior():
    """Test step behavior and return values."""
    level = load_level("two_body_problem", seed=42)
    env = InterphyreEnv(level)

    obs, info = env.reset()
    action = np.array(
        [1.0, 2.0, 0.5], dtype=np.float32
    )  # Only 1 action object with size

    obs, reward, terminated, truncated, info = env.step(action)

    assert 1 <= env.step_count <= env.max_steps
    assert env.action_placed is True
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    # Check info dictionary contents
    assert "level_name" in info
    assert "step_count" in info
    assert "action_placed" in info
    assert "success" in info
    assert "terminated" in info
    assert "truncated" in info
    assert "world_stationary" in info

    assert info["step_count"] == env.step_count
    assert info["action_placed"] is True


def test_physics_state_observation():
    """Test physics state observation format."""
    level = load_level("two_body_problem", seed=42)
    env = InterphyreEnv(level, observation_type="physics_state")

    obs, info = env.reset()

    assert isinstance(obs, dict)
    assert "objects" in obs
    assert "contacts" in obs
    assert "step_count" in obs

    # Check objects structure
    for obj_name, obj_data in obs["objects"].items():
        assert "position" in obj_data
        assert "velocity" in obj_data
        assert "angle" in obj_data
        assert "angular_velocity" in obj_data
        assert "type" in obj_data

        assert len(obj_data["position"]) == 2
        assert len(obj_data["velocity"]) == 2
        assert isinstance(obj_data["angle"], (int, float))
        assert isinstance(obj_data["angular_velocity"], (int, float))
        assert isinstance(obj_data["type"], str)

    # Check contacts structure
    assert obs["contacts"].shape == (len(level.objects), len(level.objects))
    assert obs["contacts"].dtype == np.bool_

    # Check step count
    assert obs["step_count"] == 0


def test_engine_state_improvement():
    """Test that engine.get_state() returns meaningful information."""
    level = load_level("two_body_problem", seed=42)
    env = InterphyreEnv(level)

    obs, info = env.reset()
    state = env.engine.get_state()

    assert isinstance(state, dict)
    assert "objects" in state
    assert "contacts" in state
    assert "world_properties" in state

    # Check world properties
    world_props = state["world_properties"]
    assert "gravity" in world_props
    assert "body_count" in world_props
    assert "contact_count" in world_props

    # Check objects structure
    for obj_name, obj_data in state["objects"].items():
        assert "position" in obj_data
        assert "velocity" in obj_data
        assert "angle" in obj_data
        assert "angular_velocity" in obj_data
        assert "type" in obj_data
        assert "dynamic" in obj_data


def test_error_handling():
    """Test error handling for invalid inputs."""
    level = load_level("two_body_problem", seed=42)

    # Test invalid level name (raises ModuleNotFoundError when level module doesn't exist)
    with pytest.raises(ModuleNotFoundError):
        InterphyreEnv("nonexistent_level_name")

    # Test invalid observation type
    with pytest.raises(ValueError, match="Unknown observation_type"):
        InterphyreEnv(level, observation_type="invalid")

    # Test invalid action type
    # Discrete action space should be supported (MultiDiscrete)
    env_discrete = InterphyreEnv(level, action_type="discrete")
    assert hasattr(env_discrete, "action_space")
    import gymnasium as gym

    assert isinstance(env_discrete.action_space, gym.spaces.MultiDiscrete)


def test_level_info_method():
    """Test the get_level_info method."""
    level = load_level("two_body_problem", seed=42)
    env = InterphyreEnv(level)

    level_info = env.get_level_info()

    assert isinstance(level_info, dict)
    assert "name" in level_info
    assert "action_objects" in level_info
    assert "total_objects" in level_info
    assert "object_types" in level_info
    assert "metadata" in level_info

    assert level_info["name"] == level.name
    assert level_info["action_objects"] == level.action_objects
    assert level_info["total_objects"] == len(level.objects)
    assert isinstance(level_info["object_types"], dict)
    assert isinstance(level_info["metadata"], dict)


@pytest.mark.slow
def test_simulate_method_improvement():
    """Test the improved simulate method."""
    level = load_level("two_body_problem", seed=42)
    env = InterphyreEnv(level)

    env.reset()

    # Test simulation without trace
    result = env.simulate(steps=10, return_trace=False)
    assert result is None

    # Test simulation with trace
    trace = env.simulate(steps=10, return_trace=True)
    assert isinstance(trace, list)
    assert len(trace) > 0

    # Check trace structure
    for obs, reward, done, truncated, info in trace:
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)


@pytest.mark.slow
def test_describe_scene_serializable_all_levels():
    """describe_scene() must return a json.dumps()-serializable dict for all 25 levels."""
    for level_name in list_levels():
        env = InterphyreEnv(level_name=level_name)
        env.reset()
        scene = env.describe_scene()
        # Must not raise — no numpy arrays, no Box2D objects
        json.dumps(scene)
        env.close()


@pytest.mark.slow
def test_describe_scene_structure_and_values():
    """describe_scene() values must match engine.get_state() and _get_physics_state()."""
    level = load_level("two_body_problem", seed=42)
    env = InterphyreEnv(level)
    env.reset()

    # Advance physics so positions differ from construction-time values.
    for _ in range(20):
        obs, _, done, _, _ = env.step(None)
        if done:
            break

    scene = env.describe_scene()
    phys = env._get_physics_state()
    eng = env.engine.get_state()

    assert set(scene.keys()) == {"objects", "contacts", "step_count", "success"}
    assert isinstance(scene["contacts"], list)
    assert isinstance(scene["step_count"], int)
    assert isinstance(scene["success"], bool)

    for name, sd in scene["objects"].items():
        # Required fields present
        for field in (
            "type",
            "color",
            "x",
            "y",
            "vx",
            "vy",
            "angle",
            "angular_velocity",
            "dynamic",
            "size",
        ):
            assert field in sd, f"Missing field '{field}' for object '{name}'"

        # Values match _get_physics_state
        if name in phys["objects"]:
            pd = phys["objects"][name]
            assert abs(sd["x"] - float(pd["position"][0])) < 1e-5, (
                f"x mismatch for {name}"
            )
            assert abs(sd["y"] - float(pd["position"][1])) < 1e-5, (
                f"y mismatch for {name}"
            )
            assert abs(sd["vx"] - float(pd["velocity"][0])) < 1e-5, (
                f"vx mismatch for {name}"
            )
            assert abs(sd["vy"] - float(pd["velocity"][1])) < 1e-5, (
                f"vy mismatch for {name}"
            )

        # Values match engine.get_state
        if name in eng["objects"]:
            ed = eng["objects"][name]
            assert abs(sd["x"] - float(ed["position"][0])) < 1e-5, (
                f"eng x mismatch for {name}"
            )
            assert abs(sd["y"] - float(ed["position"][1])) < 1e-5, (
                f"eng y mismatch for {name}"
            )

    env.close()
