"""
Focused environment coverage for discrete actions and invalid configurations.
"""

import numpy as np
import pytest

from interphyre.environment import InterphyreEnv
from interphyre.config import SimulationConfig
from interphyre.level import Level
from interphyre.levels import load_level
from interphyre.objects import Ball


def _make_simple_level():
    def success_condition(engine):
        return False

    objects = {
        "red_ball": Ball(x=0.0, y=0.0, radius=0.5, color="red", dynamic=True),
    }
    return Level(
        name="env_discrete_level",
        objects=objects,
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={},
    )


@pytest.mark.fast
def test_action_type_invalid_raises():
    level = _make_simple_level()
    with pytest.raises(ValueError, match="Unknown action_type"):
        InterphyreEnv(level, action_type="unknown")


@pytest.mark.fast
def test_observation_type_invalid_raises():
    level = _make_simple_level()
    with pytest.raises(ValueError, match="Unknown observation_type"):
        InterphyreEnv(level, observation_type="invalid")


@pytest.mark.fast
def test_discrete_action_conversion_valid_indices():
    level = _make_simple_level()
    env = InterphyreEnv(level, action_type="discrete")
    action = np.array([0, 0, 0], dtype=np.int64)
    converted = env._validate_action(action)
    assert converted == [(-5.0, -5.0, 0.1)]
    env.close()


@pytest.mark.fast
def test_discrete_action_list_requires_ints():
    level = _make_simple_level()
    env = InterphyreEnv(level, action_type="discrete")
    with pytest.raises(ValueError, match="must contain integer indices"):
        env._validate_action([(0.0, 0.0, 0.0)])
    env.close()


@pytest.mark.fast
def test_discrete_action_bounds_check():
    level = _make_simple_level()
    env = InterphyreEnv(level, action_type="discrete")
    # x index out of bounds
    with pytest.raises(ValueError, match="out of bounds"):
        env._validate_action(np.array([999, 0, 0], dtype=np.int64))
    env.close()


@pytest.mark.fast
def test_discrete_action_list_valid_indices():
    level = _make_simple_level()
    env = InterphyreEnv(level, action_type="discrete")
    converted = env._validate_action([(0, 0, 0)])
    assert converted == [(-5.0, -5.0, 0.1)]
    env.close()


@pytest.mark.fast
def test_discrete_action_list_requires_tuple_length():
    level = _make_simple_level()
    env = InterphyreEnv(level, action_type="discrete")
    with pytest.raises(ValueError, match="tuple/list of length 3"):
        env._validate_action([(0, 0)])
    env.close()


@pytest.mark.fast
def test_discrete_action_rejects_non_list_array():
    level = _make_simple_level()
    env = InterphyreEnv(level, action_type="discrete")
    with pytest.raises(
        ValueError, match="Action must be list of tuples or numpy array"
    ):
        env._validate_action("invalid")
    env.close()


@pytest.mark.fast
def test_no_action_objects_requires_empty_action():
    def success_condition(engine):
        return False

    level = Level(
        name="no_action",
        objects={"ball": Ball(x=0.0, y=0.0, radius=0.5)},
        action_objects=[],
        success_condition=success_condition,
        metadata={},
    )
    env = InterphyreEnv(level)
    assert env._validate_action([]) == []
    with pytest.raises(ValueError, match="No action objects"):
        env._validate_action([(-1.0, -1.0, 0.5)])
    env.close()


@pytest.mark.fast
def test_action_space_empty_for_no_action_objects_continuous():
    level = Level(
        name="no_action_continuous",
        objects={"ball": Ball(x=0.0, y=0.0, radius=0.5)},
        action_objects=[],
        success_condition=lambda engine: False,
        metadata={},
    )
    env = InterphyreEnv(level, action_type="continuous")
    assert env.action_space.shape == (0,)
    env.close()


@pytest.mark.fast
def test_action_space_empty_for_no_action_objects_discrete():
    level = Level(
        name="no_action_discrete",
        objects={"ball": Ball(x=0.0, y=0.0, radius=0.5)},
        action_objects=[],
        success_condition=lambda engine: False,
        metadata={},
    )
    env = InterphyreEnv(level, action_type="discrete")
    assert env.action_space.nvec.size == 0
    env.close()


@pytest.mark.fast
def test_observation_space_image_discrete_colors():
    level = _make_simple_level()
    env = InterphyreEnv(
        level, observation_type="image", discrete_colors=True, image_size=(32, 16)
    )
    assert env.observation_space.shape == (16, 32)
    assert env.observation_space.dtype == np.uint8
    env.close()


@pytest.mark.fast
def test_observation_space_consistency_across_modes():
    """Physics-state space used standalone must match the one nested inside 'both'."""
    level = _make_simple_level()

    env_phys = InterphyreEnv(level, observation_type="physics_state")
    env_img = InterphyreEnv(level, observation_type="image", image_size=(64, 48))
    env_both = InterphyreEnv(level, observation_type="both", image_size=(64, 48))

    # Physics-state space structure must be identical
    phys_space = env_phys.observation_space
    both_phys_space = env_both.observation_space["physics_state"]
    assert set(phys_space.spaces.keys()) == set(both_phys_space.spaces.keys())
    for key in phys_space.spaces:
        assert phys_space[key] == both_phys_space[key], f"mismatch in '{key}'"

    # Image space must match between standalone and nested
    img_space = env_img.observation_space
    both_img_space = env_both.observation_space["image"]
    assert img_space.shape == both_img_space.shape
    assert img_space.dtype == both_img_space.dtype
    assert np.array_equal(img_space.low, both_img_space.low)
    assert np.array_equal(img_space.high, both_img_space.high)

    env_phys.close()
    env_img.close()
    env_both.close()


@pytest.mark.fast
def test_reset_seed_and_interventions():
    level = _make_simple_level()
    env = InterphyreEnv(level)
    env.reset(seed=123)
    expected = np.random.default_rng(123).integers(0, 1000)
    actual = env.np_random.integers(0, 1000)
    assert actual == expected
    env.close()


@pytest.mark.fast
def test_step_raises_after_rollout_complete():
    level = _make_simple_level()
    env = InterphyreEnv(level)
    env.step([(-10.0, 0.0, 0.5)])
    with pytest.raises(RuntimeError, match="Episode already complete"):
        env.step([(-10.0, 0.0, 0.5)])
    env.close()


@pytest.mark.fast
def test_discrete_action_requires_expected_shape():
    level = _make_simple_level()
    env = InterphyreEnv(level, action_type="discrete")
    with pytest.raises(ValueError, match="Expected action shape"):
        env._validate_action(np.array([0, 0], dtype=np.int64))
    env.close()


@pytest.mark.fast
def test_discrete_action_list_requires_expected_length():
    level = _make_simple_level()
    env = InterphyreEnv(level, action_type="discrete")
    with pytest.raises(ValueError, match="Expected 1 action tuples"):
        env._validate_action([])
    env.close()


@pytest.mark.fast
def test_continuous_action_list_length_mismatch():
    level = _make_simple_level()
    env = InterphyreEnv(level, action_type="continuous")
    with pytest.raises(ValueError, match="Expected 1 action tuples"):
        env._validate_action([])
    env.close()


@pytest.mark.fast
def test_continuous_action_list_requires_numbers():
    level = _make_simple_level()
    env = InterphyreEnv(level, action_type="continuous")
    with pytest.raises(ValueError, match="coordinates must be numbers"):
        env._validate_action([("x", "y", "z")])
    env.close()


@pytest.mark.fast
def test_place_action_objects_length_mismatch():
    level = _make_simple_level()
    env = InterphyreEnv(level)
    with pytest.raises(ValueError, match="Expected 1 positions"):
        env._place_action_objects([])
    env.close()


@pytest.mark.fast
def test_get_observation_rejects_unknown_type():
    level = _make_simple_level()
    env = InterphyreEnv(level)
    env.observation_type = "unknown"
    with pytest.raises(ValueError, match="Unknown observation_type"):
        env._get_observation()
    env.close()


@pytest.mark.fast
def test_get_physics_state_empty_when_world_missing():
    level = _make_simple_level()
    env = InterphyreEnv(level)
    env.engine.world = None
    assert env._get_physics_state() == {}
    env.close()


@pytest.mark.fast
def test_get_image_observation_uses_discrete_colors(monkeypatch):
    level = _make_simple_level()
    env = InterphyreEnv(
        level, observation_type="image", discrete_colors=True, image_size=(8, 6)
    )

    class DummyRenderer:
        def __init__(self, width, height, ppm):
            self.width = width
            self.height = height
            self.ppm = ppm

        def render(self, engine):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        def render_discrete(self, engine):
            return np.ones((self.height, self.width), dtype=np.uint8)

        def close(self):
            return None

    import interphyre.render as render_module

    monkeypatch.setattr(render_module, "OpenCVRenderer", DummyRenderer)
    image = env._get_image_observation()
    assert image.shape == (6, 8)
    assert image.dtype == np.uint8
    env.close()


@pytest.mark.fast
def test_get_info_dict_success_overrides_truncation():
    level = _make_simple_level()
    env = InterphyreEnv(level)
    info = env._get_info_dict(success=True, terminated=True, truncated=True)
    assert info["terminated"] is True
    assert info["truncated"] is False
    env.close()


@pytest.mark.fast
def test_simulate_uses_default_steps():
    level = _make_simple_level()
    config = SimulationConfig(max_steps=1)
    env = InterphyreEnv(level, config=config)
    assert env.simulate() is None
    env.close()


@pytest.mark.fast
def test_simulate_requires_initialized_world():
    level = _make_simple_level()
    env = InterphyreEnv(level)
    env.engine.world = None
    with pytest.raises(ValueError, match="World is not initialized"):
        env.simulate()
    env.close()


# ============================================================================
# Verbose Output and Rendering Hooks (merged from test_environment_misc.py)
# ============================================================================


class DummyRenderer:
    def __init__(self):
        self.render_called = False
        self.close_called = False

    def render(self, engine):
        self.render_called = True

    def close(self):
        self.close_called = True


@pytest.mark.fast
def test_simulate_verbose_output(capsys):
    level = load_level("two_body_problem", seed=42)
    env = InterphyreEnv(level)
    env.simulate(steps=2, verbose=True)
    captured = capsys.readouterr()
    assert "Step 1/2" in captured.out
    env.close()


@pytest.mark.fast
def test_render_and_close_hooks():
    level = load_level("two_body_problem", seed=42)
    env = InterphyreEnv(level)
    dummy = DummyRenderer()
    env.renderer = dummy

    env.render()
    env.close()

    assert dummy.render_called is True
    assert dummy.close_called is True


@pytest.mark.fast
def test_rollback_restores_success_condition():
    """FIX-ROLLBACK-SUCCESS-CONDITION regression test.

    Modifying success_condition inside intervention_context(auto_rollback=True)
    and then raising must leave the original condition intact.
    """
    level = _make_simple_level()
    env = InterphyreEnv(level)
    env.reset(seed=0)

    original_condition = env._level.success_condition

    try:
        with env.intervention_context(auto_rollback=True) as ctx:
            ctx.modify_success_condition(lambda engine: True)
            # Confirm the mutation took effect inside the block.
            assert env._level.success_condition is not original_condition
            raise RuntimeError("deliberate exception to trigger rollback")
    except RuntimeError:
        # InterventionContext suppresses the exception when auto_rollback=True,
        # so we should not reach here. If we do, the context manager is broken.
        pytest.fail("InterventionContext should have suppressed the exception")
    except Exception as exc:
        pytest.fail(f"Unexpected exception: {exc}")

    # After the context exits with auto_rollback, original condition must be restored.
    assert env._level.success_condition is original_condition, (
        "success_condition was not restored after auto_rollback"
    )
    env.close()
