"""Integration tests for InterphyreEnv validation pipeline.

8 tests matching the spec in plans/validation_module_spec.md §Tests (test_env_validation.py).
"""

from __future__ import annotations

import json
import logging

from interphyre.environment import InterphyreEnv
from interphyre.level import Level
from interphyre.levels import _apply_scene_overrides, _level_registry
from interphyre.objects import Ball
from interphyre.validation.checks import extract_scene_dict
from interphyre.validation.oracles import _oracle_registry
from interphyre.validation.registry import SeedRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trivial_level() -> Level:
    """Level whose success condition is always True — triggers trivial warning."""
    return Level(
        name="trivial_custom",
        objects={"ball": Ball(x=0, y=0, radius=0.5, color="red", dynamic=False)},
        action_objects=["ball"],
        success_condition=lambda e: True,
    )


def _make_valid_custom_level() -> Level:
    """Level whose success condition is always False — not trivial at t=0."""
    return Level(
        name="valid_custom",
        objects={"ball": Ball(x=0, y=2.0, radius=0.5, color="blue", dynamic=False)},
        action_objects=["ball"],
        success_condition=lambda e: False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_env_default_validate_true():
    """InterphyreEnv with default validate=True returns without error; scene_dict is set."""
    env = InterphyreEnv("basket_case", seed=42)
    try:
        assert env.scene_dict is not None
        assert isinstance(env.scene_dict, dict)
    finally:
        env.close()


def test_env_variant_accessible():
    """env.variant is an integer >= 0 for a named level with default validation."""
    env = InterphyreEnv("basket_case", seed=42)
    try:
        assert isinstance(env.variant, int)
        assert env.variant >= 0
    finally:
        env.close()


def test_env_validate_false_no_scene_dict():
    """validate=False disables the validation pipeline and leaves scene_dict as None."""
    env = InterphyreEnv("basket_case", seed=42, validate=False)
    try:
        assert env.scene_dict is None
    finally:
        env.close()


def test_env_trivial_seed_produces_valid_level(tmp_path):
    """When variant=0 is trivially solved, the env constructor silently advances to variant=1.

    We pre-populate an isolated registry so no oracle needs to run in the test,
    keeping the test fast and deterministic.
    """
    from interphyre.levels import load_level

    reg = SeedRegistry(tmp_path / "test.db")

    # Simulate variant=0 being trivial.
    reg.record("basket_case", 9999, 0, "trivial")

    # Provide a valid variant=1 entry with a real scene dict.
    level_v1 = load_level("basket_case", seed=9999, variant=1)
    scene_v1 = extract_scene_dict(level_v1)
    reg.record("basket_case", 9999, 1, "valid", scene_dict=scene_v1)

    env = InterphyreEnv("basket_case", seed=9999, registry=reg)
    try:
        assert env.variant == 1
        assert env.scene_dict is not None
    finally:
        env.close()


def test_env_provenance_loggable():
    """The provenance triple (level_name, seed, variant) and scene_dict are JSON-serializable."""
    env = InterphyreEnv("basket_case", seed=42)
    try:
        log_entry = {
            "level": env._level_name,
            "seed": env._seed,
            "variant": env.variant,
            "scene": env.scene_dict,
        }
        serialized = json.dumps(log_entry)
        parsed = json.loads(serialized)

        assert parsed["level"] == "basket_case"
        assert parsed["seed"] == 42
        assert isinstance(parsed["variant"], int)
        assert isinstance(parsed["scene"], dict)
    finally:
        env.close()


def test_env_custom_level_object_trivial_warns(caplog):
    """Passing a trivial pre-built Level to InterphyreEnv logs a WARNING but does not raise."""
    level = _make_trivial_level()

    with caplog.at_level(logging.WARNING, logger="interphyre.environment"):
        env = InterphyreEnv(level, validate=True)

    try:
        trivial_warnings = [m for m in caplog.messages if "trivial" in m.lower()]
        assert trivial_warnings, "expected a trivial-level WARNING from InterphyreEnv"
    finally:
        env.close()


def test_env_custom_level_object_scene_dict_populated():
    """InterphyreEnv with a pre-built Level and validate=True extracts scene_dict from the level."""
    level = _make_valid_custom_level()
    env = InterphyreEnv(level, validate=True)
    try:
        assert env.scene_dict is not None
        assert isinstance(env.scene_dict, dict)
        assert "ball" in env.scene_dict
    finally:
        env.close()


def test_env_custom_registered_level_full_pipeline(tmp_path):
    """A user-registered level receives the full validation pipeline via InterphyreEnv.

    Registers a minimal level and a trivially-succeeding oracle so the level is
    classified as 'valid'. Verifies that env.variant and env.scene_dict are populated.
    """
    import numpy as np

    level_name = "_test_validation_pipeline"

    def build_test_level(seed=None, variant=0, scene=None):
        rng = np.random.default_rng(seed if variant == 0 else (seed, variant))
        ball = Ball(
            x=float(rng.uniform(-4, 4)),
            y=2.0,
            radius=0.5,
            color="blue",
            dynamic=True,
        )
        action_ball = Ball(x=0.0, y=4.0, radius=0.5, color="red", dynamic=True)
        level = Level(
            name=level_name,
            objects={"ball": ball, "action_ball": action_ball},
            action_objects=["action_ball"],
            # Always False at t=0 (action_ball not yet placed) — not trivial.
            success_condition=lambda e: False,
        )
        _apply_scene_overrides(level.objects, scene)
        return level

    def test_oracle(level, config, n_attempts, oracle_steps, rng):
        # Targeted oracle: always returns True for test purposes.
        return True

    _level_registry[level_name] = build_test_level
    _oracle_registry[level_name] = test_oracle

    reg = SeedRegistry(tmp_path / "test.db")

    try:
        env = InterphyreEnv(level_name, seed=0, registry=reg)
        try:
            assert isinstance(env.variant, int) and env.variant >= 0
            assert env.scene_dict is not None
            assert isinstance(env.scene_dict, dict)
        finally:
            env.close()
    finally:
        _level_registry.pop(level_name, None)
        _oracle_registry.pop(level_name, None)
