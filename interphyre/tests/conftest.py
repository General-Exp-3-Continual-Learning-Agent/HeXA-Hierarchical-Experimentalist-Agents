"""
Shared pytest fixtures for interphyre tests.
"""

import pytest

from interphyre import InterphyreEnv, SimulationConfig


@pytest.fixture
def default_config():
    """Default simulation configuration for tests."""
    return SimulationConfig(
        fps=60,
        time_step=1 / 60,
        enable_profiling=False,
    )


@pytest.fixture
def intervention_config():
    """Simulation config with interventions enabled."""
    return SimulationConfig(
        fps=60,
        time_step=1 / 60,
        enable_interventions=True,
        enable_profiling=False,
    )


@pytest.fixture
def simple_env(default_config):
    """Pre-initialized environment for the two_body_problem level."""
    env = InterphyreEnv("two_body_problem", seed=42, config=default_config)
    yield env
    env.close()


@pytest.fixture
def intervention_env(intervention_config):
    """Environment with interventions enabled."""
    env = InterphyreEnv("two_body_problem", seed=42, config=intervention_config)
    yield env
    env.close()
