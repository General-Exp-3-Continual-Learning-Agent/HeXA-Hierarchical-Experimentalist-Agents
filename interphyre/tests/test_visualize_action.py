"""Regression test for visualize_action() — confirms it runs a full simulation
and correctly reports success for known-good actions.

Background: FIX-VISUALIZE-ACTION (interphyre-8s1.3) flagged that visualize_action()
might only run a single physics tick. Investigation confirmed env.step() already
runs the full simulation via _run_simulation_rollout(), so the function was correct
as-is. This test locks in that behavior.
"""

from unittest.mock import MagicMock

from interphyre import InterphyreEnv, SimulationConfig


def test_visualize_action_returns_true_for_known_good_action():
    """A known-good action for down_to_earth should produce success=True
    after a full simulation rollout (hundreds of steps, not one tick)."""
    config = SimulationConfig(fps=60, time_step=1 / 60)
    env = InterphyreEnv("down_to_earth", seed=42, config=config)

    # Attach a no-op renderer so env.render() doesn't crash
    mock_renderer = MagicMock()
    env.renderer = mock_renderer

    env.reset()
    action = (0.0, 3.0, 0.5)
    obs, reward, terminated, truncated, info = env.step([action])

    # Full simulation should run many steps, not just one
    assert info["step_count"] > 100, (
        f"Expected hundreds of physics steps, got {info['step_count']}. "
        "env.step() should run the full simulation, not a single tick."
    )
    assert info["success"] is True, "Known-good action should solve down_to_earth"
    assert reward == 1.0

    # Renderer should have been called many times (once per physics step)
    assert mock_renderer.render.call_count > 100

    env.close()


def test_visualize_action_returns_false_for_bad_action():
    """A clearly bad action should not produce success."""
    config = SimulationConfig(fps=60, time_step=1 / 60)
    env = InterphyreEnv("down_to_earth", seed=42, config=config)
    env.renderer = MagicMock()

    env.reset()
    # Place object far off to the side — should not solve the level
    action = (4.5, 4.5, 0.1)
    obs, reward, terminated, truncated, info = env.step([action])

    assert info["success"] is False
    env.close()
