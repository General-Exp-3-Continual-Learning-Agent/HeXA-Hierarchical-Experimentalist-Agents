"""Unit tests for the blended graded reward. No LLM / scienceagent needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # HeXA (hexa_discovery, skillrl)

from hexa_discovery.reward import blended_reward, compute_trajectory_reward


def test_perfect_episode_is_max_reward():
    # norm_mse=0, expl=1 -> 0.5*1 + 0.5*1 = 1.0
    assert abs(blended_reward(1.0, 0.0) - 1.0) < 1e-9


def test_pass_line_is_zero_mse_term():
    # norm_mse exactly at threshold (0.1) -> mse_term 0; expl 0 -> reward 0
    assert abs(blended_reward(0.0, 0.1) - 0.0) < 1e-9
    # with a good explanation at the mse pass line -> 0.5*expl
    assert abs(blended_reward(0.8, 0.1) - 0.4) < 1e-9


def test_full_miss_is_floor():
    # huge error clips mse_term to -1; no explanation -> -0.5 floor
    assert abs(blended_reward(0.0, 100.0) - (-0.5)) < 1e-9


def test_reward_monotonic_in_both_terms():
    assert blended_reward(0.5, 0.02) > blended_reward(0.5, 0.08)   # lower err better
    assert blended_reward(0.9, 0.05) > blended_reward(0.2, 0.05)   # better expl better


def test_missing_components_degrade_gracefully():
    assert blended_reward(None, None) == -0.5           # nothing known -> floor
    assert blended_reward(None, 0.0) == 0.5             # perfect fit, no expl
    assert blended_reward(1.0, float("inf")) == 0.0     # perfect expl, no fit


def test_compute_trajectory_reward_prefers_stored_reward():
    traj = {"reward": 0.42, "norm_mse": 0.0, "expl_score": 1.0}
    assert compute_trajectory_reward(traj) == 0.42


def test_compute_trajectory_reward_recomputes_from_fields():
    traj = {"norm_mse": 0.0, "expl_score": 1.0}
    assert abs(compute_trajectory_reward(traj) - 1.0) < 1e-9


def test_compute_trajectory_reward_success_fallback():
    assert compute_trajectory_reward({"success": True}) == 1.0
    assert compute_trajectory_reward({"success": False}) == -0.5


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
