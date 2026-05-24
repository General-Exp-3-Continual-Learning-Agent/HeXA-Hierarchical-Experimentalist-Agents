#!/usr/bin/env python3
"""
Comprehensive test suite for performance improvements.
Tests configuration system, performance profiling, and contact tracking.
"""

import time
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from interphyre.config import SimulationConfig, PerformanceProfiler
from interphyre.levels import load_level
from interphyre.environment import InterphyreEnv


def test_configuration_system_defaults():
    config = SimulationConfig()
    assert config.fps == 60
    assert config.time_step == 1 / 60
    assert config.velocity_iters == 15
    assert config.gravity == (0, -9.8)


def test_configuration_system_custom():
    config = SimulationConfig(
        fps=120,
        time_step=1 / 120,
        velocity_iters=8,
        position_iters=4,
        gravity=(0, -20),
        enable_profiling=True,
    )
    assert config.fps == 120
    assert config.time_step == 1 / 120
    assert config.velocity_iters == 8
    assert config.position_iters == 4
    assert config.gravity == (0, -20)
    assert config.enable_profiling is True


def test_configuration_validation():
    with pytest.raises(ValueError):
        SimulationConfig(time_step=-1)
    with pytest.raises(ValueError):
        SimulationConfig(velocity_iters=0)


def test_performance_profiler_disabled():
    profiler = PerformanceProfiler(enabled=False)
    profiler.start_step()

    time.sleep(0.001)
    profiler.end_step()
    stats = profiler.get_stats()
    assert stats == {}


def test_performance_profiler_enabled():
    profiler = PerformanceProfiler(enabled=True)
    profiler.start_step()

    time.sleep(0.001)
    profiler.end_step()
    stats = profiler.get_stats()
    assert "step_times" in stats
    assert stats["step_times"]["count"] == 1
    assert stats["step_times"]["mean"] > 0
    profiler.reset()
    assert profiler.get_stats() == {}


def test_performance_profiler_batch_timing():
    profiler = PerformanceProfiler(enabled=True)
    profiler.start_step_batch()

    time.sleep(0.001)
    profiler.end_step_batch(step_count=5)
    stats = profiler.get_stats()
    assert stats["step_times"]["count"] == 5
    assert stats["step_times"]["mean"] > 0


def test_performance_profiler_wrappers():
    profiler = PerformanceProfiler(enabled=True)

    def noop():
        return "ok"

    wrapped_render = profiler.time_render(noop)
    wrapped_contact = profiler.time_contact_update(noop)

    assert wrapped_render() == "ok"
    assert wrapped_contact() == "ok"
    stats = profiler.get_stats()
    assert stats["render_times"]["count"] == 1
    assert stats["contact_update_times"]["count"] == 1


@pytest.mark.slow
def test_contact_tracking_full_and_relevant():
    config_full = SimulationConfig(
        track_all_contacts=True,
        track_relevant_contacts_only=False,
        enable_profiling=True,
    )
    level = load_level("two_body_problem", seed=42)
    env_full = InterphyreEnv(level, config=config_full)
    obs, info = env_full.reset()
    action = [(0.0, 0.0)]
    obs, reward, done, truncated, info = env_full.step(action)
    env_full.simulate(steps=100, return_trace=True)
    contact_stats = env_full.get_contact_statistics()
    assert contact_stats.get("total_events", 0) > 0
    assert contact_stats.get("unique_pairs", 0) > 0

    config_relevant = SimulationConfig(
        track_all_contacts=False,
        track_relevant_contacts_only=True,
        enable_profiling=True,
    )
    env_relevant = InterphyreEnv(level, config=config_relevant)
    obs, info = env_relevant.reset()
    obs, reward, done, truncated, info = env_relevant.step(action)
    env_relevant.simulate(steps=100, return_trace=True)
    contact_stats_relevant = env_relevant.get_contact_statistics()
    assert contact_stats_relevant.get("total_events", 0) >= 0
    assert contact_stats_relevant.get("unique_pairs", 0) >= 0


@pytest.mark.slow
def test_performance_comparison():

    level = load_level("two_body_problem", seed=42)
    fps_configs = [30, 60, 120]
    results = {}
    for fps in fps_configs:
        config = SimulationConfig(fps=fps, time_step=1 / fps, enable_profiling=True)
        env = InterphyreEnv(level, config=config)
        obs, info = env.reset()
        action = [(0.0, 0.0)]
        obs, reward, terminated, truncated, info = env.step(action)
        start_time = time.perf_counter()
        env.simulate(steps=200, return_trace=True)
        end_time = time.perf_counter()
        stats = env.get_performance_stats()
        results[fps] = {
            "wall_time": end_time - start_time,
            "step_count": stats.get("step_times", {}).get("count", 0),
            "mean_step_time": stats.get("step_times", {}).get("mean", 0),
        }
    for fps in fps_configs:
        assert results[fps]["step_count"] == 200
        assert results[fps]["mean_step_time"] >= 0


@pytest.mark.slow
def test_contact_logging():
    config = SimulationConfig(track_all_contacts=True, enable_profiling=True)
    level = load_level("two_body_problem", seed=42)
    env = InterphyreEnv(level, config=config)
    obs, info = env.reset()
    action = [(0.0, 0.0)]
    obs, reward, done, truncated, info = env.step(action)
    env.simulate(steps=50, return_trace=True)
    contact_log = env.get_contact_log()
    contact_stats = env.get_contact_statistics()
    assert isinstance(contact_log, list)
    assert isinstance(contact_stats, dict)
    assert contact_stats.get("total_events", 0) >= 0
    if contact_log:
        assert "event" in contact_log[0]
        assert "objects" in contact_log[0]


@pytest.mark.slow
def test_multiple_levels():
    levels_to_test = ["two_body_problem", "basket_case", "seesaw"]
    config = SimulationConfig(enable_profiling=True)
    for level_name in levels_to_test:
        level = load_level(level_name, seed=42)
        env = InterphyreEnv(level, config=config)
        obs, info = env.reset()
        if level.action_objects:
            action = [(0.0, 0.0) for _ in level.action_objects]
        else:
            action = []
        obs, reward, terminated, truncated, info = env.step(action)
        env.simulate(steps=100, return_trace=True)
        stats = env.get_performance_stats()
        contact_stats = env.get_contact_statistics()
        assert stats.get("step_times", {}).get("count", 0) == 100
        assert contact_stats.get("total_events", 0) >= 0
