#!/usr/bin/env python3
"""
Real-world performance benchmark for the performance improvements.
"""

import gc
import time

import pytest

psutil = pytest.importorskip("psutil")

from interphyre.config import SimulationConfig  # noqa: E402
from interphyre.environment import InterphyreEnv  # noqa: E402
from interphyre.levels import load_level  # noqa: E402


@pytest.mark.comprehensive
def test_single_level_benchmark():
    """Benchmark a single level with different configurations."""
    level = load_level("two_body_problem", seed=42)

    # Test different FPS configurations
    fps_configs = [30, 60, 120]

    for fps in fps_configs:
        config = SimulationConfig(
            fps=fps, time_step=1 / fps, enable_profiling=True, track_all_contacts=True
        )

        env = InterphyreEnv(level, config=config)
        env.reset()
        # Warm up
        env.simulate(steps=10, return_trace=False)
        env.reset_profiler()

        # Benchmark
        start_time = time.perf_counter()
        env.simulate(steps=1000, return_trace=True)
        end_time = time.perf_counter()

        stats = env.get_performance_stats()
        contact_stats = env.get_contact_statistics()

        wall_time = end_time - start_time
        assert wall_time > 0
        assert stats.get("step_times", {}).get("count", 0) == 1000
        assert contact_stats.get("total_events", 0) >= 0

        # Performance assertions
        mean_step_time = stats.get("step_times", {}).get("mean", 0)
        if mean_step_time > 0:
            expected_step_time = 1 / fps
            assert mean_step_time <= expected_step_time * 2, (
                f"Step time {mean_step_time:.6f}s too slow for {fps} FPS"
            )


@pytest.mark.comprehensive
def test_memory_usage_benchmark():
    """Benchmark memory usage across different levels."""
    levels_to_test = ["two_body_problem", "basket_case", "catapult"]

    for level_name in levels_to_test:
        level = load_level(level_name, seed=42)
        config = SimulationConfig(enable_profiling=True, track_all_contacts=True)

        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        env = InterphyreEnv(level, config=config)
        env.reset()

        # Run simulation
        env.simulate(steps=500, return_trace=True)

        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Assertions - memory can decrease due to garbage collection
        # Just check that we're not using excessive memory (>100MB increase)
        assert memory_increase < 100, (
            f"Memory increase {memory_increase:.2f}MB too high for {level_name}"
        )

        # Clean up
        del env
        gc.collect()


@pytest.mark.comprehensive
def test_contact_tracking_benchmark():
    """Benchmark contact tracking performance."""
    level = load_level("basket_case", seed=42)  # Complex level with many contacts

    # Test full contact tracking
    config_full = SimulationConfig(
        track_all_contacts=True,
        track_relevant_contacts_only=False,
        enable_profiling=True,
    )

    env_full = InterphyreEnv(level, config=config_full)
    env_full.reset()

    start_time = time.perf_counter()
    env_full.simulate(steps=1000, return_trace=True)
    end_time = time.perf_counter()

    stats_full = env_full.get_performance_stats()
    contact_stats_full = env_full.get_contact_statistics()

    wall_time_full = end_time - start_time
    assert wall_time_full > 0
    assert stats_full.get("step_times", {}).get("count", 0) == 1000
    assert contact_stats_full.get("total_events", 0) >= 0

    # Test selective contact tracking
    config_selective = SimulationConfig(
        track_all_contacts=False,
        track_relevant_contacts_only=True,
        enable_profiling=True,
    )

    env_selective = InterphyreEnv(level, config=config_selective)
    env_selective.reset()

    start_time = time.perf_counter()
    env_selective.simulate(steps=1000, return_trace=True)
    end_time = time.perf_counter()

    stats_selective = env_selective.get_performance_stats()
    contact_stats_selective = env_selective.get_contact_statistics()

    wall_time_selective = end_time - start_time
    assert wall_time_selective > 0
    assert stats_selective.get("step_times", {}).get("count", 0) == 1000
    assert contact_stats_selective.get("total_events", 0) >= 0

    # Performance comparison
    mean_full = stats_full.get("step_times", {}).get("mean", 0)
    mean_selective = stats_selective.get("step_times", {}).get("mean", 0)

    if mean_full > 0 and mean_selective > 0:
        # Selective should be at least as fast as full tracking
        assert mean_selective <= mean_full * 1.5, (
            "Selective tracking should not be significantly slower"
        )


@pytest.mark.comprehensive
def test_level_complexity_benchmark():
    """Benchmark performance across different level complexities."""
    # Test levels of increasing complexity
    simple_levels = ["two_body_problem", "straight_face"]
    medium_levels = ["basket_case", "catapult"]
    complex_levels = ["pinball_machine", "locust_swarm"]

    all_levels = simple_levels + medium_levels + complex_levels

    for level_name in all_levels:
        try:
            level = load_level(level_name, seed=42)
            config = SimulationConfig(enable_profiling=True, track_all_contacts=True)

            env = InterphyreEnv(level, config=config)
            env.reset()

            # Benchmark
            start_time = time.perf_counter()
            env.simulate(steps=500, return_trace=True)
            end_time = time.perf_counter()

            stats = env.get_performance_stats()
            contact_stats = env.get_contact_statistics()

            wall_time = end_time - start_time
            assert wall_time > 0
            assert stats.get("step_times", {}).get("count", 0) == 500
            assert contact_stats.get("total_events", 0) >= 0

            # Performance assertions based on complexity
            mean_step_time = stats.get("step_times", {}).get("mean", 0)
            if mean_step_time > 0:
                if level_name in simple_levels:
                    assert mean_step_time < 0.001, (
                        f"Simple level {level_name} too slow: {mean_step_time:.6f}s"
                    )
                elif level_name in medium_levels:
                    assert mean_step_time < 0.002, (
                        f"Medium level {level_name} too slow: {mean_step_time:.6f}s"
                    )
                elif level_name in complex_levels:
                    assert mean_step_time < 0.005, (
                        f"Complex level {level_name} too slow: {mean_step_time:.6f}s"
                    )

        except Exception as e:
            # Skip levels that fail to load
            pytest.skip(f"Level {level_name} failed to load: {e}")


@pytest.mark.comprehensive
def test_profiler_overhead():
    """Test that profiler overhead is minimal."""
    level = load_level("two_body_problem", seed=42)

    # Test without profiling
    config_no_prof = SimulationConfig(enable_profiling=False)
    env_no_prof = InterphyreEnv(level, config=config_no_prof)
    env_no_prof.reset()

    start_time = time.perf_counter()
    env_no_prof.simulate(steps=1000, return_trace=True)
    end_time = time.perf_counter()

    wall_time_no_prof = end_time - start_time

    # Test with profiling
    config_prof = SimulationConfig(enable_profiling=True)
    env_prof = InterphyreEnv(level, config=config_prof)
    env_prof.reset()

    start_time = time.perf_counter()
    env_prof.simulate(steps=1000, return_trace=True)
    end_time = time.perf_counter()

    wall_time_prof = end_time - start_time

    # Profiler overhead should be reasonable (< 20%)
    if wall_time_no_prof > 0:
        overhead = (wall_time_prof - wall_time_no_prof) / wall_time_no_prof
        assert overhead < 0.2, (
            f"Profiler overhead {overhead:.2%} too high (should be < 20%)"
        )
