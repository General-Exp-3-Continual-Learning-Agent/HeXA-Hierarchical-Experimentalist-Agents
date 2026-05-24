"""Test determinism of the physics engine.

This test verifies that two engines with the same configuration and level
produce identical results after the same number of simulation steps.
"""

import numpy as np
import pytest

from interphyre.config import SimulationConfig
from interphyre.engine import Box2DEngine
from interphyre.levels import load_level


def test_determinism(
    level_name: str = "two_body_problem", level_seed: int = 42, num_steps: int = 300
):
    """Test that two engines produce identical results.

    Args:
        level_name: Name of the level to test
        level_seed: Seed for level generation
        num_steps: Number of simulation steps to run
    """
    print(f"Testing determinism for level: {level_name}")
    print(f"  Level seed: {level_seed}")
    print(f"  Steps: {num_steps}")

    config = SimulationConfig()
    level1 = load_level(level_name, seed=level_seed)
    level2 = load_level(level_name, seed=level_seed)
    engine1 = Box2DEngine(level=level1, config=config)
    engine2 = Box2DEngine(level=level2, config=config)

    print("  Running simulation steps...")
    for _ in range(num_steps):
        engine1.world.Step(
            config.time_step, config.velocity_iters, config.position_iters
        )
        engine2.world.Step(
            config.time_step, config.velocity_iters, config.position_iters
        )
        engine1.time_update(config.time_step)
        engine2.time_update(config.time_step)

    print("  Checking body positions...")
    bodies1 = list(engine1.world.bodies)
    bodies2 = list(engine2.world.bodies)

    assert len(bodies1) == len(bodies2), (
        f"Different number of bodies: {len(bodies1)} vs {len(bodies2)}"
    )

    bodies1_sorted = sorted(
        bodies1, key=lambda b: str(b.userData) if b.userData else ""
    )
    bodies2_sorted = sorted(
        bodies2, key=lambda b: str(b.userData) if b.userData else ""
    )

    all_match = True
    for b1, b2 in zip(bodies1_sorted, bodies2_sorted):
        pos1 = (b1.position.x, b1.position.y)
        pos2 = (b2.position.x, b2.position.y)

        if not np.allclose(pos1, pos2, atol=1e-5):
            print(f"    ✗ Mismatch for body {b1.userData}: {pos1} vs {pos2}")
            all_match = False
        else:
            vel1 = (b1.linearVelocity.x, b1.linearVelocity.y)
            vel2 = (b2.linearVelocity.x, b2.linearVelocity.y)
            if not np.allclose(vel1, vel2, atol=1e-5):
                print(
                    f"    ✗ Velocity mismatch for body {b1.userData}: {vel1} vs {vel2}"
                )
                all_match = False

            if not np.allclose(b1.angle, b2.angle, atol=1e-5):
                print(
                    f"    ✗ Angle mismatch for body {b1.userData}: {b1.angle} vs {b2.angle}"
                )
                all_match = False

    if all_match:
        print("  ✓ All body positions, velocities, and angles match!")
    else:
        print("  ✗ Determinism check failed!")
        pytest.fail("Engines produced different results - determinism check failed!")


@pytest.mark.slow
def test_determinism_default():
    """Test determinism with default parameters."""
    test_determinism()


@pytest.mark.slow
def test_determinism_different_levels():
    """Test determinism with different levels and seeds."""
    test_determinism("pass_the_parcel", level_seed=100, num_steps=500)


@pytest.mark.slow
def test_determinism_extended():
    """Test determinism with extended simulation."""
    test_determinism("two_body_problem", level_seed=42, num_steps=1000)


if __name__ == "__main__":
    test_determinism_default()
    print()
    test_determinism_different_levels()
    print()
    test_determinism_extended()
    print("\n✓ All determinism tests PASSED")
