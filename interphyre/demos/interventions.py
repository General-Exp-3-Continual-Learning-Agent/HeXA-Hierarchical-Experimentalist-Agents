#!/usr/bin/env python3
"""
Interventions: Modifying simulations mid-flight.

Interventions let you change the physics state during simulation:
- Add/remove objects
- Apply forces and impulses
- Set velocities and positions
- Freeze objects

Changes can be made directly or via InterventionContext for batching.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interphyre import InterphyreEnv
from interphyre.interventions import at_step
from interphyre.objects import Ball


def demo_add_object():
    """Add a new object during simulation."""
    print("\n1. add_object()")

    env = InterphyreEnv("two_body_problem", seed=42, enable_interventions=True)
    snapshot, _ = env.run_until(at_step(50), action=(0.5, 3.0, 0.5), max_steps=100)
    env.restore(snapshot)

    env.add_object(
        "new_ball",
        Ball(x=2.0, y=3.0, radius=0.3, color="blue", dynamic=True),
    )

    print("   Added 'new_ball' at (2.0, 3.0)")
    env.close()


def demo_add_with_impulse():
    """Add object and immediately apply impulse."""
    print("\n2. add_object() with impulse")

    env = InterphyreEnv("two_body_problem", seed=42, enable_interventions=True)
    snapshot, _ = env.run_until(at_step(50), action=(0.5, 3.0, 0.5), max_steps=100)
    env.restore(snapshot)

    env.add_object(
        "fast_ball",
        Ball(x=-2.0, y=2.0, radius=0.4, color="red", dynamic=True),
        impulse=(5.0, 0.0),
    )

    vel = env.engine.bodies["fast_ball"].linearVelocity
    print(f"   Added with velocity ({vel.x:.2f}, {vel.y:.2f})")
    env.close()


def demo_apply_impulse():
    """Apply impulse to existing object."""
    print("\n3. apply_impulse()")

    env = InterphyreEnv("two_body_problem", seed=42, enable_interventions=True)
    snapshot, _ = env.run_until(at_step(50), action=(0.5, 3.0, 0.5), max_steps=100)
    env.restore(snapshot)

    # Impulse changes velocity immediately (before physics step)
    before = env.engine.bodies["green_ball"].linearVelocity
    print(f"   Before: ({before.x:.2f}, {before.y:.2f})")

    env.apply_impulse("green_ball", impulse=(10.0, 5.0))

    after = env.engine.bodies["green_ball"].linearVelocity
    print(f"   After:  ({after.x:.2f}, {after.y:.2f})")
    env.close()


def demo_set_velocity():
    """Set object velocity directly."""
    print("\n4. set_velocity()")

    env = InterphyreEnv("two_body_problem", seed=42, enable_interventions=True)
    snapshot, _ = env.run_until(at_step(50), action=(0.5, 3.0, 0.5), max_steps=100)
    env.restore(snapshot)

    before = env.engine.bodies["green_ball"].linearVelocity
    print(f"   Before: ({before.x:.2f}, {before.y:.2f})")

    env.set_velocity("green_ball", vx=5.0, vy=-3.0)
    print("   Set to (5.00, -3.00)")
    env.close()


def demo_set_position():
    """Teleport object to new position."""
    print("\n5. set_position()")

    env = InterphyreEnv("two_body_problem", seed=42, enable_interventions=True)
    snapshot, _ = env.run_until(at_step(50), action=(0.5, 3.0, 0.5), max_steps=100)
    env.restore(snapshot)

    before = env.engine.bodies["green_ball"].position
    print(f"   Before: ({before.x:.2f}, {before.y:.2f})")

    env.set_position("green_ball", x=0.0, y=0.0)

    after = env.engine.bodies["green_ball"].position
    print(f"   After:  ({after.x:.2f}, {after.y:.2f})")
    env.close()


def demo_freeze():
    """Stop an object completely."""
    print("\n6. freeze()")

    env = InterphyreEnv("two_body_problem", seed=42, enable_interventions=True)
    snapshot, _ = env.run_until(at_step(50), action=(0.5, 3.0, 0.5), max_steps=100)
    env.restore(snapshot)

    before = env.engine.bodies["green_ball"].linearVelocity
    print(f"   Before: ({before.x:.2f}, {before.y:.2f})")

    env.freeze("green_ball")

    after = env.engine.bodies["green_ball"].linearVelocity
    print(f"   After:  ({after.x:.2f}, {after.y:.2f})")
    env.close()


def demo_intervention_context():
    """Use InterventionContext for multiple changes."""
    print("\n7. intervention_context()")

    env = InterphyreEnv("two_body_problem", seed=42, enable_interventions=True)
    snapshot, _ = env.run_until(at_step(50), action=(0.5, 3.0, 0.5), max_steps=100)
    env.restore(snapshot)

    with env.intervention_context() as ctx:
        ctx.add_object(
            "helper_ball",
            Ball(x=-3.0, y=3.0, radius=0.5, color="blue", dynamic=True),
        )
        ctx.apply_impulse("helper_ball", impulse=(8.0, 0.0))
        ctx.set_velocity("green_ball", vx=0.0, vy=0.0)

    print("   Batched: add helper_ball, impulse it, stop green_ball")
    env.close()


def main():
    print("Interventions Demo")

    demo_add_object()
    demo_add_with_impulse()
    demo_apply_impulse()
    demo_set_velocity()
    demo_set_position()
    demo_freeze()
    demo_intervention_context()


if __name__ == "__main__":
    main()
