#!/usr/bin/env python3
"""
Triggers: Event detection in physics simulations.

Triggers define WHEN things should happen during simulation:
- Time-based: fire at specific step
- Contact-based: fire when objects touch
- Physics-based: fire on velocity/position thresholds
- Custom: fire on any condition you define
- Sequences: fire when events happen in order

This demo shows all trigger types with small examples.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interphyre import InterphyreEnv
from interphyre.interventions import (
    at_step,
    on_contact,
    on_contact_with,
    on_position_threshold,
    on_sequence,
    on_success,
    on_velocity_threshold,
    when,
)


def demo_time_trigger():
    """Time-based trigger: fires at a specific step."""
    print("\n1. at_step(100)")

    env = InterphyreEnv("two_body_problem", seed=42, enable_interventions=True)
    trigger = at_step(100)
    snapshot, step = env.run_until(trigger, action=(0.5, 3.0, 0.5), max_steps=200)

    if snapshot:
        print(f"   Fired at step {step}")
    else:
        print("   Did not fire")

    env.close()


def demo_contact_trigger():
    """Contact-based trigger: fires when two objects touch."""
    print("\n2. on_contact('green_ball', 'blue_ball')")

    env = InterphyreEnv("two_body_problem", seed=0, enable_interventions=True)
    trigger = on_contact("green_ball", "blue_ball")
    snapshot, step = env.run_until(trigger, action=(-4.5, 4.5, 0.5), max_steps=500)

    if snapshot:
        print(f"   Contact at step {step}")
    else:
        print("   No contact")

    env.close()


def demo_contact_with_trigger():
    """Contact-with trigger: fires when object contacts anything."""
    print("\n3. on_contact_with('green_ball')")

    env = InterphyreEnv("two_body_problem", seed=42, enable_interventions=True)
    trigger = on_contact_with("green_ball")
    snapshot, step = env.run_until(trigger, action=(0.5, 3.0, 0.5), max_steps=300)

    if snapshot:
        print(f"   green_ball contacted something at step {step}")
    else:
        print("   No contact")

    env.close()


def demo_success_trigger():
    """Success trigger: fires when level's success condition is met."""
    print("\n4. on_success()")

    env = InterphyreEnv("two_body_problem", seed=42, enable_interventions=True)
    trigger = on_success()
    snapshot, step = env.run_until(trigger, action=(0.76, 4.27, 0.58), max_steps=500)

    if snapshot:
        print(f"   Level solved at step {step}")
    else:
        print("   Not solved within max steps")

    env.close()


def demo_velocity_trigger():
    """Velocity trigger: fires when object exceeds speed threshold."""
    print("\n5. on_velocity_threshold('green_ball', 3.0)")

    env = InterphyreEnv("two_body_problem", seed=42, enable_interventions=True)
    trigger = on_velocity_threshold("green_ball", speed_threshold=3.0, above=True)
    snapshot, step = env.run_until(trigger, action=(0.5, 3.0, 0.5), max_steps=300)

    if snapshot:
        vel = env.engine.bodies["green_ball"].linearVelocity
        speed = (vel.x**2 + vel.y**2) ** 0.5
        print(f"   Exceeded threshold at step {step} (speed={speed:.2f})")
    else:
        print("   Never exceeded threshold")

    env.close()


def demo_position_trigger():
    """Position trigger: fires when object crosses position threshold."""
    print("\n6. on_position_threshold('green_ball', 'y', -2.0, 'below')")

    env = InterphyreEnv("two_body_problem", seed=42, enable_interventions=True)
    trigger = on_position_threshold(
        "green_ball", axis="y", threshold=-2.0, direction="below"
    )
    snapshot, step = env.run_until(trigger, action=(0.5, 3.0, 0.5), max_steps=500)

    if snapshot:
        pos = env.engine.bodies["green_ball"].position
        print(f"   Crossed y=-2.0 at step {step} (y={pos.y:.2f})")
    else:
        print("   Never crossed threshold")

    env.close()


def demo_custom_trigger():
    """Custom trigger: fires on any user-defined condition."""
    print("\n7. when(custom_condition)")

    env = InterphyreEnv("two_body_problem", seed=42, enable_interventions=True)

    def both_balls_low(engine):
        green_y = engine.bodies["green_ball"].position.y
        blue_y = engine.bodies["blue_ball"].position.y
        return green_y < 0 and blue_y < 0

    trigger = when(both_balls_low)
    snapshot, step = env.run_until(trigger, action=(0.5, 3.0, 0.5), max_steps=500)

    if snapshot:
        print(f"   Both balls below y=0 at step {step}")
    else:
        print("   Condition never met")

    env.close()


def demo_sequence_trigger():
    """Sequence trigger: fires when events happen in order."""
    print("\n8. on_sequence([contact1, contact2])")

    env = InterphyreEnv("two_body_problem", seed=0, enable_interventions=True)

    sequence = on_sequence(
        [
            on_contact("red_ball", "green_ball"),
            on_contact("green_ball", "blue_ball"),
        ]
    )

    snapshot, step = env.run_until(sequence, action=(-4.5, 4.5, 0.5), max_steps=500)

    if snapshot:
        print(f"   Sequence completed at step {step} (red->green->blue)")
    else:
        print("   Sequence not completed")

    env.close()


def main():
    print("Triggers Demo")

    demo_time_trigger()
    demo_contact_trigger()
    demo_contact_with_trigger()
    demo_success_trigger()
    demo_velocity_trigger()
    demo_position_trigger()
    demo_custom_trigger()
    demo_sequence_trigger()


if __name__ == "__main__":
    main()
