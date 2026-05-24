#!/usr/bin/env python3
"""
Replanning: Multi-turn simulation with checkpoints.

This demo shows the core intervention workflow:
1. Run simulation until an event (trigger)
2. Capture a checkpoint (snapshot)
3. Make modifications (add objects, apply forces)
4. Continue from the checkpoint

This enables agents to observe, decide, and act multiple times.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interphyre import InterphyreEnv
from interphyre.interventions import on_contact, on_success
from interphyre.objects import Ball


def main():
    print("Replanning Demo")

    env = InterphyreEnv("catapult", seed=0, enable_interventions=True)
    action = (-0.25, 2.5, 1.0)
    trigger = on_contact("green_ball", "black_platform")

    # Step 1: Run until trigger
    print(f"\n1. Running with action {action}")
    print(f"   Waiting for: {trigger}")

    snapshot, step = env.run_until(trigger, action=action, max_steps=500)

    if not snapshot:
        print("   Trigger did not fire")
        env.close()
        return

    print(f"   Trigger fired at step {step}")

    # Step 2: Restore and modify
    print("\n2. Restoring to checkpoint and adding intervention")
    env.restore(snapshot)

    with env.intervention_context() as ctx:
        ctx.add_object(
            "red_ball_2",
            Ball(x=-2.0, y=-3.0, radius=0.4, color="red", dynamic=True),
        )
        ctx.apply_impulse("red_ball_2", impulse=(5.0, 0.0))

    print("   Added red_ball_2 with rightward impulse")

    # Step 3: Continue
    print("\n3. Continuing simulation")
    remaining = max(500 - step, 0)
    if remaining > 0:
        obs, reward, term, trunc, info = env.step_until(
            on_success(), max_steps=remaining
        )
        success = info["success"]
    else:
        success = env.success

    print(f"   Result: {'SUCCESS' if success else 'FAILURE'}")

    env.close()


if __name__ == "__main__":
    main()
