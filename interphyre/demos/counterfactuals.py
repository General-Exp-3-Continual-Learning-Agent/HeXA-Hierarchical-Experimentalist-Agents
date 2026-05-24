#!/usr/bin/env python3
"""
Counterfactuals: Compare "what happened" vs "what could have happened".

This demo shows causal analysis through branching:
1. Run simulation to a branch point
2. Capture state
3. Run factual branch (no intervention)
4. Restore and run counterfactual branch (with intervention)
5. Compare outcomes to measure causal effect
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interphyre import InterphyreEnv
from interphyre.interventions import on_contact


def main():
    print("Counterfactual Analysis Demo")

    env = InterphyreEnv("two_body_problem", seed=0, enable_interventions=True)
    trigger = on_contact("green_ball", "blue_ball")

    # Run to branch point
    print("\n1. Running until contact")
    snapshot, step = env.run_until(trigger, action=(-4.5, 4.5, 0.5), max_steps=500)

    if not snapshot:
        print("   No contact occurred")
        env.close()
        return

    print(f"   Contact at step {step}")

    # Factual branch: no intervention
    print("\n2. Factual branch (no intervention)")
    env.restore(snapshot)
    env.step_physics(200)

    factual_pos = env.get_object_position("green_ball")
    factual_success = env.success
    print(f"   green_ball final pos: ({factual_pos[0]:.2f}, {factual_pos[1]:.2f})")
    print(f"   Success: {factual_success}")

    # Counterfactual branch: apply impulse
    print("\n3. Counterfactual branch (impulse intervention)")
    env.restore(snapshot)

    with env.intervention_context() as ctx:
        ctx.apply_impulse("green_ball", impulse=(10.0, 5.0))

    env.step_physics(200)

    cf_pos = env.get_object_position("green_ball")
    cf_success = env.success
    print(f"   green_ball final pos: ({cf_pos[0]:.2f}, {cf_pos[1]:.2f})")
    print(f"   Success: {cf_success}")

    # Compare
    print("\n4. Comparison")
    print(f"   Factual: {'SUCCESS' if factual_success else 'FAILURE'}")
    print(f"   Counterfactual: {'SUCCESS' if cf_success else 'FAILURE'}")

    dx = cf_pos[0] - factual_pos[0]
    dy = cf_pos[1] - factual_pos[1]
    divergence = (dx**2 + dy**2) ** 0.5
    print(f"   Position divergence: {divergence:.2f} units")

    env.close()


if __name__ == "__main__":
    main()
