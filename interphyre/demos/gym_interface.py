#!/usr/bin/env python3
"""
Gym Interface: Standard reinforcement learning training loop.

This demo shows how to use interphyre as a Gymnasium environment:
1. Inspecting observation and action spaces
2. Running multiple episodes
3. Random action sampling
4. Tracking episode statistics
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interphyre import InterphyreEnv

# List of levels to try
LEVELS = ["two_body_problem", "catapult", "seesaw"]


def run_episodes(level_name: str, num_episodes: int = 5, seed: int = 42):
    """Run multiple episodes on a level with random actions."""
    env = InterphyreEnv(level_name, seed=seed)

    # Inspect spaces
    print(f"\nLevel: {level_name}")
    print(f"  Observation: {env.observation_space}")
    print(f"  Action: {env.action_space}")

    # Track statistics
    successes = 0
    total_reward = 0.0

    for episode in range(num_episodes):
        obs, info = env.reset()

        # Sample a random action from the action space
        action = env.action_space.sample()

        # Take the action - simulation runs to completion
        obs, reward, terminated, truncated, info = env.step(action)

        # Track results
        success = info["success"]
        successes += int(success)
        total_reward += reward

        print(
            f"  Episode {episode + 1}: "
            f"action=({action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}) "
            f"reward={reward:+.1f} "
            f"{'SUCCESS' if success else 'FAIL'}"
        )

    # Summary
    print(f"\nResults: {successes}/{num_episodes} successful")
    print(f"Average reward: {total_reward / num_episodes:.2f}")

    env.close()
    return successes


def main():
    print("Gym Interface Demo: Random actions on multiple levels")

    total_successes = 0
    for level in LEVELS:
        total_successes += run_episodes(level, num_episodes=5, seed=42)

    print(f"\nTotal: {total_successes} successes (random actions rarely solve puzzles)")


if __name__ == "__main__":
    main()
