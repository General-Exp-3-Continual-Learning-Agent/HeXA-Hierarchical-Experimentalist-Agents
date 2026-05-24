#!/usr/bin/env python3
"""
Quickstart: Run a physics puzzle in under 20 lines.

This is the simplest possible interphyre example. It demonstrates:
1. Creating an environment
2. Resetting to get initial observation
3. Taking an action (placing a ball)
4. Checking if the puzzle was solved
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interphyre import InterphyreEnv

# Create environment for "two_body_problem" level with seed 42
env = InterphyreEnv("two_body_problem", seed=42)

# Reset to get initial observation
obs, info = env.reset()

# Take an action: place a ball at (x, y, radius)
# The simulation runs to completion after placing the ball
# Note: Not all actions solve the puzzle - finding solutions is the challenge!
action = (0.5, 3.0, 0.6)
obs, reward, terminated, truncated, info = env.step(action)

# Check result
# reward: +1 for success, -1 for failure
print("Level: two_body_problem")
print(f"Action: {action}")
print(f"Success: {info['success']}")
print(f"Reward: {reward}")

# Clean up
env.close()
