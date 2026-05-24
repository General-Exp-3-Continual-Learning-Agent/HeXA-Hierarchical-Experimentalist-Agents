# Quickstart

The simplest possible Interphyre example - create an environment, run a simulation, and check the result.

## Overview

This example demonstrates:

- Creating a `InterphyreEnv` with a level name and seed
- Resetting the environment
- Taking an action (placing an object)
- Checking if the level was solved

## Key Concepts

### InterphyreEnv

The main entry point for Interphyre. Wraps the physics simulation in a Gymnasium-compatible interface.

```python
env = InterphyreEnv("level_name", seed=42)
```

### Actions

Actions are lists of `(x, y, radius)` tuples — one per action object. Most levels have one action object, so the typical call is `env.step([(x, y, radius)])`. Specify where to place the red action ball:

- `x`: Horizontal position (-5.0 to 5.0)
- `y`: Vertical position (-5.0 to 5.0)
- `radius`: Ball size (0.1 to 1.5)

### Success

Check `info['success']` after stepping to see if the level's goal condition was met.

## Code Example

```python
from interphyre import InterphyreEnv

# Create environment for a specific level
env = InterphyreEnv("two_body_problem", seed=42)

# Reset to initial state
obs, info = env.reset()

# Take an action: place red ball at (x=0.5, y=3.0) with radius=0.6
action = [(0.5, 3.0, 0.6)]
obs, reward, terminated, truncated, info = env.step(action)

# Check result
print(f"Success: {info['success']}")
print(f"Reward: {reward}")

env.close()
```

## Running the Example

```bash
python demos/quickstart.py
```

## Expected Output

```
Level: two_body_problem
Action: (0.5, 3.0, 0.6)
Success: False
Reward: -1.0
```

## Next Steps

- Try different actions to solve the level
- Move on to [Gym Interface](gym_interface.md) for RL training patterns
