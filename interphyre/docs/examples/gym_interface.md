# Gym Interface

Standard Gymnasium (OpenAI Gym) training loop patterns for reinforcement learning research.

## Overview

This example demonstrates:

- Inspecting observation and action spaces
- Sampling random actions
- Running multiple episodes
- Tracking episode statistics

## Key Concepts

### Observation Space

A `Dict` space containing object states and contact information:

```python
print(env.observation_space)
```

### Action Space

A `Box` space with shape `(3,)` for `(x, y, radius)`:

```python
env.action_space.sample()  # Returns random (x, y, radius)
```

### Episode Structure

Each episode:

1. `reset()` - Initialize environment
2. `step(action)` - Run full simulation with placed object
3. Check `terminated` and `info['success']`

Note: Unlike most RL environments, `step()` runs the entire physics simulation. Each episode is a single decision.

## Code Example

```python
from interphyre import InterphyreEnv

env = InterphyreEnv("two_body_problem", seed=42)

# Inspect spaces
print(f"Observation: {env.observation_space}")
print(f"Action: {env.action_space}")

# Run episodes
for episode in range(5):
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"Episode {episode + 1}: "
          f"action=({action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}) "
          f"reward={reward:+.1f} "
          f"{'SUCCESS' if info['success'] else 'FAIL'}")

env.close()
```

## Running the Example

```bash
python demos/gym_interface.py
```

## Expected Output

```
Gym Interface Demo: Random actions on multiple levels

Level: two_body_problem
  Observation: Dict(...)
  Action: Box([-5. -5. 0.1], [5. 5. 1.5], (3,), float32)
  Episode 1: action=(3.86, 2.49, 0.59) reward=-0.1 FAIL
  Episode 2: action=(1.17, -1.52, 0.63) reward=-1.0 FAIL
  ...

Results: 0/5 successful
Average reward: -0.46

Total: 0 successes (random actions rarely solve puzzles)
```

## Training Tips

### Action Sampling Strategies

```python
# Uniform random (default)
action = env.action_space.sample()

# Biased toward upper region (objects fall down)
import numpy as np
action = (
    np.random.uniform(-3, 3),      # x: center-ish
    np.random.uniform(2, 4.5),     # y: upper region
    np.random.uniform(0.3, 0.7)    # radius: medium
)
```

### Multiple Levels

```python
from interphyre.levels import list_levels

for level_name in list_levels():
    env = InterphyreEnv(level_name, seed=42)
    # Train on each level...
```

### Deterministic Replay

```python
# Same seed = same initial state
env.reset(seed=42)
env.step([(0.5, 3.0, 0.6)])  # Always same result for same level
```
