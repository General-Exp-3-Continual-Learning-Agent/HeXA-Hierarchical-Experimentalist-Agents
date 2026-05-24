# Replanning

Multi-turn agent workflow - run until events, capture state, modify, and continue.

## Overview

This example demonstrates:

- `run_until(trigger, action=...)` - Run simulation until event
- `restore(snapshot)` - Return to captured state
- `step_until(trigger)` - Continue simulation until next event

## Key Concepts

### Multi-Turn Control

Unlike single-shot RL where one action runs to completion, replanning allows:

1. Run until interesting event
2. Observe state
3. Decide whether to intervene
4. Continue or restart

### run_until(trigger, action=None, max_steps=240)

Run simulation until trigger fires or max_steps reached.

```python
snapshot, step = env.run_until(
    on_contact("green_ball", "blue_ball"),
    action=[(0.5, 3.0, 0.5)],
    max_steps=500
)
```

Returns:

- `snapshot`: State at trigger, or `None` if max_steps reached
- `step`: Step number when trigger fired

### restore(snapshot)

Return simulation to captured state.

```python
env.restore(snapshot)
```

### step_until(trigger, max_steps=240)

Continue simulation (no new action) until trigger.

```python
obs, reward, term, trunc, info = env.step_until(on_success(), max_steps=300)
```

## Replanning Pattern

```python
env = InterphyreEnv("level", seed=42, enable_interventions=True)

# Phase 1: Run until event
snapshot, step = env.run_until(
    on_contact("green_ball", "platform"),
    action=[(0.5, 3.0, 0.5)]
)
env.restore(snapshot)

# Phase 2: Observe and intervene
with env.intervention_context() as ctx:
    ctx.add_object("helper", Ball(...))
    ctx.apply_impulse("helper", (5.0, 0.0))

# Phase 3: Continue to completion
obs, reward, term, trunc, info = env.step_until(on_success(), max_steps=300)
print(f"Success: {info['success']}")
```

## Running the Example

```bash
python demos/replanning.py
```

## Expected Output

```
Replanning Demo

1. Running with action (-0.25, 2.5, 1.0)
   Waiting for: EventBasedTrigger(type=contact, objects=('green_ball', 'black_platform'), once)
   Trigger fired at step 202

2. Restoring to checkpoint and adding intervention
   Added red_ball_2 with rightward impulse

3. Continuing simulation
   Result: FAILURE
```

## Advanced Patterns

### Multiple Attempts

```python
for attempt in range(10):
    snapshot, step = env.run_until(at_step(50), action=sample_action())
    env.restore(snapshot)

    if promising(env):
        env.step_until(on_success(), max_steps=300)
```

### Branching Exploration

```python
snapshot_early, _ = env.run_until(at_step(30), action=action)

for strategy in [impulse_left, impulse_right, no_action]:
    env.restore(snapshot_early)
    strategy(env)
    _, step = env.step_until(on_success(), max_steps=300)
```
