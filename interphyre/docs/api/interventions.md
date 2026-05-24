# Interventions

The interventions module provides tools for multi-turn simulation control, including triggers, state snapshots, and mid-simulation modifications.

## Quick Start

```python
from interphyre import InterphyreEnv
from interphyre.interventions import on_contact, at_step, on_success

env = InterphyreEnv("two_body_problem", seed=42, enable_interventions=True)

# Run until contact event
snapshot, step = env.run_until(
    on_contact("green_ball", "blue_ball"),
    action=[(0.5, 3.0, 0.5)],
    max_steps=500
)

if snapshot:
    env.restore(snapshot)
    env.apply_impulse("green_ball", impulse=(5.0, 0.0))
    obs, reward, term, trunc, info = env.step_until(on_success())
```

## Triggers

Triggers define WHEN something should happen during simulation. Use them with `env.run_until()` and `env.step_until()`.

### at_step(n)

Fire at a specific simulation step.

```python
from interphyre.interventions import at_step

snapshot, step = env.run_until(at_step(100), action=action)
# step == 100
```

### on_contact(obj_a, obj_b)

Fire when two specific objects touch.

```python
from interphyre.interventions import on_contact

snapshot, step = env.run_until(
    on_contact("green_ball", "blue_ball"),
    action=action
)
```

### on_contact_with(obj)

Fire when an object touches anything.

```python
from interphyre.interventions import on_contact_with

snapshot, step = env.run_until(on_contact_with("green_ball"), action=action)
```

### on_success()

Fire when the level's success condition is met.

```python
from interphyre.interventions import on_success

snapshot, step = env.run_until(on_success(), action=action, max_steps=500)
if snapshot:
    print(f"Level solved at step {step}!")
```

### on_velocity_threshold(obj, speed, above=True)

Fire when object exceeds (or drops below) a speed threshold.

```python
from interphyre.interventions import on_velocity_threshold

# Fire when green_ball goes faster than 3.0 units/second
trigger = on_velocity_threshold("green_ball", speed_threshold=3.0, above=True)
```

### on_position_threshold(obj, axis, threshold, direction)

Fire when object crosses a position threshold.

```python
from interphyre.interventions import on_position_threshold

# Fire when green_ball's y position goes below -2.0
trigger = on_position_threshold(
    "green_ball",
    axis="y",
    threshold=-2.0,
    direction="below"  # or "above"
)
```

### when(condition)

Fire when a custom condition function returns True.

```python
from interphyre.interventions import when

def both_balls_low(engine):
    green_y = engine.bodies["green_ball"].position.y
    blue_y = engine.bodies["blue_ball"].position.y
    return green_y < 0 and blue_y < 0

snapshot, step = env.run_until(when(both_balls_low), action=action)
```

### on_sequence(triggers)

Fire when multiple triggers fire in order.

```python
from interphyre.interventions import on_sequence, on_contact

# First red hits green, then green hits blue
trigger = on_sequence([
    on_contact("red_ball", "green_ball"),
    on_contact("green_ball", "blue_ball"),
])
```

### on_any(triggers)

Fire when any trigger fires.

```python
from interphyre.interventions import on_any, on_contact, at_step

# Fire on contact OR at step 200, whichever comes first
trigger = on_any([
    on_contact("green_ball", "blue_ball"),
    at_step(200),
])
```

## StateSnapshot

Captures complete simulation state for later restoration.

```python
from interphyre.interventions import StateSnapshot

# Capture via run_until (recommended)
snapshot, step = env.run_until(at_step(50), action=action)

# Restore
env.restore(snapshot)

# Low-level capture (when needed)
snapshot = StateSnapshot.capture(env.engine, metadata={"note": "custom"})
snapshot.restore(env.engine)
```

### Snapshot Contents

- All body positions, velocities, angles
- Contact listener state
- Current simulation time
- Optional metadata

## InterventionContext

Batch multiple modifications with optional auto-rollback. Access via `env.intervention_context()`.

```python
with env.intervention_context() as ctx:
    ctx.add_object("ball", Ball(x=0, y=3, radius=0.5, color="blue", dynamic=True))
    ctx.apply_impulse("ball", impulse=(5.0, 0.0))
    ctx.set_velocity("green_ball", vx=0.0, vy=0.0)
```

### Available Methods

| Method | Description |
|--------|-------------|
| `add_object(name, obj, impulse=None)` | Add physics object |
| `remove_object(name)` | Remove object |
| `apply_impulse(name, impulse, point=None)` | Apply impulse |
| `apply_force(name, force, point=None)` | Apply force |
| `set_velocity(name, vx=None, vy=None)` | Set velocity |
| `set_position(name, x=None, y=None)` | Set position |
| `freeze(name)` | Zero all velocities |
| `modify_success_condition(fn)` | Change success condition |
| `modify_metadata(**kwargs)` | Update level metadata |

### Auto-Rollback

```python
with env.intervention_context(auto_rollback=True) as ctx:
    ctx.apply_impulse("ball", impulse=(10.0, 0.0))
    # If exception occurs here, state is automatically restored
```

## Trigger Classes

For advanced use cases, you can work with trigger classes directly:

| Class | Factory Function | Description |
|-------|-----------------|-------------|
| `TimeBasedTrigger` | `at_step(n)` | Fire at step n |
| `EventBasedTrigger` | `on_contact()`, `on_contact_with()`, `on_success()` | Fire on physics events |
| `ConditionBasedTrigger` | `when(fn)` | Fire when condition is true |
| `SequenceTrigger` | `on_sequence([...])` | Fire when sequence completes |
| `AnyTrigger` | `on_any([...])` | Fire when any sub-trigger fires |

### Trigger Interface

All triggers implement:

```python
class Trigger:
    def should_fire(self, step: int, engine: Box2DEngine) -> bool:
        """Check if trigger should fire at this step."""
        ...

    def reset(self) -> None:
        """Reset trigger state (for reuse)."""
        ...
```

## Import Summary

Most users need only:

```python
from interphyre.interventions import (
    # Triggers
    at_step,
    on_contact,
    on_contact_with,
    on_success,
    on_velocity_threshold,
    on_position_threshold,
    when,
    on_sequence,
    on_any,
)
```

The `InterphyreEnv` class provides `run_until()`, `restore()`, `step_until()`, and `intervention_context()` methods that handle most intervention workflows.

## See Also

- [Environment](environment.md) - InterphyreEnv intervention methods
- [Examples: Triggers](../examples/triggers.md) - Trigger examples
- [Examples: Interventions](../examples/interventions.md) - Modification examples
- [Examples: Replanning](../examples/replanning.md) - Multi-turn workflows
- [Examples: Counterfactuals](../examples/counterfactuals.md) - Causal analysis
