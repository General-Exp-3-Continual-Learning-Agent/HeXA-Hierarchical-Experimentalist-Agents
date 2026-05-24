# Interventions

Mid-simulation modifications - add objects, apply forces, and change physics state.

## Overview

This example demonstrates:

- Adding new objects during simulation
- Applying impulses and forces
- Setting velocities and positions directly
- Freezing objects
- Batching multiple changes with `intervention_context()`

## Key Concepts

### Enabling Interventions

Interventions require `enable_interventions=True`:

```python
env = InterphyreEnv("level_name", seed=42, enable_interventions=True)
```

### Intervention Methods

All methods are available directly on `InterphyreEnv`:

| Method                                | Description                   |
| ------------------------------------- | ----------------------------- |
| `add_object(name, obj, impulse=None)` | Add new physics object        |
| `apply_impulse(name, impulse)`        | Instantaneous velocity change |
| `apply_force(name, force)`            | Continuous force              |
| `set_velocity(name, vx, vy)`          | Set exact velocity            |
| `set_position(name, x, y)`            | Teleport object               |
| `freeze(name)`                        | Stop all motion               |

### InterventionContext

Batch multiple changes:

```python
with env.intervention_context() as ctx:
    ctx.add_object("ball1", Ball(...))
    ctx.apply_impulse("ball1", (5.0, 0.0))
    ctx.freeze("other_ball")
```

## Intervention Reference

### add_object(name, object, impulse=None)

Add a new physics object to the simulation.

```python
from interphyre.objects import Ball

env.add_object(
    "new_ball",
    Ball(x=2.0, y=3.0, radius=0.3, color="blue", dynamic=True)
)

# With initial impulse
env.add_object(
    "fast_ball",
    Ball(x=-2.0, y=2.0, radius=0.4, color="red", dynamic=True),
    impulse=(5.0, 0.0)
)
```

### apply_impulse(name, impulse)

Apply an instantaneous velocity change.

```python
env.apply_impulse("green_ball", impulse=(10.0, 5.0))
```

### apply_force(name, force)

Apply a continuous force (per physics step).

```python
env.apply_force("green_ball", force=(2.0, 0.0))
```

### set_velocity(name, vx, vy)

Set exact velocity, replacing current motion.

```python
env.set_velocity("green_ball", vx=5.0, vy=-3.0)
```

### set_position(name, x, y)

Teleport object to new position.

```python
env.set_position("green_ball", x=0.0, y=0.0)
```

### freeze(name)

Stop object completely (zero velocity and angular velocity).

```python
env.freeze("green_ball")
```

### intervention_context()

Group multiple interventions:

```python
with env.intervention_context() as ctx:
    ctx.add_object(
        "helper_ball",
        Ball(x=-3.0, y=3.0, radius=0.5, color="blue", dynamic=True)
    )
    ctx.apply_impulse("helper_ball", impulse=(8.0, 0.0))
    ctx.set_velocity("green_ball", vx=0.0, vy=0.0)
```

## Running the Example

```bash
python demos/interventions.py
```

## Expected Output

```
Interventions Demo

1. add_object()
   Added 'new_ball' at (2.0, 3.0)

2. add_object() with impulse
   Added with velocity (39.79, 0.00)

3. apply_impulse()
   Before: (0.00, -8.17)
   After:  (44.96, 14.31)

4. set_velocity()
   Before: (0.00, -8.17)
   Set to (5.00, -3.00)

5. set_position()
   Before: (-3.31, -1.75)
   After:  (0.00, 0.00)

6. freeze()
   Before: (0.00, -8.17)
   After:  (0.00, 0.00)

7. intervention_context()
   Batched: add helper_ball, impulse it, stop green_ball
```
