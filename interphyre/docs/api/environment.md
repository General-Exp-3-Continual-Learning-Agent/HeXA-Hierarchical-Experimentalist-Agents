# Gymnasium Environment

## InterphyreEnv

`InterphyreEnv` is a Gymnasium-compatible environment for physics-based puzzles. It supports both standard RL usage (one-shot action placement) and intervention-based workflows (multi-turn replanning).

## Construction

### From Level Name (Recommended)

```python
from interphyre import InterphyreEnv

# Basic usage
env = InterphyreEnv("catapult", seed=42)

# With rendering
env = InterphyreEnv("catapult", seed=42, render_mode="human")

# With interventions enabled
env = InterphyreEnv("catapult", seed=42, enable_interventions=True)
```

### From Custom Level

```python
from interphyre import InterphyreEnv
from interphyre.level import Level
from interphyre.objects import Ball, Bar

level = Level(
    name="custom",
    objects={
        "ball": Ball(x=0, y=3, radius=0.5, color="green", dynamic=True),
        "platform": Bar(x=0, y=0, length=4, thickness=0.2, angle=0, dynamic=False),
    },
    action_objects=["ball"],
    success_condition=lambda engine: engine.bodies["ball"].position.y < -2,
)

env = InterphyreEnv(level)
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `level_name` | str | required | Name of level from registry |
| `seed` | int | None | Random seed for level variation |
| `config` | SimulationConfig | None | Physics simulation configuration |
| `render_mode` | str | None | `"human"`, `"rgb_array"`, or `None` |
| `observation_type` | str | `"physics_state"` | `"physics_state"`, `"image"`, or `"both"` |
| `action_type` | str | `"continuous"` | `"continuous"` or `"discrete"` |
| `image_size` | tuple | (600, 600) | Image dimensions for image observations |
| `image_ppm` | float | 60.0 | Pixels per meter for rendering |
| `discrete_colors` | bool | False | Use discrete color channels |
| `enable_interventions` | bool | False | Enable intervention API |

## Standard RL Interface

### reset()

```python
obs, info = env.reset(seed=42)
```

Returns initial observation and info dict containing:
- `level_name`: Name of the level
- `action_objects`: List of controllable object names
- `total_objects`: Number of objects in level
- `success`: Always False initially

### step()

```python
# Single action object (list with one tuple)
obs, reward, terminated, truncated, info = env.step([(0.5, 3.0, 0.6)])

# Multiple action objects
obs, reward, terminated, truncated, info = env.step([(0.5, 3.0, 0.6), (1.0, 2.0, 0.4)])
```

Actions are `(x, y, radius)` tuples specifying where to place the action object(s).

**Important**: This is a one-shot environment. `step()` runs the full simulation to completion. Call `reset()` before calling `step()` again.

### Observation Space

**physics_state** (default):
```python
{
    "objects": {
        "ball_name": {
            "position": np.array([x, y]),
            "velocity": np.array([vx, vy]),
            "angle": float,
            "angular_velocity": float,
            "type": "Ball"
        },
        ...
    },
    "contacts": np.array([[0, 1], [1, 0]]),  # Contact matrix
    "step_count": int
}
```

**image**: RGB array of shape `(height, width, 3)`

**both**: Dictionary with both `physics_state` and `image`

### Action Space

**continuous** (default): `Box([-5, -5, 0.1], [5, 5, 1.5], (3,))`

**discrete**: `MultiDiscrete` with 0.1 resolution bins

## Intervention API

Enable interventions to use multi-turn control:

```python
env = InterphyreEnv("catapult", seed=42, enable_interventions=True)
```

### run_until()

Run simulation until a trigger fires:

```python
from interphyre.interventions import on_contact, at_step

# Run with action until contact
snapshot, step = env.run_until(
    on_contact("green_ball", "blue_ball"),
    action=[(0.5, 3.0, 0.6)],
    max_steps=500
)

if snapshot:
    print(f"Contact at step {step}")
```

### restore()

Return to a captured snapshot:

```python
env.restore(snapshot)
# Simulation is now at the exact state when snapshot was captured
```

### step_until()

Continue simulation (no new action) until trigger:

```python
obs, reward, terminated, truncated, info = env.step_until(
    on_success(),
    max_steps=300
)
```

### Object Management

Add, remove, and modify objects mid-simulation:

```python
from interphyre.objects import Ball

# Add new object
env.add_object("helper", Ball(x=0, y=3, radius=0.5, color="blue", dynamic=True))
env.add_object("fast_ball", Ball(...), impulse=(5.0, 0.0))  # With initial impulse

# Remove object
env.remove_object("helper")

# Modify physics
env.apply_impulse("green_ball", impulse=(10.0, 5.0))
env.apply_force("green_ball", force=(2.0, 0.0))
env.set_velocity("green_ball", vx=5.0, vy=-3.0)
env.set_position("green_ball", x=0.0, y=0.0)
env.freeze("green_ball")  # Zero all velocities
```

### InterventionContext

Batch multiple modifications:

```python
with env.intervention_context() as ctx:
    ctx.add_object("ball", Ball(x=0, y=3, radius=0.5, color="blue", dynamic=True))
    ctx.apply_impulse("ball", impulse=(5.0, 0.0))
    ctx.set_velocity("green_ball", vx=0.0, vy=0.0)
```

With auto-rollback on exception:

```python
with env.intervention_context(auto_rollback=True) as ctx:
    ctx.apply_impulse("green_ball", impulse=(10.0, 0.0))
    # If exception occurs, state is automatically restored
```

## Properties

```python
env.level          # Current Level object
env.objects        # Dict of level objects
env.success        # Current success condition status
env.engine         # Underlying Box2DEngine
```

## Utility Methods

```python
env.get_level_info()         # Level metadata
env.get_contact_log()        # Full contact event history
env.get_contact_statistics() # Contact statistics summary
env.get_performance_stats()  # Profiler statistics
env.reset_profiler()         # Reset profiler
```

## Example: Multi-Turn Replanning

```python
from interphyre import InterphyreEnv
from interphyre.interventions import on_contact, on_success, at_step

env = InterphyreEnv("two_body_problem", seed=42, enable_interventions=True)

# Phase 1: Run until first checkpoint
snapshot, step = env.run_until(at_step(50), action=[(0.5, 3.0, 0.5)])
env.restore(snapshot)

# Phase 2: Observe and decide
pos = env.engine.bodies["green_ball"].position
if pos.y < 0:
    env.apply_impulse("green_ball", impulse=(0, 8))

# Phase 3: Continue to completion
obs, reward, term, trunc, info = env.step_until(on_success(), max_steps=300)
print(f"Success: {info['success']}")

env.close()
```

## See Also

- [Triggers](interventions.md#triggers) - Event conditions for run_until/step_until
- [Objects](objects.md) - Ball, Bar, Basket classes
- [Level](level.md) - Level model for custom levels
- [Examples](../examples/index.md) - Demo code
