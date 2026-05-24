# Custom Levels

Build your own physics puzzles with custom objects and success conditions.

## Overview

This example demonstrates:

- Creating `Level` objects from scratch
- Defining objects (Ball, Bar)
- Setting action objects (user-placeable)
- Writing custom success conditions
- Passing a custom `Level` object directly to `InterphyreEnv`

## Key Concepts

### Level Structure

```python
level = Level(
    name="my_level",
    objects={...},           # Dict of physics objects
    action_objects=[...],    # List of user-controllable object names
    success_condition=fn,    # Function(engine) -> bool
)
```

### Object Types

| Type     | Description                   |
| -------- | ----------------------------- |
| `Ball`   | Circular physics body         |
| `Bar`    | Rectangular physics body      |
| `Basket` | Container that can hold balls |

### Dynamic vs Static

- `dynamic=True` - Object moves under physics (gravity, collisions)
- `dynamic=False` - Object is fixed in place (platforms, walls)

### Success Conditions

A function that takes the physics engine and returns `True` when the goal is met:

```python
def success_condition(engine):
    return engine.is_in_contact_for_duration("green", "blue", duration)
```

## Object Reference

### Ball

```python
from interphyre.objects import Ball

Ball(
    x=0.0,          # X position
    y=2.0,          # Y position
    radius=0.5,     # Ball radius
    color="green",  # Visual color
    dynamic=True    # Moves under physics
)
```

### Bar

```python
from interphyre.objects import Bar

Bar(
    x=0.0,          # Center X position
    y=1.0,          # Center Y position
    length=4.0,     # Length of bar
    thickness=0.2,  # Thickness of bar
    angle=-20,      # Rotation in degrees
    dynamic=False   # Static platform
)
```

## Code Examples

### Simple Contact Level

```python
from interphyre import InterphyreEnv
from interphyre.level import Level
from interphyre.objects import Ball

objects = {
    "green_ball": Ball(x=-3.0, y=2.0, radius=0.5, color="green", dynamic=True),
    "blue_ball": Ball(x=3.0, y=2.0, radius=0.5, color="blue", dynamic=True),
    "red_ball": Ball(x=0.0, y=4.0, radius=0.5, color="red", dynamic=True),
}

def success_condition(engine):
    success_time = engine.config.default_success_time
    return engine.is_in_contact_for_duration("green_ball", "blue_ball", success_time)

level = Level(
    name="simple_contact",
    objects=objects,
    action_objects=["red_ball"],
    success_condition=success_condition,
)

env = InterphyreEnv(level)
env.reset()
obs, reward, term, trunc, info = env.step([(-1.0, 3.0, 0.5)])
print(f"Success: {info['success']}")
env.close()
```

### Ramp Level

```python
from interphyre.objects import Ball, Bar

objects = {
    "ball": Ball(x=-3.0, y=3.0, radius=0.3, color="green", dynamic=True),
    "ramp": Bar(x=-1.5, y=1.5, length=4.0, thickness=0.2, angle=-20, dynamic=False),
    "target": Ball(x=2.0, y=-2.0, radius=0.4, color="blue", dynamic=False),
    "action_ball": Ball(x=0.0, y=4.0, radius=0.5, color="red", dynamic=True),
}

def success_condition(engine):
    return engine.has_contact("ball", "target")

level = Level(
    name="ramp_puzzle",
    objects=objects,
    action_objects=["action_ball"],
    success_condition=success_condition,
)
```

### Complex Success Condition

```python
def success_condition(engine):
    # Both balls must be below y=0 AND touching
    a = engine.bodies.get("ball_a")
    b = engine.bodies.get("ball_b")
    if not a or not b:
        return False
    both_low = a.position.y < 0 and b.position.y < 0
    touching = engine.has_contact("ball_a", "ball_b")
    return both_low and touching
```

## Running the Example

```bash
python demos/custom_levels.py
```

## Expected Output

```
Custom Levels Demo

1. Simple contact level
   Action objects: ['red_ball']
   Success: False

2. Ramp level
   Objects: ball, ramp (static), target (static), action_ball
   Success: False

3. Platform level
   3 platforms, 1 goal, 1 pusher (action)
   Success: False

4. Custom success condition
   Condition: balls below y=0 AND touching
   Success: False
```

## Success Condition Helpers

The engine provides useful methods:

```python
def success_condition(engine):
    # Contact checks
    engine.has_contact("a", "b")
    engine.is_in_contact_for_duration("a", "b", 1.0)

    # Object access
    body = engine.bodies["object_name"]
    pos = body.position  # Vec2(x, y)
    vel = body.linearVelocity  # Vec2(x, y)

    # Configuration
    time = engine.config.default_success_time
```
