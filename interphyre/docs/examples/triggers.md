# Triggers

Event detection in physics simulations - pause execution when specific conditions are met.

## Overview

Trigger types:

- **Time-based:** `at_step(n)` - fire at specific step
- **Contact-based:** `on_contact()`, `on_contact_with()` - object collisions
- **Success-based:** `on_success()` - level goal achieved
- **Physics-based:** `on_velocity_threshold()`, `on_position_threshold()` - state thresholds
- **Custom:** `when(fn)` - arbitrary conditions
- **Sequential:** `on_sequence([...])` - events in order

## Key Concepts

### Triggers and run_until()

Triggers define WHEN something should happen. Use with `run_until()`:

```python
snapshot, step = env.run_until(trigger, action=(...), max_steps=500)
if snapshot:
    print(f"Trigger fired at step {step}")
```

### Snapshot

When a trigger fires, `run_until()` returns a `StateSnapshot` capturing the complete simulation state at that moment. Use `env.restore(snapshot)` to return to that point.

## Trigger Reference

### at_step(n)

Fire at a specific simulation step.

```python
from interphyre.interventions import at_step

trigger = at_step(100)
snapshot, step = env.run_until(trigger, action=[(0.5, 3.0, 0.5)], max_steps=200)
# step == 100
```

### on_contact(obj_a, obj_b)

Fire when two specific objects touch.

```python
from interphyre.interventions import on_contact

trigger = on_contact("green_ball", "blue_ball")
snapshot, step = env.run_until(trigger, action=[(-4.5, 4.5, 0.5)], max_steps=500)
```

### on_contact_with(obj)

Fire when an object touches anything.

```python
from interphyre.interventions import on_contact_with

trigger = on_contact_with("green_ball")
snapshot, step = env.run_until(trigger, action=[(0.5, 3.0, 0.5)], max_steps=300)
```

### on_success()

Fire when the level's success condition is met.

```python
from interphyre.interventions import on_success

trigger = on_success()
snapshot, step = env.run_until(trigger, action=[(0.76, 4.27, 0.58)], max_steps=500)
if snapshot:
    print(f"Level solved at step {step}!")
```

### on_velocity_threshold(obj, speed, above=True)

Fire when object exceeds (or drops below) a speed threshold.

```python
from interphyre.interventions import on_velocity_threshold

trigger = on_velocity_threshold("green_ball", speed_threshold=3.0, above=True)
```

### on_position_threshold(obj, axis, threshold, direction)

Fire when object crosses a position threshold.

```python
from interphyre.interventions import on_position_threshold

trigger = on_position_threshold(
    "green_ball",
    axis="y",
    threshold=-2.0,
    direction="below"
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

trigger = when(both_balls_low)
```

### on_sequence([triggers])

Fire when multiple triggers fire in order.

```python
from interphyre.interventions import on_sequence, on_contact

sequence = on_sequence([
    on_contact("red_ball", "green_ball"),
    on_contact("green_ball", "blue_ball"),
])
snapshot, step = env.run_until(sequence, action=[(-4.5, 4.5, 0.5)], max_steps=500)
```

## Running the Example

```bash
python demos/triggers.py
```

## Expected Output

```
Triggers Demo

1. at_step(100)
   Fired at step 100

2. on_contact('green_ball', 'blue_ball')
   Contact at step 343

3. on_contact_with('green_ball')
   green_ball contacted something at step 39

4. on_success()
   Level solved at step 429

5. on_velocity_threshold('green_ball', 3.0)
   Exceeded threshold at step 18 (speed=3.00)

6. on_position_threshold('green_ball', 'y', -2.0, 'below')
   Crossed y=-2.0 at step 1 (y=-2.30)

7. when(custom_condition)
   Both balls below y=0 at step 57

8. on_sequence([contact1, contact2])
   Sequence completed at step 343 (red->green->blue)
```
