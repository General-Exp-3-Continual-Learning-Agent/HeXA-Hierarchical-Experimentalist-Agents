# API Reference

This section documents the public API of Interphyre.

## Core Classes

| Module | Description |
|--------|-------------|
| [InterphyreEnv](environment.md) | Main Gymnasium environment for physics puzzles |
| [Level](level.md) | Level data model for custom puzzles |
| [Objects](objects.md) | Ball, Bar, Basket physics objects |
| [Interventions](interventions.md) | Triggers and mid-simulation modifications |

## Quick Start

```python
from interphyre import InterphyreEnv

# Standard RL usage
env = InterphyreEnv("catapult", seed=42)
obs, info = env.reset()
obs, reward, term, trunc, info = env.step([(0.5, 3.0, 0.6)])

# With interventions
env = InterphyreEnv("catapult", seed=42, enable_interventions=True)
from interphyre.interventions import on_contact
snapshot, step = env.run_until(on_contact("ball", "platform"), action=[(0.5, 3.0, 0.6)])
```

## Module Map

### Environment

- **[InterphyreEnv](environment.md)** - Gymnasium-compatible environment
  - `InterphyreEnv(level_name, seed, enable_interventions)`
  - `InterphyreEnv(level)` - From a custom `Level` object
  - `step()`, `reset()`, `render()`, `close()`
  - `run_until()`, `restore()`, `step_until()` - Intervention methods
  - `add_object()`, `apply_impulse()`, etc. - Object management

### Level Building

- **[Level](level.md)** - Level data model
  - `Level(name, objects, action_objects, success_condition)`
  - `clone()`, `move_object()`, etc.
- **[Objects](objects.md)** - Physics objects
  - `Ball`, `Bar`, `Basket`
  - `create_ball()`, `create_bar()`, `create_basket()`
- **[Level Registry](levels.md)** - Built-in levels
  - `load_level()`, `list_levels()`

### Simulation Control

- **[Interventions](interventions.md)** - Triggers and modifications
  - Triggers: `at_step()`, `on_contact()`, `on_success()`, `when()`, etc.
  - `StateSnapshot` - State capture/restore
  - `InterventionContext` - Scoped modifications

### Configuration

- **[Configuration](config.md)** - Simulation parameters
  - `SimulationConfig` - Physics settings

### Low-Level

- **[Physics Engine](engine.md)** - Box2D wrapper
  - `Box2DEngine` - Direct engine access
- **[Rendering](render.md)** - Visualization
  - `PygameRenderer`, `OpenCVRenderer`

## Import Patterns

```python
# Main entry point
from interphyre import InterphyreEnv

# For custom levels
from interphyre import InterphyreEnv, Level
from interphyre.objects import Ball, Bar, Basket

# For interventions
from interphyre.interventions import (
    at_step, on_contact, on_success, when,
    on_velocity_threshold, on_position_threshold,
    on_sequence, on_any,
)

# For configuration
from interphyre import SimulationConfig
```
