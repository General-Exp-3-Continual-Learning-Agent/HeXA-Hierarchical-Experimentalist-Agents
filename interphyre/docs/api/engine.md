# Physics Engine

## GoalContactListener

`GoalContactListener` tracks contacts, durations, and optional profiling logs. It is attached to the Box2D world by `Box2DEngine`.

Key behaviors:

- Tracks active contact pairs and their durations.
- Supports tracking all contacts or only relevant pairs.
- Records event logs when profiling is enabled.

## Box2DEngine

`Box2DEngine` owns the Box2D world, creates bodies, and advances the simulation.

Construction:

```python
from interphyre.engine import Box2DEngine
engine = Box2DEngine(level)
```

Core methods:

- `reset(level=None)`
- `place_action_objects(positions)`
- `get_state()`
- `objects()`
- `has_contact(name1, name2)`
- `world_is_stationary()`
- `is_in_basket(basket_name, target_name)`
- `is_in_contact_for_duration(a, b, success_time=None)`
- `time_update(dt)`
- `get_contact_duration(a, b)`
- `get_contact_log()`
- `get_contact_statistics()`

Notes:

- Action objects are skipped during initial world creation and must be placed with `place_action_objects`.
- Contacts and durations are tracked via the internal contact listener.
- When `SimulationConfig.enable_interventions` is True, you can attach an `InterventionScheduler` to `engine._intervention_scheduler`.
