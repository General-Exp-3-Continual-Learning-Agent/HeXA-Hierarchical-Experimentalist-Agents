# Interphyre Demos

Progressive examples demonstrating interphyre's capabilities.

## Demo Overview

| Demo | Purpose |
|------|---------|
| `quickstart.py` | Simplest usage - create env, step, check result |
| `gym_interface.py` | Standard RL training loop with Gymnasium |
| `triggers.py` | All trigger types (time, contact, velocity, etc.) |
| `interventions.py` | Mid-simulation modifications (add objects, apply forces) |
| `replanning.py` | Multi-turn workflow: run_until, restore, continue |
| `counterfactuals.py` | Causal analysis with branching simulations |
| `custom_levels.py` | Building custom levels from scratch |

## Quick Start

```bash
# Simplest example
python demos/quickstart.py

# Run all demos
for f in demos/*.py; do python $f; done
```

## Detailed Descriptions

### quickstart.py
The absolute minimum code to run a simulation. Shows how to:
- Create an environment with `InterphyreEnv("level_name", seed=42)`
- Reset and step
- Check success/failure

### gym_interface.py
Standard RL training loop pattern. Shows how to:
- Inspect observation and action spaces
- Sample random actions
- Run multiple episodes
- Track statistics

### triggers.py
Event detection in physics simulations. Demonstrates:
- `at_step(n)` - time-based
- `on_contact("a", "b")` - specific pair contact
- `on_contact_with("a")` - any contact with object
- `on_success()` - level success condition
- `on_velocity_threshold()` - physics-based
- `on_position_threshold()` - position-based
- `when(lambda)` - custom condition
- `on_sequence([...])` - sequential patterns

### interventions.py
Mid-simulation object modifications. Shows how to:
- `add_object()` - add new objects
- `apply_impulse()` / `apply_force()` - apply physics
- `set_velocity()` / `set_position()` - direct state changes
- `freeze()` - stop objects
- `intervention_context()` - batch multiple changes

### replanning.py
The core multi-turn agent workflow. Demonstrates:
- `run_until(trigger, action=...)` - run until event
- `restore(snapshot)` - return to checkpoint
- `step_until(trigger)` - continue simulation
- The observe-decide-act loop

### counterfactuals.py
Causal inference with branching. Shows how to:
- Capture state at a branch point
- Run factual branch (no intervention)
- Run counterfactual branch (with intervention)
- Compare outcomes and measure causal effect

### custom_levels.py
Building custom physics puzzles. Demonstrates:
- Creating `Level` objects with objects, action_objects, success_condition
- Using `Ball`, `Bar`, and other object types
- Writing custom success conditions
- Using `InterphyreEnv.from_level(level)` to run custom levels

## Visualization

For interactive visualization, use the viewer module:

```bash
# View a specific level and action
python -m interphyre.viewer catapult --seed 42 --action 0.5 3.0 0.6

# Run random demo
python -m interphyre.viewer --demo catapult --trials 10
```

See the [Tools documentation](../docs/tools.md) for complete CLI reference.
