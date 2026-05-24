# Examples

Progressive examples demonstrating Interphyre's capabilities for physics simulation, reinforcement learning, and causal inference research.

## Quick Start

```bash
# Run the simplest example
python demos/quickstart.py

# Run all demos
for f in demos/*.py; do python $f; done
```

## Available Examples

- **[Quickstart](quickstart.md)** - Simplest usage: create env, step, check result
- **[Gym Interface](gym_interface.md)** - Standard RL training loop with Gymnasium
- **[Triggers](triggers.md)** - Event detection (time, contact, velocity, custom conditions)
- **[Interventions](interventions.md)** - Mid-simulation modifications (add objects, apply forces)
- **[Replanning](replanning.md)** - Multi-turn workflow: run_until, restore, continue
- **[Counterfactuals](counterfactuals.md)** - Causal analysis with branching simulations
- **[Custom Levels](custom_levels.md)** - Building custom physics puzzles from scratch

## Visualization

For interactive visualization and debugging, use the viewer module:

```bash
# Visualize a specific action
python -m interphyre.viewer catapult --seed 42 --action 0.5 3.0 0.6

# Run random demo
python -m interphyre.viewer --demo catapult --trials 10
```

See [Tools](../tools.md) for complete CLI documentation.

## API Reference

For detailed API documentation, see:

- **[InterphyreEnv](../api/environment.md)** - Main environment class
- **[Interventions](../api/interventions.md)** - Triggers and modifications
- **[Objects](../api/objects.md)** - Ball, Bar, Basket classes
- **[Level](../api/level.md)** - Level model and creation
