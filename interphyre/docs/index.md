# Interphyre

Interphyre is a 2D physics puzzle simulator based on PHYRE, designed for research in physics reasoning and causal inference.

## Features

- **25 Physics Puzzles**: Hand-crafted levels with configurable parameters and procedurally generated layouts
- **Intervention System**: Apply counterfactual changes during simulation
- **Gymnasium Integration**: Standard RL environment interface
- **Multiple Rendering Backends**: Pygame (interactive) and OpenCV (headless)
- **State Snapshots**: Capture and restore complete simulation state
- **Trigger System**: Define custom success conditions and events

## Quick Start

```python
from interphyre import InterphyreEnv

# Create environment
env = InterphyreEnv(level_name="catapult")

# Run simulation
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step([0.5, 3.0, 0.6])

# Visualize
python -m interphyre.viewer catapult --seed 42 --action 0.5 3.0 0.6
```

## Documentation Structure

- **[Getting Started](getting-started.md)**: Installation and setup instructions
- **[Levels](levels.md)**: Browse the catalog of 25 physics puzzles with previews
- **[Examples](examples/index.md)**: Tutorials for interventions, replanning, counterfactuals
- **[API Reference](api/index.md)**: Complete documentation of all modules
- **[Tools](tools.md)**: CLI utilities for data collection and benchmarking
