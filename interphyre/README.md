# Interphyre

Physics-based puzzle environment for reinforcement learning and causal inference research. Built on Box2D, compatible with the Gymnasium API.

Each level is a 2D physics puzzle where the agent places one or more balls to trigger a causal chain that satisfies a goal condition (e.g. a ball reaches a target, a basket tips, a gate is cleared). Levels are procedurally generated from seeds: the same seed always produces the same geometry.

## Installation

```bash
git clone https://github.com/sankaranv/interphyre
cd interphyre
pip install -e .
```

Requires Python ≥ 3.10.

## Basic usage

```python
from interphyre import InterphyreEnv

env = InterphyreEnv("catapult", seed=42)
obs, info = env.reset()

# An action is a placement: (x, y, radius)
obs, reward, terminated, truncated, info = env.step([(0.5, 3.0, 0.5)])
print(info["success"])  # True / False
env.close()
```

The simulation runs to completion after `step()`. `reward` is +1 on success, -1 on failure. Actions are passed as a list of `(x, y, radius)` tuples — one per action object.

## Validated seeds

Every level ships with a bundle of certified seeds — (seed, solution) pairs verified to be solvable and non-trivial. Use these for training or evaluation without running the oracle:

```python
from interphyre.validation import load_valid_level, iter_valid_levels

# Load one specific valid seed
validated = load_valid_level("catapult", seed=7)
solution = validated.scene_dict  # full geometry; use validated.level for the Level object

# Iterate all valid seeds for a level
for validated in iter_valid_levels("catapult"):
    level = validated.level
    seed = validated.seed
    variant = validated.variant
```

All 25 levels have 10001 certified seeds. Solutions are stored at 4 decimal places and verified against `env.step()` before storage, ensuring they reproduce across hardware.

## Interventions

The intervention system supports mid-simulation state manipulation for counterfactual and causal analysis:

```python
from interphyre import InterphyreEnv
from interphyre.interventions import on_contact

env = InterphyreEnv("two_body_problem", seed=0, enable_interventions=True)

# Run until the two balls make contact, then capture state
snapshot, step = env.run_until(
    on_contact("green_ball", "blue_ball"),
    action=[(-4.5, 4.5, 0.5)],
    max_steps=500,
)

# Factual branch
env.restore(snapshot)
env.step_physics(200)
factual_success = env.success

# Counterfactual branch — deflect green_ball at the moment of contact
env.restore(snapshot)
with env.intervention_context() as ctx:
    ctx.apply_impulse("green_ball", impulse=(10.0, 5.0))
env.step_physics(200)
counterfactual_success = env.success

env.close()
```

Available interventions: `add_object`, `remove_object`, `apply_impulse`, `set_velocity`, `set_position`, `freeze`.

## Viewer

```bash
# Replay the certified bundle solution for a seed
python -m interphyre.viewer catapult --seed 42

# Replay a specific placement
python -m interphyre.viewer catapult --seed 42 --action 0.5 3.0 0.5

# Replay all solutions an agent wrote to a file
python -m interphyre.viewer --file results.json

# Replay file entries for one level only
python -m interphyre.viewer catapult --file results.json

# Random-placement demo
python -m interphyre.viewer catapult --demo --trials 20
```

## Levels

25 levels across varying difficulty and causal structure:

`basket_case` · `catapult` · `cliffhanger` · `dive_bomb` · `down_to_earth` · `end_of_line` · `falling_into_place` · `flagpole_sitta` · `just_a_nudge` · `keyhole` · `locust_swarm` · `marble_race` · `mind_the_gap` · `off_the_rails` · `pass_the_parcel` · `pinball_machine` · `seesaw` · `staircase` · `straight_face` · `the_cradle` · `the_funnel` · `tipping_point` · `two_body_problem` · `wedge_issue` · `zebra_crossing`

## Citation

```bibtex
@software{interphyre2026,
  title   = {Interphyre: Physics-based Puzzle Environment for RL and Causal Inference},
  author  = {Vaidyanathan, Sankaran},
  year    = {2026},
  url     = {https://github.com/sankaranv/interphyre}
}
```
