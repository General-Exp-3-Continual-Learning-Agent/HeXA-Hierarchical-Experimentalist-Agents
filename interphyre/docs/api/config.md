# Configuration

## SimulationConfig

`SimulationConfig` controls physics, contact tracking, and intervention settings. It is used by `Box2DEngine` and `InterphyreEnv`.

Key fields:

- `fps`, `time_step`, `velocity_iters`, `position_iters`
- `gravity`, `do_sleep`, `continuous_collision_detection`, `substepping`, `continuous_physics`, `warm_starting`
- `track_all_contacts`, `track_relevant_contacts_only`
- `stationary_tolerance`, `stationary_check_frames`, `default_success_time`, `max_steps`
- `enable_interventions`, `intervention_max_snapshots`, `intervention_auto_cleanup`
- `enable_profiling`, `log_step_times`

Construction:

```python
from interphyre.config import SimulationConfig
config = SimulationConfig(time_step=1/60, enable_interventions=True)
```

## PerformanceProfiler

`PerformanceProfiler` tracks step timing, render timing, and contact update timing when profiling is enabled.

Common usage:

```python
from interphyre.config import PerformanceProfiler
profiler = PerformanceProfiler(enabled=True)
```

The engine enables profiling automatically when `SimulationConfig.enable_profiling` is set.

## Constants

- `PRECISION`: rounding precision used across the simulator.
- `CONTACT_DISTANCE_TOLERANCE`: tolerance used when validating contacts.
