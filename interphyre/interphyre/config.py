import time
import warnings
from dataclasses import dataclass

# Rounding precision used across the simulator to ensure determinism.
# Note: Box2D uses float32 internally, but values are rounded here for
# deterministic input. Box2D handles float64->float32 conversion internally.
PRECISION = 8

# World bounds for level authoring.
MAX_X = 5
MAX_Y = 5
MIN_X = -5
MIN_Y = -5
WORLD_WIDTH = MAX_X - MIN_X
WORLD_HEIGHT = MAX_Y - MIN_Y


@dataclass
class SimulationConfig:
    """Configuration for Box2D simulation parameters.

    This class defines all the configurable parameters for the physics simulation,
    including timing, physics world settings, contact tracking, and performance monitoring.

    Attributes:
        fps (int): Target frames per second for rendering (default: 60)
        time_step (float): Physics time step in seconds (default: 1/60)
        velocity_iters (int): Number of velocity iterations per step (default: 15)
        position_iters (int): Number of position iterations per step (default: 20)
            Higher values improve collision resolution but are slower.
        gravity (Tuple[float, float]): Gravity vector (x, y) (default: (0, -9.8))
        do_sleep (bool): Whether to put bodies to sleep when stationary (default: True)
        continuous_collision_detection (bool): Enable CCD for fast objects (default: True)
        substepping (bool): Enable substepping for improved solver accuracy (default: False)
        continuous_physics (bool): Enable continuous physics for preventing tunneling (default: True)
        warm_starting (bool): Enable warm starting in Box2D solver (default: True)
        track_all_contacts (bool): Track all contact events for research (default: True)
        track_relevant_contacts_only (bool): Only track relevant contacts for performance (default: False)
        enable_profiling (bool): Enable performance profiling (default: False)
        log_step_times (bool): Log timing for each simulation step (default: False)
        stationary_tolerance (float): Tolerance for detecting stationary world (default: 0.0001)
        stationary_check_frames (int): Number of frames for time-based stationary detection (default: 10)
        default_success_time (float): Default time for success detection (default: 3.0)
        max_steps (int): Maximum simulation steps before timeout (default: 1000)
        verify_solutions (bool): Enable double-verification of solutions for data collection (default: False)
        enable_interventions (bool): Gate snapshot allocation for the intervention system (default: False).
            When False, no snapshots are allocated and restore() is unavailable, but trigger evaluation
            always runs regardless of this flag.
        intervention_max_snapshots (int): Maximum number of snapshots to keep (default: 100)
        intervention_auto_cleanup (bool): Automatically cleanup old snapshots (default: True)
    """

    # Time and physics settings
    fps: int = 60
    time_step: float = 1 / 60
    velocity_iters: int = 15  # Higher values improve constraint resolution
    position_iters: int = 20  # Higher values improve stability

    # Physics world settings
    gravity: tuple[float, float] = (0, -9.8)
    do_sleep: bool = True
    continuous_collision_detection: bool = True
    substepping: bool = False
    continuous_physics: bool = True
    warm_starting: bool = True

    # Contact tracking settings
    track_all_contacts: bool = True
    track_relevant_contacts_only: bool = False

    # Performance monitoring
    enable_profiling: bool = False
    log_step_times: bool = False

    # Stationary world detection
    stationary_tolerance: float = 0.0001
    stationary_check_frames: int = (
        10  # Number of frames to check for time-based stationary detection
    )
    default_success_time: float = 3.0

    # Simulation limits
    max_steps: int = 1000

    # Data collection and verification settings
    verify_solutions: bool = (
        False  # Enable double-verification of solutions (slower but safer)
    )

    # Intervention settings (opt-in)
    enable_interventions: bool = False
    intervention_max_snapshots: int = 100
    intervention_auto_cleanup: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.time_step <= 0:
            raise ValueError("time_step must be positive")
        if self.velocity_iters < 1:
            raise ValueError("velocity_iters must be at least 1")
        if self.position_iters < 1:
            raise ValueError("position_iters must be at least 1")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.intervention_max_snapshots < 1:
            raise ValueError("intervention_max_snapshots must be at least 1")
        # Warn when physics step and render rate are decoupled.
        # This is sometimes intentional but is an easy source of confusing results.
        if abs(self.time_step - 1.0 / self.fps) > 1e-9:
            warnings.warn(
                f"time_step ({self.time_step}) does not equal 1/fps (1/{self.fps} = "
                f"{1.0 / self.fps:.6f}). Physics and render rates are decoupled.",
                UserWarning,
                stacklevel=2,
            )


class PerformanceProfiler:
    """Simple performance profiler for timing simulation steps.

    This profiler tracks timing data for different aspects of the simulation,
    including step times, render times, and contact update times. It can be
    enabled/disabled and provides statistical analysis of performance data.

    Attributes:
        enabled (bool): Whether profiling is currently enabled
        step_times (List[float]): List of recorded step times in seconds
        render_times (List[float]): List of recorded render times in seconds
        contact_update_times (List[float]): List of recorded contact update times in seconds
        current_step_start (Optional[float]): Start time of current step being timed
    """

    def __init__(self, enabled: bool = False):
        """Initialize the performance profiler.

        Args:
            enabled: Whether to enable profiling from the start (default: False)
        """
        self.enabled = enabled
        self.step_times = []
        self.render_times = []
        self.contact_update_times = []
        self.current_step_start = None

    def start_step(self):
        """Start timing a simulation step.

        Records the current time to measure the duration of a single simulation step.
        Must be paired with end_step() to record the timing.
        """
        if self.enabled:
            self.current_step_start = time.perf_counter()

    def end_step(self):
        """End timing a simulation step.

        Calculates the duration since start_step() was called and records it.
        Does nothing if profiling is disabled or no step was started.
        """
        if self.enabled and self.current_step_start is not None:
            step_time = time.perf_counter() - self.current_step_start
            self.step_times.append(step_time)
            self.current_step_start = None

    def start_step_batch(self):
        """Start timing a batch of simulation steps (more efficient for long runs)."""
        if self.enabled:
            self.current_step_start = time.perf_counter()

    def end_step_batch(self, step_count: int = 1):
        """End timing a batch of simulation steps."""
        if self.enabled and self.current_step_start is not None:
            total_time = time.perf_counter() - self.current_step_start
            avg_time = total_time / step_count
            self.step_times.extend([avg_time] * step_count)
            self.current_step_start = None

    def time_render(self, func):
        """Decorator to time render calls."""
        if not self.enabled:
            return func

        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            render_time = time.perf_counter() - start
            self.render_times.append(render_time)
            return result

        return wrapper

    def time_contact_update(self, func):
        """Decorator to time contact update calls."""
        if not self.enabled:
            return func

        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            contact_time = time.perf_counter() - start
            self.contact_update_times.append(contact_time)
            return result

        return wrapper

    def get_stats(self):
        """Get performance statistics.

        Returns:
            Dict[str, Dict[str, float]]: Performance statistics including mean, max, min, and count
                for each timing category (step_times, render_times, contact_update_times).
                Returns empty dict if profiling is disabled or no data collected.
        """
        if not self.enabled:
            return {}

        stats = {}
        if self.step_times:
            stats["step_times"] = {
                "mean": sum(self.step_times) / len(self.step_times),
                "max": max(self.step_times),
                "min": min(self.step_times),
                "count": len(self.step_times),
            }
        if self.render_times:
            stats["render_times"] = {
                "mean": sum(self.render_times) / len(self.render_times),
                "max": max(self.render_times),
                "min": min(self.render_times),
                "count": len(self.render_times),
            }
        if self.contact_update_times:
            stats["contact_update_times"] = {
                "mean": sum(self.contact_update_times) / len(self.contact_update_times),
                "max": max(self.contact_update_times),
                "min": min(self.contact_update_times),
                "count": len(self.contact_update_times),
            }
        return stats

    def reset(self):
        """Reset all timing data."""
        self.step_times.clear()
        self.render_times.clear()
        self.contact_update_times.clear()
