"""Interphyre - Physics-based puzzle environment for reinforcement learning.

Example usage:
    from interphyre import InterphyreEnv

    env = InterphyreEnv("catapult", seed=42, render_mode="human")
    obs, info = env.reset()
    obs, reward, term, trunc, info = env.step([(0.5, 3.0, 0.6)])
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("interphyre")
except PackageNotFoundError:
    __version__ = "unknown"

from interphyre.level import Level
from interphyre.config import SimulationConfig
from interphyre.levels import build_level_from_scene, list_levels

# gymnasium is an optional runtime dependency; bundle generation and validation
# tooling does not require it.  Only import InterphyreEnv when available.
try:
    from interphyre.environment import InterphyreEnv, InterventionContext

    __all__ = [
        "InterphyreEnv",
        "InterventionContext",
        "Level",
        "SimulationConfig",
        "build_level_from_scene",
        "list_levels",
        "__version__",
    ]
except ImportError:
    __all__ = [
        "Level",
        "SimulationConfig",
        "build_level_from_scene",
        "list_levels",
        "__version__",
    ]
