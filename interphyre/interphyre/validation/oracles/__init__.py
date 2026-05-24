"""Oracle registry and default oracle for solvability validation.

An oracle is a function that tests whether a level is solvable by attempting
to place action objects such that the success condition is achieved within a
simulation budget. Each level may register a targeted oracle that concentrates
samples in the causally relevant region of the action space. Levels without a
targeted oracle fall back to the default uniform-random oracle.

Oracle function signature:
    oracle(level, config, n_attempts, oracle_steps, rng) -> bool

The oracle returns True as soon as any attempt achieves the success condition;
False if all n_attempts exhaust oracle_steps without success.

Targeted oracle implementations live alongside this file, one module per level
(e.g. oracles/basket_case.py). Each imports @register_oracle from this package
and is imported at the bottom of this file once implemented.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from interphyre.config import SimulationConfig
from interphyre.levels import list_levels
from interphyre.objects import Ball

if TYPE_CHECKING:
    from interphyre.environment import InterphyreEnv
    from interphyre.level import Level

# Registry mapping level name → targeted oracle function.
_oracle_registry: dict[str, Callable] = {}

# Registry mapping level name → targeted solver function.
# A solver has the same signature as an oracle but returns the winning
# action-object positions on success, or None on failure.
_solver_registry: dict[str, Callable] = {}

# Registry for per-level validation defaults.
# Oracle implementations call register_defaults() to advertise the search
# budget they were calibrated for. _bundle.py uses these when no explicit
# CLI override is passed, so test-time callers never need level-specific tuning.
_defaults_registry: dict[str, dict[str, int]] = {}
_ORACLE_DEFAULT_MAX_VARIANTS = 10
_ORACLE_DEFAULT_N_ATTEMPTS = 50

# Placement bounds for the default uniform-random oracle.
# The world spans [-5, 5]² but we stay 0.5 units inside the walls.
_PLACEMENT_MIN = -4.5
_PLACEMENT_MAX = 4.5


def register_oracle(level_name: str) -> Callable:
    """Decorator to register a targeted oracle for a level.

    The decorated function must match the oracle signature:
        fn(level, config, n_attempts, oracle_steps, rng) -> bool
    """

    def decorator(fn: Callable) -> Callable:
        _oracle_registry[level_name] = fn
        return fn

    return decorator


def register_solver(level_name: str) -> Callable:
    """Decorator to register a targeted solver for a level.

    The decorated function must match the solver signature:
        fn(level, config, n_attempts, oracle_steps, rng)
            -> list[tuple[float, float, float]] | None

    Returns the winning action-object positions (one tuple per action object)
    on success, or None if no solution was found within the attempt budget.
    """

    def decorator(fn: Callable) -> Callable:
        _solver_registry[level_name] = fn
        return fn

    return decorator


def get_oracle(level_name: str) -> Callable:
    """Return the registered oracle for level_name, or the default random oracle."""
    return _oracle_registry.get(level_name, _default_oracle)


def get_solver(level_name: str) -> Callable | None:
    """Return the registered solver for level_name, or None if not registered.

    A solver has the same signature as an oracle but returns
    list[tuple[float, float, float]] | None — the winning action-object
    positions, or None if no solution was found.
    """
    return _solver_registry.get(level_name)


def list_oracles() -> dict[str, str]:
    """Return {level_name: 'targeted'|'default'} for all levels in list_levels().

    Useful for auditing which levels have stronger solvability guarantees (via
    targeted oracles sampling the causal region) versus the uniform fallback.
    """
    return {
        level_name: ("targeted" if level_name in _oracle_registry else "default")
        for level_name in list_levels()
    }


def register_defaults(
    level_name: str, *, max_variants: int, n_attempts: int | None = None
) -> None:
    """Register per-level validation defaults for max_variants and n_attempts.

    Called at module level in targeted oracle files. These values reflect the
    search budget the oracle was calibrated for (variant count derived from the
    geometric-decay model; n_attempts from bundle analysis). _bundle.py uses
    them when no explicit CLI override is provided.
    """
    entry: dict[str, int] = {"max_variants": max_variants}
    if n_attempts is not None:
        entry["n_attempts"] = n_attempts
    _defaults_registry[level_name] = entry


def get_default_max_variants(level_name: str) -> int:
    """Return the oracle-recommended max_variants for level_name (default: 10)."""
    return _defaults_registry.get(level_name, {}).get(
        "max_variants", _ORACLE_DEFAULT_MAX_VARIANTS
    )


def get_default_n_attempts(level_name: str) -> int:
    """Return the oracle-recommended n_attempts for level_name (default: 50)."""
    return _defaults_registry.get(level_name, {}).get(
        "n_attempts", _ORACLE_DEFAULT_N_ATTEMPTS
    )


def _run_attempt(
    env: InterphyreEnv,
    positions: list[tuple[float, float, float]],
) -> bool:
    """Run a single oracle attempt via the public InterphyreEnv API.

    Calls env.reset() (which uses engine.reset_attempt() after the first call,
    preserving Box2D warm-start data) then env.step(positions) to run the full
    simulation. Returns True if info["success"] is set.

    Invalid placements are handled internally by env.step() and return False
    via info["success"]. The caller creates one InterphyreEnv before the attempt
    loop; env.reset() handles world reset between attempts without rebuilding
    static bodies.
    """
    env.reset()
    _, _, _, _, info = env.step(positions)
    return info.get("success", False)


def _default_oracle(
    level: Level,
    config: SimulationConfig,
    n_attempts: int,
    oracle_steps: int,
    rng: np.random.Generator,
) -> bool:
    """Uniform-random oracle: sample action object placements from [-4.5, 4.5]².

    For each attempt, samples independent (x, y) coordinates for every action
    object. Ball radii are held fixed at the object's configured value; size is
    ignored for bars and baskets (a placeholder 0.0 is passed as required by
    the place_action_objects API).

    Returns True on the first attempt that achieves the success condition.
    """
    from interphyre.environment import InterphyreEnv  # lazy: avoid circular import

    action_sizes = [
        level.objects[name].radius if isinstance(level.objects[name], Ball) else 0.0
        for name in level.action_objects
    ]
    n_objects = len(level.action_objects)

    env = InterphyreEnv(level, config=config)
    try:
        for _ in range(n_attempts):
            xs = rng.uniform(_PLACEMENT_MIN, _PLACEMENT_MAX, size=n_objects)
            ys = rng.uniform(_PLACEMENT_MIN, _PLACEMENT_MAX, size=n_objects)
            positions = [
                (float(x), float(y), size) for x, y, size in zip(xs, ys, action_sizes)
            ]
            if _run_attempt(env, positions):
                return True
    finally:
        env.close()
    return False


# Targeted oracle implementations — imported here so they register on package load.
from interphyre.validation.oracles import (  # noqa: E402, F401
    basket_case,
    catapult,
    cliffhanger,
    dive_bomb,
    down_to_earth,
    end_of_line,
    falling_into_place,
    flagpole_sitta,
    just_a_nudge,
    keyhole,
    locust_swarm,
    marble_race,
    mind_the_gap,
    off_the_rails,
    pass_the_parcel,
    pinball_machine,
    seesaw,
    staircase,
    straight_face,
    the_cradle,
    the_funnel,
    tipping_point,
    two_body_problem,
    wedge_issue,
    zebra_crossing,
)
