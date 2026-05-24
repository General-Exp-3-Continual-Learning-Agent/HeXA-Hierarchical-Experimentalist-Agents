"""Public API for the interphyre.validation module.

All functions that return levels return ValidatedLevel. The default global
registry is lazily initialized on first use — most callers need not instantiate
SeedRegistry directly.

Typical usage:

    from interphyre.validation import load_valid_level, iter_valid_levels

    # Load one valid level (bundled data: zero overhead for seeds 0–999)
    validated = load_valid_level("basket_case", seed=42)

    # Iterate over valid levels for an experiment loop
    for validated in iter_valid_levels("basket_case", start_seed=0):
        env = InterphyreEnv(validated.level)
        ...
        if enough_levels:
            break

    # Offline pre-warming before an experiment run
    counts = prewarm(["basket_case", "tipping_point"], range(5000), workers=8)
"""

from __future__ import annotations

import logging
import sys
import warnings
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import NamedTuple

import numpy as np

from interphyre.config import SimulationConfig
from interphyre.levels import load_level
from interphyre.validation.checks import extract_scene_dict, is_trivial
from interphyre.validation.oracles import (
    get_default_max_variants,
    get_default_n_attempts,
    get_oracle,
)
from interphyre.validation.registry import SeedRegistry, ValidatedLevel

__all__ = [
    "ValidatedLevel",
    "extract_scene_dict",
    "validate_level",
    "load_valid_level",
    "iter_valid_levels",
    "prewarm",
]

logger = logging.getLogger(__name__)

# Deterministic salt for oracle RNG seeding.
# Using sum(b"oracle") avoids Python's randomized hash() across interpreter runs.
_ORACLE_RNG_SALT = sum(b"oracle")  # 630

# Module-level default registry — created once on first use.
_default_registry: SeedRegistry | None = None


def _get_registry() -> SeedRegistry:
    """Return the default global registry, creating it on first call."""
    global _default_registry
    if _default_registry is None:
        _default_registry = SeedRegistry()
    return _default_registry


def validate_level(
    level_name: str,
    seed: int,
    variant: int,
    *,
    registry: SeedRegistry | None = None,
    config: SimulationConfig | None = None,
    n_attempts: int = 50,
    oracle_steps: int = 500,
) -> str:
    """Run the full validation pipeline for one (level_name, seed, variant) triple.

    Returns the status string: "valid", "trivial", or "impossible".

    Pipeline:
        1. Registry lookup — return cached status immediately if present.
        2. load_level(level_name, seed, variant).
        3. is_trivial check — if the success condition is met at t=0 with no
           agent action, mark "trivial". Runs before no-action-objects so that
           a level already satisfied at t=0 is never wrongly marked "impossible".
        4. No-action-objects check — levels with no action objects cannot be
           solved by the agent; mark "impossible".
        5. Oracle — run the targeted (or default) oracle; mark "valid" on
           success, "impossible" on exhaustion.

    The oracle RNG is seeded from (seed, variant, _ORACLE_RNG_SALT) so results
    are deterministic and independent of level geometry draws.
    """
    reg = registry if registry is not None else _get_registry()
    cfg = config or SimulationConfig()

    # Step 1: registry cache — skip full pipeline if already validated
    cached = reg.lookup(level_name, seed, variant)
    if cached is not None:
        return cached

    # Step 2: build the level geometry for this (seed, variant)
    level = load_level(level_name, seed=seed, variant=variant)

    # Step 3: trivial check — success already met at t=0 without any agent action.
    # Must run before the no-action-objects check: a level with no action objects
    # that already satisfies its goal is correctly "trivial", not "impossible".
    if is_trivial(level, cfg):
        reg.record(level_name, seed, variant, "trivial")
        return "trivial"

    # Step 4: levels with no action objects cannot be solved by agent placement
    if len(level.action_objects) == 0:
        reg.record(level_name, seed, variant, "impossible")
        return "impossible"

    # Step 5: oracle solvability search
    # RNG seeded from (seed, variant, salt) for reproducibility across processes.
    oracle = get_oracle(level_name)
    oracle_rng = np.random.default_rng([seed, variant, _ORACLE_RNG_SALT])
    solved = oracle(level, cfg, n_attempts, oracle_steps, oracle_rng)

    if solved:
        scene = extract_scene_dict(level)
        reg.record(level_name, seed, variant, "valid", scene_dict=scene)
        return "valid"
    else:
        reg.record(level_name, seed, variant, "impossible")
        return "impossible"


def load_valid_level(
    level_name: str,
    seed: int,
    *,
    registry: SeedRegistry | None = None,
    config: SimulationConfig | None = None,
    max_variants: int | None = None,
    n_attempts: int | None = None,
    oracle_steps: int = 1000,
) -> ValidatedLevel:
    """Return a valid level for seed, trying variants 0, 1, … until one passes.

    Variant 0 is tried first: for seeds in the bundled range (0–10000), the lookup
    hits in-memory bundled data with no I/O or oracle overhead.

    max_variants and n_attempts default to None, which resolves to the per-level
    calibrated values from register_defaults() (derived from geometric-decay analysis
    during oracle calibration). Pass explicit values only to override the calibration.

    Raises RuntimeError if no valid variant is found within max_variants tries.
    """
    reg = registry if registry is not None else _get_registry()
    cfg = config or SimulationConfig()
    # Resolve per-level calibrated search budget; fall back to conservative defaults
    # only when a level has no registered values (uncalibrated oracle).
    if max_variants is None:
        max_variants = get_default_max_variants(level_name)
    if n_attempts is None:
        n_attempts = get_default_n_attempts(level_name)

    # Warn before the oracle search loop if the bundle shows this level is
    # known-impossible (0% valid) or near-impossible (< 1% valid).
    # stacklevel=2 points the warning at the caller's site, not this function.
    valid_rate = reg.bundle_valid_rate(level_name)
    if valid_rate is not None:
        if valid_rate == 0.0:
            warnings.warn(
                f"'{level_name}' is a known-impossible level: 0% of bundled seeds "
                "have a valid placement; it is unlikely to be solvable.",
                UserWarning,
                stacklevel=2,
            )
        elif valid_rate < 0.01:
            warnings.warn(
                f"'{level_name}' is a near-impossible level: only "
                f"{valid_rate * 100:.2f}% of bundled seeds have a valid placement; "
                "it is unlikely to be solvable with valid placements.",
                UserWarning,
                stacklevel=2,
            )

    # Fast path: direct bundle lookup — O(1), no oracle call needed.
    # get_valid_entry returns the pre-computed entry for this seed (valid or
    # impossible) without scanning variants.
    bundle_entry = reg.get_valid_entry(level_name, seed)
    if bundle_entry is not None:
        if bundle_entry["status"] == "valid":
            variant = bundle_entry["variant"]
            level = load_level(level_name, seed=seed, variant=variant)
            scene = bundle_entry.get("scene") or extract_scene_dict(level)
            return ValidatedLevel(
                level=level,
                level_name=level_name,
                seed=seed,
                variant=variant,
                scene_dict=scene,
            )
        # status == "impossible": confirmed by the bundle oracle.  Fall through
        # to the scan loop so out-of-date bundles do not permanently block seeds
        # that became solvable after an oracle fix.

    for variant in range(max_variants):
        status = validate_level(
            level_name,
            seed,
            variant,
            registry=reg,
            config=cfg,
            n_attempts=n_attempts,
            oracle_steps=oracle_steps,
        )
        if status == "valid":
            # Retrieve scene dict from registry (bundled or SQLite).
            scene = reg.get_scene_dict(level_name, seed, variant)
            level = load_level(level_name, seed=seed, variant=variant)
            assert scene is not None, (
                f"registry returned status='valid' but no scene for "
                f"({level_name!r}, seed={seed}, variant={variant})"
            )
            return ValidatedLevel(
                level=level,
                level_name=level_name,
                seed=seed,
                variant=variant,
                scene_dict=scene,
            )

    # Improve the error message when the bundle confirms the level is impossible
    # or near-impossible, so researchers understand why this failed.
    if valid_rate is not None and valid_rate < 0.01:
        impossibility = (
            "known-impossible (0% valid in bundle)"
            if valid_rate == 0.0
            else f"near-impossible ({valid_rate * 100:.2f}% valid in bundle)"
        )
        raise RuntimeError(
            f"load_valid_level: '{level_name}' seed={seed} has no valid variant in "
            f"[0, {max_variants}). This level is {impossibility} — "
            "it may not be solvable with valid placements."
        )
    raise RuntimeError(
        f"load_valid_level: '{level_name}' seed={seed} has no valid variant in "
        f"[0, {max_variants}). Increase max_variants or audit oracle coverage "
        "with list_oracles()."
    )


def iter_valid_levels(
    level_name: str,
    start_seed: int = 0,
    *,
    registry: SeedRegistry | None = None,
    config: SimulationConfig | None = None,
    max_variants: int | None = None,
    n_attempts: int | None = None,
    oracle_steps: int = 1000,
    max_attempts: int = 10000,
) -> Iterator[ValidatedLevel]:
    """Yield valid levels for seed, seed+1, seed+2, … without bound.

    The primary interface for experiment loops. Seed is incremented after each
    yield; variant search is handled transparently by load_valid_level.

    Args:
        level_name: Name of the level to iterate over.
        start_seed: First seed to try (default 0).
        registry: Optional SeedRegistry; defaults to the module-level registry.
        config: Optional SimulationConfig; defaults to SimulationConfig().
        max_variants: Maximum variants to try per seed (None = use per-level calibrated
            value from register_defaults); passed through to load_valid_level.
        n_attempts: Oracle attempts per variant (None = use per-level calibrated value
            from register_defaults); passed through to load_valid_level.
        oracle_steps: Simulation steps per oracle attempt; passed through to
            load_valid_level.
        max_attempts: Maximum number of consecutive seeds that can fail before a
            RuntimeError is raised (default 10000). This guards against infinite
            loops when the level is impossible or extremely unlikely to produce a
            valid geometry. Known-impossible levels such as ``the_cradle`` will
            raise after at most ``max_attempts`` seeds rather than hanging forever.

    Raises:
        RuntimeError: If ``max_attempts`` consecutive seeds all fail to produce a
            valid level. This most likely means the level itself is impossible (no
            seed will ever be valid) rather than a transient failure. Use
            ``prewarm()`` to audit oracle coverage for the level.
    """
    reg = registry if registry is not None else _get_registry()
    cfg = config or SimulationConfig()
    seed = start_seed

    # Track consecutive failures to detect impossible or near-impossible levels.
    consecutive_failures = 0

    while True:
        try:
            yield load_valid_level(
                level_name,
                seed,
                registry=reg,
                config=cfg,
                max_variants=max_variants,
                n_attempts=n_attempts,
                oracle_steps=oracle_steps,
            )
            # A successful yield resets the consecutive failure counter.
            consecutive_failures = 0
        except RuntimeError:
            # Seed exhausted all variants — skip and continue to the next seed.
            # Callers who need to know the exhaustion rate should use prewarm().
            consecutive_failures += 1
            if consecutive_failures >= max_attempts:
                raise RuntimeError(
                    f"iter_valid_levels: '{level_name}' produced no valid level in "
                    f"{max_attempts} consecutive seeds (starting at seed "
                    f"{seed - max_attempts + 1}). The level may be impossible — "
                    "verify with prewarm() before iterating. To extend the search, "
                    "pass a larger max_attempts value."
                ) from None
        seed += 1


class _PrewarmArgs(NamedTuple):
    """Arguments passed to each _prewarm_worker invocation.

    Using a NamedTuple makes call sites self-documenting and guards against
    positional argument reordering bugs when the argument list changes.
    """

    level_name: str
    seed: int
    cache_path: str
    config: SimulationConfig
    max_variants: int
    n_attempts: int
    oracle_steps: int


def _prewarm_worker(args: _PrewarmArgs) -> tuple[str, int, str, int]:
    """Validate one (level_name, seed) pair in a worker process.

    Each worker creates its own SeedRegistry — WAL mode on the shared SQLite
    database ensures safe concurrent writes without coordination.

    Returns (level_name, seed, outcome, winning_variant) where outcome is one of:
      "valid"      — a valid variant was found; winning_variant is its index
      "trivial"    — all max_variants returned "trivial" (no agent needed)
      "impossible" — all max_variants returned "impossible" (oracle failed)
      "exhausted"  — max_variants exhausted with a mix of trivial and impossible
    For non-valid outcomes, winning_variant is -1.
    """
    level_name, seed, cache_path, config, max_variants, n_attempts, oracle_steps = args

    # Top-level function required for spawn-compatible pickling (macOS/Windows).
    from interphyre.validation import validate_level  # noqa: PLC0415
    from interphyre.validation.registry import SeedRegistry  # noqa: PLC0415

    registry = SeedRegistry(cache_path)

    # Use batched() so all variant writes for this seed are committed in one
    # fsync rather than one commit per variant — reduces I/O by up to max_variants×.
    # If the worker is killed mid-seed the partial writes are rolled back and the
    # seed will be re-run on the next prewarm call, which is the same recovery
    # behaviour as before.
    statuses: list[str] = []
    with registry.batched():
        for variant in range(max_variants):
            status = validate_level(
                level_name,
                seed,
                variant,
                registry=registry,
                config=config,
                n_attempts=n_attempts,
                oracle_steps=oracle_steps,
            )
            if status == "valid":
                return level_name, seed, "valid", variant
            statuses.append(status)

    # Exhausted all max_variants — categorise by what was found.
    unique = set(statuses)
    if unique == {"trivial"}:
        return level_name, seed, "trivial", -1
    elif unique == {"impossible"}:
        return level_name, seed, "impossible", -1
    else:
        return level_name, seed, "exhausted", -1


def _seed_outcome_from_registry(
    reg: SeedRegistry, level_name: str, seed: int, max_variants: int
) -> tuple[str, int] | None:
    """Return the prewarm outcome for seed if all max_variants are in the registry.

    Returns None if any variant is not yet validated — the seed must be sent
    to the worker pool.

    Returns (outcome, winning_variant) where:
      - outcome is "valid", "trivial", "impossible", or "exhausted"
      - winning_variant is the variant index when outcome is "valid", else -1
    """
    statuses: list[str] = []
    for variant in range(max_variants):
        status = reg.lookup(level_name, seed, variant)
        if status is None:
            return None  # Still has unchecked variants — needs worker
        if status == "valid":
            return "valid", variant  # capture the first valid variant index
        statuses.append(status)

    # All max_variants checked, none valid.
    unique = set(statuses)
    if unique == {"trivial"}:
        return "trivial", -1
    elif unique == {"impossible"}:
        return "impossible", -1
    else:
        return "exhausted", -1


def prewarm(
    level_names: list[str],
    seeds: range | list[int],
    *,
    registry: SeedRegistry | None = None,
    config: SimulationConfig | None = None,
    workers: int = 4,
    max_variants: int = 10,
    n_attempts: int = 50,
    oracle_steps: int = 500,
    progress: bool = True,
) -> dict[str, dict[str, int | dict[int, int]]]:
    """Pre-validate seeds for multiple levels in parallel.

    Idempotent: seeds fully resolved in the registry are counted without re-running
    the oracle. Seeds with partial coverage have only uncached variants validated.

    Uses ProcessPoolExecutor for CPU-bound oracle work. Shows a tqdm progress bar
    when progress=True and tqdm is installed; falls back to periodic stderr prints.

    Returns per-level outcome counts:
        {level_name: {
            "valid": N, "trivial": N, "impossible": N, "exhausted": N,
            "variant_hist": {variant_index: count_of_seeds_first_valid_here}
        }}
    variant_hist records the distribution of first-valid-variant values across all
    valid seeds. A mean of 0.0 and max of 0 means every seed's variant=0 geometry
    is solvable; higher values indicate the oracle relies on variant fallbacks.
    """
    reg = registry if registry is not None else _get_registry()
    cfg = config or SimulationConfig()
    cache_path = str(reg.db_path)
    seeds_list = list(seeds)

    counts: dict[str, dict[str, int | dict[int, int]]] = {
        name: {
            "valid": 0,
            "trivial": 0,
            "impossible": 0,
            "exhausted": 0,
            "variant_hist": {},
        }
        for name in level_names
    }

    def _record_outcome(level_name: str, outcome: str, winning_variant: int) -> None:
        counts[level_name][outcome] += 1  # type: ignore[operator]
        if outcome == "valid":
            hist = counts[level_name]["variant_hist"]
            hist[winning_variant] = hist.get(winning_variant, 0) + 1  # type: ignore[union-attr,index]

    # Partition seeds: fully resolved in registry vs. needs worker processing.
    pending: list[_PrewarmArgs] = []
    for level_name in level_names:
        for seed in seeds_list:
            result = _seed_outcome_from_registry(reg, level_name, seed, max_variants)
            if result is not None:
                outcome, winning_variant = result
                _record_outcome(level_name, outcome, winning_variant)
            else:
                pending.append(
                    _PrewarmArgs(
                        level_name=level_name,
                        seed=seed,
                        cache_path=cache_path,
                        config=cfg,
                        max_variants=max_variants,
                        n_attempts=n_attempts,
                        oracle_steps=oracle_steps,
                    )
                )

    if not pending:
        return counts

    # Set up progress reporting — tqdm if available, else periodic stderr prints.
    use_tqdm = False
    pbar = None
    if progress:
        try:
            from tqdm import tqdm as _tqdm  # noqa: PLC0415

            pbar = _tqdm(total=len(pending), desc="prewarm", unit="seed")
            use_tqdm = True
        except ImportError:
            pass

    completed = 0
    log_interval = max(1, len(pending) // 20)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_prewarm_worker, args): args for args in pending}
        for future in as_completed(futures):
            level_name, _seed, outcome, winning_variant = future.result()
            _record_outcome(level_name, outcome, winning_variant)
            completed += 1
            if use_tqdm and pbar is not None:
                pbar.update(1)
            elif progress and completed % log_interval == 0:
                print(
                    f"prewarm: {completed}/{len(pending)} done",
                    file=sys.stderr,
                )

    if use_tqdm and pbar is not None:
        pbar.close()

    return counts
