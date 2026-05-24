"""Offline bundle generator for interphyre/data/levels/.

Runs the same validation pipeline as validate_level but writes output to
lzma-compressed JSON files that ship with the package. These files feed the
O(1) bundled lookup tier in SeedRegistry.

Not part of the public API. Must not be imported by runtime code.

Usage (run once at release):
    python -m interphyre.validation._bundle --levels all --seeds 0:1000 --workers 8
"""

from __future__ import annotations

import argparse
import json
import lzma
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import numpy as np

from interphyre.config import SimulationConfig
from interphyre.levels import build_level_from_scene, list_levels, load_level
from interphyre.validation import _ORACLE_RNG_SALT
from interphyre.validation.checks import extract_scene_dict, is_trivial
from interphyre.validation.oracles import (
    get_default_max_variants,
    get_default_n_attempts,
    get_oracle,
    get_solver,
)
from interphyre.validation.registry import _compute_schema_hash

# Output directory for bundled lzma files.
_BUNDLE_DIR = Path(__file__).parent.parent / "data" / "levels"

# Stored solutions are rounded to this many decimal places before storage and
# verified to still solve the level.  Physics coordinates live in [-5, 5], so
# 4dp gives 0.0001 precision — identical to tools/collect_data.py, and coarse
# enough that IEEE-754 FP differences between CPU architectures (AMD vs Intel)
# do not shift the stored value.  A solution that fails after rounding was
# knife-edge and is rejected in favour of a more robust placement.
_SOLUTION_ROUND_DIGITS = 4


def _git_short_hash() -> str:
    """Return the short git hash of HEAD at bundle generation time.

    Stored as bundle metadata['oracle_commit'] so future engineers can identify
    which oracle version produced the bundle. Falls back to 'unknown' if git is
    unavailable (e.g., in CI environments without a .git directory).
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


_DEFAULT_MAX_VARIANTS = 10
_DEFAULT_N_ATTEMPTS = 50
_DEFAULT_ORACLE_STEPS = 500
_DEFAULT_WORKERS = 4

# Hard cap: bundles must only contain seeds in [0, _MAX_SEED] (inclusive).
# Canonical seed universe is seeds 0–10000 (10001 seeds total).
_MAX_SEED = 10_000


class _ValidateSeedArgs(NamedTuple):
    """Arguments for one _validate_seed worker invocation.

    NamedTuple guards against positional reordering bugs between the work_items
    construction site and the _validate_seed unpack.
    """

    level_name: str
    seed: int
    max_variants: int
    n_attempts: int
    oracle_steps: int


def _oracle_rng(seed: int, variant: int) -> np.random.Generator:
    """Return a reproducible RNG for the oracle, seeded from (seed, variant, salt).

    Three-integer list seeding matches validate_level in __init__.py, ensuring
    that bundle and live paths produce identical oracle RNG sequences for every
    (seed, variant) pair.
    """
    return np.random.default_rng([seed, variant, _ORACLE_RNG_SALT])


def _validate_seed(args: _ValidateSeedArgs) -> dict:
    """Validate one (level_name, seed) pair, returning exactly one entry.

    Top-level function required for ProcessPoolExecutor pickling on macOS (spawn).

    Returns a single dict {seed, variant, status, scene, solution}:
    - status "valid":      first variant that passes the oracle; scene and
                           solution are populated.
    - status "impossible": all variants exhausted; variant is recorded as 0
                           (the specific failed variant is irrelevant).

    Intermediate failed variants are not recorded — one entry per seed is all
    the bundle needs, keeping bundle files compact and lookups O(1) by seed.
    """
    level_name, seed, max_variants, n_attempts, oracle_steps = args
    config = SimulationConfig(max_steps=oracle_steps)
    level_solver = get_solver(level_name)
    oracle = get_oracle(level_name)

    # Lazy import: avoid circular import (environment → validation → environment).
    from interphyre.environment import InterphyreEnv

    for variant in range(max_variants):
        level = load_level(level_name, seed=seed, variant=variant)

        # Trivial check: skip variants where the success condition is already met
        # at t=0 without any agent action.  These are not valid training examples
        # (no placement was needed) and must not appear in the bundle.
        if is_trivial(level, config):
            continue

        # Solver path: prefer get_solver so the winning placement is captured.
        # Oracle bool path: fall back when no solver is registered.
        rng = _oracle_rng(seed, variant)
        if level_solver is not None:
            sol = level_solver(level, config, n_attempts, oracle_steps, rng)
            if sol is not None:
                # Round to _SOLUTION_ROUND_DIGITS before storage and verify
                # the rounded placement still solves the level.  Full-precision
                # (15–16dp) solutions can be knife-edge: tiny FP differences
                # between CPU architectures (AMD vs Intel SIMD) shift the
                # trajectory just enough to miss.  A solution that fails after
                # rounding was not robust and is rejected; try the next variant.
                solution_json = [
                    [
                        round(pos[0], _SOLUTION_ROUND_DIGITS),
                        round(pos[1], _SOLUTION_ROUND_DIGITS),
                        round(pos[2], _SOLUTION_ROUND_DIGITS),
                    ]
                    for pos in sol
                ]
                env_check = InterphyreEnv(level, config=config)
                try:
                    env_check.reset()
                    _, _, _, _, info = env_check.step(solution_json)
                finally:
                    env_check.close()
                if not info.get("success", False):
                    # Rounded solution doesn't reproduce — knife-edge; try next variant.
                    continue
                scene_dict = extract_scene_dict(level)
                return {
                    "seed": seed,
                    "variant": variant,
                    "status": "valid",
                    "scene": scene_dict,
                    "solution": solution_json,
                }
        else:
            # Oracle-only path: solution is unavailable.
            if oracle(level, config, n_attempts, oracle_steps, rng):
                scene_dict = extract_scene_dict(level)
                return {
                    "seed": seed,
                    "variant": variant,
                    "status": "valid",
                    "scene": scene_dict,
                    "solution": None,
                }

    # All variants exhausted without finding a valid placement.
    return {
        "seed": seed,
        "variant": 0,
        "status": "impossible",
        "scene": None,
        "solution": None,
    }


def _assert_round_trip(
    level_name: str, seed: int, variant: int, scene_dict: dict
) -> None:
    """Assert that build_level_from_scene reconstructs the scene bit-identically.

    Aborts with RuntimeError on any mismatch — this is a hard invariant for
    reproducibility. A failure means extract_scene_dict is missing an attribute
    that affects geometry, and the bundle must not be written until fixed.
    """
    reconstructed = build_level_from_scene(level_name, scene_dict)
    scene_dict_2 = extract_scene_dict(reconstructed)

    if scene_dict == scene_dict_2:
        return

    mismatches = [
        f"  {name}.{attr}: {scene_dict[name][attr]!r} != {scene_dict_2.get(name, {}).get(attr)!r}"
        for name in scene_dict
        for attr in scene_dict[name]
        if scene_dict[name][attr] != scene_dict_2.get(name, {}).get(attr)
    ]
    raise RuntimeError(
        f"Round-trip assertion failed for {level_name} seed={seed} variant={variant}.\n"
        f"Mismatched attributes:\n" + "\n".join(mismatches)
    )


def _read_existing_bundle(bundle_path: Path) -> dict | None:
    """Read an existing bundle file, returning the parsed data dict or None if absent."""
    if not bundle_path.exists():
        return None
    with lzma.open(bundle_path, "rb") as fh:
        return json.load(fh)


def _extend_level_bundle(
    level_name: str,
    *,
    target_valid: int,
    max_variants: int,
    n_attempts: int,
    oracle_steps: int,
    workers: int,
) -> None:
    """Extend an existing bundle until it contains target_valid valid entries.

    Reads the existing bundle, estimates how many more seeds are needed at the
    observed valid rate, generates them starting from max_seed+1, merges with
    existing entries, and rewrites the file. If the bundle already meets
    target_valid the function returns immediately without touching the file.
    """
    bundle_path = _BUNDLE_DIR / f"{level_name}.json.lzma"
    existing = _read_existing_bundle(bundle_path)
    if existing is None:
        raise FileNotFoundError(
            f"No existing bundle for {level_name}. Use --seeds to generate from scratch."
        )

    existing_entries = existing["entries"]
    existing_valid_seeds = {
        e["seed"] for e in existing_entries if e["status"] == "valid"
    }
    n_valid = len(existing_valid_seeds)

    if n_valid >= target_valid:
        print(
            f"[{level_name}] Already has {n_valid} valid seeds (target={target_valid}). "
            "Nothing to do."
        )
        return

    # Estimate how many new seeds are needed.  Use the observed valid fraction
    # with a 1.25× safety margin so we over-generate rather than under-generate.
    n_unique_seeds = len({e["seed"] for e in existing_entries})
    observed_rate = n_valid / n_unique_seeds if n_unique_seeds else 0.1
    extra_needed = target_valid - n_valid
    # Guard against near-zero rates producing absurd estimates.
    capped_rate = max(observed_rate, 0.02)
    extra_seeds = int(extra_needed / capped_rate * 1.25) + 50

    max_seed = max(e["seed"] for e in existing_entries)
    if max_seed >= _MAX_SEED:
        print(
            f"[{level_name}] Already at seed ceiling ({max_seed} = {_MAX_SEED}). "
            f"Cannot extend further — {n_valid} valid is the maximum achievable."
        )
        return

    new_start = max_seed + 1
    new_stop = min(new_start + extra_seeds, _MAX_SEED + 1)
    new_seeds = range(new_start, new_stop)

    print(
        f"[{level_name}] Has {n_valid}/{target_valid} valid. "
        f"Valid rate {observed_rate:.1%}. "
        f"Generating {len(new_seeds)} new seeds ({new_start}:{new_stop})..."
    )

    _build_level_bundle(
        level_name,
        new_seeds,
        max_variants=max_variants,
        n_attempts=n_attempts,
        oracle_steps=oracle_steps,
        workers=workers,
        _existing_entries=existing_entries,
    )


def _checkpoint_write(
    bundle_path: Path,
    entries: list[dict],
    schema_hash: str,
    oracle_commit: str,
) -> None:
    """Write entries to the bundle file atomically via a temp file.

    Called periodically during generation so that a killed job loses at most
    one checkpoint interval of work rather than everything.
    """
    tmp = bundle_path.with_suffix(".lzma.tmp")
    with lzma.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(
            {
                "schema_hash": schema_hash,
                "oracle_commit": oracle_commit,
                "entries": entries,
            },
            fh,
        )
    tmp.replace(bundle_path)


def _build_level_bundle(
    level_name: str,
    seeds: range,
    *,
    max_variants: int,
    n_attempts: int,
    oracle_steps: int,
    workers: int,
    _existing_entries: list[dict] | None = None,
    _output_path: Path | None = None,
) -> None:
    """Validate all seeds for one level and write the lzma bundle file.

    When _existing_entries is supplied the new entries are merged with the
    existing ones before writing, enabling the --extend workflow.

    Checkpoints are written every 100 seeds so that a preempted or killed job
    loses at most 100 seeds of work. The checkpoint file is a valid bundle that
    can be extended with --extend if the job is interrupted.
    """
    print(f"[{level_name}] Validating {len(seeds)} seeds with {workers} workers...")

    _BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = (
        _output_path
        if _output_path is not None
        else _BUNDLE_DIR / f"{level_name}.json.lzma"
    )
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    schema_hash = _compute_schema_hash(level_name)
    oracle_commit = _git_short_hash()
    base_entries = list(_existing_entries) if _existing_entries is not None else []

    work_items = [
        _ValidateSeedArgs(level_name, seed, max_variants, n_attempts, oracle_steps)
        for seed in seeds
    ]

    all_entries: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_validate_seed, item): item[1] for item in work_items}
        completed = 0
        for future in as_completed(futures):
            seed = futures[future]
            try:
                entry = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"[{level_name}] Worker failed for seed={seed}: {exc}"
                ) from exc
            all_entries.append(entry)
            completed += 1
            if completed % 100 == 0:
                print(f"[{level_name}]   {completed}/{len(seeds)} seeds done")
                # Checkpoint: write merged entries sorted by seed so a killed job
                # leaves a valid, readable bundle on disk.
                checkpoint = sorted(
                    base_entries + all_entries, key=lambda e: (e["seed"], e["variant"])
                )
                _checkpoint_write(bundle_path, checkpoint, schema_hash, oracle_commit)

    # Round-trip assertion: every valid entry's scene must reconstruct identically.
    valid_entries = [e for e in all_entries if e["status"] == "valid"]
    print(
        f"[{level_name}] Verifying round-trip for {len(valid_entries)} valid entries..."
    )
    for entry in valid_entries:
        _assert_round_trip(level_name, entry["seed"], entry["variant"], entry["scene"])

    # Final write: merge with pre-existing entries and sort.
    all_entries = sorted(
        base_entries + all_entries, key=lambda e: (e["seed"], e["variant"])
    )
    _checkpoint_write(bundle_path, all_entries, schema_hash, oracle_commit)

    statuses = [e["status"] for e in all_entries]
    print(
        f"[{level_name}] Done — valid: {statuses.count('valid')}, "
        f"impossible: {statuses.count('impossible')} "
        f"→ {bundle_path}"
    )


def _parse_seeds(seeds_arg: str) -> range:
    """Parse Python slice notation into a range. Accepts start:stop or start:stop:step."""
    parts = seeds_arg.split(":")
    try:
        if len(parts) == 2:
            result = range(int(parts[0]), int(parts[1]))
        elif len(parts) == 3:
            result = range(int(parts[0]), int(parts[1]), int(parts[2]))
        else:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid seeds format '{seeds_arg}'. Expected start:stop or start:stop:step."
        )
    if result.stop > _MAX_SEED + 1:
        raise argparse.ArgumentTypeError(
            f"Seed range {seeds_arg} exceeds the canonical universe [0, {_MAX_SEED}]. "
            f"Bundles must not contain seeds above {_MAX_SEED}."
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate bundled validation data for interphyre/data/levels/."
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["all"],
        help="Level names to bundle, or 'all' for every registered level.",
    )
    parser.add_argument(
        "--seeds",
        default="0:1000",
        help="Seed range in slice notation: start:stop or start:stop:step.",
    )
    parser.add_argument("--workers", type=int, default=_DEFAULT_WORKERS)
    parser.add_argument(
        "--max-variants",
        type=int,
        default=None,
        help="Max variants per seed (default: per-oracle recommendation from register_defaults).",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=None,
        help="Oracle attempts per seed (default: per-oracle recommendation from register_defaults).",
    )
    parser.add_argument("--oracle-steps", type=int, default=_DEFAULT_ORACLE_STEPS)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Write bundle to this path instead of the default interphyre/data/levels/ location. "
            "Useful for parallel chunk jobs: write each chunk to a temp file, then merge with "
            "the bundle-merge script."
        ),
    )
    parser.add_argument(
        "--extend",
        action="store_true",
        help=(
            "Extend an existing bundle by adding seeds beyond its current max. "
            "Reads the existing bundle, estimates how many more seeds are needed "
            "to reach --target-valid, generates them, and merges. "
            "Requires the bundle file to already exist. Ignores --seeds."
        ),
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help=(
            "Merge new seeds (from --seeds) into an existing bundle. "
            "Reads the existing bundle, validates the specified seeds, and writes "
            "the union. Use this to add specific seeds (e.g. 10000:10001) without "
            "replacing the existing data. Requires the bundle file to already exist."
        ),
    )
    parser.add_argument(
        "--target-valid",
        type=int,
        default=10000,
        help="Target number of valid seeds when using --extend (default: 10000).",
    )

    args = parser.parse_args()

    level_names = list_levels() if args.levels == ["all"] else args.levels

    if args.extend:
        print(
            f"Extending bundles: {len(level_names)} levels, "
            f"target_valid={args.target_valid}, workers={args.workers}"
        )
        for level_name in sorted(level_names):
            _extend_level_bundle(
                level_name,
                target_valid=args.target_valid,
                max_variants=args.max_variants
                if args.max_variants is not None
                else get_default_max_variants(level_name),
                n_attempts=args.attempts
                if args.attempts is not None
                else get_default_n_attempts(level_name),
                oracle_steps=args.oracle_steps,
                workers=args.workers,
            )
    elif args.merge:
        seeds = _parse_seeds(args.seeds)
        print(
            f"Merging seeds {args.seeds} into bundles: {len(level_names)} levels, "
            f"workers={args.workers}"
        )
        for level_name in sorted(level_names):
            bundle_path = _BUNDLE_DIR / f"{level_name}.json.lzma"
            existing = _read_existing_bundle(bundle_path)
            if existing is None:
                raise FileNotFoundError(
                    f"No existing bundle for {level_name}. "
                    "Use --seeds (without --merge) to generate from scratch."
                )
            _build_level_bundle(
                level_name,
                seeds,
                max_variants=args.max_variants
                if args.max_variants is not None
                else get_default_max_variants(level_name),
                n_attempts=args.attempts
                if args.attempts is not None
                else get_default_n_attempts(level_name),
                oracle_steps=args.oracle_steps,
                workers=args.workers,
                _existing_entries=existing["entries"],
                _output_path=args.output,
            )
    else:
        seeds = _parse_seeds(args.seeds)
        print(
            f"Building bundles: {len(level_names)} levels, "
            f"seeds {args.seeds}, workers={args.workers}"
        )
        for level_name in sorted(level_names):
            _build_level_bundle(
                level_name,
                seeds,
                max_variants=args.max_variants
                if args.max_variants is not None
                else get_default_max_variants(level_name),
                n_attempts=args.attempts
                if args.attempts is not None
                else get_default_n_attempts(level_name),
                oracle_steps=args.oracle_steps,
                workers=args.workers,
                _output_path=args.output,
            )

    print("All bundles complete.")


if __name__ == "__main__":
    main()
