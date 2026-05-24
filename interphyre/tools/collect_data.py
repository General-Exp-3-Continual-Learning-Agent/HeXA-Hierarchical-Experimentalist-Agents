#!/usr/bin/env python3
"""
Fast and reliable data collection for Interphyre levels.

This is a rewrite of collect_training_data.py with focus on:
- Deterministic simulation (no false positives/negatives)
- Performance (environment reuse, single simulation per action)
- Clean architecture (separation of concerns)
- Optional verification mode for debugging

Key improvements:
1. Single environment instance per seed (no repeated creation/destruction)
2. No double verification by default (optional verify_mode flag)
3. Contact validation happens once per physics step (no race conditions)
4. Relaxed contact tolerance (0.05 instead of 0.01)
5. Time-based stationary detection (no floating-point jitter)
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.random_agent import RandomAgent
from interphyre.config import SimulationConfig
from interphyre.environment import InterphyreEnv
from interphyre.levels import load_level


class CEMAgent:
    """Cross-Entropy Method optimizer for continuous actions."""

    def __init__(
        self,
        seed: Optional[int] = None,
        population: int = 128,
        elite_frac: float = 0.1,
        iterations: int = 5,
        min_std: float = 0.02,
    ):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.population = population
        self.elite_frac = elite_frac
        self.iterations = iterations
        self.min_std = min_std
        self.action_space = None
        self._evaluator = None

    def set_action_space(self, action_space) -> None:
        self.action_space = action_space

    def set_evaluator(self, evaluator) -> None:
        """Set evaluator callback: evaluator(action_array) -> (reward, success)."""
        self._evaluator = evaluator

    def set_seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def get_action(self, observation: Any) -> np.ndarray:
        if self.action_space is None or self._evaluator is None:
            raise ValueError("Action space or evaluator not set for CEMAgent.")

        low = np.asarray(self.action_space.low, dtype=np.float32)
        high = np.asarray(self.action_space.high, dtype=np.float32)
        dim = low.shape[0]

        mean = (low + high) / 2.0
        std = (high - low) / 2.0

        elite_count = max(1, int(self.population * self.elite_frac))
        best_success = None
        best_reward = float("-inf")

        for _ in range(self.iterations):
            samples = self.rng.normal(mean, std, size=(self.population, dim))
            samples = np.clip(samples, low, high)
            samples = np.round(samples, 4)

            rewards = []
            successes = []
            for sample in samples:
                reward, success = self._evaluator(sample)
                rewards.append(reward)
                successes.append(success)

                if success and reward > best_reward:
                    best_reward = reward
                    best_success = sample.copy()

            if best_success is not None:
                return best_success

            rewards_np = np.asarray(rewards)
            elite_indices = rewards_np.argsort()[-elite_count:]
            elite_samples = samples[elite_indices]

            mean = elite_samples.mean(axis=0)
            std = elite_samples.std(axis=0)
            std = np.maximum(std, self.min_std)

        raise ValueError("CEM did not find a successful action.")


def _log(message: str) -> None:
    """Log message to stderr."""
    print(message, file=sys.stderr)


def _flatten_action(action: Sequence[Any]) -> List[float]:
    """Flatten action to list of floats."""
    if isinstance(action, np.ndarray):
        return action.astype(float).reshape(-1).tolist()

    if not action:
        return []

    first = action[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        flat: List[float] = []
        for triplet in action:
            if len(triplet) != 3:
                raise ValueError(f"Expected action triplet, got {triplet}")
            flat.extend([float(triplet[0]), float(triplet[1]), float(triplet[2])])
        return flat

    return [float(x) for x in action]


def _normalize_action(
    action: Sequence[Any], fix_ball_size: Optional[float] = None
) -> List[float]:
    """Normalize action to list of floats with optional ball size fixing."""
    flat = _flatten_action(action)
    if len(flat) % 3 != 0:
        raise ValueError(f"Action length must be multiple of 3, got {len(flat)}")

    if fix_ball_size is not None:
        for i in range(2, len(flat), 3):
            flat[i] = float(fix_ball_size)

    return [round(float(x), 4) for x in flat]


def _action_to_tuples(action: Sequence[Any]) -> List[Tuple[float, float, float]]:
    """Convert action to list of (x, y, radius) tuples."""
    flat = _flatten_action(action)
    if len(flat) % 3 != 0:
        raise ValueError(f"Action length must be multiple of 3, got {len(flat)}")

    return [
        (float(flat[i]), float(flat[i + 1]), float(flat[i + 2]))
        for i in range(0, len(flat), 3)
    ]


class SolutionStorage:
    """Handle loading/saving solution files with validation."""

    def __init__(self, level_name: str, output_dir: Path):
        """Initialize solution storage.

        Args:
            level_name: Name of the level
            output_dir: Output directory for solutions
        """
        self.level_name = level_name
        self.level_dir = output_dir / level_name
        self.successes_path = self.level_dir / "successes.json"
        self.failures_path = self.level_dir / "failures.json"
        self.csv_path = self.level_dir / "dataset.csv"

    def load(self) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """Load existing solutions from JSON files.

        Returns:
            Tuple of (success_map, failure_map) where keys are seed strings
        """
        success_map: Dict[str, List[float]] = {}
        failure_map: Dict[str, List[float]] = {}

        if self.successes_path.exists():
            try:
                with self.successes_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if self.level_name in data and "solutions" in data[self.level_name]:
                        for k, v in data[self.level_name]["solutions"].items():
                            if v and len(v) > 0:
                                success_map[k] = _normalize_action(v[0])
            except Exception as e:
                _log(f"Warning: Failed to load successes: {e}")

        if self.failures_path.exists():
            try:
                with self.failures_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if self.level_name in data and "solutions" in data[self.level_name]:
                        for k, v in data[self.level_name]["solutions"].items():
                            if v and len(v) > 0:
                                failure_map[k] = _normalize_action(v[0])
            except Exception as e:
                _log(f"Warning: Failed to load failures: {e}")

        return success_map, failure_map

    def save(
        self,
        success_map: Dict[str, List[float]],
        failure_map: Dict[str, List[float]],
    ) -> None:
        """Save solutions to JSON files.

        Args:
            success_map: Map of seed -> success action
            failure_map: Map of seed -> failure action
        """
        self.level_dir.mkdir(parents=True, exist_ok=True)

        with self.successes_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    self.level_name: {
                        "solutions": {k: [v] for k, v in success_map.items()}
                    }
                },
                f,
                indent=2,
            )

        with self.failures_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    self.level_name: {
                        "solutions": {k: [v] for k, v in failure_map.items()}
                    }
                },
                f,
                indent=2,
            )

    def regenerate_csv(
        self,
        success_map: Dict[str, List[float]],
        failure_map: Dict[str, List[float]],
    ) -> None:
        """Regenerate CSV from solution maps.

        Args:
            success_map: Map of seed -> success action
            failure_map: Map of seed -> failure action
        """
        self.level_dir.mkdir(parents=True, exist_ok=True)

        seeds = sorted(set(success_map.keys()) & set(failure_map.keys()), key=int)
        rows = []
        for seed in seeds:
            img_rel = f"images/seed_{seed}.png"
            rows.append(
                {
                    "image": img_rel,
                    "action": json.dumps(failure_map[seed]),
                    "label": 0,
                    "level": self.level_name,
                    "seed": seed,
                    "attempt_index": 1,
                    "reward": -0.1,
                }
            )
            rows.append(
                {
                    "image": img_rel,
                    "action": json.dumps(success_map[seed]),
                    "label": 1,
                    "level": self.level_name,
                    "seed": seed,
                    "attempt_index": 1,
                    "reward": 1.0,
                }
            )

        rows.sort(key=lambda r: (int(r["seed"]), r["label"]))
        fieldnames = [
            "image",
            "action",
            "label",
            "level",
            "seed",
            "attempt_index",
            "reward",
        ]
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def validate_consistency(
        self,
        success_map: Dict[str, List[float]],
        failure_map: Dict[str, List[float]],
    ) -> List[str]:
        """Check that successes and failures don't overlap.

        Args:
            success_map: Map of seed -> success action
            failure_map: Map of seed -> failure action

        Returns:
            List of error messages for overlapping actions
        """
        errors = []
        for seed in set(success_map.keys()) & set(failure_map.keys()):
            if success_map[seed] == failure_map[seed]:
                errors.append(
                    f"Seed {seed} has identical action in successes and failures: {success_map[seed]}"
                )
        return errors


class DataCollector:
    """Core data collection engine with determinism guarantees."""

    def __init__(
        self,
        level_name: str,
        config: SimulationConfig,
        agent: Any,
        verify_mode: bool = False,
        fix_ball_size: Optional[float] = None,
    ):
        """Initialize data collector.

        Args:
            level_name: Name of the level
            config: Simulation configuration
            agent: Agent to use for action generation
            verify_mode: If True, double-check each solution (slower but safer)
            fix_ball_size: If provided, fix ball radius to this value
        """
        self.level_name = level_name
        self.config = config
        self.agent = agent
        self.verify_mode = verify_mode
        self.fix_ball_size = fix_ball_size
        self.env: Optional[InterphyreEnv] = None
        self.current_seed: Optional[int] = None

    def _create_env(self, seed: int) -> InterphyreEnv:
        """Create environment for a seed.

        Args:
            seed: Random seed for level generation

        Returns:
            InterphyreEnv instance
        """
        level = load_level(self.level_name, seed=seed)
        env = InterphyreEnv(
            level,
            config=self.config,
            observation_type="image",
            action_type="continuous",
            image_size=(224, 224),
            image_ppm=60.0,
            discrete_colors=False,
        )
        return env

    def collect_seed(
        self,
        seed: int,
        need_success: bool,
        max_attempts: int = 50000,
        agent_override: Optional[Any] = None,
    ) -> Optional[List[float]]:
        """Collect single success or failure for a seed.

        Args:
            seed: Random seed
            need_success: If True, find success; if False, find failure
            max_attempts: Maximum attempts before giving up
            agent_override: Optional agent to use instead of the default agent

        Returns:
            Action list if found, None otherwise
        """
        # Create or refresh environment for this seed
        if self.env is None or self.current_seed != seed:
            if self.env is not None:
                self.env.close()
            self.env = self._create_env(seed)
            self.current_seed = seed

        # Set up agent
        obs, _ = self.env.reset(seed=seed)
        agent = agent_override or self.agent
        agent.set_action_space(self.env.action_space)
        if hasattr(agent, "set_evaluator"):
            agent.set_evaluator(lambda action: self._evaluate_action(seed, action))

        # Try to find a solution
        for attempt in range(max_attempts):
            attempt_seed = seed + attempt + 1
            agent.set_seed(attempt_seed)

            try:
                # Get action from agent
                raw_action = agent.get_action(obs)
                action_list = _normalize_action(
                    raw_action, fix_ball_size=self.fix_ball_size
                )

                reward, is_success = self._evaluate_action(seed, action_list)
                if reward is None:
                    continue

                # Check if this matches what we need
                if need_success and is_success:
                    if self.verify_mode:
                        if self._verify_action(seed, action_list):
                            return action_list
                    else:
                        return action_list
                elif not need_success and not is_success:
                    if self.verify_mode:
                        if not self._verify_action(seed, action_list):
                            return action_list
                    else:
                        return action_list

            except Exception:
                continue

        return None

    def _evaluate_action(
        self, seed: int, action: Sequence[Any]
    ) -> Tuple[Optional[float], bool]:
        """Validate and run a single action, returning (reward, success)."""
        if self.env is None:
            return None, False

        try:
            action_list = _normalize_action(action, fix_ball_size=self.fix_ball_size)
            action_tuples = _action_to_tuples(action_list)
        except Exception:
            return None, False

        validation = self.env.validate_action(action_tuples)
        if validation.get("invalid", False):
            return None, False

        self.env.reset(seed=seed)
        _, reward, _, _, info = self.env.step(action_tuples)
        return reward, bool(info.get("success", False))

    def _verify_action(self, seed: int, action: List[float]) -> bool:
        """Verify an action by running it again in a fresh environment.

        This is used in verify_mode to double-check results.

        Args:
            seed: Random seed
            action: Action to verify

        Returns:
            True if action succeeds, False otherwise
        """
        try:
            level = load_level(self.level_name, seed=seed)
            verify_env = InterphyreEnv(
                level,
                config=self.config,
                observation_type="physics_state",
                action_type="continuous",
                discrete_colors=False,
            )
            verify_env.reset(seed=seed)
            action_tuples = _action_to_tuples(action)
            _, _, _, _, info = verify_env.step(action_tuples)
            verify_env.close()
            return bool(info.get("success", False))
        except Exception:
            return False

    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()
            self.env = None
            self.current_seed = None


@dataclass(frozen=True)
class CollectConfig:
    """Configuration for data collection."""

    level_name: str
    output_dir: Path
    min_seed: int
    max_seed: int
    agent_type: str
    max_attempts: int
    log_frequency: int
    fix_ball_size: Optional[float]
    verify_mode: bool
    overwrite: bool
    workers: int = 1
    explicit_seeds: Optional[List[int]] = (
        None  # If provided, only collect these specific seeds
    )
    cem_population: int = 128
    cem_elite_frac: float = 0.1
    cem_iterations: int = 5


def _collect_seed_worker(
    args: Tuple[
        str,
        int,
        int,
        Optional[float],
        bool,
        str,
        int,
        float,
        int,
    ],
) -> Tuple[int, Optional[List[float]], Optional[List[float]]]:
    """Worker function for parallel seed collection.

    Args:
        args: Tuple of (level_name, seed, max_attempts, fix_ball_size, verify_mode)

    Returns:
        Tuple of (seed, success_action, failure_action)
    """
    (
        level_name,
        seed,
        max_attempts,
        fix_ball_size,
        verify_mode,
        agent_type,
        cem_population,
        cem_elite_frac,
        cem_iterations,
    ) = args

    # Create collector for this worker
    sim_config = SimulationConfig(max_steps=1000, verify_solutions=verify_mode)
    if agent_type == "cem":
        success_agent = CEMAgent(
            seed=seed,
            population=cem_population,
            elite_frac=cem_elite_frac,
            iterations=cem_iterations,
        )
        failure_agent = RandomAgent(seed=seed)
    else:
        success_agent = RandomAgent(seed=seed)
        failure_agent = success_agent
    collector = DataCollector(
        level_name,
        sim_config,
        success_agent,
        verify_mode=verify_mode,
        fix_ball_size=fix_ball_size,
    )

    # Collect both success and failure
    success_action = collector.collect_seed(
        seed, need_success=True, max_attempts=max_attempts
    )
    failure_action = collector.collect_seed(
        seed,
        need_success=False,
        max_attempts=max_attempts,
        agent_override=failure_agent,
    )

    collector.close()

    return seed, success_action, failure_action


def verify_solutions(
    level_name: str,
    output_dir: Path,
    min_seed: int,
    max_seed: int,
    log_frequency: int = 100,
) -> Dict[str, Any]:
    """Verify existing solutions without regenerating.

    Args:
        level_name: Level name
        output_dir: Output directory
        min_seed: Minimum seed
        max_seed: Maximum seed
        log_frequency: Log every N seeds

    Returns:
        Dictionary with verification results
    """
    storage = SolutionStorage(level_name, output_dir)
    success_map, failure_map = storage.load()

    _log(f"Loaded {len(success_map)} successes, {len(failure_map)} failures")

    # Check for overlaps
    errors = storage.validate_consistency(success_map, failure_map)
    if errors:
        _log(f"Found {len(errors)} consistency errors:")
        for error in errors[:10]:
            _log(f"  {error}")

    # Verify each solution
    wrong_successes: List[int] = []
    wrong_failures: List[int] = []
    missing_successes: List[int] = []
    missing_failures: List[int] = []

    config = SimulationConfig(max_steps=1000)
    collector = DataCollector(level_name, config, RandomAgent(), verify_mode=False)

    total_seeds = max_seed - min_seed + 1
    pbar = tqdm(total=total_seeds, desc="Verifying", unit="seed", file=sys.stderr)

    for seed in range(min_seed, max_seed + 1):
        seed_str = str(seed)

        if seed_str in success_map:
            if not collector._verify_action(seed, success_map[seed_str]):
                wrong_successes.append(seed)
        else:
            missing_successes.append(seed)

        if seed_str in failure_map:
            if collector._verify_action(seed, failure_map[seed_str]):
                wrong_failures.append(seed)
        else:
            missing_failures.append(seed)

        if seed % log_frequency == 0:
            _log(f"  Verified seed {seed}/{max_seed}...")

        pbar.update(1)

    pbar.close()
    collector.close()

    _log("\nVerification Summary:")
    _log(f"  Total seeds checked: {total_seeds}")
    _log(f"  Correct successes: {len(success_map) - len(wrong_successes)}")
    _log(f"  Wrong successes: {len(wrong_successes)}")
    _log(f"  Missing successes: {len(missing_successes)}")
    _log(f"  Correct failures: {len(failure_map) - len(wrong_failures)}")
    _log(f"  Wrong failures: {len(wrong_failures)}")
    _log(f"  Missing failures: {len(missing_failures)}")

    if wrong_successes:
        _log(f"\n  Wrong success seeds (first 20): {wrong_successes[:20]}")
    if wrong_failures:
        _log(f"\n  Wrong failure seeds (first 20): {wrong_failures[:20]}")

    return {
        "level": level_name,
        "total_seeds": total_seeds,
        "correct_successes": len(success_map) - len(wrong_successes),
        "wrong_successes": wrong_successes,
        "missing_successes": missing_successes,
        "correct_failures": len(failure_map) - len(wrong_failures),
        "wrong_failures": wrong_failures,
        "missing_failures": missing_failures,
        "consistency_errors": errors,
    }


def collect_for_level(config: CollectConfig) -> Dict[str, Any]:
    """Collect contrastive data for a level across a seed range.

    Args:
        config: Collection configuration

    Returns:
        Dictionary with collection results
    """
    storage = SolutionStorage(config.level_name, config.output_dir)
    success_map, failure_map = storage.load()

    _log(
        f"Loaded {len(success_map)} existing successes, {len(failure_map)} existing failures"
    )

    # Determine which seeds need processing
    if config.explicit_seeds is not None:
        base_seeds = list(config.explicit_seeds)
    else:
        base_seeds = list(range(config.min_seed, config.max_seed + 1))

    if config.overwrite:
        seeds_to_process = base_seeds
    else:
        # Verify existing solutions and drop invalid ones so they get regenerated.
        verifier = DataCollector(
            config.level_name,
            SimulationConfig(max_steps=1000),
            RandomAgent(seed=0),
            verify_mode=False,
        )
        for seed in base_seeds:
            seed_str = str(seed)
            if seed_str in success_map:
                if not verifier._verify_action(seed, success_map[seed_str]):
                    _log(f"  Replacing invalid success for seed {seed}")
                    del success_map[seed_str]
            if seed_str in failure_map:
                if verifier._verify_action(seed, failure_map[seed_str]):
                    _log(f"  Replacing invalid failure for seed {seed}")
                    del failure_map[seed_str]
        verifier.close()

        seeds_to_process = [
            seed
            for seed in base_seeds
            if str(seed) not in success_map or str(seed) not in failure_map
        ]

    if not seeds_to_process:
        _log("All seeds already have both success and failure - nothing to do!")
        return {
            "level": config.level_name,
            "successes": len(success_map),
            "failures": len(failure_map),
            "consistency_errors": 0,
        }

    _log(f"Processing {len(seeds_to_process)} seeds for {config.level_name}...")
    _log(f"Using {config.workers} worker(s)")

    missing_successes: List[int] = []
    missing_failures: List[int] = []

    if config.workers > 1:
        # Parallel collection using multiprocessing
        worker_args = [
            (
                config.level_name,
                seed,
                config.max_attempts,
                config.fix_ball_size,
                config.verify_mode,
                config.agent_type,
                config.cem_population,
                config.cem_elite_frac,
                config.cem_iterations,
            )
            for seed in seeds_to_process
        ]

        with mp.Pool(processes=config.workers) as pool:
            pbar = tqdm(
                total=len(seeds_to_process), desc="Seeds", unit="seed", file=sys.stderr
            )

            # Process seeds in parallel with progress tracking
            for result in pool.imap_unordered(_collect_seed_worker, worker_args):
                seed, success_action, failure_action = result
                seed_str = str(seed)

                if success_action:
                    success_map[seed_str] = success_action
                elif config.overwrite and seed_str in success_map:
                    del success_map[seed_str]
                if not success_action:
                    missing_successes.append(seed)

                if failure_action:
                    failure_map[seed_str] = failure_action
                elif config.overwrite and seed_str in failure_map:
                    del failure_map[seed_str]
                if not failure_action:
                    missing_failures.append(seed)

                # Save periodically (every result)
                if pbar.n % 1 == 0:
                    storage.save(success_map, failure_map)
                    storage.regenerate_csv(success_map, failure_map)

                pbar.update(1)

            pbar.close()

        if missing_successes:
            _log(
                "Missing success for seeds: "
                + ", ".join(str(seed) for seed in sorted(missing_successes))
            )
        if missing_failures:
            _log(
                "Missing failure for seeds: "
                + ", ".join(str(seed) for seed in sorted(missing_failures))
            )

    else:
        # Sequential collection (original code path)
        sim_config = SimulationConfig(
            max_steps=1000, verify_solutions=config.verify_mode
        )
        if config.agent_type == "cem":
            success_agent = CEMAgent(
                seed=42,
                population=config.cem_population,
                elite_frac=config.cem_elite_frac,
                iterations=config.cem_iterations,
            )
            failure_agent = RandomAgent(seed=42)
        else:
            success_agent = RandomAgent(seed=42)
            failure_agent = success_agent
        collector = DataCollector(
            config.level_name,
            sim_config,
            success_agent,
            verify_mode=config.verify_mode,
            fix_ball_size=config.fix_ball_size,
        )

        pbar = tqdm(
            total=len(seeds_to_process), desc="Seeds", unit="seed", file=sys.stderr
        )

        for seed in seeds_to_process:
            seed_str = str(seed)
            if config.overwrite:
                success_map.pop(seed_str, None)
                failure_map.pop(seed_str, None)

            # Check if we need success
            if config.overwrite or seed_str not in success_map:
                if seed % config.log_frequency == 0:
                    _log(f"  Finding success for seed {seed}...")

                action = collector.collect_seed(
                    seed, need_success=True, max_attempts=config.max_attempts
                )
                if action:
                    success_map[seed_str] = action
                else:
                    _log(f"  Warning: Could not find success for seed {seed}")
                    missing_successes.append(seed)

            # Check if we need failure
            if config.overwrite or seed_str not in failure_map:
                if seed % config.log_frequency == 0:
                    _log(f"  Finding failure for seed {seed}...")

                action = collector.collect_seed(
                    seed,
                    need_success=False,
                    max_attempts=config.max_attempts,
                    agent_override=failure_agent,
                )
                if action:
                    failure_map[seed_str] = action
                else:
                    _log(f"  Warning: Could not find failure for seed {seed}")
                    missing_failures.append(seed)

            # Save periodically (every result)
            if pbar.n % 1 == 0:
                storage.save(success_map, failure_map)
                storage.regenerate_csv(success_map, failure_map)

            pbar.update(1)

        pbar.close()
        collector.close()

        if missing_successes:
            _log(
                "Missing success for seeds: "
                + ", ".join(str(seed) for seed in sorted(missing_successes))
            )
        if missing_failures:
            _log(
                "Missing failure for seeds: "
                + ", ".join(str(seed) for seed in sorted(missing_failures))
            )

    _log("Saving final solutions...")
    storage.save(success_map, failure_map)
    storage.regenerate_csv(success_map, failure_map)

    _log(f"Complete: {len(success_map)} successes, {len(failure_map)} failures")

    # Final consistency check
    errors = storage.validate_consistency(success_map, failure_map)
    if errors:
        _log(f"WARNING: Found {len(errors)} consistency errors in final data!")

    return {
        "level": config.level_name,
        "successes": len(success_map),
        "failures": len(failure_map),
        "consistency_errors": len(errors),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Collect contrastive training data (v2)"
    )
    parser.add_argument("--level", required=True, help="Level name")
    parser.add_argument("--output-dir", required=True, help="Output directory")

    # Seed specification - either range or explicit list
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument(
        "--min-seed", type=int, help="Minimum seed (use with --max-seed)"
    )
    seed_group.add_argument(
        "--seeds",
        type=str,
        help="Comma-separated list of seeds (e.g., '42,69,123,256')",
    )

    parser.add_argument(
        "--max-seed", type=int, help="Maximum seed (use with --min-seed)"
    )
    parser.add_argument(
        "--max-attempts", type=int, default=50000, help="Max attempts per seed"
    )
    parser.add_argument(
        "--log-frequency", type=int, default=100, help="Log every N seeds"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify solutions, don't regenerate",
    )
    verify_group = parser.add_mutually_exclusive_group()
    verify_group.add_argument(
        "--verify-mode",
        action="store_true",
        help="Enable double-verification (default)",
    )
    verify_group.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable double-verification (faster but riskier)",
    )
    parser.add_argument(
        "--fix-ball-size",
        type=float,
        default=None,
        help="Fix ball size/radius to this value (optional)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, use 4-8 for speedup)",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Keep existing successes/failures (default: overwrite)",
    )
    parser.add_argument(
        "--agent",
        choices=["random", "cem"],
        default="random",
        help="Action search strategy for successes (default: random)",
    )
    parser.add_argument(
        "--cem-population",
        type=int,
        default=128,
        help="CEM population size per iteration (default: 128)",
    )
    parser.add_argument(
        "--cem-elite-frac",
        type=float,
        default=0.1,
        help="CEM elite fraction (default: 0.1)",
    )
    parser.add_argument(
        "--cem-iterations",
        type=int,
        default=5,
        help="CEM iterations per attempt (default: 5)",
    )
    return parser


def main() -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Determine seed range
    if args.seeds:
        # Parse comma-separated seed list
        try:
            explicit_seeds = [int(s.strip()) for s in args.seeds.split(",")]
            min_seed = min(explicit_seeds)
            max_seed = max(explicit_seeds)
            _log(
                f"Using explicit seed list ({len(explicit_seeds)} seeds): {explicit_seeds}"
            )
        except ValueError as e:
            _log(f"Error parsing seeds: {e}")
            _log("Seeds must be comma-separated integers (e.g., '42,69,123')")
            return
    else:
        # Use range
        if args.max_seed is None:
            _log("Error: --max-seed is required when using --min-seed")
            parser.print_help()
            return
        min_seed = args.min_seed
        max_seed = args.max_seed
        explicit_seeds = None

    verify_mode = not args.no_verify
    if args.verify_mode:
        verify_mode = True

    if args.verify_only:
        result = verify_solutions(
            level_name=args.level,
            output_dir=output_dir,
            min_seed=min_seed,
            max_seed=max_seed,
            log_frequency=args.log_frequency,
        )
    else:
        config = CollectConfig(
            level_name=args.level,
            output_dir=output_dir,
            min_seed=min_seed,
            max_seed=max_seed,
            agent_type=args.agent,
            max_attempts=args.max_attempts,
            log_frequency=args.log_frequency,
            fix_ball_size=args.fix_ball_size,
            verify_mode=verify_mode,
            overwrite=not args.no_overwrite,
            workers=args.workers,
            explicit_seeds=explicit_seeds,
            cem_population=args.cem_population,
            cem_elite_frac=args.cem_elite_frac,
            cem_iterations=args.cem_iterations,
        )
        result = collect_for_level(config)

    _log(f"\nResult: {result}")


if __name__ == "__main__":
    main()
