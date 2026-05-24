import argparse
import os
import sys
import time
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import importlib

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interphyre.environment import InterphyreEnv
from interphyre.config import SimulationConfig
from interphyre.levels import _level_registry, load_level
from agents.random_agent import RandomAgent
from agents.evaluation import EpisodeResult, Evaluator


def get_action_space_size(env) -> int:
    """Get the action space size from environment."""
    try:
        if (
            hasattr(env, "action_space")
            and env.action_space is not None
            and hasattr(env.action_space, "shape")
        ):
            return env.action_space.shape[0]
    except (AttributeError, TypeError, IndexError):
        # Action space not available or malformed, return default
        return 3  # Default action space size for red ball (x, y, radius)
    return 0


def get_available_levels() -> List[str]:
    """Get all available level names."""

    levels_dir = os.path.join(os.path.dirname(__file__), "..", "interphyre", "levels")
    level_files = [
        f for f in os.listdir(levels_dir) if f.endswith(".py") and f != "__init__.py"
    ]

    for level_file in level_files:
        level_name = level_file[:-3]  # Remove .py extension
        try:
            importlib.import_module(f"interphyre.levels.{level_name}")
        except Exception as e:
            print(f"Warning: Could not import {level_name}: {e}")

    return list(_level_registry.keys())


def run_level_baseline(
    level_name: str,
    num_episodes: int = 100,
    max_attempts: int = 100,  # Maximum attempts per episode
    seed: int = 42,
    image_size: tuple = (128, 128),
    cycle_seeds: bool = False,
) -> Dict[str, Any]:
    """Run baseline evaluation for a single level.

    Each episode consists of up to max_attempts valid attempts:
    1. Agent provides action (object placement)
    2. If invalid action: sample new action without resetting environment (until valid)
    3. If valid action: count as attempt and run full simulation
    4. If success: episode ends
    5. If timeout: reset environment with same seed and try again (up to max_attempts)

    All episodes and attempts contain only valid actions.
    Each episode uses a unique environment seed, retries use the same seed.
    """
    print(f"Evaluating level: {level_name}")

    try:
        # Load level
        level = load_level(level_name)

        # Create environment
        config = SimulationConfig(max_steps=1000)
        env = InterphyreEnv(
            level=level,
            config=config,
            observation_type="image",
            action_type="continuous",
            image_size=image_size,
            image_ppm=60.0,
            discrete_colors=False,
        )

        # Create agent
        agent = RandomAgent(seed=seed)
        agent.set_action_space(env.action_space)

        # Create evaluator
        evaluator = Evaluator()

        print(f"  Action objects: {level.action_objects}")
        print(f"  Total objects: {len(level.objects)}")
        action_space_size = get_action_space_size(env)
        print(f"  Action space size: {action_space_size}")

        # Run episodes until we get the target number of valid episodes
        valid_episodes = 0
        total_episodes_run = 0

        # Create progress bar for episodes
        pbar = tqdm(total=num_episodes, desc=f"Episodes ({level_name})", unit="ep")

        while valid_episodes < num_episodes:
            episode_start_time = time.time()
            success = False
            attempts = 0
            steps_to_success = 0

            # Reset environment with unique seed for this episode
            episode_seed = seed + total_episodes_run
            obs, info = env.reset(seed=episode_seed)

            # Try up to max_attempts per episode
            for attempt in range(max_attempts):
                # Use different seed for each attempt if cycling is enabled
                if cycle_seeds:
                    attempt_seed = episode_seed + attempt
                    obs, info = env.reset(seed=attempt_seed)
                else:
                    # Reset environment for this attempt with the same episode seed
                    obs, info = env.reset(seed=episode_seed)

                # Keep trying until we get a valid action (with safety limit)
                max_invalid_retries = 1000  # Safety limit to prevent infinite loops
                invalid_retries = 0

                while invalid_retries < max_invalid_retries:
                    # Get action from agent (with different seed for each attempt)
                    attempt_seed = episode_seed + attempt + 1
                    agent.set_seed(attempt_seed)
                    action = agent.get_action(obs)

                    # Execute action (runs full simulation to completion)
                    obs, reward, terminated, truncated, info = env.step(action)

                    # Check if action was valid
                    if terminated and reward < 0:  # Invalid action
                        invalid_retries += 1
                        # Reset environment and try another action
                        obs, info = env.reset(seed=episode_seed)
                        continue
                    else:
                        # Valid action found - count as attempt and break out of retry loop
                        attempts += 1
                        break

                if invalid_retries >= max_invalid_retries:
                    print(
                        f"  Warning: Hit safety limit for invalid actions in episode {total_episodes_run}"
                    )
                    break

                # Check if successful
                if terminated and reward > 0:  # Success
                    success = True
                    steps_to_success = info.get("step_count", 0)
                    break  # Success! End episode
                # If not successful, continue to next attempt (environment already reset above)

            # Episode completed (either success or max attempts reached)
            valid_episodes += 1
            total_episodes_run += 1
            episode_time = time.time() - episode_start_time

            # Record episode result
            result = EpisodeResult(
                success=success,
                attempts=attempts,
                steps_to_success=steps_to_success,
                total_steps=info.get("step_count", 0),
                reward=reward,
                episode_time=episode_time,
                action_space_size=get_action_space_size(env),
                episode_type="valid",  # All episodes are valid now
            )
            evaluator.add_episode_result(result)

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix(
                {
                    "success": success,
                    "attempts": attempts,
                    "success_rate": f"{sum(1 for r in evaluator.results if r.success) / len(evaluator.results) * 100:.1f}%",
                }
            )

        # Close progress bar
        pbar.close()

        # Get metrics
        metrics = evaluator.get_metrics()

        # Close environment
        env.close()

        print(f"  Episodes completed: {valid_episodes}")
        print(f"  Success rate: {metrics.success_rate:.2%}")
        print(f"  Average attempts per episode: {metrics.avg_attempts_to_success:.2f}")

        # Return level results
        return {
            "level_name": level_name,
            "success_rate": metrics.success_rate,
            "avg_attempts": metrics.avg_attempts_to_success,
            "avg_steps_to_success": metrics.avg_steps_to_success,
            "avg_total_steps": metrics.avg_total_steps,
            "avg_reward": metrics.avg_reward,
            "avg_episode_time": metrics.avg_episode_time,
            "successful_episodes": metrics.successful_episodes,
            "failed_episodes": metrics.failed_episodes,
            "total_episodes": valid_episodes,
            "target_episodes": num_episodes,
            "action_space_size": metrics.action_space_size,
            "action_objects": level.action_objects,
            "total_objects": len(level.objects),
        }

    except Exception as e:
        print(f"  Error: {e}")
        return {
            "level_name": level_name,
            "error": str(e),
            "success_rate": 0.0,
            "avg_attempts": 0.0,
            "avg_steps_to_success": 0.0,
            "avg_total_steps": 0.0,
            "avg_reward": 0.0,
            "avg_episode_time": 0.0,
            "successful_episodes": 0,
            "failed_episodes": 0,
            "valid_episodes": 0,
            "invalid_episodes": 0,
            "total_episodes_run": 0,
            "target_valid_episodes": num_episodes,
            "action_space_size": 0,
            "action_objects": [],
            "total_objects": 0,
        }


def run_multiple_trials(
    level_name: str,
    num_trials: int = 10,
    num_episodes_per_trial: int = 100,
    max_attempts: int = 100,
    seed: int = 42,
    image_size: tuple = (128, 128),
    cycle_seeds: bool = False,
) -> Dict[str, Any]:
    """Run multiple trials for distribution analysis."""
    print(f"Running {num_trials} trials for level: {level_name}")
    print(f"Episodes per trial: {num_episodes_per_trial}")

    trial_results = []

    for trial_id in range(num_trials):
        print(f"  Trial {trial_id + 1}/{num_trials}")
        result = run_level_baseline(
            level_name=level_name,
            num_episodes=num_episodes_per_trial,
            max_attempts=max_attempts,
            seed=seed + trial_id * 1000,
            image_size=image_size,
            cycle_seeds=cycle_seeds,
        )
        trial_results.append(result)

    # Calculate distribution statistics
    success_rates = [r["success_rate"] for r in trial_results]
    avg_attempts = [r["avg_attempts"] for r in trial_results]
    avg_steps = [r["avg_steps_to_success"] for r in trial_results]

    def calc_stats(values):
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
            "q25": np.percentile(values, 25),
            "q75": np.percentile(values, 75),
        }

    distribution_stats = {
        "level_name": level_name,
        "num_trials": num_trials,
        "episodes_per_trial": num_episodes_per_trial,
        "success_rate": calc_stats(success_rates),
        "avg_attempts": calc_stats(avg_attempts),
        "avg_steps_to_success": calc_stats(avg_steps),
        "trial_results": trial_results,
    }

    return distribution_stats


def run_all_levels_baseline(
    level_names: Optional[List[str]] = None,
    num_episodes: int = 100,
    max_attempts: int = 100,
    seed: int = 42,
    image_size: tuple = (128, 128),
    cycle_seeds: bool = False,
    save_results: bool = True,
    results_file: str = "results/all_levels_baseline_results.csv",
) -> None:
    """Run baseline evaluation across multiple levels."""
    if level_names is None:
        level_names = get_available_levels()

    print(f"Running baseline evaluation across {len(level_names)} levels...")
    print(f"Episodes per level: {num_episodes}, Max attempts: {max_attempts}")
    print(f"Image size: {image_size}, Seed: {seed}")
    print()

    level_results = []
    successful_levels = 0
    failed_levels = 0

    # Create progress bar for levels
    level_pbar = tqdm(level_names, desc="Levels", unit="level")

    for i, level_name in enumerate(level_pbar):
        level_pbar.set_description(f"Level: {level_name}")
        result = run_level_baseline(
            level_name=level_name,
            num_episodes=num_episodes,
            max_attempts=max_attempts,
            seed=seed,
            image_size=image_size,
            cycle_seeds=cycle_seeds,
        )
        level_results.append(result)

        if "error" not in result:
            successful_levels += 1
            level_pbar.set_postfix(
                {
                    "success_rate": f"{result['success_rate']:.1%}",
                    "completed": f"{successful_levels}/{len(level_names)}",
                }
            )
        else:
            failed_levels += 1
            level_pbar.set_postfix(
                {
                    "status": "ERROR",
                    "completed": f"{successful_levels}/{len(level_names)}",
                }
            )

    # Calculate summary statistics
    successful_results = [r for r in level_results if "error" not in r]

    if successful_results:
        success_rates = [r["success_rate"] for r in successful_results]
        avg_attempts = [r["avg_attempts"] for r in successful_results]
        avg_steps_to_success = [r["avg_steps_to_success"] for r in successful_results]
        avg_total_steps = [r["avg_total_steps"] for r in successful_results]
        avg_rewards = [r["avg_reward"] for r in successful_results]
        avg_episode_times = [r["avg_episode_time"] for r in successful_results]

        def calc_stats(values):
            return {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values),
                "q25": np.percentile(values, 25),
                "q75": np.percentile(values, 75),
            }

        summary_stats = {
            "success_rate": calc_stats(success_rates),
            "avg_attempts": calc_stats(avg_attempts),
            "avg_steps_to_success": calc_stats(avg_steps_to_success),
            "avg_total_steps": calc_stats(avg_total_steps),
            "avg_reward": calc_stats(avg_rewards),
            "avg_episode_time": calc_stats(avg_episode_times),
        }
    else:
        summary_stats = {}

    # Print summary
    print("=" * 80)
    print("ALL LEVELS BASELINE SUMMARY")
    print("=" * 80)
    print(f"Total levels evaluated: {len(level_names)}")
    print(f"Successful evaluations: {successful_levels}")
    print(f"Failed evaluations: {failed_levels}")

    if successful_results:
        print("\nSuccess Rate Across Levels:")
        print(
            f"  Mean: {summary_stats['success_rate']['mean']:.2%} ± {summary_stats['success_rate']['std']:.2%}"
        )
        print(
            f"  Range: [{summary_stats['success_rate']['min']:.2%}, {summary_stats['success_rate']['max']:.2%}]"
        )
        print(f"  Median: {summary_stats['success_rate']['median']:.2%}")

        print("\nAverage Attempts Across Levels:")
        print(
            f"  Mean: {summary_stats['avg_attempts']['mean']:.2f} ± {summary_stats['avg_attempts']['std']:.2f}"
        )
        print(
            f"  Range: [{summary_stats['avg_attempts']['min']:.2f}, {summary_stats['avg_attempts']['max']:.2f}]"
        )

        print("\nAverage Steps to Success Across Levels:")
        print(
            f"  Mean: {summary_stats['avg_steps_to_success']['mean']:.2f} ± {summary_stats['avg_steps_to_success']['std']:.2f}"
        )
        print(
            f"  Range: [{summary_stats['avg_steps_to_success']['min']:.2f}, {summary_stats['avg_steps_to_success']['max']:.2f}]"
        )

    # Print individual level results
    print("\nIndividual Level Results:")
    print("-" * 80)
    print(
        f"{'Level Name':<20} {'Success Rate':<12} {'Attempts':<10} {'Steps':<8} {'Total':<6} {'Success':<7}"
    )
    print("-" * 80)

    for result in level_results:
        if "error" not in result:
            print(
                f"{result['level_name']:<20} {result['success_rate']:>10.2%} {result['avg_attempts']:>8.2f} {result['avg_steps_to_success']:>6.0f} {result['total_episodes']:>4d} {result['successful_episodes']:>6d}"
            )
        else:
            print(
                f"{result['level_name']:<20} {'ERROR':<12} {'-':<10} {'-':<8} {'-':<6} {'-':<7}"
            )

    # Save results if requested
    if save_results:
        # Convert to DataFrame for CSV export
        df = pd.DataFrame(level_results)
        df.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")

        # Also save summary statistics
        summary_file = results_file.replace(".csv", "_summary.csv")
        if summary_stats:
            summary_df = pd.DataFrame(summary_stats)
            summary_df.to_csv(summary_file, index=False)
            print(f"Summary statistics saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Random Agent Benchmark Tool")

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--level", type=str, help="Single level to evaluate")
    mode_group.add_argument("--levels", nargs="*", help="Specific levels to evaluate")
    mode_group.add_argument(
        "--all-levels", action="store_true", help="Evaluate all available levels"
    )

    # Evaluation parameters
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of episodes per level"
    )
    parser.add_argument(
        "--max-attempts", type=int, default=100, help="Maximum attempts per episode"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--cycle-seeds",
        action="store_true",
        help="Use different seeds for each attempt (generates new level variations)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[128, 128],
        help="Image size (width height)",
    )

    # Multiple trials mode
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of trials for distribution analysis",
    )

    # Output options
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save results to file"
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="results/baseline_results.csv",
        help="Results file path",
    )

    args = parser.parse_args()

    # Convert image size to tuple
    image_size = tuple(args.image_size)

    # Determine mode and run appropriate evaluation
    if args.level:
        if args.trials > 1:
            # Multiple trials mode
            result = run_multiple_trials(
                level_name=args.level,
                num_trials=args.trials,
                num_episodes_per_trial=args.episodes,
                max_attempts=args.max_attempts,
                seed=args.seed,
                image_size=image_size,
                cycle_seeds=args.cycle_seeds,
            )

            # Print distribution analysis
            print("\n" + "=" * 80)
            print("DISTRIBUTION ANALYSIS")
            print("=" * 80)
            print(f"Level: {result['level_name']}")
            print(
                f"Trials: {result['num_trials']}, Episodes per trial: {result['episodes_per_trial']}"
            )
            print("\nSuccess Rate Distribution:")
            print(
                f"  Mean: {result['success_rate']['mean']:.2%} ± {result['success_rate']['std']:.2%}"
            )
            print(
                f"  Range: [{result['success_rate']['min']:.2%}, {result['success_rate']['max']:.2%}]"
            )
            print(f"  Median: {result['success_rate']['median']:.2%}")

            # Save results if requested
            if not args.no_save:
                results_file = f"results/{args.level}_distribution_analysis.csv"
                df = pd.DataFrame(result["trial_results"])
                df.to_csv(results_file, index=False)
                print(f"\nTrial results saved to: {results_file}")
        else:
            # Single level mode
            result = run_level_baseline(
                level_name=args.level,
                num_episodes=args.episodes,
                max_attempts=args.max_attempts,
                seed=args.seed,
                image_size=image_size,
                cycle_seeds=args.cycle_seeds,
            )

            # Save results if requested
            if not args.no_save:
                results_file = f"results/{args.level}_baseline_results.csv"
                df = pd.DataFrame([result])
                df.to_csv(results_file, index=False)
                print(f"\nResults saved to: {results_file}")

    elif args.levels:
        # Specific levels mode
        run_all_levels_baseline(
            level_names=args.levels,
            num_episodes=args.episodes,
            max_attempts=args.max_attempts,
            seed=args.seed,
            image_size=image_size,
            cycle_seeds=args.cycle_seeds,
            save_results=not args.no_save,
            results_file=args.results_file,
        )

    elif args.all_levels:
        # All levels mode
        run_all_levels_baseline(
            level_names=None,
            num_episodes=args.episodes,
            max_attempts=args.max_attempts,
            seed=args.seed,
            image_size=image_size,
            cycle_seeds=args.cycle_seeds,
            save_results=not args.no_save,
            results_file=args.results_file,
        )


if __name__ == "__main__":
    main()
