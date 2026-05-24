import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class EpisodeResult:
    """Results from a single episode."""

    success: bool
    attempts: int
    steps_to_success: int
    total_steps: int
    reward: float
    episode_time: float
    action_space_size: int
    episode_type: str  # 'valid' or 'invalid'


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics across multiple episodes."""

    success_rate: float
    avg_attempts_to_success: float
    avg_steps_to_success: float
    avg_total_steps: float
    avg_reward: float
    avg_episode_time: float
    action_space_size: int
    total_episodes: int
    successful_episodes: int
    failed_episodes: int

    def __str__(self):
        return f"""
Evaluation Metrics:
==================
Success Rate: {self.success_rate:.2%}
Average Attempts to Success: {self.avg_attempts_to_success:.2f}
Average Steps to Success: {self.avg_steps_to_success:.2f}
Average Total Steps: {self.avg_total_steps:.2f}
Average Reward: {self.avg_reward:.3f}
Average Episode Time: {self.avg_episode_time:.3f}s
Action Space Size: {self.action_space_size}
Total Episodes: {self.total_episodes}
Successful Episodes: {self.successful_episodes}
Failed Episodes: {self.failed_episodes}
"""


class Evaluator:
    """Evaluates agent performance across multiple episodes."""

    def __init__(self):
        self.results: List[EpisodeResult] = []

    def add_episode_result(self, result: EpisodeResult):
        """Add a single episode result."""
        self.results.append(result)

    def get_metrics(self) -> EvaluationMetrics:
        """Calculate aggregated metrics from all episode results."""
        if not self.results:
            raise ValueError("No episode results available")

        # Separate valid episodes
        valid_results = [r for r in self.results if r.episode_type == "valid"]

        # Calculate metrics only on valid episodes
        successful_results = [r for r in valid_results if r.success]
        failed_results = [r for r in valid_results if not r.success]

        success_rate = (
            len(successful_results) / len(valid_results) if valid_results else 0
        )

        # Calculate averages (only on valid episodes)
        avg_attempts = (
            np.mean([r.attempts for r in valid_results]) if valid_results else 0
        )
        avg_steps_to_success = (
            np.mean([r.steps_to_success for r in successful_results])
            if successful_results
            else 0
        )
        avg_total_steps = (
            np.mean([r.total_steps for r in valid_results]) if valid_results else 0
        )
        avg_reward = np.mean([r.reward for r in valid_results]) if valid_results else 0
        avg_episode_time = (
            np.mean([r.episode_time for r in valid_results]) if valid_results else 0
        )

        # Action space size should be the same for all episodes
        action_space_size = self.results[0].action_space_size if self.results else 0

        return EvaluationMetrics(
            success_rate=success_rate,
            avg_attempts_to_success=avg_attempts,
            avg_steps_to_success=avg_steps_to_success,
            avg_total_steps=avg_total_steps,
            avg_reward=avg_reward,
            avg_episode_time=avg_episode_time,
            action_space_size=action_space_size,
            total_episodes=len(self.results),
            successful_episodes=len(successful_results),
            failed_episodes=len(failed_results),
        )

    def reset(self):
        """Clear all episode results."""
        self.results.clear()

    def save_results(self, filepath: str):
        """Save results to a CSV file."""
        import pandas as pd

        # Convert results to DataFrame
        data = []
        for r in self.results:
            data.append(
                {
                    "success": r.success,
                    "attempts": r.attempts,
                    "steps_to_success": r.steps_to_success,
                    "total_steps": r.total_steps,
                    "reward": r.reward,
                    "episode_time": r.episode_time,
                    "action_space_size": r.action_space_size,
                    "episode_type": r.episode_type,
                }
            )

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
