import numpy as np
from typing import Any, Optional
import gymnasium as gym


class RandomAgent:
    """
    Random agent that samples actions uniformly from the environment's action space.
    Used for establishing baseline performance metrics.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the random agent.

        Args:
            seed: Random seed for reproducibility
        """
        self.action_space = None

    def set_action_space(self, action_space: gym.Space):
        """
        Set the action space for the agent.

        Args:
            action_space: Gymnasium action space
        """
        self.action_space = action_space

    def get_action(self, observation: Any) -> np.ndarray:
        """
        Sample a random action from the action space with bounds-aware sampling.

        Args:
            observation: Current environment observation (not used by random agent)

        Returns:
            Random action as numpy array
        """
        if self.action_space is None:
            raise ValueError("Action space not set. Call set_action_space() first.")

        # Sample from action space but ensure bounds compliance
        action = self.action_space.sample()

        # Ensure the action respects the world bounds considering radius
        # Action space is [-5, 5] but valid placement is [-5+radius, 5-radius]
        if len(action) > 0:
            # Reshape to (num_objects, 3) for easier processing
            num_objects = len(action) // 3
            action_reshaped = action.reshape(num_objects, 3)

            for i in range(num_objects):
                x, y, radius = action_reshaped[i]
                # Clamp coordinates to valid bounds considering radius
                min_coord = -5.0 + radius
                max_coord = 5.0 - radius
                x = np.clip(x, min_coord, max_coord)
                y = np.clip(y, min_coord, max_coord)
                action_reshaped[i] = [x, y, radius]

            action = action_reshaped.flatten()

        return action

    def set_seed(self, seed: int):
        """
        Set a new random seed for the agent.

        Args:
            seed: New random seed
        """
        if self.action_space is None:
            raise ValueError("set_seed() called before set_action_space()")
        self.action_space.seed(seed)

    def reset(self):
        """Reset the agent state (no-op for random agent)."""
        pass
