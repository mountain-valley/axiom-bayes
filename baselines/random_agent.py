"""Random baseline agent for sanity checking."""

import numpy as np


class RandomAgent:
    """Agent that selects actions uniformly at random."""

    def __init__(self, num_actions: int, seed: int = 0):
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

    def observe(self, observation: np.ndarray, reward: float):
        pass

    def act(self) -> int:
        return self.rng.integers(self.num_actions)
