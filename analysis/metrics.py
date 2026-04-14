"""Metrics for evaluating AXIOM and baselines."""

import numpy as np


def cumulative_reward(rewards: np.ndarray) -> float:
    """Total reward over all steps."""
    return float(np.sum(rewards))


def moving_average_reward(rewards: np.ndarray, window: int = 1000) -> np.ndarray:
    """Per-step reward smoothed with a moving average (as in Figure 3 of the paper)."""
    if len(rewards) < window:
        return np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
    kernel = np.ones(window) / window
    return np.convolve(rewards, kernel, mode="valid")


def reward_summary(rewards: np.ndarray) -> dict:
    """Compute summary statistics matching Table 1 format."""
    return {
        "cumulative": cumulative_reward(rewards),
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
    }
