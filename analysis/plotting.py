"""Plotting utilities for learning curves, ablations, and interpretability figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from analysis.metrics import moving_average_reward

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def setup_style():
    """Set consistent plot aesthetics."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams["figure.dpi"] = 150


def plot_learning_curves(
    results: dict[str, dict[str, np.ndarray]],
    game: str,
    window: int = 1000,
    save: bool = True,
):
    """
    Plot moving-average reward curves for multiple agents on one game.

    Args:
        results: {agent_name: {"rewards": (seeds, steps) array}}
        game: game name for title and filename
        window: moving average window size
        save: whether to save to results/figures/
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    for agent_name, data in results.items():
        rewards = data["rewards"]
        curves = np.array([moving_average_reward(r, window) for r in rewards])
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        steps = np.arange(len(mean)) + window
        ax.plot(steps, mean, label=agent_name)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2)

    ax.set_xlabel("Step")
    ax.set_ylabel(f"Average Reward ({window})")
    ax.set_title(game.capitalize())
    ax.legend()
    fig.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / f"learning_curve_{game}.png")
        print(f"Saved: {FIGURES_DIR / f'learning_curve_{game}.png'}")

    return fig, ax


def plot_cumulative_reward_table(
    results: dict[str, dict[str, float]],
    save: bool = True,
):
    """
    Create a bar chart of cumulative rewards across games (replicating Table 1).

    Args:
        results: {agent_name: {game_name: cumulative_reward}}
    """
    setup_style()
    # TODO: implement when results are available
    raise NotImplementedError


if __name__ == "__main__":
    print("Run with: make figures")
    print("No results to plot yet.")
