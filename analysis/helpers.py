"""Analysis helpers: load CSV results, compute summary statistics, group by parameter."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_results_dir(results_dir: str | Path) -> pd.DataFrame:
    """Load all CSVs in a directory into a single DataFrame.

    Each CSV gets additional columns extracted from the filename:
    - param_value: the parameter value (parsed from filename)
    - seed: the seed number (parsed from filename)

    Expected filename pattern: {param}_{value}_seed{N}.csv
    """
    results_dir = Path(results_dir)
    frames = []

    for csv_path in sorted(results_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        parts = csv_path.stem.rsplit("_seed", maxsplit=1)
        if len(parts) == 2:
            param_value_str, seed_str = parts
            df["source_file"] = csv_path.name
            df["param_value"] = param_value_str
            df["seed"] = int(seed_str)
        else:
            df["source_file"] = csv_path.name
            df["param_value"] = csv_path.stem
            df["seed"] = 0
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def moving_average(series: np.ndarray | pd.Series, window: int = 1000) -> np.ndarray:
    """Compute moving average matching the paper's Figure 3 convention."""
    arr = np.asarray(series, dtype=float)
    if len(arr) < window:
        return np.cumsum(arr) / np.arange(1, len(arr) + 1)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def summary_stats(
    df: pd.DataFrame,
    reward_col: str = "Reward",
    group_cols: list[str] | None = None,
    last_n: int = 1000,
) -> pd.DataFrame:
    """Compute per-group summary statistics over the last N steps.

    Returns a DataFrame with columns: cumulative_reward, mean_reward,
    std_reward, and (if present) final_num_components.
    """
    if group_cols is None:
        group_cols = ["param_value"]

    def _summarize(g: pd.DataFrame) -> pd.Series:
        tail = g.tail(last_n)
        result = {
            "cumulative_reward": tail[reward_col].sum(),
            "mean_reward": tail[reward_col].mean(),
            "std_reward": tail[reward_col].std(),
            "num_steps": len(g),
        }
        if "Num Components" in g.columns:
            result["final_num_components"] = g["Num Components"].iloc[-1]
        return pd.Series(result)

    return df.groupby(group_cols + ["seed"]).apply(_summarize).reset_index()
