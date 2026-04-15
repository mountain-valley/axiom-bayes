"""Analysis helpers for AXIOM sweep result loading and aggregation."""

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


def group_by_parameter(
    df: pd.DataFrame,
    parameter_col: str = "param_value",
    reward_col: str = "Reward",
    step_col: str = "Step",
) -> pd.DataFrame:
    """Group sweep data by parameter and step for comparison plots."""
    needed = [parameter_col, step_col, reward_col]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for grouping: {missing}")

    grouped = (
        df.groupby([parameter_col, step_col], dropna=False)[reward_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "reward_mean",
                "std": "reward_std",
                "count": "n",
                parameter_col: "parameter_value",
            }
        )
    )
    grouped["reward_sem"] = grouped["reward_std"] / np.sqrt(grouped["n"].clip(lower=1))
    grouped["reward_ci95"] = 1.96 * grouped["reward_sem"]
    return grouped


def summary_stats(
    df: pd.DataFrame,
    reward_col: str = "Reward",
    group_cols: list[str] | None = None,
    last_n: int = 1000,
) -> pd.DataFrame:
    """Compute per-group summary statistics over the last N steps.

    Returns seed-aggregated metrics with mean/std/95% CI across seeds.
    """
    if group_cols is None:
        group_cols = ["param_value"]

    def _per_seed_summary(g: pd.DataFrame) -> pd.Series:
        tail = g.tail(last_n)
        result = {
            "cumulative_reward": tail[reward_col].sum(),
            "mean_reward": tail[reward_col].mean(),
            "num_steps": len(g),
        }
        if "Num Components" in g.columns:
            result["final_num_components"] = g["Num Components"].iloc[-1]
        return pd.Series(result)

    per_seed = (
        df.groupby(group_cols + ["seed"], dropna=False)
        .apply(_per_seed_summary, include_groups=False)
        .reset_index()
    )

    agg_spec: dict[str, list[str]] = {
        "cumulative_reward": ["mean", "std", "count"],
        "mean_reward": ["mean", "std"],
        "num_steps": ["mean"],
    }
    if "final_num_components" in per_seed.columns:
        agg_spec["final_num_components"] = ["mean", "std"]

    grouped = per_seed.groupby(group_cols, dropna=False).agg(agg_spec)
    grouped.columns = ["_".join(col).rstrip("_") for col in grouped.columns.to_flat_index()]
    grouped = grouped.reset_index()

    n = grouped["cumulative_reward_count"].clip(lower=1)
    grouped["cumulative_reward_sem"] = grouped["cumulative_reward_std"].fillna(0.0) / np.sqrt(n)
    grouped["cumulative_reward_ci95"] = 1.96 * grouped["cumulative_reward_sem"]
    grouped["mean_reward_sem"] = grouped["mean_reward_std"].fillna(0.0) / np.sqrt(n)
    grouped["mean_reward_ci95"] = 1.96 * grouped["mean_reward_sem"]
    return grouped
