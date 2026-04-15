"""Test experiment infrastructure and analysis helpers (Roadmap Task 1)."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from analysis.helpers import load_results_dir, moving_average, summary_stats


@pytest.fixture
def tmp_csv_dir(tmp_path):
    """Create a temporary directory with sample CSV files."""
    for val in ["0.1", "0.5"]:
        for seed in range(2):
            df = pd.DataFrame({
                "Step": range(100),
                "Reward": np.random.randn(100),
                "Num Components": np.arange(100) // 10 + 1,
            })
            df.to_csv(tmp_path / f"info_gain_{val}_seed{seed}.csv", index=False)
    return tmp_path


class TestLoadResultsDir:
    def test_loads_all_csvs(self, tmp_csv_dir):
        df = load_results_dir(tmp_csv_dir)
        assert len(df) == 4 * 100  # 4 files × 100 rows

    def test_parses_param_and_seed(self, tmp_csv_dir):
        df = load_results_dir(tmp_csv_dir)
        assert set(df["seed"].unique()) == {0, 1}
        param_vals = df["param_value"].unique()
        assert len(param_vals) == 2

    def test_empty_dir(self, tmp_path):
        df = load_results_dir(tmp_path)
        assert len(df) == 0


class TestMovingAverage:
    def test_basic(self):
        rewards = np.ones(2000)
        ma = moving_average(rewards, window=1000)
        assert len(ma) == 1001
        np.testing.assert_allclose(ma, 1.0)

    def test_short_series(self):
        rewards = np.array([1.0, 2.0, 3.0])
        ma = moving_average(rewards, window=1000)
        expected = np.cumsum(rewards) / np.arange(1, 4)
        np.testing.assert_allclose(ma, expected)

    def test_known_values(self):
        rewards = np.array([0.0] * 999 + [1.0])
        ma = moving_average(rewards, window=1000)
        assert len(ma) == 1
        assert ma[0] == pytest.approx(0.001)


class TestSummaryStats:
    def test_computes_stats(self, tmp_csv_dir):
        df = load_results_dir(tmp_csv_dir)
        stats = summary_stats(df, last_n=50)
        assert "cumulative_reward" in stats.columns
        assert "mean_reward" in stats.columns
        assert "final_num_components" in stats.columns
        assert len(stats) == 4  # 2 param_values × 2 seeds
