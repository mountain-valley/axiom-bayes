"""Test experiment infrastructure and analysis helpers (Roadmap Task 1)."""

import numpy as np
import pandas as pd
import pytest

from analysis.helpers import group_by_parameter, load_results_dir, moving_average, summary_stats
from experiments import run_sweep as sweep_runner


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
        assert "cumulative_reward_mean" in stats.columns
        assert "mean_reward_mean" in stats.columns
        assert "cumulative_reward_ci95" in stats.columns
        assert "final_num_components_mean" in stats.columns
        assert len(stats) == 2  # aggregated across seeds


class TestGroupByParameter:
    def test_groups_by_step_and_value(self):
        df = pd.DataFrame(
            {
                "param_value": ["0.1", "0.1", "0.1", "0.1"],
                "seed": [0, 1, 0, 1],
                "Step": [0, 0, 1, 1],
                "Reward": [1.0, 3.0, 2.0, 4.0],
            }
        )
        grouped = group_by_parameter(df)
        assert len(grouped) == 2
        assert set(grouped["parameter_value"]) == {"0.1"}
        step0 = grouped[grouped["Step"] == 0].iloc[0]
        assert step0["reward_mean"] == pytest.approx(2.0)
        assert step0["reward_ci95"] > 0


class TestSweepRunner:
    def test_build_extra_args_special_cases(self):
        args, label = sweep_runner.build_extra_args("scale_factor", "2.0", seed=1)
        assert args[:2] == ["--scale", "0.15,0.15,1.5,1.5,1.5"]
        assert args[-2:] == ["--seed", "1"]
        assert label == "2.0"

        args, _ = sweep_runner.build_extra_args("discrete_alpha_scale", "10.0", seed=2)
        assert args[0] == "--discrete_alphas"
        assert args[-2:] == ["--seed", "2"]
        assert len(args[1:-2]) == 6

        args, _ = sweep_runner.build_extra_args("reward_alpha", "0.1", seed=3)
        assert args[0] == "--discrete_alphas"
        assert args[5] == "0.1"

    def test_run_sweep_writes_expected_structure(self, tmp_path, monkeypatch):
        fake_axiom_csv = tmp_path / "explode.csv"
        pd.DataFrame({"Step": [0], "Reward": [1.0]}).to_csv(fake_axiom_csv, index=False)

        def fake_run_single(game, steps, extra_args, fast):
            return fake_axiom_csv

        monkeypatch.setattr(sweep_runner, "run_single", fake_run_single)

        output_dir = tmp_path / "results"
        sweep_runner.run_sweep(
            param="info_gain",
            values=["0.0", "0.1"],
            game="Explode",
            seeds=2,
            steps=10,
            fast=True,
            output_dir=output_dir,
        )

        expected_files = {
            "info_gain_0.0_seed0.csv",
            "info_gain_0.0_seed1.csv",
            "info_gain_0.1_seed0.csv",
            "info_gain_0.1_seed1.csv",
        }
        actual_files = {p.name for p in output_dir.glob("*.csv")}
        assert actual_files == expected_files
