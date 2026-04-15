"""
Run AXIOM parameter sweeps for the analysis project.

Calls vendor/axiom/main.py with varied hyperparameters, collecting
one CSV per (param_value, seed) combination.

Usage:
    python experiments/run_sweep.py \
        --param info_gain --values 0.0 0.05 0.1 0.5 1.0 \
        --game Explode --seeds 3 --steps 5000 --fast

    python experiments/run_sweep.py \
        --config experiments/configs/prior_sensitivity_smm.yaml \
        --game Explode --seeds 3 --steps 5000 --fast
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AXIOM_DIR = PROJECT_ROOT / "vendor" / "axiom"
RESULTS_DIR = PROJECT_ROOT / "results"

FAST_ARGS = [
    "--planning_horizon", "16",
    "--planning_rollouts", "16",
    "--num_samples_per_rollout", "1",
    "--bmr_pairs", "200",
    "--bmr_samples", "200",
]


def run_single(game: str, steps: int, extra_args: list[str], fast: bool) -> Path:
    """Run a single AXIOM experiment and return the output CSV path."""
    cmd = [
        sys.executable, "main.py",
        "--game", game,
        "--num_steps", str(steps),
    ]
    if fast:
        cmd.extend(FAST_ARGS)
    cmd.extend(extra_args)

    env = {**os.environ, "WANDB_MODE": "disabled"}
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(AXIOM_DIR), env=env, check=True)

    csv_name = f"{game.lower()}.csv"
    return AXIOM_DIR / csv_name


def run_sweep(
    param: str,
    values: list[str],
    game: str,
    seeds: int,
    steps: int,
    fast: bool,
    output_dir: Path,
):
    """Sweep a single parameter across values and seeds."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for val in values:
        for seed in range(seeds):
            print(f"\n=== {param}={val} seed={seed} ===")
            extra = [f"--{param}", str(val), "--seed", str(seed)]
            csv_src = run_single(game, steps, extra, fast)

            dest = output_dir / f"{param}_{val}_seed{seed}.csv"
            if csv_src.exists():
                shutil.copy2(csv_src, dest)
                print(f"  Saved: {dest}")
            else:
                print(f"  WARNING: expected {csv_src} not found")


def run_from_config(config_path: str, game: str, seeds: int, steps: int, fast: bool):
    """Load a YAML config and run all sweeps defined in it."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    experiment_name = config.get("experiment_name", Path(config_path).stem)
    output_dir = RESULTS_DIR / experiment_name

    for sweep in config.get("sweeps", []):
        param = sweep["param"]
        values = [str(v) for v in sweep["values"]]
        print(f"\n--- Sweeping {param}: {values} ---")
        param_dir = output_dir / param
        run_sweep(param, values, game, seeds, steps, fast, param_dir)


def main():
    parser = argparse.ArgumentParser(description="Run AXIOM parameter sweeps")
    parser.add_argument("--config", type=str, help="YAML config defining parameter grids")
    parser.add_argument("--param", type=str, help="Single parameter to sweep (alternative to --config)")
    parser.add_argument("--values", nargs="+", help="Values for --param")
    parser.add_argument("--game", type=str, default="Explode")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--fast", action="store_true", default=False)
    args = parser.parse_args()

    if args.config:
        run_from_config(args.config, args.game, args.seeds, args.steps, args.fast)
    elif args.param and args.values:
        output_dir = RESULTS_DIR / args.param
        run_sweep(args.param, args.values, args.game, args.seeds, args.steps, args.fast, output_dir)
    else:
        parser.error("Provide either --config or both --param and --values")


if __name__ == "__main__":
    main()
