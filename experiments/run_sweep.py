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

DEFAULT_SMM_SCALE = [0.075, 0.075, 0.75, 0.75, 0.75]
DEFAULT_RMM_DISCRETE_ALPHAS = [1e-4, 1e-4, 1e-4, 1e-4, 1.0, 1e-4]


def _format_value_for_filename(value: str) -> str:
    """Sanitize parameter values so they are safe and readable in filenames."""
    return str(value).replace("/", "-").replace(" ", "")


def build_extra_args(param: str, value: str, seed: int) -> tuple[list[str], str]:
    """Translate sweep params to AXIOM CLI args.

    Returns:
        extra_args: CLI args to pass to vendor/axiom/main.py
        value_label: canonical value label used in result filenames
    """
    if param == "scale_factor":
        factor = float(value)
        scaled = [x * factor for x in DEFAULT_SMM_SCALE]
        scale_arg = ",".join(f"{x:g}" for x in scaled)
        return ["--scale", scale_arg, "--seed", str(seed)], value

    if param == "discrete_alpha_scale":
        factor = float(value)
        scaled = [x * factor for x in DEFAULT_RMM_DISCRETE_ALPHAS]
        discrete_alpha_args = [f"{x:g}" for x in scaled]
        return ["--discrete_alphas", *discrete_alpha_args, "--seed", str(seed)], value

    if param == "reward_alpha":
        reward_alpha = float(value)
        alphas = [1e-4, 1e-4, 1e-4, 1e-4, reward_alpha, 1e-4]
        discrete_alpha_args = [f"{x:g}" for x in alphas]
        return ["--discrete_alphas", *discrete_alpha_args, "--seed", str(seed)], value

    return [f"--{param}", str(value), "--seed", str(seed)], value


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
            extra, value_label = build_extra_args(param, val, seed)
            csv_src = run_single(game, steps, extra, fast)

            safe_value = _format_value_for_filename(value_label)
            dest = output_dir / f"{param}_{safe_value}_seed{seed}.csv"
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
        sweep_name = sweep.get("name", param)
        print(f"\n--- Sweeping {sweep_name} ({param}): {values} ---")
        param_dir = output_dir / sweep_name
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
