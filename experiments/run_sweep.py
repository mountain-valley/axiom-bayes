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
import tempfile
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


def run_single(
    game: str,
    steps: int,
    extra_args: list[str],
    fast: bool,
    dest_csv: Path,
    no_video: bool = True,
    with_prediction_error: bool = False,
) -> None:
    """Run one AXIOM experiment and persist its CSV output.

    Notes:
    - Each run executes in an isolated temp directory to avoid concurrent
      file clobbering on Slurm arrays.
    - AXIOM can fail after CSV write (e.g., ffmpeg missing during video export);
      we still copy the CSV if present and non-empty.
    """
    entry_script = (
        str(PROJECT_ROOT / "experiments" / "run_with_prediction_error.py")
        if with_prediction_error
        else str(AXIOM_DIR / "main.py")
    )
    cmd = [
        sys.executable,
        entry_script,
        "--game", game,
        "--num_steps", str(steps),
    ]
    if fast:
        cmd.extend(FAST_ARGS)
    if no_video:
        cmd.append("--no_video")
    cmd.extend(extra_args)

    env = {**os.environ, "WANDB_MODE": "disabled"}
    print(f"  Running: {' '.join(cmd)}")
    tmp_root = RESULTS_DIR / ".tmp_axiom_runs"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{game.lower()}_", dir=str(tmp_root)) as run_dir:
        result = subprocess.run(cmd, cwd=run_dir, env=env, check=False)
        csv_src = Path(run_dir) / f"{game.lower()}.csv"

        if csv_src.exists() and csv_src.stat().st_size > 0:
            dest_csv.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(csv_src, dest_csv)
            print(f"  Saved: {dest_csv}")
        else:
            print(
                "  WARNING: expected non-empty CSV not found at "
                f"{csv_src} (return code {result.returncode})"
            )

        if result.returncode != 0:
            print(
                "  WARNING: AXIOM exited non-zero "
                f"({result.returncode}); keeping copied CSV if available."
            )


def run_sweep(
    param: str,
    values: list[str],
    game: str,
    seeds: int,
    steps: int,
    fast: bool,
    output_dir: Path,
    no_video: bool = True,
    with_prediction_error: bool = False,
):
    """Sweep a single parameter across values and seeds."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for val in values:
        for seed in range(seeds):
            print(f"\n=== {param}={val} seed={seed} ===")
            extra, value_label = build_extra_args(param, val, seed)

            safe_value = _format_value_for_filename(value_label)
            dest = output_dir / f"{param}_{safe_value}_seed{seed}.csv"
            run_single(
                game,
                steps,
                extra,
                fast,
                dest,
                no_video,
                with_prediction_error=with_prediction_error,
            )


def run_from_config(
    config_path: str,
    game: str,
    seeds: int,
    steps: int,
    fast: bool,
    no_video: bool = True,
    with_prediction_error: bool = False,
):
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
        run_sweep(
            param,
            values,
            game,
            seeds,
            steps,
            fast,
            param_dir,
            no_video,
            with_prediction_error=with_prediction_error,
        )


def run_one(
    param: str,
    value: str,
    seed: int,
    game: str,
    steps: int,
    fast: bool,
    output_dir: Path,
    no_video: bool = True,
    with_prediction_error: bool = False,
):
    """Run a single (param, value, seed) combination — used by Slurm array jobs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    extra, value_label = build_extra_args(param, value, seed)

    safe_value = _format_value_for_filename(value_label)
    dest = output_dir / f"{param}_{safe_value}_seed{seed}.csv"
    run_single(
        game,
        steps,
        extra,
        fast,
        dest,
        no_video,
        with_prediction_error=with_prediction_error,
    )


def main():
    parser = argparse.ArgumentParser(description="Run AXIOM parameter sweeps")
    parser.add_argument("--config", type=str, help="YAML config defining parameter grids")
    parser.add_argument("--param", type=str, help="Single parameter to sweep (alternative to --config)")
    parser.add_argument("--values", nargs="+", help="Values for --param")
    parser.add_argument("--game", type=str, default="Explode")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--fast", action="store_true", default=False)
    parser.add_argument(
        "--with-video",
        action="store_true",
        default=False,
        help="Enable AXIOM video/media output (disabled by default for sweeps).",
    )
    parser.add_argument(
        "--with-prediction-error",
        action="store_true",
        default=False,
        help="Use custom runner that logs Next-State Prediction Error column.",
    )

    # Single-run mode for Slurm array jobs
    parser.add_argument(
        "--run-one",
        action="store_true",
        default=False,
        help="Run a single (param, value, seed) instead of a full sweep. "
        "Requires --param, --value, --seed, and --output-dir.",
    )
    parser.add_argument("--value", type=str, help="Single value (used with --run-one)")
    parser.add_argument("--seed", type=int, help="Single seed (used with --run-one)")
    parser.add_argument("--output-dir", type=str, help="Output directory (used with --run-one)")

    args = parser.parse_args()
    no_video = not args.with_video

    if args.run_one:
        if not all([args.param, args.value is not None, args.seed is not None, args.output_dir]):
            parser.error("--run-one requires --param, --value, --seed, and --output-dir")
        run_one(
            args.param,
            args.value,
            args.seed,
            args.game,
            args.steps,
            args.fast,
            Path(args.output_dir),
            no_video,
            with_prediction_error=args.with_prediction_error,
        )
    elif args.config:
        run_from_config(
            args.config,
            args.game,
            args.seeds,
            args.steps,
            args.fast,
            no_video,
            with_prediction_error=args.with_prediction_error,
        )
    elif args.param and args.values:
        output_dir = RESULTS_DIR / args.param
        run_sweep(
            args.param,
            args.values,
            args.game,
            args.seeds,
            args.steps,
            args.fast,
            output_dir,
            no_video,
            with_prediction_error=args.with_prediction_error,
        )
    else:
        parser.error("Provide --run-one, --config, or both --param and --values")


if __name__ == "__main__":
    main()
