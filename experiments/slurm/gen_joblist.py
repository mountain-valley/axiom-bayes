"""Generate a Slurm job list from a sweep YAML config.

Each line is a tab-separated record that run_sweep_array.sh reads to
dispatch one AXIOM run per SLURM_ARRAY_TASK_ID.

Usage:
    python experiments/slurm/gen_joblist.py \
        experiments/configs/prior_sensitivity_smm.yaml \
        --game Explode --seeds 3 --steps 10000 > jobs.txt

    N=$(wc -l < jobs.txt)
    sbatch --array=1-${N} --export=JOBLIST=jobs.txt \
        experiments/slurm/run_sweep_array.sh
"""

import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Generate Slurm job list from sweep config")
    parser.add_argument("config", type=str, help="YAML config file")
    parser.add_argument("--game", type=str, default="Explode")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--fast", action="store_true", default=False)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    experiment_name = config.get("experiment_name", Path(args.config).stem)

    for sweep in config.get("sweeps", []):
        param = sweep["param"]
        sweep_name = sweep.get("name", param)
        output_dir = f"results/{experiment_name}/{sweep_name}"

        for value in sweep["values"]:
            for seed in range(args.seeds):
                fast_flag = "1" if args.fast else "0"
                fields = [
                    param,
                    str(value),
                    str(seed),
                    args.game,
                    str(args.steps),
                    output_dir,
                    fast_flag,
                ]
                print("\t".join(fields))


if __name__ == "__main__":
    main()
