#!/bin/bash
#SBATCH --job-name=axiom-sweep
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#
#
# Slurm array job wrapper for AXIOM parameter sweeps.
#
# Each array task reads one line from a job list file and delegates to
# run_sweep.py --run-one, which handles special parameter transforms
# (scale_factor, discrete_alpha_scale, reward_alpha) identically to the
# local sequential runner.
#
# Usage:
#   python experiments/slurm/gen_joblist.py \
#       experiments/configs/prior_sensitivity_smm.yaml \
#       --game Explode --seeds 3 --steps 10000 > jobs.txt
#
#   N=$(wc -l < jobs.txt)
#   sbatch --array=1-${N} --export=JOBLIST=jobs.txt \
#       experiments/slurm/run_sweep_array.sh
#   # If your cluster requires explicit QoS/account:
#   # sbatch --qos=<your-qos> --account=<your-account> --array=1-${N} --export=JOBLIST=jobs.txt \
#   #       experiments/slurm/run_sweep_array.sh
#
# Job list format (tab-separated, one line per task):
#   param  value  seed  game  steps  output_dir  fast

set -euo pipefail

# Slurm may execute a copied script from /var/spool; use submit directory first.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

JOBLIST="${JOBLIST:?JOBLIST environment variable must point to a job list file}"

if [[ ! -f "$JOBLIST" ]]; then
    echo "ERROR: job list not found: $JOBLIST" >&2
    exit 1
fi

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBLIST")
if [[ -z "$LINE" ]]; then
    echo "ERROR: no line $SLURM_ARRAY_TASK_ID in $JOBLIST" >&2
    exit 1
fi

IFS=$'\t' read -r PARAM VALUE SEED GAME STEPS OUTPUT_DIR FAST <<< "$LINE"

mkdir -p "$PROJECT_ROOT/slurm_logs"

# --- Modules (adjust names to match your cluster) ---
# module load python/3.11 cuda/12.x cudnn/8.x

# --- Activate venv ---
source "$PROJECT_ROOT/.venv/bin/activate"

# --- Environment ---
export WANDB_MODE=disabled
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

echo "[sweep-array] task=$SLURM_ARRAY_TASK_ID / $(wc -l < "$JOBLIST")"
echo "[sweep-array] PARAM=$PARAM VALUE=$VALUE SEED=$SEED GAME=$GAME STEPS=$STEPS FAST=$FAST"
echo "[sweep-array] OUTPUT_DIR=$OUTPUT_DIR"
echo ""

# --- Build and run via run_sweep.py --run-one ---
CMD=(
    python "$PROJECT_ROOT/experiments/run_sweep.py"
    --run-one
    --param "$PARAM"
    --value "$VALUE"
    --seed "$SEED"
    --game "$GAME"
    --steps "$STEPS"
    --output-dir "$PROJECT_ROOT/$OUTPUT_DIR"
)

if [[ "$FAST" == "1" ]]; then
    CMD+=(--fast)
fi

echo "[sweep-array] CMD: ${CMD[*]}"
echo ""

"${CMD[@]}"
