#!/bin/bash
#SBATCH --job-name=axiom
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --qos=dw87
#
# Run a single AXIOM experiment on a compute node.
#
# Delegates to run_sweep.py --run-one so that special parameter transforms
# (scale_factor, discrete_alpha_scale, reward_alpha) work identically
# to the local runner.  For baseline runs (no PARAM), calls main.py directly.
#
# Usage:
#   sbatch --export=GAME=Explode,STEPS=10000,SEED=0 experiments/slurm/run_single.sh
#
#   # With a specific parameter override:
#   sbatch --export=GAME=Explode,STEPS=10000,SEED=0,PARAM=info_gain,VALUE=0.5 \
#       experiments/slurm/run_single.sh
#
#   # Override wall-time from the command line:
#   sbatch --time=02:00:00 --export=GAME=Explode,STEPS=5000,SEED=0 \
#       experiments/slurm/run_single.sh
#
# Environment variables (all optional, defaults shown):
#   GAME       — Gameworld game name           (default: Explode)
#   STEPS      — number of environment steps   (default: 10000)
#   SEED       — random seed                   (default: 0)
#   PARAM      — AXIOM CLI parameter to set    (default: empty → baseline run)
#   VALUE      — value for PARAM               (default: empty)
#   FAST       — if "1", use reduced CPU/BMR settings (default: 0)
#   OUTPUT_DIR — results subdirectory          (default: results/slurm_runs)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GAME="${GAME:-Explode}"
STEPS="${STEPS:-10000}"
SEED="${SEED:-0}"
PARAM="${PARAM:-}"
VALUE="${VALUE:-}"
FAST="${FAST:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-results/slurm_runs}"

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

echo "[run_single] $(date)"
echo "[run_single] GAME=$GAME STEPS=$STEPS SEED=$SEED PARAM=$PARAM VALUE=$VALUE FAST=$FAST"
echo ""

if [[ -n "$PARAM" && -n "$VALUE" ]]; then
    # Delegate to run_sweep.py --run-one for correct parameter translation
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

    echo "[run_single] CMD: ${CMD[*]}"
    echo ""
    "${CMD[@]}"
else
    # Baseline run — call main.py directly
    CMD=(
        python main.py
        --game "$GAME"
        --num_steps "$STEPS"
        --seed "$SEED"
    )
    if [[ "$FAST" == "1" ]]; then
        CMD+=(
            --planning_horizon 16
            --planning_rollouts 16
            --num_samples_per_rollout 1
            --bmr_pairs 200
            --bmr_samples 200
        )
    fi

    echo "[run_single] CMD: ${CMD[*]}"
    echo ""

    cd "$PROJECT_ROOT/vendor/axiom"
    "${CMD[@]}"

    CSV_NAME="$(echo "$GAME" | tr '[:upper:]' '[:lower:]').csv"
    DEST_DIR="$PROJECT_ROOT/$OUTPUT_DIR"
    mkdir -p "$DEST_DIR"
    DEST="$DEST_DIR/${GAME,,}_baseline_seed${SEED}.csv"
    cp "$CSV_NAME" "$DEST"
    echo ""
    echo "[run_single] Saved: $DEST"
fi
