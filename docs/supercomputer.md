# Supercomputer Guide (BYU RC)

This guide covers running AXIOM experiments on the BYU Research Computing
supercomputer. For general project setup and local usage, see [README.md](../README.md).

---

## 1 Cluster Basics

### 1.1 Login Nodes

Login nodes have internet access and are used to prepare compute jobs.
All `pip install`, `git clone`, and `make setup-*` commands run here.

```bash
ssh username@ssh.rc.byu.edu
```

Two-factor authentication is required. SSH multiplexing avoids repeated prompts:
<https://rc.byu.edu/wiki/index.php?page=SSH+Multiplexing>

To avoid typing the full hostname each time, add an entry to `~/.ssh/config`
on your local machine:

```
Host orc
    HostName ssh.rc.byu.edu
    User <your-username>
```

Then connect with just `ssh orc`.

### 1.2 Compute Nodes

Compute nodes have GPUs but **no internet access**. This means:
- All dependencies must be installed beforehand (on login nodes).
- `WANDB_MODE=disabled` is mandatory (wandb cannot phone home).
- No git operations, pip installs, or HTTP requests will succeed.

The login node has a GPU visible in the system but it is **not usable** for
computation. To force CPU-only execution when testing on a login node, set
`CUDA_VISIBLE_DEVICES=""` (or use `JAX_PLATFORMS=cpu` for JAX specifically).

**Interactive session** (debugging only — avoid leaving GPUs idle):

```bash
salloc --time=4:00:00 --qos=dw87 --gpus=1 --mem=32G --cpus-per-gpu=8
```

**Batch jobs** (preferred for all experiments):

```bash
sbatch experiments/slurm/run_single.sh
```

### 1.3 Job Management

```bash
squeue -u $USER              # list your running/pending jobs
sacct -j JOBID --format=...  # detailed info on a finished job
scancel JOBID                # cancel one job
scancel -u $USER             # cancel all your jobs
```

---

## 2 First-Time Setup (Login Node)

Run these steps once on a login node. Everything here requires internet.

### 2.1 Clone the Project

```bash
cd ~  # or your preferred workspace; see §5 for storage notes
git clone <your-repo-url> axiom-bayes
cd axiom-bayes
```

### 2.2 Load Modules

The exact module names depend on what BYU RC provides. Check available
modules with `module avail` and adjust as needed:

```bash
module load python/3.11 cuda/12.x cudnn/8.x
```

Verify the Python version (must be 3.10 or 3.11; AXIOM requires `<3.12`):

```bash
python3 --version
```

### 2.3 Create the Environment

```bash
make setup                    # creates .venv, installs analysis deps
source .venv/bin/activate
make setup-gameworld          # clones & installs Gameworld
make setup-axiom              # clones & installs official AXIOM

# Install CUDA-enabled JAX (the default pip install may pull CPU-only)
pip install --upgrade "jax[cuda12]"
```

### 2.4 Verify

```bash
source .venv/bin/activate
python -c "import jax; print(jax.default_backend())"   # should print 'gpu' on a compute node
make test                                                # runs on login node (CPU)
```

`jax.default_backend()` will return `cpu` on login nodes (no GPU) — that is
expected. The important thing is that `jax[cuda12]` is installed so it picks
up the GPU on compute nodes.

---

## 3 Environment Variables

Set these in your Slurm scripts (the templates in `experiments/slurm/`
already include them):

```bash
export WANDB_MODE=disabled                    # mandatory — no internet on compute
export JAX_PLATFORMS=cuda                     # use GPU backend

# Memory management — prevents OOM during long sweeps
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

For testing on a **login node** (CPU-only, no sbatch):

```bash
export CUDA_VISIBLE_DEVICES=""               # hide the unusable login-node GPU
export JAX_PLATFORMS=cpu                     # force JAX to CPU backend
export WANDB_MODE=disabled
```

---

## 4 Running Experiments

### 4.1 Quick Sanity Check (Login Node)

Before submitting batch jobs, run a short CPU smoke test on the login node
to catch import errors, missing files, or config issues:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu WANDB_MODE=disabled \
    python vendor/axiom/main.py --game Explode --num_steps 20 \
    --planning_horizon 8 --planning_rollouts 8 --num_samples_per_rollout 1
```

Cancel with Ctrl-C after a few steps — the goal is to verify the environment
loads and the run starts, not to produce results. This avoids wasting queue
time on jobs that fail immediately due to missing dependencies.

### 4.2 Single Run

Submit a single AXIOM run using the provided template:

```bash
sbatch --export=GAME=Explode,STEPS=10000,SEED=0 \
    experiments/slurm/run_single.sh
```

Override `#SBATCH` defaults from the command line:

```bash
sbatch --time=02:00:00 --export=GAME=Bounce,STEPS=5000,SEED=0 \
    experiments/slurm/run_single.sh
```

### 4.3 Sweep via Array Jobs

On the cluster, sweeps should run in parallel (one Slurm task per
param-value-seed combination) rather than sequentially as `run_sweep.py`
does locally.

**Step 1 — Generate a job list** from a YAML config:

```bash
python experiments/slurm/gen_joblist.py \
    experiments/configs/prior_sensitivity_smm.yaml \
    --game Explode --seeds 3 --steps 10000 \
    > jobs.txt
```

Each line in `jobs.txt` is a tab-separated record:
`param  value  seed  game  steps  output_dir`

**Step 2 — Submit the array job:**

```bash
N=$(wc -l < jobs.txt)
sbatch --array=1-${N} \
    --export=JOBLIST=jobs.txt \
    experiments/slurm/run_sweep_array.sh
```

This launches N parallel tasks. Each task reads its line from `jobs.txt`
using `$SLURM_ARRAY_TASK_ID` and runs the corresponding AXIOM configuration.

**Full-phase shortcuts** (generate + submit in one go):

```bash
# Phase 1 — prior sensitivity (all three configs)
for cfg in experiments/configs/prior_sensitivity_*.yaml; do
    python experiments/slurm/gen_joblist.py "$cfg" \
        --game Explode --seeds 3 --steps 10000 > jobs.txt
    N=$(wc -l < jobs.txt)
    sbatch --array=1-${N} --export=JOBLIST=jobs.txt \
        experiments/slurm/run_sweep_array.sh
done

# Phase 2 — BMR ablation
python experiments/slurm/gen_joblist.py \
    experiments/configs/bmr_ablation.yaml \
    --game Explode --seeds 3 --steps 10000 > jobs.txt
N=$(wc -l < jobs.txt)
sbatch --array=1-${N} --export=JOBLIST=jobs.txt \
    experiments/slurm/run_sweep_array.sh

# Phase 3 — info-gain sweep
python experiments/slurm/gen_joblist.py \
    experiments/configs/info_gain_sweep.yaml \
    --game Explode --seeds 5 --steps 10000 > jobs.txt
N=$(wc -l < jobs.txt)
sbatch --array=1-${N} --export=JOBLIST=jobs.txt \
    experiments/slurm/run_sweep_array.sh
```

### 4.4 Local vs. Cluster Workflow

| | Local (laptop / single GPU) | Cluster (Slurm) |
|---|---|---|
| Runner | `python experiments/run_sweep.py --config ...` | `sbatch --array=... experiments/slurm/run_sweep_array.sh` |
| Parallelism | Sequential (subprocess per config) | One Slurm task per config |
| Config format | Same YAML files in `experiments/configs/` | Same YAML files (parsed by `gen_joblist.py`) |
| Results | Written to `results/` | Written to `results/` (same layout) |

---

## 5 Storage and Data Transfer

### 5.1 Where to Put the Repo

See <https://rc.byu.edu/wiki/?id=Storage> for filesystem details. Typical
layout:

- **Home (`~`)** — limited quota, backed up. Good for the repo itself.
- **Scratch / compute storage** — larger quota, faster I/O, not backed up.
  Good for `results/` if output volume gets large (symlink `results/` to
  scratch if needed).

### 5.2 Cleanup Etiquette

Cluster storage is shared. After completing a round of experiments:
- Transfer results to your local machine (see below).
- Delete large intermediate files (`.mp4` videos, scratch CSVs) from the
  cluster once safely copied.
- Remove old Slurm logs: `rm slurm_logs/*.out` or keep only recent ones.

### 5.3 Transferring Results

From your local machine:

```bash
rsync -avz --include='*/' --include='*.csv' --include='*.png' \
    --exclude='*' \
    username@ssh.rc.byu.edu:~/axiom-bayes/results/ \
    ./results/
```

Or pull everything (including videos):

```bash
rsync -avz username@ssh.rc.byu.edu:~/axiom-bayes/results/ ./results/
```

---

## 6 Resource Estimates

Timing references from local benchmarks (update GPU column after first
cluster run):

| Steps | CPU (fast settings) | GPU (estimated) |
|-------|---------------------|-----------------|
| 200   | ~320 s              | TBD             |
| 5,000 | ~90 min             | TBD             |
| 10,000| ~3 h                | TBD             |

**Sweep totals** (rough job counts at 3 seeds per config):

| Phase | Configs | Seeds | Total jobs | Est. GPU-hours (TBD) |
|-------|---------|-------|------------|----------------------|
| Phase 1 (sMM + tMM + rMM) | 16 + 12 + 15 = 43 | 3 | 129 | — |
| Phase 2 (BMR ablation)     | 2 + 5 + 5 + 5 = 17 | 3 | 51  | — |
| Phase 3 (info-gain)        | 6 + 4 = 10          | 3 | 30  | — |
| **Total**                  |                      |   | **210** | — |

Fill in the GPU column after your first batch completes. Use
`sacct -j JOBID --format=JobID,Elapsed,MaxRSS,MaxVMSize` to check
actual wall-time and memory.

---

## 7 Job Monitoring and Debugging

### 7.1 Check Job Status

```bash
squeue -u $USER                         # running and pending jobs
squeue -u $USER -t PENDING              # just pending
sacct --starttime=today -u $USER        # today's completed jobs
```

### 7.2 Read Logs

Slurm output goes to `slurm_logs/<jobname>_<jobid>.out`:

```bash
tail -f slurm_logs/axiom_12345.out      # follow a running job
ls -lt slurm_logs/ | head               # most recent logs
```

### 7.3 SSH into a Running Compute Node

Useful for checking GPU utilization or debugging hangs:

```bash
squeue -u $USER                         # note the NODELIST column
ssh <node-name>                         # e.g. ssh m9-31-4
nvidia-smi                              # GPU utilization on that node
```

### 7.4 Common Issues

- **OOM**: Reduce `XLA_PYTHON_CLIENT_MEM_FRACTION` or request more memory
  (`--mem=64G`).
- **Job stuck in PENDING**: Check `squeue -u $USER -t PENDING` — the
  REASON column shows why (e.g., `QOSMaxJobsPerUserLimit`, `Resources`).
- **Module not found**: Run `module avail <keyword>` on a login node to
  find the correct module name. Module names change across cluster updates.
- **Import errors on compute node**: The venv was likely built with
  different modules loaded. Rebuild with the same `module load` commands
  that your Slurm scripts use.
- **CUDA version mismatch**: JAX (like PyTorch) is sensitive to the CUDA
  version it was built against. If `jax[cuda12]` was installed but the
  cluster loads a different CUDA version via `module load`, you may get
  silent failures or crashes. Verify with `python -c "import jax; jax.devices()"` 
  on a compute node. If it fails, reinstall JAX after loading the correct
  CUDA module.
