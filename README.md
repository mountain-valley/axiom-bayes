# AXIOM Bayesian Analysis

A Bayesian analysis of **AXIOM: Active eXpanding Inference with Object-centric Models** (Heins et al., 2025) for CS 677 Bayesian Statistics.

AXIOM is an active inference agent that learns object-centric world models using four expanding Bayesian mixture models (sMM, iMM, tMM, rMM) and masters pixel-based games within 10,000 steps — without gradient-based optimization. This project installs the [official codebase](https://github.com/VersesTech/axiom) and conducts three focused analyses:

1. **Prior and Hyperparameter Sensitivity** — How do prior specifications and model hyperparameters affect perception, dynamics, and reward?
2. **Bayesian Model Reduction** — How does principled model selection via expected free energy compare to naive pruning?
3. **Exploration vs. Exploitation** — How does the information-gain term (KL on Dirichlet posteriors) shape learning?

## Quickstart

```bash
make setup                  # Create venv, install analysis dependencies
source .venv/bin/activate   # Activate the environment
make setup-gameworld        # Clone & install Gameworld environments
make setup-axiom            # Clone & install official AXIOM codebase
make baseline GAME=Explode  # Run a baseline agent
make test                   # Run tests
```

### Reproducing the Exact Environment

The quickstart above (`make setup`) installs the latest compatible versions.
To reproduce the exact package versions used during development, use the lock
file instead:

```bash
make setup-locked               # Pinned versions from requirements-lock.txt
source .venv/bin/activate
make setup-gameworld && make setup-axiom
```

**Requirements:** Python 3.10 or 3.11 (AXIOM's JAX dependency requires `<3.12`).

### Vendor code (Gameworld / AXIOM) and local patches

The `vendor/` directory is gitignored in this repo: `vendor/gameworld` and `vendor/axiom`
are normal Git clones with their own history. Edits there are **not** visible in
`git status` at the project root until you record them (see below). If you do not patch
upstream, use the Quickstart as-is and you can skip the fork steps.

**If you need to change Gameworld or AXIOM and keep those changes reproducible:**

1. On GitHub, fork [VersesTech/gameworld](https://github.com/VersesTech/gameworld) and/or
   [VersesTech/axiom](https://github.com/VersesTech/axiom) to your account (one fork per
   repo you intend to patch).
2. **Point each vendor repo at your fork** so `git push` goes to your GitHub copy, not
   VersesTech. Pick the case that matches your disk:

   - **You do not have `vendor/gameworld` or `vendor/axiom` yet** (or you are happy to
     delete them and clone again). From the project root:

     ```bash
     rm -rf vendor/gameworld vendor/axiom   # omit if those folders do not exist
     GAMEWORLD_GIT_URL=https://github.com/<you>/gameworld.git make setup-gameworld
     AXIOM_GIT_URL=https://github.com/<you>/axiom.git make setup-axiom
     ```

     `make setup-*` only runs `git clone` when the folder is missing, so the env vars
     must be set **before** that first clone. After this, `origin` is your fork.

   - **Those folders already exist** (for example you followed Quickstart and `origin`
     is still `VersesTech/...`). `make setup-gameworld` will not clone again, so change
     the remote in place—same files on disk, new push destination:

     ```bash
     cd vendor/gameworld && git remote set-url origin https://github.com/<you>/gameworld.git && git fetch origin
     cd ../axiom && git remote set-url origin https://github.com/<you>/axiom.git && git fetch origin
     ```

     (Adjust if you only forked one of the two.) Use the first bullet instead if you
     prefer a clean re-clone from your fork rather than repointing `origin`.
3. **Develop in the vendor repo:** `cd vendor/gameworld` (or `vendor/axiom`), create a
   branch if you like, commit, and `git push origin <branch>` so your fork on GitHub
   holds the commits.
4. **Pin what this project expects:** back at the project root, run `make vendor-lock`.
   That rewrites [docs/vendor_versions.txt](docs/vendor_versions.txt) with each vendor’s
   `origin` URL, branch name, and full commit SHA. Commit `docs/vendor_versions.txt` in
   **this** repo so collaborators (and future you) know which revision to check out.

To match those pins on another machine after the repos exist:
`cd vendor/gameworld && git fetch origin && git checkout <commit from vendor_versions.txt>`,
then repeat for `vendor/axiom`, then `pip install -e vendor/gameworld -e vendor/axiom`.

This workflow keeps patches in your fork while the analysis repo only stores a small
lock file. **Other approaches (not used in this repo):** a git submodule (records vendor
SHA inside this repo’s tree but needs `.gitignore` changes), or tracked `.patch` files
applied during `make setup-*`.

**Cursor / VS Code — Git UI for vendor repos:** The default “open folder” view only
surfaces the root repo’s Git state; `vendor/` is ignored there. To get **separate Source
Control entries** for this project, Gameworld, and AXIOM, use **File → Open Workspace
from File…** and choose [axiom-bayes.code-workspace](axiom-bayes.code-workspace) at the
repo root. After `make setup-gameworld` and `make setup-axiom`, all three folders exist
and each nested clone appears in the SCM sidebar under its own root name.

## Repository Layout

```
vendor/axiom/       Official AXIOM source (JAX, installed editable)
vendor/gameworld/   Official Gameworld environment suite
experiments/        Sweep runner, YAML configs for each analysis phase
analysis/           Plotting utilities, helpers, Jupyter notebooks
results/            Output CSVs, BMR logs, generated figures
tests/              Setup verification, infrastructure tests
docs/               Roadmap, paper notes, replication log, report, presentation
```

### Data Flow

```
experiments/  →  runs AXIOM with varied configs  →  writes to  →  results/
                                                                     │
analysis/     →  reads results  →  generates  →  results/figures/
```

- **`experiments/`** — Sweep scripts and YAML configs that run AXIOM (`vendor/axiom/main.py`) with different hyperparameter settings. Each run produces a CSV with per-step reward, expected utility, expected info gain, and component count.

- **`results/`** — Output of experiments, organized by phase (`prior_sensitivity/`, `bmr_ablation/`, `info_gain_sweep/`). Raw CSVs are gitignored; generated figures in `results/figures/` are tracked.

- **`analysis/`** — Code that consumes results: loads CSVs, computes statistics, generates figures. Jupyter notebooks produce the final plots for the report.

## Key Commands

| Command                        | Description                                        |
|--------------------------------|----------------------------------------------------|
| `make help`                    | List all available targets                         |
| `make baseline GAME=X`        | Run a baseline AXIOM agent on game X               |
| `make sweep-phase1`           | Run prior and hyperparameter sensitivity experiments |
| `make sweep-phase2`           | Run BMR ablation experiments                       |
| `make sweep-phase3`           | Run info-gain sweep experiments                    |
| `make figures`                | Regenerate plots from saved results                |
| `make test`                   | Run tests                                          |
| `make lint`                   | Lint analysis and experiment code                  |
| `make vendor-lock`            | Write `docs/vendor_versions.txt` from vendor repos |

All sweep commands accept `GAME=X`, `SEEDS=N`, `STEPS=N`, and `FAST=1` (default) / `FAST=0` (full runs).

## Task 1 Infrastructure Notes

Task 1 adds the reusable experiment pipeline used by later roadmap tasks.

### Sweep runner

- Entry point: `experiments/run_sweep.py`
- Supports either:
  - direct parameter/value sweep, or
  - YAML-defined sweeps in `experiments/configs/*.yaml`

Example direct sweep:

```bash
.venv/bin/python experiments/run_sweep.py \
  --param info_gain --values 0.0 0.05 0.1 0.5 1.0 \
  --game Explode --seeds 3 --steps 5000 --fast
```

Example config-driven sweep:

```bash
.venv/bin/python experiments/run_sweep.py \
  --config experiments/configs/prior_sensitivity_smm.yaml \
  --game Explode --seeds 3 --steps 5000 --fast
```

Enable next-state prediction error logging (optional):

```bash
.venv/bin/python experiments/run_sweep.py \
  --config experiments/configs/prior_sensitivity_tmm.yaml \
  --game Explode --seeds 3 --steps 5000 --fast \
  --with-prediction-error
```

### Output layout and filenames

- Results are written under `results/<experiment_name>/<sweep_name>/`.
- Each run writes one CSV:
  - `<param>_<value>_seed<N>.csv`
- `make sweep-phase1`, `make sweep-phase2`, and `make sweep-phase3` are wrappers around this runner.

### Built-in roadmap-specific parameter transforms

The runner includes special handling for parameters represented as vectors in AXIOM:

- `scale_factor` -> translates to `--scale <scaled_default_vector>`
- `discrete_alpha_scale` -> translates to scaled `--discrete_alphas ...`
- `reward_alpha` -> translates to reward-specific `--discrete_alphas ...`

### Verified baseline CSV columns (required for downstream metrics)

From baseline AXIOM output, the currently available columns are:

- `Step`
- `Reward`
- `Average Reward`
- `Cumulative Reward`
- `Expected Utility`
- `Expected Info Gain`
- `Num Components`

`next-state prediction error` is not present in standard CSV output and should be treated as unavailable unless custom logging is added.
When `--with-prediction-error` is used, the custom runner writes
`Next-State Prediction Error` as an additional CSV column.

Observed overhead from local A/B benchmarking (CPU, `Explode`, 200 steps, seed 0):
- baseline runner: `319.56s`
- with prediction-error logging: `348.57s`
- relative overhead: ~`9.1%`

### GPU memory note for sweeps

On constrained GPUs, JAX preallocation can cause OOM during long sweep batches.
If needed, run sweeps with:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

## Games (Gameworld 10k)

Aviate · Bounce · Cross · Drive · Explode · Fruits · Gold · Hunt · Impact · Jump

See the [Gameworld repo](https://github.com/VersesTech/gameworld) for environment details.

## References

- Paper: `AXIOM_paper.pdf` in this repo
- Official code: https://github.com/VersesTech/axiom
- Gameworld: https://github.com/VersesTech/gameworld
