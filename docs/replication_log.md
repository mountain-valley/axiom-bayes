# Replication Log

Running record of progress, decisions, and findings. Append new entries at the top.

---

## 2026-04-15 — Vendor Git workflow (Gameworld / AXIOM pins)

- Adopted **Option A** (fork or upstream + pinned SHA documented in-repo): added
  [vendor_versions.txt](vendor_versions.txt) with `origin` URL, branch, and
  full commit for `vendor/gameworld` and `vendor/axiom`.
- Added `make vendor-lock` to regenerate that file from the current vendor clones.
- Extended `Makefile` so first-time clones can use `GAMEWORLD_GIT_URL` and
  `AXIOM_GIT_URL` (defaults remain VersesTech upstream).
- Documented workflow in [README.md](../README.md) (Vendor code section).

---

## 2026-04-14 — Task 0 completed (full baseline finished)

- Full baseline run completed via dedicated terminal command:
  - `WANDB_MODE=disabled make baseline`
- Verified Task 0 baseline artifacts in `results/baseline/`:
  - `explode_baseline.csv`
  - `explode_baseline.mp4`
- Verified baseline CSV shape and columns:
  - 5000 rows
  - columns include: `Step`, `Reward`, `Average Reward`, `Cumulative Reward`, `Expected Utility`, `Expected Info Gain`, `Num Components`
- Verified roadmap-required test command after Task 0 completion:
  - `make test` -> **11 passed**.
- Marked Task 0 complete in `docs/roadmap.md`.

---

## 2026-04-14 — Task 0 verification + baseline output routing

- Re-verified Task 0 setup on this machine by reinstalling editable upstream deps:
  - `.venv/bin/python -m pip install -e vendor/gameworld -e vendor/axiom`
- Fixed reproducibility issue in `Makefile` test command:
  - Switched `PYTEST` from a direct script path to `$(PYTHON) -m pytest` so stale shebangs in `.venv/bin/pytest` do not break `make test` if the project path changes.
- Verified tests via roadmap-required command:
  - `make test` -> **11 passed**.
- Updated baseline artifact handling so outputs land in `results/`:
  - `make baseline` now copies both `explode.csv` and `explode.mp4` to:
    - `results/baseline/explode_baseline.csv`
    - `results/baseline/explode_baseline.mp4`
- Confirmed the current root `explode.csv` is still the 100-step smoke artifact (100 rows), so full baseline output is not yet present.
- Full 5000-step baseline run is expected to be long on this host (~90+ min from observed early throughput); hand off to dedicated terminal:
  - `WANDB_MODE=disabled make baseline`

---

## 2026-04-15 — Task 0 setup fixed (Python 3.11, vendor install, smoke run)

- Recreated `.venv` with Python 3.11.15 (AXIOM/Gameworld require Python `<3.12`).
- Cloned official upstream repos into `vendor/axiom` and `vendor/gameworld`.
- Installed editable packages in the project venv:
  - `pip install -e ".[dev]"`
  - `pip install -e vendor/gameworld`
  - `pip install -e vendor/axiom`
- Added local ffmpeg binary in venv (`.venv/bin/ffmpeg`) so AXIOM's `mediapy` export works.
- Updated `tests/test_setup.py` to validate real upstream entry points:
  - AXIOM args parser import via `vendor/axiom/defaults.py`.
  - Gameworld env creation via `gymnasium.make("Gameworld-Explode-v0")` after `import gameworld.envs`.
- Verified setup tests: `pytest tests/test_setup.py -q` -> **4 passed**.
- Verified smoke run:
  - `python vendor/axiom/main.py --game=Explode --num_steps 100 --planning_horizon 8 --planning_rollouts 8 --num_samples_per_rollout 1`
  - Produced `explode.csv` and `explode.mp4`.
  - Copied CSV to `results/baseline/explode_smoke.csv`.
- Remaining for Task 0 completion: run the full 5k-step baseline and copy final CSV to
  `results/baseline/explode_baseline.csv`.

---

## 2026-04-15 — Clean up old stubs; align project with analysis roadmap

- **Deleted** old reimplementation stub directories: `axiom/`, `envs/`, `baselines/`.
  These were from the original plan to reimplement AXIOM from scratch. The project
  now uses the official codebase at `vendor/axiom/`.
- **Deleted** old test files (`test_smm.py`, `test_imm.py`, `test_rmm.py`,
  `test_tmm.py`, `test_agent.py`, `test_inference.py`) that imported from the
  now-removed `axiom` package.
- **Deleted** old experiment scripts (`run_experiment.py`, `sweep.py`,
  `run_ablation.py`) and configs (`default.yaml`, `bounce.yaml`) that referenced
  the old training loop and `envs.gameworld`.
- **Deleted** old analysis notebooks (`01_learning_curves.ipynb`,
  `02_interpretability.ipynb`, `03_ablations.ipynb`) that were designed for the
  reimplementation workflow.
- **Created** `experiments/run_sweep.py` — new sweep runner that calls
  `vendor/axiom/main.py` via subprocess with varied hyperparameters.
- **Created** five YAML config files matching the roadmap's three analysis phases:
  `prior_sensitivity_smm.yaml`, `prior_sensitivity_tmm.yaml`,
  `prior_sensitivity_rmm.yaml`, `bmr_ablation.yaml`, `info_gain_sweep.yaml`.
- **Created** `analysis/helpers.py` with `load_results_dir()`, `moving_average()`,
  and `summary_stats()` for loading and analyzing AXIOM CSV output.
- **Created** `tests/test_infrastructure.py` (Task 1 analysis helper tests).
- **Created** three phase-specific notebooks: `phase1_prior_sensitivity.ipynb`,
  `phase2_bmr.ipynb`, `phase3_exploration.ipynb` with figure stubs matching the
  roadmap's figure specifications.
- **Kept** `analysis/metrics.py` and `analysis/plotting.py` (generic utilities
  that remain useful for the analysis project).
- Next: complete Task 0 (install AXIOM + Gameworld) and Task 1 (verify CSV
  columns, finalize sweep runner).

---

## 2025-04-14 — Project scaffolding

- Created repo structure: `axiom/`, `envs/`, `baselines/`, `experiments/`, `analysis/`, `tests/`
- Set up `pyproject.toml` with editable install and CLI entry point
- Added Makefile with targets for setup, train, sweep, ablation, test, figures
- Created `.cursor/rules/project.mdc` for agent context
- Next: implement core mixture models (start with sMM)
