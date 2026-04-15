# Replication Log

Running record of progress, decisions, and findings. Append new entries at the top.

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
