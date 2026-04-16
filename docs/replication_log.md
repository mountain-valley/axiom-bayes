# Replication Log

Running record of progress, decisions, and findings. Append new entries at the top.

---

## 2026-04-15 — Parallel no-video smoke test passed; arrays resubmitted cleanly

- Ran a short end-to-end concurrent smoke test with two local jobs using
  `experiments/run_sweep.py --run-one` (30 steps, `--fast`) to validate that:
  - outputs do not collide under the isolated temp working-directory strategy,
  - CSVs are saved to `results/` correctly.
- Smoke outputs written successfully:
  - `results/smoke_no_video/dof_offset_2.0_seed0.csv` (30 rows)
  - `results/smoke_no_video/dof_offset_50.0_seed1.csv` (30 rows)
- After successful smoke verification, cancelled and cleanly resubmitted Phase 1 arrays:
  - cancelled: `11514163`, `11514164`, `11514165`
  - resubmitted:
    - Task 2 sMM (5k): `11514396`
    - Task 3 tMM (5k): `11514397`
    - Task 4 rMM (5k): `11514398`
- These runs now use the updated sweep runner defaults (`--no_video` via
  `run_sweep.py`, unless `--with-video` is explicitly requested).

---

## 2026-04-15 — Added `--no_video` mode for cleaner sweep runs

- Confirmed AXIOM had no built-in no-video CLI flag.
- Added `--no_video` to AXIOM config/CLI parsing in `vendor/axiom/defaults.py`
  and wired it into `ExperimentConfig`.
- Updated `vendor/axiom/main.py` so when `config.no_video` is enabled it:
  - skips frame accumulation into `observations`,
  - skips `mediapy.show_videos(...)`,
  - skips wandb media/scalar logging block at the end.
- Updated `experiments/run_sweep.py` to pass `--no_video` by default for sweep
  runs; added `--with-video` to opt back in when needed.
- This makes cluster sweeps cleaner and avoids ffmpeg/video side-effects while
  preserving CSV outputs.

---

## 2026-04-15 — Slurm sweep reliability fix + full Phase 1 rerun at 5k

- Diagnosed sweep-output loss on cluster runs:
  - AXIOM writes CSV first, then attempts video export via `mediapy`/`ffmpeg`.
  - On compute nodes without `ffmpeg`, `main.py` exits non-zero after CSV write.
  - `experiments/run_sweep.py` used `subprocess.run(..., check=True)`, so CSV copy
    into `results/` was skipped on non-zero exit.
  - All array tasks also shared `cwd=vendor/axiom`, so concurrent runs clobbered
    `explode.csv` (`mode="w"`), causing race conditions and data loss.
- Implemented reliability fix in `experiments/run_sweep.py`:
  - run each AXIOM invocation in an isolated temporary working directory under
    `results/.tmp_axiom_runs/` (prevents concurrent overwrite),
  - use `check=False`, copy CSV if present and non-empty even when process exits
    non-zero (e.g., video export failure),
  - warn on non-zero exit so failures remain visible in logs.
- Cancelled broken arrays and resubmitted all Phase 1 sweeps at 5000 steps:
  - cancelled: `11510724` (Task 2), `11512513` (Task 3), `11512514` (Task 4),
  - resubmitted:
    - Task 2 (sMM, 5k): `jobs_task2_smm_5k_rerun.txt` -> `11514163` (48 jobs)
    - Task 3 (tMM, 5k): `jobs_task3_tmm_5k_rerun.txt` -> `11514164` (42 jobs)
    - Task 4 (rMM, 5k): `jobs_task4_rmm_5k_rerun.txt` -> `11514165` (60 jobs)
- Decision: keep full planning/BMR defaults and use 5000 steps across Phase 1
  for compute-budget consistency.

---

## 2026-04-15 — Tasks 3 & 4 resubmitted at 5k steps; compute-budget policy documented

- **Rationale:** To manage cluster time, adopted a "reduce steps, not planning
  quality" policy. The AXIOM paper shows most learning occurs in the first 5k
  steps. Using `--fast` (16 rollouts, 200 BMR pairs) would degrade planner and
  BMR fidelity, confounding the hyperparameter sensitivity signal. Reducing to
  5000 steps preserves full inference quality (512 rollouts, 2000 BMR pairs)
  while halving wall-clock time per job.
- Cancelled original Task 3 array (`11510773`, 10k steps, 42 jobs).
  Task `11510773_1` had already completed before cancellation; its 10k-step
  result is in `results/prior_sensitivity/tmm/` but will not be used for
  analysis (mismatched step count with the rest of the sweep).
- Resubmitted Task 3 at 5000 steps:
  - `jobs_task3_tmm_5k.txt` (42 jobs)
  - job id: `11512513`
- Submitted Task 4 (rMM prior sensitivity) at 5000 steps:
  - `experiments/configs/prior_sensitivity_rmm.yaml` (Explode, 3 seeds)
  - `jobs_task4_rmm_5k.txt` (60 jobs)
  - job id: `11512514`
- Created `results/prior_sensitivity/rmm/` directory.
- Documented the 5k-step compute-budget guidance in:
  - `docs/roadmap.md` (Compute Notes section)
  - `docs/supercomputer.md` (Resource Estimates section)

---

## 2026-04-15 — Task 3 originally launched on Slurm (tMM — cancelled, see above)

- Started Task 3 (`docs/roadmap.md`) experiment execution on the cluster using
  `experiments/configs/prior_sensitivity_tmm.yaml` (Explode, 3 seeds, 10,000 steps).
- Created Task 3 results directory before submission:
  - `results/prior_sensitivity/tmm/`
- Generated Task 3 array job list with 42 runs (5 `sigma_sqr` + 5
  `logp_threshold` + 4 `n_total_components`, each x 3 seeds):
  - `jobs_task3_tmm.txt`
- Submitted Slurm array job (no `--qos` flag):
  - `sbatch --array=1-42 --export=JOBLIST=/home/tbday/axiom-bayes/jobs_task3_tmm.txt experiments/slurm/run_sweep_array.sh`
  - job id: `11510773`
- At submission time:
  - baseline `11510639` still running,
  - Task 2 array `11510724` mostly running with some pending tasks,
  - Task 3 array `11510773` pending behind current QOS/user node limits.

---

## 2026-04-15 — Task 2 launched on Slurm (sMM prior sensitivity)

- Started Task 2 (`docs/roadmap.md`) experiment execution on the cluster using
  `experiments/configs/prior_sensitivity_smm.yaml` (Explode, 3 seeds, 10,000 steps).
- Created missing results directories before submission:
  - `results/`
  - `results/prior_sensitivity/smm/`
- Generated Task 2 array job list with 48 runs (5 `dof_offset` + 6
  `smm_eloglike_threshold` + 5 `scale_factor`, each x 3 seeds):
  - `jobs_task2_smm.txt`
- Submitted Slurm array job (no `--qos` flag):
  - `sbatch --array=1-48 --export=JOBLIST=/home/tbday/axiom-bayes/jobs_task2_smm.txt experiments/slurm/run_sweep_array.sh`
  - job id: `11510724`
- Baseline job `11510639` remains running; Task 2 array is currently pending and
  queued to start as resources become available.

---

## 2026-04-15 — Prediction-error overhead benchmark documented

- Ran a controlled A/B benchmark to quantify the cost of logging
  `Next-State Prediction Error`.
- Benchmark setup:
  - game: `Explode`
  - steps: `200`
  - seed: `0`
  - planning settings: `--planning_horizon 16 --planning_rollouts 16 --num_samples_per_rollout 1`
  - BMR settings: `--bmr_pairs 200 --bmr_samples 200`
  - backend used for this measurement: `JAX_PLATFORMS=cpu`
- Results:
  - baseline runner (`vendor/axiom/main.py`): `319.56s`
  - prediction-error runner (`experiments/run_with_prediction_error.py`): `348.57s`
  - added overhead: `+29.01s` (~`9.1%`)
- Added the same summary to `README.md` under Task 1 infrastructure notes.

---

## 2026-04-15 — Optional next-state prediction-error logging added

- Added `experiments/run_with_prediction_error.py`, an optional AXIOM runner that
  mirrors the standard loop and writes an extra CSV column:
  `Next-State Prediction Error`.
- Added `--with-prediction-error` to `experiments/run_sweep.py`:
  - default behavior remains unchanged (uses upstream `vendor/axiom/main.py`),
  - when enabled, sweeps call the custom runner to include prediction-error output.
- Extended `tests/test_infrastructure.py`:
  - updated mocked sweep test for the new runner signature,
  - added a test ensuring prediction-error mode selects the custom entrypoint.
- Updated `README.md` Task 1 infrastructure docs with usage example and column behavior.
- Existing results are unaffected; this applies to future runs only.

---

## 2026-04-15 — Task 1 documentation finalized (README usage + outputs)

- Added a dedicated **Task 1 Infrastructure Notes** section to `README.md` to document:
  - how to run `experiments/run_sweep.py` directly and via YAML configs,
  - expected output directory/filename conventions in `results/`,
  - special parameter transforms (`scale_factor`, `discrete_alpha_scale`, `reward_alpha`),
  - verified baseline CSV columns available for downstream analysis,
  - GPU OOM mitigation environment variables for long sweep batches.
- This closes remaining user-facing documentation gaps for Task 1 usage.

---

## 2026-04-15 — Task 1 completed (experiment infrastructure + column audit)

- Completed Task 1 infrastructure work:
  - Enhanced `experiments/run_sweep.py` to support roadmap-specific derived sweeps:
    - `scale_factor` -> `--scale <scaled_vector>`
    - `discrete_alpha_scale` -> scaled `--discrete_alphas`
    - `reward_alpha` -> reward-specific `--discrete_alphas` sweep
  - Added safer filename normalization for parameter values and optional config `name`
    field support so repeated parameters (e.g., multiple `prune_every` sweeps) do not
    overwrite each other.
  - Updated `experiments/configs/bmr_ablation.yaml` to use named sweep blocks
    (`prune_every_replication`, `prune_every_frequency`, `bmr_samples`, `bmr_pairs`).
- Extended `analysis/helpers.py`:
  - Added `group_by_parameter()` for per-step parameter-comparison plotting.
  - Upgraded `summary_stats()` to aggregate across seeds and report mean/std/SEM/95% CI.
  - Kept moving-average helper aligned with paper convention (`window=1000`).
- Expanded `tests/test_infrastructure.py`:
  - Added coverage for `group_by_parameter()`.
  - Added mocked sweep-runner tests validating special-parameter CLI translation and
    expected output CSV naming/layout.
- Verified roadmap-required tests:
  - `make test` -> **14 passed**.
- Verified currently available AXIOM CSV columns from `vendor/axiom/explode.csv`
  (5000 rows):
  - `Step`, `Reward`, `Average Reward`, `Cumulative Reward`,
    `Expected Utility`, `Expected Info Gain`, `Num Components`
- Optional-metric availability note for downstream tasks:
  - Next-state prediction error is **not** present in the standard CSV output and should
    be treated as unavailable unless extra logging is added in a later task.

---

## 2026-04-14 — Task 0 run finalized (GPU + env API compatibility)

- Completed full baseline run workflow on this machine using GPU-backed JAX:
  - `WANDB_MODE=disabled JAX_PLATFORMS=cuda make baseline`
- Resolved CUDA backend bring-up issue by fixing NVIDIA driver/library mismatch
  (`580.126.09` now loaded for both kernel module and userspace).
- Installed CUDA-enabled JAX packages in project venv and verified:
  - `jax.default_backend()` reports `gpu`
  - `jax.devices()` includes `CudaDevice(id=0)`
- Updated local editable `vendor/gameworld` env `reset` signatures to
  `reset(self, *, seed=None, options=None)` and added `super().reset(seed=seed)`
  calls, eliminating Gymnasium reset API deprecation warnings during run setup.
- Added baseline stage messages in `Makefile` so post-progress-bar finalization is
  explicit (`starting AXIOM run` -> `model run complete; finalizing output artifacts`).

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
