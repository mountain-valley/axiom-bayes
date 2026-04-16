# Project Roadmap

This project performs a Bayesian analysis of AXIOM (Heins et al., 2025) — an
active inference agent that uses Bayesian mixture models, variational
inference, Bayesian model reduction, and uncertainty-aware planning to master
pixel-based games without gradient-based optimization.

Rather than reimplementing AXIOM (the source code is available at
`github.com/VersesTech/axiom`), this project installs the official codebase and
conducts three focused analyses that connect the system to CS 677 Bayesian
statistics concepts:

1. **Prior and Hyperparameter Sensitivity** — How do prior specifications and
   model hyperparameters affect perception, dynamics learning, and reward discovery?
2. **Bayesian Model Reduction (BMR) Deep Dive** — How does principled Bayesian
   model selection via expected free energy compare to naive pruning?
3. **Exploration vs. Exploitation** — How does the information-gain term
   (KL divergence on Dirichlet posteriors) affect learning behavior?

**Modular milestone structure:** Each analysis theme is its own phase with
experiments and figures. You can write a report at any stopping point. Phase 0
(setup + baseline) is a prerequisite; after that, Phases 1-3 are largely
independent — they can be done in any order, though doing them in sequence
builds a natural narrative arc (prior specification → model selection →
decision theory).

**Replication vs. novel experiments:** Some experiments replicate results already
presented in the paper (Table 1 ablations, Figure 4b-c). These serve as
verification that our experimental setup reproduces known results before
extending the analysis. They are labeled **(Replication)** below. All other
experiments are novel extensions not present in the paper, labeled **(Novel)**.
Phase 1 is entirely novel. Phases 2 and 3 each begin with a replication anchor
and then extend into new territory.

The old implementation-focused roadmap is preserved in
`docs/roadmap_implementation.md`. The full 10-game replication roadmap is in
`docs/roadmap_replication.md`.

Each task below is self-contained. Point an LLM agent to a specific task by
saying:

> "Read `docs/roadmap.md` and work on Task N."

After completing any task, the agent must:
1. Run `make test` and confirm all tests pass.
2. Append a dated summary to `docs/replication_log.md`.
3. Check off the task in the Progress section below.

---

## Progress

The checklist below is a quick visual summary. The source of truth for what has
actually been completed is `docs/replication_log.md` — it contains dated entries
with details on what was done, what worked, and any open issues. If the checklist
and the log disagree, trust the log.

### Phase 0: Setup and Baseline
- [x] Task 0 — Install AXIOM + Gameworld and verify baseline run
- [x] Task 1 — Experiment infrastructure (sweep runner, logging, analysis helpers)

### Phase 1: Prior and Hyperparameter Sensitivity
- [ ] Task 2 — sMM prior sensitivity experiments
- [ ] Task 3 — tMM hyperparameter sensitivity experiments
- [ ] Task 4 — rMM prior sensitivity experiments
- [ ] Task 5 — Phase 1 figures and analysis

### Phase 2: Bayesian Model Reduction Deep Dive
- [ ] Task 6 — Instrument BMR logging
- [ ] Task 7 — BMR ablation experiments (with/without, frequency, sample count)
- [ ] Task 8 — Naive pruning baseline for comparison
- [ ] Task 9 — Phase 2 figures and analysis

### Phase 3: Exploration vs. Exploitation
- [ ] Task 10 — Info-gain sweep experiments
- [ ] Task 11 — Temporal decomposition analysis
- [ ] Task 12 — Phase 3 figures and analysis

### Report and Presentation
- [ ] Task 13 — Draft the report (at whatever phase you've reached)
- [ ] Task 14 — Create the presentation

---

## What a Report Looks Like at Each Stopping Point

| Stop after   | Report angle                                                      | Key figures                                                                                             |
| --------------| -------------------------------------------------------------------| ---------------------------------------------------------------------------------------------------------|
| Phase 0 only | Baseline AXIOM performance on Gameworld                           | Learning curve, component counts, basic diagnostics                                                     |
| Phase 1      | + Prior and hyperparameter sensitivity of Bayesian mixture models | + Heatmaps of reward vs. prior/hyperparameter settings, convergence curves, component-count sensitivity |
| Phase 2      | + Bayesian model selection via BMR                                | + Component count over time (BMR vs. no-BMR), merge-score distributions, performance comparison         |
| Phase 3      | + Bayesian decision theory: exploration vs. exploitation          | + Reward curves by info-gain coefficient, utility/info-gain timeseries, behavior comparison             |
| All phases   | Complete Bayesian analysis of AXIOM across three themes           | All of the above, with a unifying narrative                                                             |

Each row is a complete story. Later phases add depth, but every phase stands alone.

---

## Project Framing for CS 677

The report and presentation should be framed around these Bayesian inference themes
(all covered in CS 677) rather than as an RL project:

1. **Priors and posterior updates** — Most AXIOM models use exponential-family
   likelihoods with conjugate priors: NIW for the iMM and rMM Gaussians,
   Dirichlet for Categoricals, Gamma for sMM color variances. The tMM is an
   exception — it uses uniform (flat) priors on its linear parameters D, b and
   fixes the covariance to 2I, so it is not a textbook conjugate Bayesian
   regression model. Phase 1 asks: how sensitive is the system to these prior
   specifications and structural hyperparameters?

2. **Bayesian model selection** — BMR evaluates the expected free energy change
   of merging redundant model components. Phase 2 asks: how effective is principled
   model selection compared to heuristic alternatives?

3. **Decision-making under uncertainty** — The planner balances reward-seeking
   (utility) with information-seeking (KL divergence on Dirichlet posteriors).
   Phase 3 asks: how does the epistemic exploration bonus affect learning?

4. **Variational inference** — AXIOM minimizes variational free energy via
   coordinate-ascent mean-field VI. This is the underlying inference engine
   and appears throughout all three analyses.

5. **Nonparametric Bayes** — Stick-breaking priors allow automatic model
   expansion. Component counts appear as a diagnostic in every phase.

---

## Compute Notes

AXIOM is JAX-based and benefits from GPU acceleration. All experiments can be
run on CPU with reduced hyperparameters for faster iteration:

```bash
# Fast CPU run (~5-10 min): reduced planning and BMR
python main.py --game=Explode \
    --planning_horizon 16 --planning_rollouts 16 \
    --num_samples_per_rollout 1 --num_steps 5000 \
    --bmr_pairs 200 --bmr_samples 200

# Full run (~30-60 min on GPU, hours on CPU)
python main.py --game=Explode --num_steps 10000
```

For sweeps, start with fast CPU settings to verify the pipeline, then do final
runs with full settings. The experiment scripts should support both modes via
a `--fast` flag.

**Compute-budget guidance (5k vs 10k steps):** When compute time is limited,
prefer reducing step count to 5000 over using `--fast`. The `--fast` flag
degrades planning quality (512 → 16 rollouts) and BMR quality (2000 → 200
pairs/samples), which confounds prior/hyperparameter sensitivity results.
Reducing to 5000 steps preserves full inference fidelity while still capturing
the majority of AXIOM's learning — the paper notes that "AXIOM often reaches
most of its final reward within the first 5k steps." Use "last 1000 steps"
metrics (steps 4000–5000) for summary statistics. All runs *within* a sweep
must use the same step count so comparisons are fair; different tasks may use
different step counts.

---

## Game Selection Rationale

Each analysis phase uses a different primary game, chosen so the Bayesian
mechanism under study has the largest, most interpretable effect. The paper's
ablation results (Table 1, Appendix E.2) show that BMR, information gain, and
prior sensitivity have game-dependent impacts. Using the same game everywhere
would produce weak signal for Phases 2 and 3.

| Phase | Primary Game | Paper Evidence (Table 1) |
| ----- | ------------ | ------------------------ |
| Phase 1 (Priors) | **Explode** | Canonical demo game; moderate object count; already used for baseline and in-progress sweeps. No ablation data for prior sensitivity (all Phase 1 experiments are novel), so any game with nontrivial reward dynamics works. |
| Phase 2 (BMR) | **Gold** | Cumulative reward drops from 190 to 45 without BMR (76% decline). The 2D free-movement arena requires spatial generalization of reward clusters — exactly what BMR's merge procedure enables. Hunt shows a similar effect (206 → 48) and is a strong secondary choice. |
| Phase 3 (Info-gain) | **Bounce** | Cumulative reward drops from 27 to 8 without IG (70% decline). The Pong-like mechanics create a natural explore-then-exploit arc: the agent must actively seek ball–paddle interactions to learn dynamics before it can score. |

**Why not the same game for all phases?** Explode shows only a 9% reward drop
without BMR and a modest IG effect concentrated in the first few hundred steps
(Figure 4c). Using it for Phases 2–3 would yield flat, uninformative sweep
curves. Matching each phase to the game where its mechanism matters most
produces clearer figures and a more compelling report.

**Secondary games (if time permits):** Each task below names an optional
secondary game for generalization checks. Running a second game adds breadth
but is not required for a complete analysis.

---

## Phase 0: Setup and Baseline

**Milestone:** AXIOM runs end-to-end on at least one Gameworld game, producing
a reward CSV and gameplay video. Experiment infrastructure is in place for
subsequent phases.

### Task 0 — Install AXIOM + Gameworld and Verify Baseline Run

**Depends on:** nothing

**What to do:**

1. Clone the official AXIOM repo into `vendor/axiom`:
   ```bash
   mkdir -p vendor
   git clone https://github.com/VersesTech/axiom.git vendor/axiom
   ```

2. Clone Gameworld:
   ```bash
   git clone https://github.com/VersesTech/gameworld.git vendor/gameworld
   ```

3. Install both into the project venv (AXIOM requires Python 3.10-3.11):
   ```bash
   pip install -e vendor/gameworld
   pip install -e vendor/axiom
   ```

4. Verify with a short test run:
   ```bash
   cd vendor/axiom
   python main.py --game=Explode --num_steps 100 \
       --planning_horizon 8 --planning_rollouts 8 \
       --num_samples_per_rollout 1
   ```
   This should produce `explode.csv` and `explode.mp4` in the vendor/axiom directory.

5. Run a full baseline (fast CPU settings):
   ```bash
   python main.py --game=Explode --num_steps 5000 \
       --planning_horizon 16 --planning_rollouts 16 \
       --num_samples_per_rollout 1 \
       --bmr_pairs 200 --bmr_samples 200
   ```
   Copy the output CSV to `results/baseline/explode_baseline.csv`.

**Tests to write (`tests/test_setup.py`):**
- Import `axiom` and `gameworld.envs` without errors.
- Create a Gameworld environment, reset, take one step, verify observation shape.
- Import AXIOM's `defaults.parse_args` and verify it produces a valid config.

**Notes:**
- AXIOM depends on `jax`, `equinox`, `wandb`, `mediapy`, and `gameworld`.
  These are pulled in via `pip install -e vendor/axiom`.
- If wandb prompts for login during runs, set `WANDB_MODE=disabled` in the
  environment to suppress it.
- The project's own `pyproject.toml` should list `jax`, `matplotlib`, `seaborn`,
  `pandas`, `numpy`, `scipy` as direct dependencies (for analysis scripts).
  AXIOM's own deps are handled by its own pyproject.toml.

---

### Task 1 — Experiment Infrastructure

**Files:** `experiments/run_sweep.py`, `experiments/configs/`, `analysis/helpers.py`
**Depends on:** Task 0

**What to build:**

1. **Sweep runner** (`experiments/run_sweep.py`): A script that takes a parameter
   name, a list of values, a game, number of seeds, and step count, then runs
   AXIOM for each combination. Each run should:
   - Call `vendor/axiom/main.py` (via subprocess or by importing and calling
     `main()` directly) with the appropriate CLI args.
   - Save the output CSV to `results/<experiment_name>/<param>_<value>_seed<N>.csv`.
   - Support a `--fast` flag that applies the reduced CPU hyperparameters.

   Example usage:
   ```bash
   python experiments/run_sweep.py \
       --param info_gain --values 0.0 0.05 0.1 0.5 1.0 \
       --game Explode --seeds 3 --steps 5000 --fast
   ```

2. **Config templates** (`experiments/configs/`): YAML files defining the
   parameter grids for each phase:
   - `prior_sensitivity_smm.yaml` — sMM sweep parameters
   - `prior_sensitivity_tmm.yaml` — tMM sweep parameters
   - `prior_sensitivity_rmm.yaml` — rMM sweep parameters
   - `bmr_ablation.yaml` — BMR experiment parameters
   - `info_gain_sweep.yaml` — exploration/exploitation sweep

3. **Analysis helpers** (`analysis/helpers.py`): Functions to:
   - Load a directory of CSV results into a single DataFrame.
   - Compute summary statistics (mean, std, CI across seeds).
   - Compute moving-average reward (window=1000, matching the paper).
   - Group results by parameter value for comparison plots.

4. **Makefile targets**: Add targets for running each phase's experiments and
   generating figures (see updated Makefile).

5. **Verify available CSV columns**: Before building analysis code, inspect the
   CSV output from a baseline run to determine exactly which columns the official
   AXIOM code logs. Document the available columns in `docs/replication_log.md`.
   Any metric in later tasks marked "optional — only if verified available" depends
   on this step. If a desired metric (e.g., next-state prediction error) is not
   logged, mark it as unavailable so downstream tasks skip it rather than fail.

**Tests to write (`tests/test_infrastructure.py`):**
- `load_results_dir` correctly aggregates multiple CSVs.
- Moving-average reward computation matches a hand-computed example.
- Sweep runner generates the expected directory structure (mock the actual runs).

---

## Phase 1: Prior and Hyperparameter Sensitivity

**Milestone:** Systematic experiments showing how prior hyperparameters and
structural settings in the sMM, tMM, and rMM affect AXIOM's performance, with
figures connecting each parameter to its Bayesian interpretation.

**Bayesian theme:** Most AXIOM models use conjugate priors (NIW, Dirichlet,
Gamma) that enable gradient-free posterior updates, while the tMM uses flat
priors with a fixed covariance. All have hyperparameters that encode prior
beliefs or structural assumptions about object appearance (sMM), dynamics
regularity (tMM), and interaction structure (rMM). How sensitive is the agent
to these specifications?

**All Phase 1 experiments are novel.** The paper uses fixed hyperparameters and
does not analyze prior or hyperparameter sensitivity.

### Task 2 — sMM Prior Sensitivity Experiments

**Files:** `experiments/configs/prior_sensitivity_smm.yaml`, analysis notebook
**Depends on:** Task 1

**What to do:** Run AXIOM on Explode while varying sMM prior hyperparameters
one at a time, holding all others at default values.

**Parameters to sweep (from `defaults.py` and `SMMConfig`):**

1. **`dof_offset`** (default: 10.0) — Degrees of freedom offset for the Gamma
   prior on observation noise. Higher values → stronger prior belief that
   observations are precise.
   - Sweep: `[2.0, 5.0, 10.0, 20.0, 50.0]`
   - CLI: `--dof_offset <value>`

2. **`scale`** (default: `[0.075, 0.075, 0.75, 0.75, 0.75]`) — Prior scale for
   each observation dimension (x, y, r, g, b). Controls expected observation
   variance. Position dims (0.075) are tight; color dims (0.75) are loose.
   - Sweep: scale all values by `[0.25, 0.5, 1.0, 2.0, 4.0]` (i.e., multiply
     the default vector by the factor).
   - CLI: `--scale <comma-separated>`

3. **`smm_eloglike_threshold`** (default: 5.7) — Expected log-likelihood
   threshold for slot expansion. Lower → more aggressive expansion (more slots
   created); higher → more conservative.
   - Sweep: `[3.0, 4.0, 5.0, 5.7, 7.0, 10.0]`
   - CLI: `--smm_eloglike_threshold <value>`

**Run configuration:**
- Game: Explode (primary). If time permits: Gold (enables cross-phase
  comparison with Phase 2) or Bounce.
- Seeds: 3 per configuration (more for final figures)
- Steps: 5000 (fast) or 10000 (full)

**Metrics to extract from each run's CSV:**
- Cumulative reward (sum of last 1000 steps)
- Average reward (mean of last 1000 steps)
- Number of active sMM components at end of run (from `Num Components` column)

**Bayesian interpretation (for report):**
- `dof_offset` controls the strength of the Gamma prior on observation noise
  σ_c. A large offset makes the agent confident about low noise, which sharpens
  slot assignments but can cause poor segmentation when the prior is wrong.
- `scale` encodes the expected observation variance per dimension. Misspecified
  scale (too tight on color → everything looks identical; too loose on position
  → objects blur together) directly illustrates the effect of prior misspecification.
- `smm_eloglike_threshold` controls the Bayesian nonparametric expansion: it is
  the bar a new observation must clear to justify a new component rather than
  being explained by existing ones. This is the stick-breaking prior in action.

**What to save:** CSVs to `results/prior_sensitivity/smm/`.

---

### Task 3 — tMM Hyperparameter Sensitivity Experiments

**Files:** `experiments/configs/prior_sensitivity_tmm.yaml`, analysis notebook
**Depends on:** Task 1

**What to do:** Sweep tMM hyperparameters that control dynamics learning.
Note: the tMM does not use conjugate priors — it uses uniform (flat) priors on
its linear parameters D, b with a fixed covariance of 2I. The parameters swept
here are structural and likelihood hyperparameters, not prior hyperparameters.

**Parameters to sweep (from `defaults.py` and `TMMConfig`):**

1. **`sigma_sqr`** (default: 2.0) — Gaussian likelihood variance for transition
   matching. Controls how precisely a transition must match an existing mode to
   be explained by it. Lower → tighter match required → more modes created.
   - Sweep: `[0.5, 1.0, 2.0, 4.0, 8.0]`
   - CLI: `--sigma_sqr <value>`

2. **`logp_threshold`** (default: -0.00001) — Log-probability threshold below
   which a new tMM mode is created. More negative → harder to trigger expansion.
   - Sweep: `[-0.1, -0.01, -0.001, -0.00001, -0.000001]`
   - CLI: `--logp_threshold <value>`

3. **`n_total_components`** (default: 500) — Maximum number of tMM modes.
   This is the truncation level of the stick-breaking prior.
   - Sweep: `[50, 100, 200, 500]`
   - CLI: `--n_total_components <value>`

**Metrics to extract:**
- Cumulative reward (last 1000 steps)
- Number of active tMM modes at end of run

**Optional metrics (only if verified available in Task 1):**
- Quality of next-state predictions (requires the official code to log
  prediction error; skip if not exposed)

**Bayesian interpretation (for report):**
- Unlike the sMM and rMM, the tMM does not use conjugate priors — its linear
  parameters have flat priors and fixed covariance (2I). The parameters swept
  here are therefore likelihood and structural hyperparameters.
- `sigma_sqr` is the fixed likelihood variance in the tMM's linear-Gaussian
  model. It plays the same role as the noise variance in Bayesian linear
  regression: too small → overfitting (every transition is unique), too large →
  underfitting (all transitions look the same, single mode explains everything).
- `logp_threshold` is the stick-breaking expansion criterion — the posterior-
  predictive density must fall below this to justify a new component. This is
  Bayesian model comparison: the agent decides whether existing modes suffice
  or a new one is needed.
- `n_total_components` is the truncation level of the nonparametric prior.
  It sets an upper bound on model complexity.

**What to save:** CSVs to `results/prior_sensitivity/tmm/`.

---

### Task 4 — rMM Prior Sensitivity Experiments

**Files:** `experiments/configs/prior_sensitivity_rmm.yaml`, analysis notebook
**Depends on:** Task 1

**What to do:** Sweep rMM hyperparameters that control interaction and reward
learning.

**Parameters to sweep (from `defaults.py` and `RMMConfig`):**

1. **`discrete_alphas`** (default: `[1e-4, 1e-4, 1e-4, 1e-4, 1.0, 1e-4]`) —
   Dirichlet prior concentrations for the six discrete features (identity,
   switch, action, reward, and others). The 5th element (1.0, for reward) is
   much larger than the others (1e-4), encoding a strong prior that reward is
   common.
   - Sweep the uniform scale: multiply all alphas by `[0.01, 0.1, 1.0, 10.0, 100.0]`.
   - Also sweep the reward-specific alpha (5th element) independently:
     `[1e-4, 0.01, 0.1, 1.0, 10.0]` while holding others at 1e-4.
   - CLI: `--discrete_alphas <space-separated>`

2. **`cont_scale_switch`** (default: 75.0) — Precision scaling for the
   continuous features in the rMM Gaussian components. Higher → each component
   is more localized in feature space.
   - Sweep: `[10.0, 25.0, 75.0, 150.0, 300.0]`
   - CLI: `--cont_scale_switch <value>`

3. **`r_interacting`** (default: 0.075) — Radius threshold for object
   interaction detection. Objects closer than this are considered interacting.
   - Sweep: `[0.025, 0.05, 0.075, 0.15, 0.3]`
   - CLI: `--r_interacting <value>`

**Metrics to extract:**
- Cumulative reward (last 1000 steps)
- Number of active rMM components at end of run
- Reward prediction accuracy (fraction of correctly predicted reward signs)

**Bayesian interpretation (for report):**
- `discrete_alphas` are Dirichlet prior concentrations. Small α → sparse
  posterior (most mass on one category), large α → diffuse posterior (uniform
  over categories). The asymmetry (1.0 for reward vs. 1e-4 for others) encodes
  the prior belief that reward is more predictable than other discrete features.
  Varying this illustrates how Dirichlet priors shape categorical inference.
- `cont_scale_switch` controls the precision (inverse variance) of the Gaussian
  likelihood in the rMM's continuous dimensions. This is the observation noise
  assumption for interaction features — analogous to σ² in Bayesian regression.
- `r_interacting` is a design choice (not a Bayesian prior per se) that
  determines which observations reach the rMM. It controls the data the
  Bayesian model sees, illustrating how preprocessing interacts with inference.

**What to save:** CSVs to `results/prior_sensitivity/rmm/`.

---

### Task 5 — Phase 1 Figures and Analysis

**Files:** `analysis/notebooks/phase1_prior_sensitivity.ipynb`, `analysis/plotting.py`
**Depends on:** Tasks 2-4

**Figures to produce:**

1. **Reward sensitivity heatmap (one per model):** X-axis = parameter value,
   Y-axis = parameter name (within each model subplot), color = mean cumulative
   reward (last 1000 steps). Three heatmaps side by side: sMM parameters, tMM
   parameters, rMM parameters. If secondary games were run, add them as
   additional rows.
   Save as `results/figures/phase1_reward_heatmap.png`.

2. **Component count vs. parameter value:** Line plots showing how the number
   of active components (sMM slots, tMM modes, rMM clusters) changes as a
   function of the corresponding expansion/prior parameter. Error bars from
   seeds. Save as `results/figures/phase1_component_sensitivity.png`.

3. **Learning curve comparison:** For each model's most impactful parameter,
   overlay the learning curves (moving-average reward vs. step) for 3-5
   parameter values on the same plot. Highlight the default value.
   Save as `results/figures/phase1_learning_curves.png`.

4. **Parameter-reward scatter with Bayesian interpretation:** For 1-2 key
   parameters, plot reward vs. parameter value with annotations explaining
   the under-regularized and over-regularized regimes.
   Save as `results/figures/phase1_interpretation.png`.

**Analysis notebook structure:**
- Load all results from `results/prior_sensitivity/`.
- Compute summary statistics grouped by (model, parameter, value).
- Generate each figure with clear axis labels and legends.
- Write 1-2 paragraphs per figure interpreting the results through the Bayesian
  lens (these paragraphs feed directly into the report).

---

## Phase 2: Bayesian Model Reduction Deep Dive

**Milestone:** Detailed analysis of how BMR works inside AXIOM — what gets
merged, what the merge scores look like, and how performance compares to
simpler alternatives.

**Bayesian theme:** BMR is a sampling-based greedy model-reduction procedure.
Every 500 frames, AXIOM samples up to 2000 used rMM components, scores their
mutual expected log-likelihoods on data generated through ancestral sampling,
and greedily tests merge candidates. A merge is accepted if it decreases the
expected free energy of the multinomial distributions over reward and next-tMM
switch, conditioned on the sampled data; otherwise it is rolled back. This is a
principled Bayesian-style model-reduction procedure, but it is not a simple
closed-form Bayes-factor calculation — it relies on sampled data and greedy
search.

### Task 6 — Instrument BMR Logging

**Files:** Patch to `vendor/axiom/axiom/models/rmm.py`, new `analysis/bmr_logger.py`
**Depends on:** Task 0

**What to do:** Add lightweight logging to the BMR code path so that each BMR
round records its internal decisions. This is the only modification to the
official AXIOM source code.

**Where BMR happens in the code:**
- `main.py` calls `ax.reduce_fn_rmm()` every `prune_every` steps.
- `reduce_fn_rmm` (in `axiom/infer.py`) calls `rmm_tools.run_bmr()`.
- `run_bmr` (in `axiom/models/rmm.py`) evaluates merge candidates and accepts
  or rejects them based on expected free energy.

**What to log per BMR round:**
- Timestep when BMR fires.
- Number of active rMM components before and after.
- Number of candidate pairs evaluated.
- Number of merges accepted.
- For each evaluated pair: the merge score (free energy change) and whether
  it was accepted.
- Total active component count after merging.

**Implementation approach:**
- Create `analysis/bmr_logger.py` with a `BMRLogger` class that writes to a
  JSON-lines file (`results/bmr_logs/<run_name>.jsonl`).
- Patch `vendor/axiom/axiom/models/rmm.py`'s `run_bmr` function to accept an
  optional logger callback. Alternatively, create a thin wrapper
  `experiments/run_with_bmr_logging.py` that monkeypatches the function to
  capture the needed data via JAX's callback mechanisms.
- Keep patches minimal and clearly commented so they can be rebased if the
  upstream code changes.

**Tests to write (`tests/test_bmr_logger.py`):**
- BMRLogger writes valid JSONL.
- Loading a JSONL log and computing summary statistics works correctly.

---

### Task 7 — BMR Ablation Experiments

**Files:** `experiments/configs/bmr_ablation.yaml`
**Depends on:** Tasks 1, 6

**What to do:** Run AXIOM with different BMR configurations.

**Why Gold?** Gold was chosen because the player moves freely across a 2D arena,
so reward-relevant interactions (collecting coins, hitting dogs) occur at many
spatial locations. BMR enables spatial generalization by merging single-event
rMM clusters — without it, cumulative reward drops from 190 to 45 (Table 1),
the largest BMR effect in the benchmark.

**Experiments:**

1. **(Replication)** **BMR enabled vs. disabled** — corresponds to the "no BMR"
   row in Table 1 and the component-count plot in Figure 4b:
   - Enabled (default): `--prune_every 500`
   - Disabled: `--prune_every 999999` (effectively never)
   - Game: Gold (primary). If time permits: Hunt (similar 4x BMR effect) or
     Explode (for comparison with Figure 4b, which uses Explode).
   - Seeds: 3-5
   - Steps: 10000

2. **(Novel)** **BMR frequency sweep:**
   - Values: `--prune_every` in `[250, 500, 1000, 2000, 5000]`
   - Game: Gold
   - Seeds: 3

3. **(Novel)** **BMR sample count sweep:**
   - Values: `--bmr_samples` in `[100, 500, 1000, 2000, 5000]`
   - Also vary `--bmr_pairs` with the same values.
   - Game: Gold
   - Seeds: 3

**Metrics to extract:**
- Cumulative reward (last 1000 steps)
- rMM component count at end of run
- rMM component count over time (from logged data if BMR logging is active)

**What to save:** CSVs to `results/bmr_ablation/`. BMR logs (if instrumented)
to `results/bmr_logs/`.

---

### Task 8 — Naive Pruning Baseline

**Files:** `experiments/run_naive_pruning.py`
**Depends on:** Task 7

**What to do:** Implement a simple alternative to BMR for comparison: count-based
pruning. Every `prune_every` steps, remove rMM components that have been assigned
fewer than N data points since the last prune.

**Implementation:**
- This requires a small patch to the AXIOM main loop. After the standard
  `step_fn`, track per-component assignment counts. At prune intervals, zero out
  components below the count threshold instead of running BMR.
- Alternatively, implement this as a post-hoc analysis: run without BMR, save
  the rMM state at prune intervals, and simulate what count-based pruning would
  have done. This avoids modifying the inference loop.

**Experiments:**
- Run with count-based pruning at threshold `[1, 5, 10, 50]`.
- Compare against BMR and no-pruning on Gold with the same seed combinations.

**Bayesian interpretation (for report):**
- BMR scores merge candidates by sampling data from the generative model and
  evaluating whether merging two rMM components decreases expected free energy
  over reward and tMM-switch multinomials. Count-based pruning uses a
  frequentist heuristic (remove rare components). Comparing them illustrates
  whether a principled, sampling-based Bayesian model-reduction procedure
  outperforms simple heuristics — and under what conditions.

**What to save:** CSVs to `results/bmr_ablation/naive_pruning/`.

---

### Task 9 — Phase 2 Figures and Analysis

**Files:** `analysis/notebooks/phase2_bmr.ipynb`, `analysis/plotting.py`
**Depends on:** Tasks 7-8

**Figures to produce:**

1. **(Replication + Extension — Figure 4b)** **Component count over time
   (BMR vs. no-BMR):** Two lines on the same plot showing rMM component count
   over training steps. The BMR line should show the characteristic sawtooth
   pattern (growth, then sharp drops at prune intervals). Note: Figure 4b in
   the paper uses Explode; reproducing this on Gold is a novel extension that
   tests whether the same pattern appears in a game where BMR has a much
   larger impact.
   Save as `results/figures/phase2_component_count.png`.

2. **(Novel)** **Merge-score distribution:** Histogram of expected free energy
   changes across all evaluated pairs in one BMR round. Mark the accept/reject
   threshold. Show that most pairs are clearly reject (very different components)
   with a smaller peak of accepted merges.
   Save as `results/figures/phase2_merge_scores.png`.

3. **(Novel)** **BMR frequency and sample-count sensitivity:** Two line plots
   showing cumulative reward as a function of `prune_every` and `bmr_samples`.
   Save as `results/figures/phase2_bmr_params.png`.

4. **(Novel)** **BMR vs. naive pruning comparison:** Bar chart comparing
   cumulative reward for (a) full BMR, (b) no pruning, (c) count-based pruning
   at best threshold.
   Save as `results/figures/phase2_bmr_vs_naive.png`.

5. **(Novel)** **Performance vs. model complexity:** Scatter plot of final
   cumulative reward vs. final rMM component count, across all BMR
   configurations. Show that BMR achieves good reward with fewer components
   (Occam's razor in action).
   Save as `results/figures/phase2_reward_vs_complexity.png`.

**Analysis notebook structure:**
- Load CSVs from `results/bmr_ablation/` and BMR logs from `results/bmr_logs/`.
- Compute per-configuration summary statistics.
- Generate all figures.
- Write interpretive paragraphs connecting results to Bayesian model selection.

---

## Phase 3: Exploration vs. Exploitation

**Milestone:** Empirical analysis of how AXIOM's information-gain term
(epistemic exploration via KL divergence on Dirichlet posteriors) affects
learning behavior and reward.

**Bayesian theme:** Active inference balances pragmatic value (expected reward)
with epistemic value (information gain — how much the agent would learn about
its world model). The information gain is computed as a KL divergence between
Dirichlet posteriors before and after imagined observations. This is pure
Bayesian decision theory.

### Task 10 — Info-Gain Sweep Experiments

**Files:** `experiments/configs/info_gain_sweep.yaml`
**Depends on:** Task 1

**What to do:** Sweep the `info_gain` coefficient that weights the epistemic
exploration bonus in the planner's objective.

**Parameters:**
- **(Replication — Table 1 "no IG" ablation)** `--info_gain 0.0` vs. `0.1`:
  Reproduces the paper's comparison of full AXIOM against the "no information
  gain" variant.
- **(Novel)** Additional `--info_gain` values `[0.01, 0.05, 0.5, 1.0]`:
  The paper only tests on/off; this sweep reveals the full sensitivity curve.
- Full sweep list: `--info_gain` in `[0.0, 0.01, 0.05, 0.1, 0.5, 1.0]`
  - 0.0 = pure exploitation (no exploration bonus)
  - 0.1 = default
  - 1.0 = heavy exploration
- Game: Bounce (primary). If time permits: Cross (shows the opposite effect —
  IG hurts due to negative-reward car collisions being information-rich,
  providing a compelling contrast).
- Seeds: 3-5
- Steps: 10000 (or 5000 with fast settings)

**Why Bounce?** Bounce shows the clearest IG benefit in the paper (Table 1:
27 vs. 8 without IG, a 70% drop). The Pong-like mechanics create a natural
explore-then-exploit arc: the agent must actively seek ball–paddle interactions
to learn dynamics before it can reliably score.

**Also sweep (if time permits):**
- **(Novel — appendix has sample-count ablation but not horizon)**
  Planning horizon: `--planning_horizon` in `[8, 16, 24, 32]` at default
  info_gain, to see how planning depth interacts with exploration.

**Metrics to extract from each run's CSV:**
- Reward (per-step and cumulative)
- Expected Utility (logged per step)
- Expected Info Gain (logged per step)
- Num Components (logged per step)

**What to save:** CSVs to `results/info_gain_sweep/`.

---

### Task 11 — Temporal Decomposition Analysis

**Files:** `analysis/notebooks/phase3_exploration.ipynb`
**Depends on:** Task 10

**What to do:** Analyze the temporal dynamics of exploration vs. exploitation
using the per-step data from the info-gain sweep.

**Analysis to perform:**

1. **(Replication + Extension — Figure 4c)** **Utility vs. info-gain over
   time:** For the default info_gain=0.1 run, plot expected utility and expected
   info gain as separate timeseries. The paper's Figure 4c (on Explode) shows
   info gain decreasing over time while utility increases. Reproducing this on
   Bounce is a novel extension: we expect the same qualitative pattern but
   potentially a longer exploration phase given Bounce's richer interaction
   dynamics. Verify whether the crossover pattern holds.

2. **(Novel)** **Exploration phase duration:** For each info_gain coefficient,
   estimate when the agent transitions from exploration-dominated to
   exploitation-dominated behavior. Define this as the first timestep where
   expected utility exceeds expected info gain (scaled by the coefficient). Plot
   this transition point as a function of info_gain.

3. **(Novel)** **Component discovery rate:** Plot the rate of new component
   creation (sMM, tMM, rMM) over time for different info_gain values. The
   hypothesis is that higher info_gain leads to faster component discovery
   (the agent actively seeks novel situations), but the paper only tests the
   default value and a no-IG ablation — this sweep will reveal whether the
   relationship is actually monotonic.

4. **(Novel)** **Reward efficiency:** Compute the cumulative reward at step
   2000, 5000, and 10000 for each info_gain value. The hypothesis is that a
   higher exploration bonus hurts early reward but helps late reward. Plot the
   reward trajectories to show whether this explore-exploit tradeoff holds
   across the coefficient range.

**Save intermediate analysis DataFrames** to `results/info_gain_sweep/analysis/`.

---

### Task 12 — Phase 3 Figures and Analysis

**Files:** `analysis/notebooks/phase3_exploration.ipynb`, `analysis/plotting.py`
**Depends on:** Tasks 10-11

**Figures to produce:**

1. **(Novel)** **Learning curves by info-gain coefficient:** Overlay moving-
   average reward curves for all info_gain values on one plot. Highlight the
   default (0.1). The paper only compares on vs. off; this shows the full curve.
   Save as `results/figures/phase3_reward_by_info_gain.png`.

2. **(Replication + Extension — Figure 4c)** **Utility vs. info-gain
   timeseries:** Two-panel plot. Top: expected utility over time. Bottom:
   expected info gain over time. The default (info_gain=0.1) curve on Bounce
   is a novel extension of Figure 4c (which uses Explode); additional curves
   for other coefficient values are entirely novel.
   Save as `results/figures/phase3_utility_infogain_timeseries.png`.

3. **(Novel)** **Exploration-exploitation transition:** Plot the
   exploration→exploitation transition timestep as a function of info_gain
   coefficient. Annotate with the Bayesian interpretation: higher info_gain
   weights epistemic value more heavily, which we expect to delay the transition,
   though the shape of this relationship is an open question.
   Save as `results/figures/phase3_transition_point.png`.

4. **(Novel)** **Cumulative reward at early/mid/late training:** Grouped bar
   chart showing reward at steps 2000, 5000, 10000 for each info_gain value.
   Tests whether exploration has a short-term cost but long-term benefit.
   Save as `results/figures/phase3_reward_over_time.png`.

5. **(Novel)** **Component discovery comparison:** Line plot of active component
   count over time for info_gain=0 vs. default vs. high. Tests the hypothesis
   that exploration-seeking agents discover more structure.
   Save as `results/figures/phase3_component_discovery.png`.

**Analysis notebook structure:**
- Load CSVs from `results/info_gain_sweep/`.
- Run the temporal decomposition analysis from Task 11.
- Generate all figures.
- Write interpretive paragraphs connecting to Bayesian decision theory.

---

## Report and Presentation

### Task 13 — Draft the Report

**File:** `docs/report/report.tex`
**Depends on:** whichever phases you've completed

The report is 4-5 pages, 12pt, 1in margins, single-spaced. The audience is a
CS 677 student. The narrative arc: AXIOM is a strongly Bayesian/probabilistic active-inference system built from Bayesian mixture modules and variational updates.; we analyze three aspects of its Bayesian machinery.

**Section guide:**

**13a — Introduction (~0.5 page):**
- AXIOM masters pixel-based games using only Bayesian mixture models — no
  gradient-based learning.
- We analyze three
  Bayesian aspects: prior and hyperparameter sensitivity, model selection, and exploration.
- Brief preview of findings.

**13b — Background (~1 page):**
- Bayesian priors in AXIOM: conjugate pairs (NIW, Dirichlet, Gamma) where
  applicable and flat/fixed priors for the tMM. Natural parameter updates and
  why this enables gradient-free learning.
- Bayesian model selection: sampling-based greedy expected-free-energy merge test (BMR).
- Decision theory under uncertainty: expected utility + information gain.
  KL divergence as exploration bonus.
- Keep accessible — a CS 677 student should follow without RL background.

**13c — AXIOM Architecture (~0.75 page):**
- Brief overview of the four models (sMM, iMM, tMM, rMM) and how they compose.
- Emphasize the Bayesian structure: each model is a mixture with specified
  priors (conjugate where possible, flat/fixed for the tMM), updated via
  coordinate-ascent VI.
- One concrete example: walk through how the sMM processes a frame, showing
  the E-step (responsibilities) and M-step (natural parameter update).

**13d — Analysis and Results (~1.5 pages):**
- **Prior and hyperparameter sensitivity** (Phase 1): key findings, 1-2 figures.
- **BMR analysis** (Phase 2): key findings, 1-2 figures.
- **Exploration/exploitation** (Phase 3): key findings, 1-2 figures.
- For each: state the question, describe the experiment, show the figure,
  interpret through the Bayesian lens.

**13e — Discussion + Conclusion (~0.5 page):**
- What the analyses reveal about AXIOM's Bayesian design.
- Practical implications: when do priors matter? When does BMR help?
- Connection to CS 677 concepts (summarize the three themes).
- Limitations and what further analysis would add.

**Compile:** `cd docs/report && make report`

---

### Task 14 — Create the Presentation

**Files:** `docs/presentation/slides.md`, `docs/presentation/script.md`
**Depends on:** Task 13

1. ~8-10 slides: motivation, AXIOM overview, three analysis themes (1-2 slides
   each with key figures), takeaways.
2. ~8 minutes. Focus on Bayesian connections and results interpretation.
3. Record and post to Discord #final-project channel.

**Slide outline:**
- Slide 1: Title + overview
- Slide 2: What is AXIOM? (architecture diagram, emphasize Bayesian structure)
- Slide 3: Prior sensitivity — question + key figure
- Slide 4: Prior sensitivity — interpretation (Bayesian priors in practice)
- Slide 5: BMR — question + key figure
- Slide 6: BMR — interpretation (Bayesian model selection in practice)
- Slide 7: Exploration/exploitation — question + key figure
- Slide 8: Exploration/exploitation — interpretation (Bayesian decision theory)
- Slide 9: Summary + what we learned about Bayesian inference from AXIOM
- Slide 10: References

---

## Quick Reference: Task Dependencies

```
Task 0  (install AXIOM) ──> Task 1 (experiment infrastructure)
                                │
            ┌───────────────────┼───────────────────┐
            v                   v                   v
    Task 2 (sMM priors)  Task 6 (instrument BMR)  Task 10 (info-gain sweep)
    Task 3 (tMM priors)  Task 7 (BMR experiments) Task 11 (temporal analysis)
    Task 4 (rMM priors)  Task 8 (naive pruning)   Task 12 (Phase 3 figures)
    Task 5 (Phase 1 figs) Task 9 (Phase 2 figs)
            │                   │                   │
            └───────────────────┼───────────────────┘
                                v
                        Task 13 (report)
                                │
                                v
                        Task 14 (presentation)
```

Tasks 0-1 are prerequisites for everything.
Phases 1, 2, 3 are independent of each other (can be done in any order).
Within Phase 1, Tasks 2-4 can be done in parallel; Task 5 depends on all of them.
Within Phase 2, Task 6 is a prerequisite for Tasks 7-8; Task 9 depends on both.
Within Phase 3, Task 10 → Task 11 → Task 12 is sequential.
Task 13 can start as soon as any phase is complete.
