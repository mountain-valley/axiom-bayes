# Project Roadmap

This document defines every task needed to replicate the AXIOM paper. Each task is self-contained: it states what to build, where the
code lives, what math is involved, what to test, and what to document. Point an LLM
agent to a specific task by saying:

> "Read `docs/roadmap.md` and work on Task N."

Before starting any task, the agent should check the Progress section below and read
`docs/replication_log.md` to see what has already been completed.

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

### Phase 1: Core Models
- [x] Task 1 — sMM (slot mixture model)
- [x] Task 2 — iMM (identity mixture model)
- [ ] Task 3 — tMM (transition mixture model)

### Phase 2: Inference Engine
- [ ] Task 4 — Variational inference loop
- [ ] Task 5 — rMM (recurrent mixture model)

### Phase 3: Structure Learning and Planning
- [ ] Task 6 — Structure learning for all models
- [ ] Task 7 — Bayesian Model Reduction (BMR)
- [ ] Task 8 — Active inference planner

### Phase 4: Integration
- [ ] Task 9 — Environment wrapper
- [ ] Task 10 — Agent + training loop

### Phase 5: Experiments and Analysis
- [ ] Task 11 — Run replication experiments
- [ ] Task 12 — Run ablation experiments
- [ ] Task 13 — Generate figures and tables

### Phase 6: Report and Presentation
- [ ] Task 14 — Draft the report (14a-14e)
- [ ] Task 15 — Create the presentation

---

## Phase 1: Core Models

### Task 1 — Slot Mixture Model (sMM)

**File:** `axiom/models/smm.py`
**Tests:** `tests/test_smm.py`
**Depends on:** nothing (first task)
**Status:** Completed (2026-04-14)

**What to build:** Implement the `SlotMixtureModel` class. The sMM parses an RGB image
into object-centric slot latents via a Gaussian mixture over pixel tokens.

**Input:** An (H, W, 3) RGB image, tokenized into (N, 5) pixel tokens [x, y, r, g, b]
by `envs/utils.py:image_to_tokens`.

**Math (paper Eq. 2, Appendix A.2):**
- Each slot k has continuous features x^(k) encoding position p^(k), color c^(k), and
  spatial extent e^(k).
- Pixel n's likelihood under slot k:
  `p(y^n | x^(k)) = N(A x^(k), diag(B x^(k), sigma_c^(k)))`
  where A selects [position, color] and B selects [extent].
- Pixel-to-slot assignments z^n_smm are Categorical with mixing weights pi_smm.
- Mixing weights have a truncated stick-breaking prior: Dir(1, ..., 1, alpha_smm).
- E-step: compute posterior responsibilities for each pixel under each slot.
- M-step: update slot position, color, and extent from assigned pixels.

**Implement these methods:**
- `initialize(image_shape)` — set up projection matrices A and B.
- `infer(pixel_tokens)` — E-step: assign pixels to slots, return assignments + slot features.
- `update_params(pixel_tokens, assignments)` — M-step: update slot parameters.
- `expand_if_needed(pixel_tokens)` — grow a new slot if unexplained pixels exist (use
  the threshold logic from `axiom/inference/structure_learning.py`).

**Tests to write:**
- Correct pixel assignments on a synthetic image with 2-3 colored rectangles on black.
- Slot parameter recovery after several E-M iterations on known data.
- Slot expansion when a new object color appears mid-sequence.

---

### Task 2 — Identity Mixture Model (iMM)

**File:** `axiom/models/imm.py`
**Tests:** `tests/test_imm.py`
**Depends on:** Task 1 (uses slot color+shape features from sMM)
**Status:** Completed (2026-04-14)

**What to build:** Implement the `IdentityMixtureModel` class. The iMM assigns a
discrete identity code (object type) to each slot based on its color and shape features.

**Input:** Per-slot 5-D vector [c^(k), e^(k)] (3 color dims + 2 extent dims).

**Math (paper Eqs. 3-4, Appendix A.6):**
- Each identity type j has parameters (mu_j, Sigma_j) with a conjugate
  Normal-Inverse-Wishart (NIW) prior: NIW(m_j, kappa_j, U_j, n_j).
- Likelihood: `p(c^(k), e^(k) | z_type^(k)=j) = N(mu_j, Sigma_j)`
- Assignments z_type ~ Cat(pi_type) with stick-breaking prior Dir(1,...,1, alpha_imm).
- E-step: compute type responsibilities using the NIW posterior predictive, which is
  a multivariate Student-t distribution.
- M-step: update NIW sufficient statistics (m, kappa, U, n) from assigned slots.

**Implement these methods:**
- `infer_identity(color_shape_features)` — return (K,) integer identity codes.
- `update_params(color_shape_features, assignments)` — update NIW parameters.
- `expand_if_needed(color_shape_features)` — grow a new type if needed.

**Tests to write:**
- Correct identity assignment given slots with 2-3 distinct color/shape clusters.
- NIW posterior update correctness: hand-compute the posterior for a known prior +
  one data point and compare against the implementation.
- Type expansion when a novel object type appears.

---

### Task 3 — Transition Mixture Model (tMM)

**File:** `axiom/models/tmm.py`
**Tests:** `tests/test_tmm.py`
**Depends on:** Task 1 (uses slot latent states from sMM)

**What to build:** Implement the `TransitionMixtureModel` class. The tMM models
per-slot dynamics as a switching linear dynamical system (SLDS) with L shared linear
modes.

**Input:** Previous slot state x^(k)_{t-1} and a switch variable selecting which
linear mode is active.

**Math (paper Eq. 5, Appendix A.7):**
- Each mode l has parameters (D_l, b_l). Prediction:
  `x^(k)_t = D_l x^(k)_{t-1} + b_l + noise`, with fixed covariance 2I.
- The L linear modes are shared across all K slots.
- Switch variable s^(k)_{t,tmm} selects which mode is active for slot k at time t.
- Mixing weights pi_tmm have a stick-breaking prior Dir(1,...,1, alpha_tmm).
- M-step: update D_l, b_l using linear regression sufficient statistics from
  (x_{t-1}, x_t) pairs assigned to mode l.

**Implement these methods:**
- `predict(x_prev, switch_state)` — return predicted next state.
- `update_params(x_prev, x_curr, switch_state)` — update linear parameters for mode.
- `expand_if_needed(x_prev, x_curr)` — grow a new mode if the transition is not
  explained by existing modes.

**Tests to write:**
- Linear prediction correctness: set known D, b, verify x_next = D @ x_prev + b.
- Parameter recovery: generate data from a known linear system, run M-step updates,
  verify D and b converge to the true values.
- Mode expansion when a qualitatively different trajectory type appears.
- Mode sharing: verify that two different slots using the same motion pattern share
  the same mode.

---

## Phase 2: Inference Engine

### Task 4 — Variational Inference Loop

**File:** `axiom/inference/variational.py`
**Tests:** `tests/test_inference.py`
**Depends on:** Tasks 1-3 (all three models must be functional)

**What to build:** Implement the `e_step` and `m_step` functions that wire together
sMM, iMM, and tMM into a single-frame variational inference cycle.

**Math (paper Eqs. 8-9, Section 2 "Variational inference"):**
- Mean-field factorization: q(Z, Theta) = q(Theta) * prod_t [ prod_n q(z^n_smm) * prod_k q(O^(k)) ]
- E-step runs per-timestep in sequence:
  1. sMM: assign pixels to slots, yielding slot latents O_t.
  2. iMM: infer identity codes from slot color/shape features.
  3. tMM: predict next states given current switch states.
- M-step: update parameters of all three models using sufficient statistics.
- These E-M updates run once per frame (streaming coordinate-ascent VI).

**Implement:**
- `e_step(observation, smm, imm, tmm, rmm)` — run the full E-step for one frame
  (the rmm argument can be ignored until Task 5).
- `m_step(latent_states, observation, smm, imm, tmm, rmm)` — update all parameters.

**Tests to write:**
- End-to-end test: create a synthetic 2-object scene, run 10 E-M iterations, verify
  that slot positions converge to the true object positions.
- Verify that M-step updates decrease the variational free energy.

---

### Task 5 — Recurrent Mixture Model (rMM)

**File:** `axiom/models/rmm.py`
**Tests:** `tests/test_rmm.py`
**Depends on:** Tasks 1-4 (needs all models + inference loop)

**What to build:** Implement the `RecurrentMixtureModel` class. The rMM models the
joint distribution of continuous and discrete slot features, capturing object-object
interactions and conditioning the tMM switch states.

**Input:** A tuple of continuous features f^(k) (slot position, distance to nearest
object, etc.) and discrete features d^(k) (identity code, tMM switch, action, reward).

**Math (paper Eqs. 6-7, Appendix A.8):**
- Each rMM component m has a factorized likelihood:
  `p(f, d | s_rmm=m) = N(f; mu_m, Sigma_m) * prod_i Cat(d_i; alpha_{m,i})`
- Continuous features f^(k) are computed from:
  - Projection C of slot k's own state (position, velocity)
  - Interaction function g(x^(1:K)) returning distance to nearest object
- Discrete features d^(k) include: identity code z_type, tMM switch s_tmm, action a, reward r.
- Mixing weights with stick-breaking prior Dir(1,...,1, alpha_rmm).
- E-step: infer rMM component assignment, which implies a posterior over tMM switch states.
- M-step: update Gaussian parameters (mu, Sigma) and Categorical parameters (alpha)
  per component.

**Implement these methods:**
- `infer_switch(continuous_features, discrete_features)` — return posterior over
  rMM components and implied tMM switch distribution.
- `update_params(continuous_features, discrete_features, assignment)` — update
  Gaussian and Categorical parameters.
- `expand_if_needed(continuous_features, discrete_features)` — grow a new component.

**Tests to write:**
- Switch inference with known component parameters: verify correct assignment.
- Gaussian x Categorical likelihood computation against manual calculation.
- Component expansion when a new interaction type appears.

**Then update:** `axiom/inference/variational.py` — integrate the rMM into `e_step`
and `m_step` so the full inference loop uses all four models.

---

## Phase 3: Structure Learning and Planning

### Task 6 — Structure Learning for All Models

**File:** `axiom/inference/structure_learning.py`
**Tests:** `tests/test_inference.py`
**Depends on:** Tasks 1-5

**What to build:** Extend the existing growing heuristic (already partially implemented
for threshold computation and assign-or-expand logic) to be called by all four mixture
models during inference.

**Math (paper Section 2.1 "Fast structure learning"):**
- For each new data point, compute posterior-predictive log-density under each
  existing component: l_{t,c} = E_q[log p(y_t | Theta_c)].
- New-component threshold: tau_t = log p_0(y_t) + log alpha, where p_0 is the prior
  predictive under an empty component.
- If max_c(l_{t,c}) < tau_t and capacity remains, create a new component.
- Each model (sMM, iMM, tMM, rMM) needs its own prior predictive density p_0 and
  its own max-component cap.

**What to implement:**
- A `posterior_predictive_log_density` method (or helper) for each model.
- Wire `expand_if_needed` in each model to use the shared `assign_or_expand` logic.
- Verify the threshold computation matches the truncated stick-breaking / CRP rule.

**Tests to write:**
- For each model: verify that expansion fires at the right threshold.
- Verify that capacity limits are respected.
- End-to-end: run inference on a sequence where a new object appears at frame 50;
  verify slot count increases.

---

### Task 7 — Bayesian Model Reduction (BMR)

**File:** `axiom/inference/bmr.py`
**Tests:** `tests/test_inference.py` (add BMR-specific tests)
**Depends on:** Task 5 (needs a working rMM)

**What to build:** Implement Bayesian Model Reduction for the rMM. Every 500 frames,
sample component pairs, score whether merging them decreases expected free energy, and
greedily merge the best candidates.

**Math (paper Section 2.1 "Bayesian Model Reduction"):**
- Every Delta_T_BMR = 500 frames, sample up to n_pair = 2000 active rMM component pairs.
- For each pair (a, b), generate data from the model via ancestral sampling.
- Score the merge: does the reduced model (with a and b merged) decrease the expected
  free energy of the multinomial distributions over reward and tMM switch?
- Accept the merge if free energy decreases; otherwise roll back.
- This enables generalization (e.g., learning that negative reward occurs when a ball
  hits the bottom, by merging multiple single-event clusters).

**Implement:**
- `select_merge_candidates(num_active, max_pairs, rng)` — sample component pairs.
- `score_merge(component_a, component_b, sampled_data)` — compute delta free energy.
- `perform_bmr(rmm_params, num_active, max_pairs, rng)` — run one BMR round.

**Tests to write:**
- Verify that merging two identical components always decreases free energy.
- Verify that merging two very different components does not pass the threshold.
- Verify that active component count decreases after BMR on a model with redundant
  components.

---

### Task 8 — Active Inference Planner

**File:** `axiom/planning/active_inference.py`
**Tests:** add `tests/test_planning.py`
**Depends on:** Tasks 1-5 (needs a functional world model for rollouts)

**What to build:** Implement the `Planner` class that selects actions by evaluating
candidate policies through imagined rollouts scored by expected free energy.

**Math (paper Eq. 10, Appendix A.11):**
- For each candidate policy pi (a sequence of H actions):
  - Roll out imagined trajectories through the world model from current slot latents.
  - Expected utility: E_q[log p(r_tau | O_tau, pi)] summed over horizon.
  - Information gain: D_KL(q(alpha_rmm | O, pi) || q(alpha_rmm)) — how much the
    rMM Dirichlet counts would change.
  - Expected free energy = -utility - info_gain (lower is better).
- Select the policy with lowest expected free energy; return its first action.

**Implement:**
- `select_action(slot_latents, world_model, rng)` — sample num_rollouts random
  action sequences, score each, return best first action.
- `compute_expected_utility(imagined_rewards)` — sum log-reward probabilities.
- `compute_information_gain(rmm_dirichlet_counts, imagined_assignments)` — compute
  KL divergence on Dirichlet parameters.

**Tests to write:**
- With a mock world model that always predicts +1 reward for action 0 and -1 for
  others, verify the planner selects action 0.
- Verify that information gain is non-negative.
- Verify that with info_gain_weight=0, the planner is purely reward-seeking.

---

## Phase 4: Integration

### Task 9 — Environment Wrapper

**File:** `envs/gameworld.py`
**Tests:** manual verification (requires gameworld installed)
**Depends on:** nothing (can be done in parallel with Phase 1)

**What to build:** Complete the `GameworldEnv` wrapper so it provides a consistent
reset/step interface for AXIOM. Requires `make setup-gameworld` first.

**Implement:**
- `reset()` — initialize the game, return first observation as a dict with "image"
  (H,W,3 ndarray) and "info".
- `step(action)` — return (observation, reward, done, info).
- `num_actions` — query the underlying environment for the action space size.
- `observation_shape` — return (210, 160, 3) for Gameworld.

**Verify:** Run a quick manual loop — reset, take 100 random actions, print rewards.

---

### Task 10 — Agent + Training Loop

**File:** `axiom/agent.py`, `experiments/run_experiment.py`
**Tests:** `tests/test_agent.py`
**Depends on:** Tasks 1-8 (full model), Task 9 (environment)

**What to build:** Complete the `AXIOMAgent.observe` method and the training loop in
`run_experiment.py`.

**`AXIOMAgent.observe(observation, reward)` should:**
1. Tokenize the image via `envs/utils.py:image_to_tokens`.
2. Run E-step: sMM -> iMM -> rMM -> tMM (infer latents for this frame).
3. Run M-step: update all four model parameters.
4. Call `expand_if_needed` on each model (structure learning).
5. If `should_run_bmr()`, call BMR on the rMM.
6. Store slot latents for the planner.

**Training loop in `run_experiment.py`:**
1. Load config (default.yaml + game override).
2. Initialize environment and agent.
3. Loop for `steps` iterations: observe -> act -> step -> log reward.
4. Save reward array to `results/{game}_seed{seed}.npy`.
5. Print cumulative reward at the end.

**Tests to write:**
- Smoke test: agent + mock environment, 100 steps, no crash.
- Verify observe increments step_count and triggers BMR at the right interval.

---

## Phase 5: Experiments and Analysis

### Task 11 — Run Replication Experiments

**Depends on:** Task 10

**What to do:**
1. Run AXIOM on all 10 Gameworld games, 10 seeds each, 10k steps: `make sweep`.
2. Run the random baseline on the same games for comparison.
3. Save all reward arrays to `results/`.
4. Log wall-clock timing per game.

---

### Task 12 — Run Ablation Experiments

**Depends on:** Task 11

**What to do:**
1. Run three ablation variants using `make ablation`:
   - **no_bmr:** set `bmr_interval=0` (disable BMR).
   - **no_ig:** set `info_gain_weight=0` (disable information gain in planning).
   - **fixed_distance:** use a fixed interaction distance hyperparameter instead of
     learning it in the rMM.
2. Save results to `results/` with variant labels.

---

### Task 13 — Generate Figures and Tables

**Files:** `analysis/plotting.py`, `analysis/metrics.py`, `analysis/notebooks/`
**Depends on:** Tasks 11-12

**What to do:**
1. Replicate **Figure 3** (learning curves): moving-average reward per step for AXIOM
   vs baselines on all 10 games. Use `analysis/plotting.py:plot_learning_curves`.
2. Replicate **Table 1** (cumulative reward): mean +/- std over 10 seeds per game per
   model variant. Use `analysis/metrics.py:reward_summary`.
3. Generate **Figure 4a** (interpretability): imagined trajectories and reward-conditioned
   rMM clusters for one game (e.g., Impact).
4. Generate **Figure 4b** (BMR pruning): rMM component count over training for Explode.
5. Generate **Figure 4c** (exploration-exploitation): info gain vs utility over time.
6. Save all figures to `results/figures/`.
7. Copy final figures to `docs/report/figures/` for the LaTeX report.

---

## Phase 6: Report and Presentation

### Task 14 — Draft the Report

**File:** `docs/report/report.tex`
**Depends on:** Tasks 11-13 (need results and figures)

The report is 4-5 pages, 12pt font, 1-inch margins, single-spaced. The audience is a
CS 677 student. The LaTeX template and section stubs are already in place.

**Sections to draft (one at a time):**

**14a — Introduction (~0.5 page):**
- Deep RL is data-hungry; humans use object-level priors to learn fast.
- AXIOM combines Bayesian inference with object-centric representations.
- This project replicates AXIOM's results on the Gameworld 10k benchmark.

**14b — Background (~1 page):**
- Mixture models and conjugate priors (NIW, Dirichlet) — connect to CS 677 material.
- Variational inference and the ELBO / free energy.
- Nonparametric Bayes: stick-breaking priors and automatic model complexity.
- Active inference: planning as minimizing expected free energy.

**14c — Methods (~1.5 pages):**
- High-level AXIOM architecture: four mixture models and their roles.
- Detailed walkthrough of one model (e.g., sMM) to show the Bayesian mechanics.
- Structure learning (growing + BMR).
- Planning with utility + information gain.
- Any deviations from the paper in the replication.

**14d — Results (~1 page):**
- Learning curves (Figure 3 replication).
- Cumulative reward table (Table 1 replication).
- Ablation findings: what happens without BMR, without info gain.
- Interpretability examples if space permits.

**14e — Discussion + Conclusion (~0.5 page):**
- How well does the replication match? Where does it differ and why?
- What the ablations reveal about each Bayesian component.
- Connection back to CS 677: why conjugate priors and variational inference matter.
- Limitations and future work.

**Compile with:** `cd docs/report && make report`

---

### Task 15 — Create the Presentation

**Files:** `docs/presentation/slides.md`, `docs/presentation/script.md`
**Depends on:** Task 14 (report content feeds the presentation)

**What to do:**
1. Create ~8-10 slides covering: motivation, Bayesian inference connections, AXIOM
   architecture, key results, and takeaways.
2. Refine the script in `script.md` to hit exactly ~8 minutes.
3. Focus on visual results (learning curves, cluster visualizations).
4. Record and post to the Discord #final-project channel.

---

## Quick Reference: Task Dependencies

```
Task 1  (sMM) ─────┐
Task 2  (iMM) ──────┤
Task 3  (tMM) ──────┼──> Task 4 (VI loop) ──> Task 5 (rMM) ──┐
Task 9  (env)  ─────┘                                         │
                    Task 6 (structure learning) <──────────────┤
                    Task 7 (BMR) <─────────────────────────────┤
                    Task 8 (planning) <────────────────────────┘
                              │
                              v
                    Task 10 (agent + training loop)
                              │
                              v
                    Task 11 (replication experiments)
                              │
                              v
                    Task 12 (ablation experiments)
                              │
                              v
                    Task 13 (figures and tables)
                              │
                              v
                    Task 14 (report) ──> Task 15 (presentation)
```

Tasks 1, 2, 3, and 9 can be done in parallel.
Tasks 6, 7, and 8 can be done in parallel after Task 5.
