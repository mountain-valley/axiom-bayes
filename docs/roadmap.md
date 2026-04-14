# Project Roadmap

This project studies whether the Bayesian components of AXIOM (Heins et al., 2025) —
variational updating, mixture-based latent dynamics, and uncertainty-aware action
selection — can be replicated in a simplified game setting, and evaluates which
components most affect sample efficiency.

**Modular milestone structure:** Each phase ends with its own experiments and figures,
so you can write a report at any stopping point. Later phases build on earlier ones
and make the story richer, but no phase is wasted. If time runs short, stop at the
latest completed phase and write up what you have.

The full 10-game replication roadmap is preserved in `docs/roadmap_replication.md`.

Each task below is self-contained. Point an LLM agent to a specific task by saying:

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

### Phase 1: Bayesian Mixture Models for Object Perception
- [ ] Task 1 — sMM (slot mixture model)
- [ ] Task 2 — iMM (identity mixture model)
- [ ] Task 3 — Phase 1 experiments and figures

### Phase 2: Variational Inference for Dynamics Learning
- [ ] Task 4 — tMM (transition mixture model)
- [ ] Task 5 — Coordinate-ascent VI loop
- [ ] Task 6 — Phase 2 experiments and figures

### Phase 3: Nonparametric Bayes — Growing and Pruning
- [ ] Task 7 — rMM (recurrent mixture model)
- [ ] Task 8 — Online model expansion
- [ ] Task 9 — Bayesian Model Reduction (BMR)
- [ ] Task 10 — Phase 3 experiments and figures

### Phase 4: Uncertainty-Aware Decision Making (stretch goal)
- [ ] Task 11 — Expected free energy planner
- [ ] Task 12 — Full agent + Gameworld integration
- [ ] Task 13 — Ablation experiments and figures

### Report and Presentation
- [ ] Task 14 — Draft the report (at whatever phase you've reached)
- [ ] Task 15 — Create the presentation

---

## What a Report Looks Like at Each Stopping Point

| Stop after | Report angle | Key figures |
|---|---|---|
| Phase 1 | Bayesian mixture models for object segmentation | Pixel assignments, parameter convergence, prior sensitivity |
| Phase 2 | + Variational inference for object-centric dynamics | + Free energy curves, trajectory modes, slot tracking |
| Phase 3 | + Nonparametric Bayes for adaptive model complexity | + Component growth/pruning, BMR impact |
| Phase 4 | + Full active inference agent with ablations | + Learning curves, ablation comparison, exploration vs. exploitation |

Each row is a complete story. Later phases add depth, but every phase stands alone.

---

## Project Framing for CS 677

The report and presentation should be framed around these Bayesian inference themes
(all covered in CS 677) rather than as an RL replication:

1. **Conjugate priors and posterior updates** — Every model uses exponential-family
   likelihoods with conjugate priors (NIW for Gaussians, Dirichlet for Categoricals).
   The M-step is a natural parameter update — no gradients.

2. **Variational inference and the ELBO** — AXIOM minimizes variational free energy
   via coordinate-ascent mean-field VI.

3. **Mixture models and EM** — All four modules are mixture models. The E-step assigns
   data to components; the M-step updates parameters. This is EM with Bayesian priors.

4. **Nonparametric Bayes / model complexity** — Stick-breaking priors allow automatic
   model expansion. BMR prunes via closed-form Bayes factors.

5. **Decision-making under uncertainty** — The planner balances reward-seeking (utility)
   with information-seeking (epistemic exploration via KL divergence on Dirichlet posteriors).

---

## Phase 1: Bayesian Mixture Models for Object Perception

**Milestone:** At the end of this phase, you can segment game frames into objects
using conjugate Bayesian mixture models, and have figures showing it works.

### Task 1 — Slot Mixture Model (sMM)

**File:** `axiom/models/smm.py`
**Tests:** `tests/test_smm.py`
**Depends on:** nothing

**What to build:** Implement the `SlotMixtureModel` class. The sMM is a Gaussian
mixture model that parses an RGB image into object-centric slot latents.

**Input:** An (H, W, 3) RGB image, tokenized into (N, 5) pixel tokens [x, y, r, g, b]
by `envs/utils.py:image_to_tokens`.

**Math (paper Eq. 2):**
- Each slot k has continuous features x^(k) encoding position p^(k), color c^(k), and
  spatial extent e^(k).
- Pixel n's likelihood under slot k:
  `p(y^n | x^(k)) = N(A x^(k), diag(B x^(k), sigma_c^(k)))`
  where A selects [position, color] and B selects [spatial extent].
- Pixel-to-slot assignments z^n_smm ~ Cat(pi_smm).
- Mixing weights have a truncated stick-breaking prior: Dir(1, ..., 1, alpha_smm).
- E-step: compute posterior responsibilities for each pixel under each slot.
- M-step: update slot position, color, and extent from assigned pixels.

**Bayesian connection:** Standard Gaussian mixture with conjugate priors. The E/M cycle
is the same conjugate EM from class, applied to image segmentation.

**Implement:** `initialize`, `infer`, `update_params`, `expand_if_needed`.

**Tests to write:**
- Correct pixel assignments on a synthetic image with 2-3 colored rectangles.
- Slot parameter recovery after several E-M iterations.
- Slot expansion when a new object color appears.

---

### Task 2 — Identity Mixture Model (iMM)

**File:** `axiom/models/imm.py`
**Tests:** `tests/test_imm.py`
**Depends on:** Task 1

**What to build:** Implement the `IdentityMixtureModel` class. The iMM assigns a
discrete identity code (object type) to each slot from its color and shape features,
using a Gaussian mixture with Normal-Inverse-Wishart (NIW) priors.

**Input:** Per-slot 5-D vector [c^(k), e^(k)] (3 color + 2 extent dims).

**Math (paper Eqs. 3-4):**
- Each identity type j has parameters (mu_j, Sigma_j) with conjugate
  NIW prior: NIW(m_j, kappa_j, U_j, n_j).
- Likelihood: `p(c^(k), e^(k) | type=j) = N(mu_j, Sigma_j)`.
- Assignments z_type ~ Cat(pi_type) with stick-breaking prior.
- E-step: type responsibilities via NIW posterior predictive (multivariate Student-t).
- M-step: update NIW sufficient statistics (m, kappa, U, n).

**Bayesian connection:** Textbook conjugate Bayesian Gaussian mixture. The NIW prior
is the conjugate prior for a multivariate Gaussian with unknown mean and covariance.
The posterior predictive is a Student-t, naturally accounting for parameter uncertainty.

**Implement:** `infer_identity`, `update_params`, `expand_if_needed`.

**Tests to write:**
- Correct identity assignment with 2-3 distinct color/shape clusters.
- NIW posterior update: hand-compute for known prior + one data point, compare.
- Type expansion when a novel object type appears.

---

### Task 3 — Phase 1 Experiments and Figures

**Files:** `analysis/notebooks/01_learning_curves.ipynb` (repurpose or create new),
`analysis/plotting.py`
**Depends on:** Tasks 1-2

**What to do:** Demonstrate the sMM and iMM on both synthetic and real data.

**Experiments:**
1. **Synthetic images:** Create 3-4 synthetic images with colored rectangles on a
   black background. Run sMM for 20 E-M iterations. Show that slots converge to
   the correct object positions, colors, and extents.
2. **Real Gameworld frames:** Install gameworld (`make setup-gameworld`), grab a few
   frames from Bounce or Explode, run sMM on them. Show pixel-to-slot assignments
   overlaid on the original image.
3. **Prior sensitivity:** Vary alpha_smm (e.g., 0.1, 1.0, 10.0) and show how it
   affects the number of slots created. This demonstrates the role of the
   stick-breaking prior in controlling model complexity.
4. **Identity clustering:** After sMM segments objects, run iMM on the slot features.
   Show that objects of the same type get the same identity code.

**Figures to produce:**
- Original image vs. slot assignments (color-coded by slot) — side by side.
- Slot parameter convergence over E-M iterations (position error vs. iteration).
- Number of active slots vs. alpha_smm (prior sensitivity plot).
- iMM identity clusters in color-shape feature space.

Save to `results/figures/phase1_*`.

---

## Phase 2: Variational Inference for Dynamics Learning

**Milestone:** At the end of this phase, you can track objects over time and learn
their dynamics using streaming variational inference.

### Task 4 — Transition Mixture Model (tMM)

**File:** `axiom/models/tmm.py`
**Tests:** `tests/test_tmm.py`
**Depends on:** Task 1

**What to build:** Implement the `TransitionMixtureModel` class. The tMM models
per-slot dynamics as a switching linear dynamical system with L shared modes.

**Input:** Previous slot state x^(k)_{t-1} and switch variable.

**Math (paper Eq. 5):**
- Each mode l has parameters (D_l, b_l): `x_t = D_l x_{t-1} + b_l + noise`.
- Fixed covariance 2I. L modes shared across all K slots.
- Switch variable s^(k)_{t,tmm} selects the active mode.
- Stick-breaking prior on mixing weights.
- M-step: Bayesian linear regression update for D_l, b_l.

**Bayesian connection:** Each linear mode is a Bayesian linear regression with a
conjugate prior on (D, b). The switching mechanism is a latent-variable model akin
to a hidden Markov model over dynamics modes.

**Implement:** `predict`, `update_params`, `expand_if_needed`.

**Tests to write:**
- Linear prediction correctness with known D, b.
- Parameter recovery from synthetic linear trajectory data.
- Mode expansion when a new trajectory type appears.
- Mode sharing: two slots with the same motion use the same mode.

---

### Task 5 — Coordinate-Ascent VI Loop

**File:** `axiom/inference/variational.py`
**Tests:** `tests/test_inference.py`
**Depends on:** Tasks 1-2, Task 4

**What to build:** Implement `e_step` and `m_step` that wire sMM, iMM, and tMM
into a single-frame variational inference cycle.

**Math (paper Eqs. 8-9):**
- Mean-field factorization over slots, time, and parameters.
- E-step per frame: sMM (assign pixels) → iMM (infer identities) → tMM (predict dynamics).
- M-step: update all model parameters using sufficient statistics.
- One E-M pass per frame — streaming coordinate-ascent VI.

**Bayesian connection:** This is CAVI with a mean-field approximation. The free energy
being minimized is the negative ELBO. Gradient-free natural parameter updates are
possible because every likelihood-prior pair is conjugate.

**Implement:** `e_step(observation, smm, imm, tmm, rmm)` and
`m_step(latent_states, observation, smm, imm, tmm, rmm)`. The rmm argument
can be a no-op until Phase 3.

**Tests to write:**
- End-to-end: synthetic 2-object scene, 10 E-M iterations, slot positions converge.
- M-step updates decrease variational free energy.

---

### Task 6 — Phase 2 Experiments and Figures

**Files:** `analysis/notebooks/`, `analysis/plotting.py`
**Depends on:** Tasks 4-5

**What to do:** Demonstrate streaming VI on sequences — synthetic and real.

**Experiments:**
1. **Synthetic bouncing ball:** Generate a short sequence (50-100 frames) of a ball
   following a linear trajectory with one bounce (direction change). Run the full
   VI loop. Show that the sMM tracks the ball's position and the tMM discovers
   two linear modes (pre-bounce and post-bounce).
2. **Real Gameworld rollout:** Run 200 random-action steps on Bounce. Feed the
   frames through the VI loop. Show slot tracking over time.
3. **Free energy convergence:** Plot variational free energy per frame over the
   sequence. Show it decreases as the model improves.
4. **Learned trajectory modes:** Visualize the tMM linear modes (D, b vectors) —
   show that they correspond to interpretable motions (e.g., "moving right",
   "moving down").

**Figures to produce:**
- Slot positions overlaid on frames at t=0, t=25, t=50 (tracking over time).
- Variational free energy vs. frame number.
- tMM mode visualization (direction vectors in 2D position space).
- Number of active tMM modes over the sequence.

Save to `results/figures/phase2_*`.

---

## Phase 3: Nonparametric Bayes — Growing and Pruning

**Milestone:** At the end of this phase, you can show that the model automatically
adapts its complexity — growing when new objects/dynamics appear and pruning when
components are redundant.

### Task 7 — Recurrent Mixture Model (rMM)

**File:** `axiom/models/rmm.py`
**Tests:** `tests/test_rmm.py`
**Depends on:** Tasks 1-5

**What to build:** Implement the `RecurrentMixtureModel`. The rMM models the joint
distribution of continuous slot features and discrete game state, capturing
object-object interactions.

**Input:** Continuous features f^(k) (position, distance to nearest object) and
discrete features d^(k) (identity code, tMM switch, action, reward).

**Math (paper Eqs. 6-7):**
- Each component m: `p(f, d | s_rmm=m) = N(f; mu_m, Sigma_m) * prod_i Cat(d_i; alpha_{m,i})`
- Continuous features from projection C of slot state + interaction function g(x^(1:K)).
- Stick-breaking prior on mixing weights.
- E-step: infer rMM component → posterior over tMM switch states.
- M-step: update Gaussian (mu, Sigma) and Categorical (alpha) parameters.

**Bayesian connection:** Mixture model with mixed continuous-discrete likelihoods.
Gaussian components have NIW priors; Categorical components have Dirichlet priors.

**Implement:** `infer_switch`, `update_params`, `expand_if_needed`.
Then update `axiom/inference/variational.py` to integrate rMM into the VI loop.

**Tests to write:**
- Switch inference with known component parameters.
- Gaussian x Categorical likelihood against manual calculation.
- Component expansion when a new interaction type appears.

---

### Task 8 — Online Model Expansion

**File:** `axiom/inference/structure_learning.py`
**Tests:** `tests/test_inference.py`
**Depends on:** Tasks 1-7

**What to build:** Extend the growing heuristic so all four models use it.

**Math (paper Section 2.1):**
- Posterior-predictive log-density vs. threshold tau = log p_0(y) + log alpha.
- If best score < tau and capacity remains, create a new component.
- Deterministic MAP version of the CRP assignment rule.

**Bayesian connection:** The expansion threshold comes from the stick-breaking prior
(finite Dirichlet process approximation). Growing vs. reusing is Bayesian model
comparison: marginal likelihood under existing component vs. new one.

**Implement:**
- `posterior_predictive_log_density` helper for each model.
- Wire each model's `expand_if_needed` to the shared `assign_or_expand` logic.

**Tests to write:**
- Expansion fires at the right threshold for each model.
- Capacity limits respected.
- New object at frame 50 → slot count increases.

---

### Task 9 — Bayesian Model Reduction (BMR)

**File:** `axiom/inference/bmr.py`
**Tests:** `tests/test_inference.py`
**Depends on:** Task 7

**What to build:** Implement BMR for the rMM. Every 500 frames, evaluate merges.

**Math (paper Section 2.1):**
- Sample up to 2000 rMM component pairs.
- Score merge: does the reduced model decrease expected free energy of the
  multinomial over reward and tMM switch?
- Accept if free energy decreases; roll back otherwise.

**Bayesian connection:** BMR is closed-form Bayesian model comparison — computing the
Bayes factor for merging two components using Dirichlet-Categorical conjugacy. This
is principled model selection without cross-validation.

**Implement:** `select_merge_candidates`, `score_merge`, `perform_bmr`.

**Tests to write:**
- Merging identical components always decreases free energy.
- Merging very different components does not pass threshold.
- Active component count decreases with redundant components.

---

### Task 10 — Phase 3 Experiments and Figures

**Files:** `analysis/notebooks/`, `analysis/plotting.py`
**Depends on:** Tasks 7-9

**What to do:** Demonstrate automatic model complexity adaptation.

**Experiments:**
1. **Object appearance/disappearance:** Run a sequence where a new object appears
   at frame 100. Show slot count growing. Then remove an object and show the
   model adapts (unused slot counter from Appendix A.5).
2. **BMR pruning dynamics:** Run the full model on a 2000+ frame Gameworld rollout
   (random actions). Plot rMM component count over frames. Show the sharp decline
   when BMR fires at every 500-frame interval.
3. **With vs. without BMR:** Run the same sequence with BMR enabled and disabled.
   Compare: component count over time, and (if possible) prediction quality or
   free energy. This is the first ablation.
4. **Generalization via merging:** Show that after BMR, the model has merged
   single-event clusters into general rules (e.g., "ball near bottom → negative
   reward" regardless of exact x-position).

**Figures to produce:**
- Active rMM component count vs. frame (with and without BMR).
- Active slot count over a sequence with appearing/disappearing objects.
- rMM clusters in 2D space, colored by predicted reward (before and after BMR).
- Free energy comparison with/without BMR.

Save to `results/figures/phase3_*`.

---

## Phase 4: Uncertainty-Aware Decision Making (Stretch Goal)

**Milestone:** Full AXIOM agent playing Gameworld games, with ablation experiments
showing which Bayesian components matter most.

### Task 11 — Expected Free Energy Planner

**File:** `axiom/planning/active_inference.py`
**Tests:** add `tests/test_planning.py`
**Depends on:** Tasks 1-9

**What to build:** Implement the `Planner` class.

**Math (paper Eq. 10):**
- pi* = argmin_pi sum_tau [ -E[log p(r|O,pi)] - D_KL(q(alpha_rmm|O,pi) || q(alpha_rmm)) ]
- First term: expected utility. Second term: information gain.

**Bayesian connection:** The agent seeks information to reduce uncertainty about its
world model. Information gain is a KL divergence on Dirichlet posteriors — quantifying
epistemic uncertainty. This is decision-making under uncertainty with a principled
Bayesian exploration bonus.

**Implement:** `select_action`, `compute_expected_utility`, `compute_information_gain`.

**Tests to write:**
- Mock world: +1 reward for action 0 → planner selects action 0.
- Information gain is non-negative.
- info_gain_weight=0 → purely reward-seeking.
- utility_weight=0 → purely information-seeking.

---

### Task 12 — Full Agent + Gameworld Integration

**Files:** `axiom/agent.py`, `experiments/run_experiment.py`, `envs/gameworld.py`
**Tests:** `tests/test_agent.py`
**Depends on:** Tasks 1-11

**What to build:** Complete the agent, environment wrapper, and training loop.
Target 1-2 games (Bounce and Explode recommended).

**Agent `observe` method:**
1. Tokenize image.
2. E-step: sMM → iMM → rMM → tMM.
3. M-step: update all parameters.
4. Structure learning: expand if needed.
5. BMR every 500 steps.

**Training loop:** Load config, init env + agent, loop 10k steps, save rewards.

**Tests:** Smoke test — mock environment, 100 steps, no crash.

---

### Task 13 — Ablation Experiments and Figures

**Files:** `experiments/run_ablation.py`, `analysis/plotting.py`
**Depends on:** Task 12

**What to do:** Run three variants on 1-2 games, 5+ seeds each:
1. **Full AXIOM** — all components active.
2. **No BMR** — `bmr_interval=0`.
3. **No information gain** — `info_gain_weight=0`.

**Figures to produce:**
- Learning curves: moving-average reward for full vs. no-BMR vs. no-info-gain.
- Exploration-exploitation tradeoff: info gain and utility per step over training.
- Cumulative reward summary (mean ± std across seeds).

Save to `results/figures/phase4_*`.

---

## Report and Presentation

### Task 14 — Draft the Report

**File:** `docs/report/report.tex`
**Depends on:** whichever phase you've completed

The report is 4-5 pages, 12pt, 1in margins, single-spaced. The audience is a CS 677
student. Write about what you have — the report scope scales with the phases completed.

**Section guide:**

**14a — Introduction (~0.5 page):**
- Humans learn fast using object priors; RL agents don't.
- AXIOM uses Bayesian mixture models with conjugate priors — no gradients.
- This project replicates the Bayesian core and evaluates which components matter.

**14b — Background (~1 page):**
- Gaussian mixtures with conjugate priors (NIW, Dirichlet).
- Variational inference: ELBO / free energy, CAVI, mean-field.
- Nonparametric Bayes: stick-breaking priors, automatic model complexity.
- (If Phase 4 reached) Active inference: planning as free energy minimization.
- Keep accessible — no RL background assumed.

**14c — Methods (~1.5 pages):**
- AXIOM architecture: whichever models you implemented.
- Walk through one model in detail (sMM) to show the Bayesian mechanics.
- Structure learning if Phase 3 reached; planning if Phase 4 reached.

**14d — Results (~1 page):**
- Figures from whichever phases you completed.
- If Phase 1 only: mixture model segmentation + prior sensitivity.
- If Phase 2: + dynamics learning + free energy convergence.
- If Phase 3: + BMR ablation + model complexity adaptation.
- If Phase 4: + game-playing learning curves + full ablations.

**14e — Discussion + Conclusion (~0.5 page):**
- What worked, what the results show about the Bayesian components.
- Connection to CS 677 concepts.
- What further phases would add (point to `roadmap_replication.md` for full scope).

**Compile:** `cd docs/report && make report`

---

### Task 15 — Create the Presentation

**Files:** `docs/presentation/slides.md`, `docs/presentation/script.md`
**Depends on:** Task 14

1. ~8-10 slides: motivation, Bayesian connections (the 5 themes above), architecture,
   results from completed phases, takeaways.
2. ~8 minutes. Focus on Bayesian connections and results interpretation.
3. Record and post to Discord #final-project channel.

---

## Quick Reference: Task Dependencies

```
Task 1  (sMM) ──┐
Task 2  (iMM) ──┼──> Task 3 (Phase 1 figures)
                │
Task 4  (tMM) ──┤
                └──> Task 5 (VI loop) ──> Task 6 (Phase 2 figures)
                                    │
                                    v
                              Task 7 (rMM) ──> Task 8 (expansion)
                                          └──> Task 9 (BMR)
                                                │
                                                v
                                          Task 10 (Phase 3 figures)
                                                │
                                                v
                                          Task 11 (planner)
                                                │
                                                v
                                          Task 12 (full agent)
                                                │
                                                v
                                          Task 13 (Phase 4 figures)
                                                │
                                                v
                                    Task 14 (report) ──> Task 15 (presentation)
```

Tasks 1, 2, and 4 can be done in parallel.
Tasks 8 and 9 can be done in parallel after Task 7.
Task 14 can be started after any phase's figures are complete.
