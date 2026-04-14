# Project Roadmap

This project studies whether the Bayesian components of AXIOM — variational updating,
mixture-based latent dynamics, and uncertainty-aware action selection — can be replicated
in a simplified game setting, and evaluates which components most affect sample efficiency.

The scope is deliberately narrowed from a full AXIOM replication to a **Bayesian core
slice**: implement the mixture models with conjugate priors, variational inference, online
structure learning (growing + Bayesian Model Reduction), and uncertainty-aware planning.
Test on 1-2 Gameworld environments, not the full 10-game benchmark. The focus is on
understanding and demonstrating the Bayesian machinery, not on matching every number
in the paper.

The full 10-game, all-baselines replication roadmap is preserved in `docs/roadmap_extended.md`
for reference or future extension.

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

### Phase 1: Bayesian Mixture Models
- [ ] Task 1 — sMM (slot mixture model)
- [ ] Task 2 — iMM (identity mixture model)
- [ ] Task 3 — tMM (transition mixture model)

### Phase 2: Variational Inference
- [ ] Task 4 — Coordinate-ascent VI loop (E-step + M-step)
- [ ] Task 5 — rMM (recurrent mixture model)

### Phase 3: Structure Learning
- [ ] Task 6 — Online model expansion (growing heuristic)
- [ ] Task 7 — Bayesian Model Reduction (BMR)

### Phase 4: Uncertainty-Aware Planning
- [ ] Task 8 — Expected free energy planner (utility + information gain)

### Phase 5: Integration and Experiments
- [ ] Task 9 — Agent + environment + training loop (1-2 games)
- [ ] Task 10 — Ablation experiments (no BMR, no info gain)
- [ ] Task 11 — Generate figures

### Phase 6: Report and Presentation
- [ ] Task 12 — Draft the report
- [ ] Task 13 — Create the presentation

---

## Project Framing for CS 677

The report and presentation should be framed around these Bayesian inference themes
 rather than as an RL replication:

1. **Conjugate priors and posterior updates** — Every model in AXIOM uses exponential-
   family likelihoods with conjugate priors (Normal-Inverse-Wishart for Gaussians,
   Dirichlet for Categoricals). The M-step is just a natural parameter update — no
   gradients. This is the same conjugate Bayesian updating from class, applied to
   a structured generative model.

2. **Variational inference and the ELBO** — AXIOM minimizes variational free energy
   (= negative ELBO) via coordinate-ascent mean-field VI. Connect this to the
   variational inference material from class.

3. **Mixture models and EM** — All four modules are mixture models. The E-step assigns
   data to components; the M-step updates component parameters. This is EM with
   Bayesian priors — a natural extension of the EM intuition from class.

4. **Nonparametric Bayes / model complexity** — The stick-breaking priors allow
   automatic model expansion (like a Dirichlet process). BMR prunes redundant
   components. Together they control model complexity in a principled Bayesian way.

5. **Decision-making under uncertainty** — The planner uses expected free energy,
   which balances reward-seeking (utility) with information-seeking (epistemic
   exploration). The information gain term is a KL divergence on Dirichlet
   posteriors — quantifying uncertainty about the world model.

---

## Phase 1: Bayesian Mixture Models

### Task 1 — Slot Mixture Model (sMM)

**File:** `axiom/models/smm.py`
**Tests:** `tests/test_smm.py`
**Depends on:** nothing

**What to build:** Implement the `SlotMixtureModel` class. The sMM is a Gaussian
mixture model that parses an RGB image into object-centric slot latents. This is
the perception front-end of AXIOM.

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

**Bayesian connection:** This is a standard Gaussian mixture model with Bayesian
priors on mixing weights. The E-step / M-step cycle is conjugate EM — the same
algorithm taught in class, applied to image segmentation.

**Implement:** `initialize`, `infer`, `update_params`, `expand_if_needed`.

**Tests to write:**
- Correct pixel assignments on a synthetic image with 2-3 colored rectangles.
- Slot parameter recovery after several E-M iterations.
- Slot expansion when a new object color appears.

---

### Task 2 — Identity Mixture Model (iMM)

**File:** `axiom/models/imm.py`
**Tests:** `tests/test_imm.py`
**Depends on:** Task 1 (uses slot color+shape features)

**What to build:** Implement the `IdentityMixtureModel` class. The iMM assigns a
discrete identity code (object type) to each slot from its color and shape features,
using a Gaussian mixture with Normal-Inverse-Wishart (NIW) priors.

**Input:** Per-slot 5-D vector [c^(k), e^(k)] (3 color + 2 extent dims).

**Math (paper Eqs. 3-4):**
- Each identity type j has parameters (mu_j, Sigma_j) with conjugate
  NIW prior: NIW(m_j, kappa_j, U_j, n_j).
- Likelihood: `p(c^(k), e^(k) | type=j) = N(mu_j, Sigma_j)`.
- Assignments z_type ~ Cat(pi_type) with stick-breaking prior.
- E-step: compute type responsibilities via the NIW posterior predictive
  (a multivariate Student-t distribution).
- M-step: update NIW sufficient statistics (m, kappa, U, n).

**Bayesian connection:** This is a textbook conjugate Bayesian Gaussian mixture.
The NIW prior is the conjugate prior for a multivariate Gaussian with unknown
mean and covariance — a core topic from class. The posterior predictive is a
Student-t, which naturally accounts for parameter uncertainty.

**Implement:** `infer_identity`, `update_params`, `expand_if_needed`.

**Tests to write:**
- Correct identity assignment with 2-3 distinct color/shape clusters.
- NIW posterior update: hand-compute for a known prior + one data point, compare.
- Type expansion when a novel object type appears.

---

### Task 3 — Transition Mixture Model (tMM)

**File:** `axiom/models/tmm.py`
**Tests:** `tests/test_tmm.py`
**Depends on:** Task 1 (uses slot latent states)

**What to build:** Implement the `TransitionMixtureModel` class. The tMM models
dynamics as a switching linear dynamical system (SLDS) with L shared linear modes.

**Input:** Previous slot state x^(k)_{t-1} and a switch variable.

**Math (paper Eq. 5):**
- Each mode l has parameters (D_l, b_l): `x^(k)_t = D_l x^(k)_{t-1} + b_l + noise`.
- Fixed covariance 2I for all components.
- L modes are shared across all K slots.
- Switch variable s^(k)_{t,tmm} selects the active mode.
- Stick-breaking prior on mixing weights.
- M-step: Bayesian linear regression update for D_l, b_l.

**Bayesian connection:** Each linear mode is a Bayesian linear regression — conjugate
prior on (D, b) updated with observed (x_prev, x_curr) pairs. The switching mechanism
is a latent-variable model akin to a hidden Markov model over dynamics modes.

**Implement:** `predict`, `update_params`, `expand_if_needed`.

**Tests to write:**
- Linear prediction correctness with known D, b.
- Parameter recovery from synthetic linear trajectory data.
- Mode expansion when a new trajectory type appears.
- Mode sharing: two slots with the same motion use the same mode.

---

## Phase 2: Variational Inference

### Task 4 — Coordinate-Ascent VI Loop

**File:** `axiom/inference/variational.py`
**Tests:** `tests/test_inference.py`
**Depends on:** Tasks 1-3

**What to build:** Implement the `e_step` and `m_step` functions that wire sMM, iMM,
and tMM into a single-frame variational inference cycle.

**Math (paper Eqs. 8-9):**
- Mean-field factorization: q(Z, Theta) = q(Theta) * prod_t [prod_n q(z^n_smm) * prod_k q(O^(k))].
- E-step per frame:
  1. sMM: assign pixels to slots, yielding slot latents.
  2. iMM: infer identity codes from color/shape features.
  3. tMM: predict next states given current switch states.
- M-step: update all model parameters using sufficient statistics.
- One E-M pass per frame — streaming coordinate-ascent VI.

**Bayesian connection:** This is coordinate-ascent variational inference (CAVI) with
a mean-field approximation. The free energy being minimized is the negative ELBO.
The gradient-free natural parameter updates are possible because every likelihood-prior
pair is conjugate. This is the same CAVI algorithm from class, applied to a structured
multi-component generative model.

**Implement:** `e_step(observation, smm, imm, tmm, rmm)` and
`m_step(latent_states, observation, smm, imm, tmm, rmm)`. The rmm argument
can be a no-op until Task 5.

**Tests to write:**
- End-to-end: synthetic 2-object scene, 10 E-M iterations, verify slot positions converge.
- Verify M-step updates decrease variational free energy.

---

### Task 5 — Recurrent Mixture Model (rMM)

**File:** `axiom/models/rmm.py`
**Tests:** `tests/test_rmm.py`
**Depends on:** Tasks 1-4

**What to build:** Implement the `RecurrentMixtureModel` class. The rMM is the most
complex module — it models the joint distribution of continuous slot features and
discrete game state (identity, dynamics mode, action, reward), capturing object-object
interactions.

**Input:** Continuous features f^(k) (slot position, distance to nearest object) and
discrete features d^(k) (identity code, tMM switch, action, reward).

**Math (paper Eqs. 6-7):**
- Each rMM component m has a factorized likelihood:
  `p(f, d | s_rmm=m) = N(f; mu_m, Sigma_m) * prod_i Cat(d_i; alpha_{m,i})`
- Continuous features from projection C of slot state + interaction function g(x^(1:K)).
- Stick-breaking prior on mixing weights.
- E-step: infer rMM component, which implies a posterior over tMM switch states.
- M-step: update Gaussian (mu, Sigma) and Categorical (alpha) parameters.

**Bayesian connection:** This is a mixture model with mixed continuous-discrete
likelihoods. Each component's Gaussian has an NIW prior; each Categorical has a
Dirichlet prior. The joint inference over continuous and discrete latents is the
same variational EM machinery as the other models, just applied to a richer data type.

**Implement:** `infer_switch`, `update_params`, `expand_if_needed`.

**Then update** `axiom/inference/variational.py` to integrate the rMM into the
E-step and M-step.

**Tests to write:**
- Switch inference with known component parameters.
- Gaussian x Categorical likelihood computation against manual calculation.
- Component expansion when a new interaction type appears.

---

## Phase 3: Structure Learning

### Task 6 — Online Model Expansion

**File:** `axiom/inference/structure_learning.py`
**Tests:** `tests/test_inference.py`
**Depends on:** Tasks 1-5

**What to build:** Extend the existing growing heuristic (threshold + assign-or-expand
are partially implemented) so all four models use it during inference.

**Math (paper Section 2.1):**
- For each data point, compute posterior-predictive log-density under each component.
- Threshold: tau = log p_0(y) + log alpha (prior predictive under empty component + concentration).
- If best component score < tau and capacity remains, create a new component.
- This is a deterministic MAP version of the Chinese Restaurant Process assignment rule.

**Bayesian connection:** The expansion threshold comes directly from the stick-breaking
prior (a finite approximation to a Dirichlet process). The decision to grow vs. reuse
is a Bayesian model comparison: is the marginal likelihood higher under an existing
component or a new one? This is nonparametric Bayes in action.

**Implement:**
- `posterior_predictive_log_density` helper for each model.
- Wire each model's `expand_if_needed` to use the shared `assign_or_expand` logic.

**Tests to write:**
- Expansion fires at the right threshold for each model.
- Capacity limits are respected.
- End-to-end: new object appears at frame 50, slot count increases.

---

### Task 7 — Bayesian Model Reduction (BMR)

**File:** `axiom/inference/bmr.py`
**Tests:** `tests/test_inference.py`
**Depends on:** Task 5

**What to build:** Implement BMR for the rMM. Every 500 frames, evaluate whether
merging pairs of rMM components decreases expected free energy, and greedily merge
the best candidates.

**Math (paper Section 2.1):**
- Sample up to 2000 active rMM component pairs.
- For each pair, score the merge: does the reduced model decrease expected free energy
  of the multinomial distributions over reward and tMM switch?
- Accept merge if free energy decreases; roll back otherwise.
- This enables generalization — e.g., learning that negative reward occurs when the
  ball hits the bottom, by merging multiple single-event clusters.

**Bayesian connection:** BMR is a closed-form Bayesian model comparison between a
full model and a reduced (merged) model. It computes the Bayes factor for the merge
analytically using the Dirichlet-Categorical conjugacy. This is one of the most
distinctly Bayesian pieces of AXIOM — it's principled model selection without
cross-validation or heuristic pruning.

**Implement:** `select_merge_candidates`, `score_merge`, `perform_bmr`.

**Tests to write:**
- Merging two identical components always decreases free energy.
- Merging two very different components does not pass the threshold.
- Active component count decreases after BMR with redundant components.

---

## Phase 4: Uncertainty-Aware Planning

### Task 8 — Expected Free Energy Planner

**File:** `axiom/planning/active_inference.py`
**Tests:** add `tests/test_planning.py`
**Depends on:** Tasks 1-5

**What to build:** Implement the `Planner` class. It selects actions by rolling out
imagined trajectories through the world model and scoring them with expected free energy.

**Math (paper Eq. 10):**
- pi* = argmin_pi sum_tau [ -E[log p(r|O,pi)] - D_KL(q(alpha_rmm|O,pi) || q(alpha_rmm)) ]
- First term: expected utility (reward seeking).
- Second term: information gain — how much the rMM Dirichlet posteriors would change.
- Lower expected free energy = better policy.

**Bayesian connection:** This is the heart of active inference. The agent doesn't just
maximize reward — it also seeks information to reduce uncertainty about its own model.
The information gain is a KL divergence between Dirichlet posteriors, which quantifies
epistemic uncertainty. This is decision-making under uncertainty with a Bayesian
exploration bonus, not an ad-hoc epsilon-greedy strategy.

**Implement:** `select_action`, `compute_expected_utility`, `compute_information_gain`.

**Tests to write:**
- Mock world model: always +1 reward for action 0 → planner selects action 0.
- Information gain is non-negative.
- With info_gain_weight=0, planner is purely reward-seeking.
- With utility_weight=0, planner is purely information-seeking.

---

## Phase 5: Integration and Experiments

### Task 9 — Agent + Environment + Training Loop

**Files:** `axiom/agent.py`, `experiments/run_experiment.py`, `envs/gameworld.py`
**Tests:** `tests/test_agent.py`
**Depends on:** Tasks 1-8

**What to build:** Complete the agent, environment wrapper, and training loop.
Run on **1-2 games** (suggest Bounce and Explode — they show different dynamics
and are among the clearest in the paper's results).

**Agent `observe` method:**
1. Tokenize image via `envs/utils.py:image_to_tokens`.
2. E-step: sMM → iMM → rMM → tMM.
3. M-step: update all parameters.
4. Structure learning: expand models if needed.
5. BMR every 500 steps.

**Training loop:** Load config, init env + agent, loop for 10k steps, save rewards.

**Environment wrapper:** Complete `envs/gameworld.py` reset/step interface.
Requires `make setup-gameworld`.

**Tests:** Smoke test with mock environment, 100 steps, no crash.

---

### Task 10 — Ablation Experiments

**Files:** `experiments/run_ablation.py`
**Depends on:** Task 9

**What to do:** Run three variants on 1-2 games, 5+ seeds each:

1. **Full AXIOM** — all components active (baseline).
2. **No BMR** — disable Bayesian Model Reduction (set `bmr_interval=0`).
3. **No information gain** — disable epistemic exploration (set `info_gain_weight=0`).

These ablations directly answer: "Which Bayesian components matter most for sample
efficiency?" This is the central question of the project.

Save reward arrays to `results/`.

---

### Task 11 — Generate Figures

**Files:** `analysis/plotting.py`, `analysis/metrics.py`
**Depends on:** Task 10

**What to generate:**

1. **Learning curves** (paper Figure 3 style): moving-average reward for full AXIOM
   vs. no-BMR vs. no-info-gain on the 1-2 chosen games. This is the main result.

2. **BMR pruning plot** (paper Figure 4b style): rMM component count over training,
   showing how BMR compresses the model.

3. **Exploration-exploitation plot** (paper Figure 4c style): info gain vs. utility
   over training, showing the shift from exploration to exploitation.

4. **Optional:** rMM cluster visualization (paper Figure 4a style) for interpretability.

Save to `results/figures/` and copy to `docs/report/figures/`.

---

## Phase 6: Report and Presentation

### Task 12 — Draft the Report

**File:** `docs/report/report.tex`
**Depends on:** Task 11

The report is 4-5 pages, 12pt, 1in margins, single-spaced. The audience is a CS 677
student who knows conjugate priors, posterior updates, variational inference, and
mixture models from class, but nothing about active inference or RL.

**Section guide (draft one section at a time):**

**12a — Introduction (~0.5 page):**
- Humans learn games fast using prior knowledge about objects; RL agents don't.
- AXIOM is a Bayesian agent that uses mixture models with conjugate priors to build
  an object-centric world model — no gradient-based optimization.
- This project replicates the Bayesian core of AXIOM on a simplified setting and
  evaluates which components drive sample efficiency.

**12b — Background (~1 page):**
- Gaussian mixture models with conjugate priors (NIW, Dirichlet) — connect to class.
- Variational inference: ELBO / free energy, coordinate-ascent VI, mean-field.
- Nonparametric Bayes: stick-breaking priors for automatic model complexity.
- Active inference: planning as free energy minimization, exploration via info gain.
- Keep this accessible — define terms, avoid assuming RL background.

**12c — Methods (~1.5 pages):**
- AXIOM architecture at a high level: four mixture models and their roles.
- Walk through one model in detail (e.g., sMM) to show the Bayesian mechanics:
  likelihood, prior, E-step, M-step, expansion.
- Structure learning: growing heuristic + BMR as Bayesian model comparison.
- Planning: expected free energy = utility + information gain.
- Note the scope reduction: 1-2 games, focus on Bayesian components.

**12d — Results (~1 page):**
- Learning curves: full AXIOM vs. ablations on 1-2 games.
- What the ablations reveal: does BMR help? Does info gain help?
- BMR pruning and exploration-exploitation dynamics.
- Use figures, not tables of numbers.

**12e — Discussion + Conclusion (~0.5 page):**
- Which Bayesian components matter most and why.
- Connection to CS 677: conjugate priors make online learning tractable, variational
  inference scales to structured models, uncertainty quantification drives exploration.
- Limitations: simplified setting, fewer games than paper, no deep RL baselines.
- What extending to the full benchmark would require (point to `roadmap_extended.md`).

**Compile:** `cd docs/report && make report`

---

### Task 13 — Create the Presentation

**Files:** `docs/presentation/slides.md`, `docs/presentation/script.md`
**Depends on:** Task 12

**What to do:**
1. ~8-10 slides: motivation, Bayesian inference connections (the 5 themes from the
   "Project Framing" section above), AXIOM architecture, ablation results, takeaways.
2. Refine script to ~8 minutes. Spend the most time on the Bayesian connections
   and results interpretation — these are the highest-weighted grading dimensions.
3. Emphasize visual results (learning curves, BMR pruning, exploration-exploitation).
4. Record and post to Discord #final-project channel.

---

## Quick Reference: Task Dependencies

```
Task 1  (sMM) ─────┐
Task 2  (iMM) ──────┼──> Task 4 (VI loop) ──> Task 5 (rMM) ──┐
Task 3  (tMM) ──────┘                                         │
                    Task 6 (structure learning) <──────────────┤
                    Task 7 (BMR) <─────────────────────────────┤
                    Task 8 (planning) <────────────────────────┘
                              │
                              v
                    Task 9 (agent + training, 1-2 games)
                              │
                              v
                    Task 10 (ablation experiments)
                              │
                              v
                    Task 11 (figures)
                              │
                              v
                    Task 12 (report) ──> Task 13 (presentation)
```

Tasks 1, 2, and 3 can be done in parallel.
Tasks 6, 7, and 8 can be done in parallel after Task 5.
