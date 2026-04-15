# Project Roadmap

This project studies whether the Bayesian components of AXIOM (Heins et al., 2025) —
variational updating, mixture-based latent dynamics, and uncertainty-aware action
selection — can be replicated in a simplified game setting, and evaluates which
components most affect sample efficiency.

**Modular milestone structure:** Each phase ends with its own experiments and figures,
so you can write a report at any stopping point. Later phases build on earlier ones
and make the story richer, but no phase is wasted. If time runs short, stop at the
latest completed phase and write up what you have.

**Paper fidelity:** Implementation tasks follow the paper's algorithms faithfully —
streaming one-pass-per-frame inference, the exact prior specifications, and the
paper's specific matrix structures. The scope is reduced (1-2 games instead of 10),
but the algorithmic core is preserved. Pedagogical experiments (prior sensitivity
analysis, batch EM demos, etc.) are included as optional additions at each phase
for the CS 677 report, clearly separated from the paper-faithful experiments.

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
| Phase 1 | Bayesian mixture models for object segmentation | Slot assignments on Gameworld frames, identity clustering, streaming convergence |
| Phase 2 | + Variational inference for object-centric dynamics | + Free energy curves, trajectory modes, slot tracking over sequences |
| Phase 3 | + Nonparametric Bayes for adaptive model complexity | + Component growth/pruning, BMR impact on model complexity |
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

**Milestone:** At the end of this phase, you can process Gameworld frames through the
sMM and iMM in the paper's streaming fashion, segmenting objects and assigning identity
codes, with figures showing it works.

### Task 1 — Slot Mixture Model (sMM)

**File:** `axiom/models/smm.py`
**Tests:** `tests/test_smm.py`
**Depends on:** nothing

**What to build:** Implement the `SlotMixtureModel` class exactly as specified in the
paper. The sMM processes one frame at a time in a streaming fashion. It is NOT batch
EM — there is one forward pass per frame, and slot latents carry forward as priors
for the next frame.

**Input:** An (H, W, 3) RGB image, tokenized into (N, 5) pixel tokens [x, y, r, g, b]
by `envs/utils.py:image_to_tokens`.

**Math (paper Eq. 2, Eqs. 12-15 in Appendix A.2):**

The slot latent x^(k) is a vector encoding position p^(k) (2D), color c^(k) (3D),
spatial extent e^(k) (2D), and potentially other features. The sMM likelihood for
pixel n under slot k is:

```
p(y^n | x^(1:K), sigma_c^(1:K), z^n_smm) = prod_k N(A x^(k), diag(B x^(k), sigma_c^(k)))^{z^n_{k,smm}}
```

The projection matrices A and B are **fixed** (not learned):
- `A = [I_5, 0_{5x5}]` — selects position (2D) and color (3D) from the slot latent.
- `B = [0_{2x8}, I_2]` — selects spatial extent (2D) from the slot latent.

These define the Gaussian mean (position + color) and spatial covariance (extent).

Per-slot color variance sigma_c^(k) has a **Gamma prior** per RGB channel:
`p(sigma_c^(k)) = prod_{j in R,G,B} Gamma(gamma_0, 1)` (Eq. 15).

Pixel-to-slot assignments: `z^n_smm | pi_smm ~ Cat(pi_smm)`.
Mixing weights: `pi_smm ~ Dir(1, ..., 1, alpha_0_smm)` — truncated stick-breaking
prior with K-1 ones and a final concentration alpha_0_smm (Eq. 14).

**Streaming inference (one pass per frame):**
- E-step: compute posterior responsibilities for each pixel under each slot, using
  current slot latents (carried forward from previous frame).
- M-step: update slot latent features (position, color, extent) and color variance
  sigma_c from assigned pixels using natural parameter updates.
- Slot latents persist across frames — the posterior from frame t becomes the prior
  for frame t+1.

**Bayesian connection (for report):** This is a Gaussian mixture model with conjugate
priors (Gamma on variance, Dirichlet on mixing weights) applied to image segmentation.
The streaming update is online Bayesian learning — each frame refines the posterior
over slot parameters.

**Implement:** `initialize`, `infer`, `update_params`, `expand_if_needed`.

**Tests to write:**
- Correct pixel-to-slot assignments on a single synthetic frame with 2-3 colored objects.
- Streaming convergence: process 20 identical frames sequentially; verify slot latents
  converge to the true object positions/colors (this tests the streaming prior mechanism,
  NOT batch EM iteration).
- Slot expansion: process a frame with a new object not seen before; verify a new slot
  is created using the stick-breaking threshold.
- Verify A and B projection matrices have the correct fixed structure.

---

### Task 2 — Identity Mixture Model (iMM)

**File:** `axiom/models/imm.py`
**Tests:** `tests/test_imm.py`
**Depends on:** Task 1

**What to build:** Implement the `IdentityMixtureModel` class as specified in the paper.
The iMM assigns a discrete identity code (object type) to each slot based on its 5-D
color+shape features, using a Gaussian mixture with NIW priors.

**Input:** Per-slot 5-D vector [c^(k), e^(k)] (3 color + 2 extent dims).

**Math (paper Eqs. 3-4, Appendix A.6):**
- Each identity type j has parameters (mu_j, Sigma_j) with conjugate
  NIW prior: `p(mu_j, Sigma_j^{-1}) = NIW(m_j, kappa_j, U_j, n_j)`.
- Likelihood: `p(c^(k), e^(k) | type=j) = N(mu_j, Sigma_j)`.
- Assignments: `z_type^(k) | pi_type ~ Cat(pi_type)`.
- Mixing weights: `pi_type ~ Dir(1, ..., 1, alpha_0_type)` — stick-breaking prior
  allowing up to V types.
- E-step: compute type responsibilities using the NIW posterior predictive, which is
  a multivariate Student-t distribution.
- M-step: update NIW sufficient statistics (m, kappa, U, n) from assigned slots.
- Identity codes are NOT slot-specific — the same identity model is shared across all
  K slots, enabling type-specific (not instance-specific) dynamics learning.

**Bayesian connection (for report):** Textbook conjugate Bayesian Gaussian mixture. The
NIW prior is the conjugate prior for a multivariate Gaussian with unknown mean and
covariance. The posterior predictive is a Student-t, naturally accounting for parameter
uncertainty.

**Implement:** `infer_identity`, `update_params`, `expand_if_needed`.

**Tests to write:**
- Correct identity assignment given slots with 2-3 distinct color/shape clusters.
- NIW posterior update correctness: hand-compute for a known prior + one data point,
  compare against the implementation.
- Type expansion when a novel object type appears (new color/shape combination).
- Verify identity codes are shared across slots (two red balls → same type).

---

### Task 3 — Phase 1 Experiments and Figures

**Files:** `analysis/notebooks/`, `analysis/plotting.py`
**Depends on:** Tasks 1-2

**Paper-faithful experiments:**

1. **Streaming segmentation on Gameworld frames:** Install gameworld
   (`make setup-gameworld`). Collect 50 consecutive frames from Bounce using random
   actions. Process them streaming (one pass per frame) through the sMM. Show
   pixel-to-slot assignments at frames 1, 10, 25, 50. Demonstrate that the model
   improves its segmentation over the sequence as it accumulates evidence.

2. **Identity assignment on Gameworld:** After streaming the sMM on the Bounce
   sequence, run iMM on the resulting slot color/shape features. Show that objects
   of the same type (e.g., all balls) get the same identity code, while distinct
   types (ball vs. paddle vs. wall) get different codes.

3. **Slot latent convergence:** Plot the sMM slot positions over the 50-frame sequence.
   Show them converging to the true object positions within the first few frames.

**Figures to produce:**
- Gameworld frame with slot assignments overlaid (color-coded by slot) at t=1, 10, 50.
- Slot position traces over the 50-frame streaming sequence (convergence plot).
- iMM identity clusters visualized in 2D (PCA or first two dims of color-shape space).

**Optional pedagogical experiments (for CS 677 report):**

4. **Batch EM baseline:** Run a standard (non-streaming) Gaussian mixture model on a
   single Gameworld frame for 20 EM iterations. Compare convergence behavior against
   AXIOM's streaming single-pass approach. This illustrates how streaming Bayesian
   updating trades per-frame computation for temporal accumulation of evidence.

5. **Prior sensitivity:** Vary alpha_smm (e.g., 0.1, 1.0, 10.0) and show how the
   stick-breaking concentration affects the number of slots created. This demonstrates
   the Bayesian nonparametric mechanism for controlling model complexity.

6. **NIW prior effect:** Show what happens to iMM identity assignment with a very
   tight NIW prior (high kappa) vs. a diffuse one (low kappa). Illustrate how the
   prior balances prior knowledge and observed evidence.

Save to `results/figures/phase1_*`.

---

## Phase 2: Variational Inference for Dynamics Learning

**Milestone:** At the end of this phase, you can process Gameworld frame sequences
through the full sMM → iMM → tMM pipeline in a streaming fashion, tracking objects
and learning their dynamics modes.

### Task 4 — Transition Mixture Model (tMM)

**File:** `axiom/models/tmm.py`
**Tests:** `tests/test_tmm.py`
**Depends on:** Task 1

**What to build:** Implement the `TransitionMixtureModel` class as specified in the
paper. The tMM models per-slot dynamics as a switching linear dynamical system (SLDS)
with L shared linear modes.

**Input:** Previous slot state x^(k)_{t-1} and switch variable s^(k)_{t,tmm}.

**Math (paper Eq. 5, Appendix A.7, Eqs. 26-28):**

Each mode l has linear parameters (D_l, b_l):
```
p(x^(k)_t | x^(k)_{t-1}, s_{t,tmm}, D_{1:L}, b_{1:L}) = prod_l N(D_l x^(k)_{t-1} + b_l, G_t^{-1} * 2I)^{s_{t,l,tmm}}
```

Key details from the paper:
- The covariance is `G_t^{-1} * 2I`, where G_t is a per-slot "moving and present" gate
  (Appendix A.3). G_t = 1 for active/moving objects, effectively giving covariance 2I.
  G_t → 0 for absent/stationary objects, inflating covariance to ignore their transitions.
- D_l, b_l have **uniform (flat) priors** (Eq. 28) — NOT informative conjugate priors.
  The update is still via sufficient statistics of a linear Gaussian model, but the prior
  is non-informative.
- The L modes are **shared across all K slots** — not slot-specific.
- Switch variable: `s^(k)_{t,tmm} | pi_tmm ~ Cat(pi_tmm)`.
- Mixing weights: `pi_tmm ~ Dir(1, ..., 1, alpha_0_tmm)` — stick-breaking prior.

**Bayesian connection (for report):** The tMM is a switching linear dynamical system
where each mode is a linear-Gaussian model with flat priors. The sufficient-statistics
update is a special case of Bayesian linear regression with a non-informative prior.
The switching mechanism is a latent-variable model analogous to an HMM over dynamics
modes, with the stick-breaking prior enabling automatic discovery of new modes.

**Implement:** `predict`, `update_params`, `expand_if_needed`. Include the G_t gating
mechanism.

**Tests to write:**
- Linear prediction correctness: set known D, b, verify x_next = D @ x_prev + b.
- Parameter recovery: generate data from a known linear system, run sufficient-statistics
  updates over a sequence of (x_prev, x_curr) pairs, verify D and b converge.
- G_t gating: verify that transitions for stationary/absent objects (G_t ≈ 0) do not
  update the mode parameters.
- Mode expansion when a qualitatively different trajectory appears.
- Mode sharing: verify that two slots following the same motion pattern use the same mode.

---

### Task 5 — Coordinate-Ascent VI Loop

**File:** `axiom/inference/variational.py`
**Tests:** `tests/test_inference.py`
**Depends on:** Tasks 1-2, Task 4

**What to build:** Implement `e_step` and `m_step` that wire sMM, iMM, and tMM into
a single-frame variational inference cycle, exactly as described in the paper.

**Math (paper Eqs. 8-9, Section 2 "Variational inference"):**

AXIOM uses a mean-field factorization:
```
q(Z_{0:T}, Theta) = q(Theta) * prod_t [ prod_n q(z^n_{t,smm}) * prod_k q(O^(k)_t) ]
q(Theta) = q(Theta_smm) * q(Theta_imm) * q(Theta_tmm) * q(Theta_rmm)
```

The E-step and M-step run **once per timestep** — this is streaming coordinate-ascent
VI, not iterative EM. Per frame t:

1. **E-step (forward filtering):**
   - sMM: assign pixels to slots using current slot latents → yields O_t.
   - iMM: infer identity codes from slot color/shape features.
   - (tMM switch inference handled by rMM in Phase 3; for now, use the most likely
     mode based on prediction error.)
   - tMM: predict next slot states given switch states.

2. **M-step (natural parameter updates):**
   - Update sMM slot latents (position, color, extent, sigma_c).
   - Update iMM NIW parameters from slot features.
   - Update tMM linear parameters (D_l, b_l) from (x_{t-1}, x_t) transition pairs.

These updates use sufficient statistics accumulated from the E-step and the current
observation. The gradient-free form inherits from the exponential-family structure.

**Bayesian connection (for report):** This is coordinate-ascent variational inference
(CAVI) with a mean-field approximation. The free energy being minimized is the negative
ELBO. Each update step has a closed-form solution because every likelihood-prior pair
is conjugate (or flat, for tMM).

**Implement:** `e_step(observation, smm, imm, tmm, rmm)` and
`m_step(latent_states, observation, smm, imm, tmm, rmm)`. The rmm argument
can be a no-op until Phase 3.

**Tests to write:**
- Streaming convergence: process 50 frames of a synthetic 2-object sequence (one pass
  per frame); verify slot positions track the objects across frames.
- Verify M-step updates decrease variational free energy across frames.
- Verify the VI loop processes one frame at a time (no iterating on a single frame).

---

### Task 6 — Phase 2 Experiments and Figures

**Files:** `analysis/notebooks/`, `analysis/plotting.py`
**Depends on:** Tasks 4-5

**Paper-faithful experiments:**

1. **Streaming VI on Gameworld Bounce:** Run 500 random-action steps on Bounce. Process
   the frame sequence through the full VI loop (sMM → iMM → tMM), one frame at a time.
   Show that the sMM tracks object positions across frames and the tMM discovers distinct
   linear modes (e.g., ball moving right, ball moving down-left after bounce).

2. **Learned tMM modes:** After processing the sequence, visualize the tMM linear
   parameters (D_l, b_l). Plot the implied displacement vectors in 2D position space.
   Show that each mode corresponds to an interpretable motion pattern.

3. **Free energy over the sequence:** Plot variational free energy per frame. Show it
   generally decreases over time as the model accumulates evidence and improves.

4. **Active tMM mode count:** Plot the number of active tMM modes over the sequence.
   Show modes being created as the model encounters new dynamics.

**Figures to produce:**
- Slot positions overlaid on frames at t=1, t=100, t=300, t=500 (tracking over time).
- Variational free energy vs. frame number.
- tMM mode displacement vectors in 2D position space (arrow plot).
- Number of active tMM modes vs. frame number.

**Optional pedagogical experiments (for CS 677 report):**

5. **Synthetic bouncing ball:** Generate a clean synthetic sequence (100 frames) of a
   single ball on a linear trajectory with one bounce (direction change). Process
   through the VI loop. Show that the tMM discovers exactly two modes (pre-bounce and
   post-bounce). This is a simplified illustration of mode discovery without the
   complexity of a real game environment.

6. **Streaming vs. batch comparison:** Process the same 50-frame sequence in two ways:
   (a) streaming one-pass-per-frame (paper-faithful), and (b) batch EM with 10
   iterations per frame. Compare convergence speed and final accuracy. This illustrates
   the tradeoff between per-frame computation and temporal evidence accumulation.

Save to `results/figures/phase2_*`.

---

## Phase 3: Nonparametric Bayes — Growing and Pruning

**Milestone:** At the end of this phase, the full four-model world model (sMM + iMM +
tMM + rMM) runs on Gameworld sequences with automatic model expansion and BMR pruning.

### Task 7 — Recurrent Mixture Model (rMM)

**File:** `axiom/models/rmm.py`
**Tests:** `tests/test_rmm.py`
**Depends on:** Tasks 1-5

**What to build:** Implement the `RecurrentMixtureModel` as specified in the paper.
The rMM models the joint distribution of continuous and discrete slot features,
capturing object-object interactions and conditioning the tMM switch states.

**Input:** Per slot k, a tuple of continuous features f^(k) and discrete features d^(k).

**Math (paper Eqs. 6-7, Appendix A.8):**

The rMM is a mixture model over mixed continuous-discrete data:
```
p(f^(k)_{t-1}, d^(k)_{t-1} | s^(k)_{t,rmm}) = prod_m [ N(f; mu_m, Sigma_m) * prod_i Cat(d_i; alpha_{m,i}) ]^{s_{t,m,rmm}}
```

**Continuous features** f^(k) are computed from (Appendix A.8):
- Projection C of slot k's own state: position, velocity-like features.
- Interaction function g(x^(1:K)): X and Y displacement to the nearest object,
  identity code of the nearest object, and other interaction features.
  The function g computes nearest-neighbor distances to populate the z_interacting
  latent (Appendix A.4).

**Discrete features** d^(k) include:
- Identity code z_type^(k) (from iMM).
- tMM switch state s^(k)_{t,tmm}.
- Action a_{t-1}.
- Reward r_t.

Each component's Gaussian has **NIW priors**; each Categorical has **Dirichlet priors**.
Mixing weights: `pi_rmm ~ Dir(1, ..., 1, alpha_0_rmm)`.

The rMM's assignment variable s^(k)_{t,rmm} implies a posterior over the tMM switch
state — this is how object interactions influence dynamics predictions.

**Bayesian connection (for report):** The rMM is a mixture model with a factorized
Gaussian x Categorical likelihood. Each factor has a conjugate prior (NIW for Gaussian,
Dirichlet for Categorical), enabling closed-form variational updates. The joint
inference over continuous and discrete latents is a natural extension of standard
mixture model EM to richer data types.

**Implement:** `infer_switch`, `update_params`, `expand_if_needed`.
Then update `axiom/inference/variational.py` to integrate the rMM into the E-step
and M-step, replacing the temporary tMM switch inference.

**Tests to write:**
- Switch inference with known component parameters: verify correct assignment.
- Gaussian x Categorical likelihood computation against manual calculation.
- Component expansion when a new interaction type appears.
- Verify that rMM assignment implies a posterior over tMM switch states.

---

### Task 8 — Online Model Expansion

**File:** `axiom/inference/structure_learning.py`
**Tests:** `tests/test_inference.py`
**Depends on:** Tasks 1-7

**What to build:** Extend the growing heuristic (partially implemented) so all four
models use it, following the paper's procedure exactly.

**Math (paper Section 2.1 "Fast structure learning"):**

The paper's growing procedure, applied identically to each model:

1. For each new data point y_t, compute the variational posterior-predictive
   log-density under each existing component c:
   `l_{t,c} = E_{q(Theta_c)}[log p(y_t | Theta_c)]`.

2. The truncated stick-breaking prior defines the new-component threshold:
   `tau_t = log p_0(y_t) + log alpha`
   where p_0 is the prior predictive density under an **empty** (fresh prior) component.

3. Select c* = argmax_c l_{t,c}. If l_{t,c*} >= tau_t, hard-assign y_t to c*.
   Otherwise, if C_{t-1} < C_max, instantiate a new component and assign to it.

4. Update the chosen component's parameters via the M-step.

This is a deterministic MAP version of the CRP assignment rule.

**Bayesian connection (for report):** The expansion threshold comes from the stick-breaking
prior (a finite approximation to a Dirichlet process). The grow-vs-reuse decision is
Bayesian model comparison: the posterior-predictive log-density is a marginal likelihood,
and tau is the marginal likelihood under a new component.

**Implement:**
- `posterior_predictive_log_density` for each model (sMM, iMM, tMM, rMM).
- Wire each model's `expand_if_needed` to the shared `assign_or_expand` logic.

**Tests to write:**
- Expansion fires at the correct threshold for each model type.
- Capacity limits (C_max) are respected.
- Process a Gameworld sequence where a new object appears mid-sequence; verify slot
  count increases at the right frame.

---

### Task 9 — Bayesian Model Reduction (BMR)

**File:** `axiom/inference/bmr.py`
**Tests:** `tests/test_inference.py`
**Depends on:** Task 7

**What to build:** Implement BMR for the rMM as specified in the paper. Every
Delta_T_BMR = 500 frames, evaluate whether merging rMM component pairs decreases
expected free energy.

**Math (paper Section 2.1 "Bayesian Model Reduction"):**

1. Every 500 frames, sample up to n_pair = 2000 pairs of active rMM components.
2. For each pair, generate data from the model via **ancestral sampling**.
3. Score the merge: compute the expected free energy of the multinomial distributions
   over reward and tMM switch under the full model and the reduced (merged) model.
4. Greedily accept merges that decrease expected free energy; roll back otherwise.

This enables generalization — e.g., learning that negative reward occurs when the ball
hits the bottom, by merging multiple single-event clusters that each saw one instance
of this event.

**Bayesian connection (for report):** BMR is closed-form Bayesian model comparison.
It computes the Bayes factor for merging two components analytically using the
Dirichlet-Categorical conjugacy. This is principled model selection — the model
automatically discovers the right level of granularity without cross-validation or
heuristic pruning.

**Implement:** `select_merge_candidates`, `score_merge`, `perform_bmr`.

**Tests to write:**
- Merging two identical components always decreases free energy.
- Merging two very different components does not pass the threshold.
- Active component count decreases after BMR on a model with redundant components.

---

### Task 10 — Phase 3 Experiments and Figures

**Files:** `analysis/notebooks/`, `analysis/plotting.py`
**Depends on:** Tasks 7-9

**Paper-faithful experiments:**

1. **Full world model on Gameworld:** Run the complete four-model pipeline
   (sMM → iMM → rMM → tMM) on a 2000+ frame Gameworld rollout (Bounce or Explode,
   random actions). Process streaming, one frame at a time with structure learning
   active. This is the paper's online learning setup minus the planner.

2. **BMR pruning dynamics (replicating Figure 4b):** Plot rMM component count over
   frames. Show the initial growth phase as new interactions are encountered, then
   the sharp decline when BMR fires at 500-frame intervals and merges redundant
   components.

3. **With vs. without BMR:** Run the same 2000-frame sequence with BMR enabled and
   disabled. Compare rMM component count over time and prediction free energy. This
   replicates the "no BMR" ablation from Table 1 at the world-model level.

4. **rMM cluster visualization (replicating Figure 4a right panel):** After processing,
   plot rMM clusters in 2D position space, colored by predicted reward (green =
   positive, red = negative). Show that the model has learned where in space reward
   and punishment occur.

**Figures to produce:**
- rMM component count vs. frame (with and without BMR) — replicates Figure 4b.
- rMM clusters in 2D space colored by reward prediction — replicates Figure 4a (right).
- Active slot count and tMM mode count over the sequence.
- Free energy over time with and without BMR.

**Optional pedagogical experiments (for CS 677 report):**

5. **Expansion threshold sensitivity:** Vary alpha_0_rmm and show how it affects the
   initial growth rate of rMM components. Higher alpha → more components before BMR
   intervenes. This illustrates the role of the Dirichlet process concentration
   parameter in controlling model complexity.

6. **Merge score distribution:** After one BMR round, histogram the merge scores
   across all evaluated pairs. Show the distribution of Bayes factors and where the
   accept/reject threshold falls. This makes the Bayesian model comparison step
   concrete and visual.

Save to `results/figures/phase3_*`.

---

## Phase 4: Uncertainty-Aware Decision Making (Stretch Goal)

**Milestone:** Full AXIOM agent playing Gameworld games, with ablation experiments
showing which Bayesian components matter most.

### Task 11 — Expected Free Energy Planner

**File:** `axiom/planning/active_inference.py`
**Tests:** add `tests/test_planning.py`
**Depends on:** Tasks 1-9

**What to build:** Implement the `Planner` class as specified in the paper.

**Math (paper Eq. 10, Appendix A.11):**
```
pi* = argmin_pi sum_{tau=0}^{H} [ -E_{q(O_tau|pi)}[log p(r_tau | O_tau, pi)] - D_KL(q(alpha_rmm | O_tau, pi) || q(alpha_rmm)) ]
```

- First term: **expected utility** — evaluated using the learned model and slot latents,
  accumulated over the planning horizon H.
- Second term: **information gain** — computed using the posterior Dirichlet counts of
  the rMM. Scores how much information about rMM switch states would be gained by
  taking the policy under consideration.
- Policies are sequences of H actions. The paper samples 64 to 512 random action
  sequences and selects the one with lowest expected free energy.
- Rollouts are imagined trajectories through the learned world model from the current
  slot latents.

**Bayesian connection (for report):** The agent seeks information to reduce uncertainty
about its world model. The information gain term is a KL divergence between Dirichlet
posteriors (before and after imagined observations), quantifying epistemic uncertainty.
This is decision-making under uncertainty with a principled Bayesian exploration bonus,
not an ad-hoc epsilon-greedy strategy.

**Implement:** `select_action`, `compute_expected_utility`, `compute_information_gain`.

**Tests to write:**
- Mock world model that always predicts +1 reward for action 0 → planner selects action 0.
- Information gain is non-negative (KL divergence property).
- info_gain_weight=0 → purely reward-seeking.
- utility_weight=0 → purely information-seeking.

---

### Task 12 — Full Agent + Gameworld Integration

**Files:** `axiom/agent.py`, `experiments/run_experiment.py`, `envs/gameworld.py`
**Tests:** `tests/test_agent.py`
**Depends on:** Tasks 1-11

**What to build:** Complete the agent, environment wrapper, and training loop.
Target 1-2 games (Bounce and Explode recommended).

**Agent `observe` method (paper's per-step procedure):**
1. Tokenize image via `envs/utils.py:image_to_tokens`.
2. E-step: sMM (assign pixels → slot latents) → iMM (infer identities) →
   rMM (infer switch states from interaction features) → tMM (predict next states).
3. M-step: update all four models' parameters via natural parameter updates.
4. Structure learning: run `expand_if_needed` on each model.
5. If step % 500 == 0: run BMR on the rMM.
6. Store slot latents for the planner.

**Training loop:** Load config, init env + agent, loop 10k steps (observe → act → step),
save reward array to `results/`.

**Environment wrapper:** Complete `envs/gameworld.py` reset/step interface. AXIOM
operates on full 210x160 frames (not downscaled like the baselines).

**Tests:** Smoke test — mock environment, 100 steps, no crash.

---

### Task 13 — Ablation Experiments and Figures

**Files:** `experiments/run_ablation.py`, `analysis/plotting.py`
**Depends on:** Task 12

**Paper-faithful experiments (replicating Table 1 ablations on 1-2 games):**

Run three variants on Bounce and Explode, 5+ seeds each:
1. **Full AXIOM** — all components active.
2. **No BMR** — `bmr_interval=0` (disable Bayesian Model Reduction).
3. **No information gain** — `info_gain_weight=0` (disable epistemic exploration).

**Figures to produce:**
- Learning curves (replicating Figure 3): moving-average reward per step for full AXIOM
  vs. no-BMR vs. no-info-gain. Mean ± std over seeds.
- Exploration-exploitation tradeoff (replicating Figure 4c): per-step info gain and
  expected utility over training. Show info gain decreasing and utility increasing.
- Cumulative reward summary (replicating Table 1 format): mean ± std over seeds.

**Optional pedagogical experiments (for CS 677 report):**

4. **Random agent baseline:** Run a random agent on the same games for comparison.
   This provides a floor to show that AXIOM is actually learning, without requiring
   the full BBF/DreamerV3 baseline infrastructure.

5. **Planning horizon sensitivity:** Vary the planning horizon H (e.g., 1, 4, 8, 16)
   and show how it affects cumulative reward. This illustrates the tradeoff between
   planning depth and computational cost in active inference.

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
- Note scope: 1-2 games, faithful algorithm, reduced benchmark.

**14d — Results (~1 page):**
- Figures from whichever phases you completed.
- If Phase 1 only: streaming segmentation on Gameworld + identity clustering.
- If Phase 2: + dynamics learning + free energy convergence.
- If Phase 3: + BMR pruning dynamics + model complexity adaptation.
- If Phase 4: + game-playing learning curves + ablation analysis.
- Include any optional pedagogical experiments that strengthen the Bayesian narrative.

**14e — Discussion + Conclusion (~0.5 page):**
- What worked, what the results show about the Bayesian components.
- Connection to CS 677 concepts.
- What further phases or full-benchmark replication would add.

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
