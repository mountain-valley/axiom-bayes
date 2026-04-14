# AXIOM Paper Notes

Key equations, design decisions, and open questions from Heins et al. (2025).

## Architecture Summary

AXIOM is a POMDP agent with four Bayesian mixture model modules:

### 1. Slot Mixture Model (sMM)
- Parses RGB images into object-centric latents (position, color, shape/extent)
- Each pixel assigned to one of K slots via Categorical assignment
- Gaussian likelihood with fixed linear projections A (position+color) and B (shape)
- Truncated stick-breaking prior on mixing weights (enables automatic slot expansion)

### 2. Identity Mixture Model (iMM)
- Infers discrete identity codes per object from color+shape features
- Mixture of up to V Gaussians with Normal-Inverse-Wishart priors
- Enables shared dynamics across slots (type-specific, not instance-specific)

### 3. Transition Mixture Model (tMM)
- Switching linear dynamical system (SLDS) per slot
- L linear modes shared across all K slots (e.g., "ball falling", "paddle left")
- Fixed covariance (2I) for all components
- Truncated stick-breaking prior enables growing new modes online

### 4. Recurrent Mixture Model (rMM)
- Models joint distribution of continuous slot features + discrete game state
- Conditions tMM switch states on multi-object interactions
- Inputs: slot positions, distance to nearest object, identity codes, action, reward
- Factorized Gaussian × Categorical likelihood per component

## Inference

- Coordinate-ascent variational inference (mean-field approximation)
- Forward-only filtering, one frame at a time (streaming)
- E-step updates latent states, M-step updates parameters via natural parameter updates
- Gradient-free: all models are exponential family with conjugate priors

## Structure Learning

- **Growing**: compare posterior-predictive log-density vs new-component threshold τ
- **BMR**: every 500 frames, greedily merge rMM components if it decreases expected free energy

## Planning (Eq. 10)

π* = argmin_π Σ [ -E[log p(r|O,π)] - D_KL(q(α_rmm|O,π) || q(α_rmm)) ]

- First term: expected utility (reward seeking)
- Second term: information gain (epistemic exploration)
- Rollouts: 64 to 512 planning rollouts per step

## Key Hyperparameters

- ΔT_BMR = 500 (BMR interval)
- n_pair = 2000 (BMR candidate pairs)
- α₀ values per module (expansion propensity)
- Planning horizon H, number of rollouts
- Per-game configs needed (interaction distance in fixed_distance ablation)

## Open Questions

- (Add questions as they arise during replication)
