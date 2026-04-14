# Presentation Script (~8 minutes)

Target: CS 677 classmates on Discord. Focus on motivation, Bayesian inference connections, and results.

## Slide 1: Title (~30s)
- Project title, your name, CS 677

## Slide 2: Motivation (~1 min)
- Deep RL is powerful but data-hungry
- Humans use prior knowledge about objects to learn games in minutes
- Can we build agents that do the same using Bayesian inference?

## Slide 3: What is AXIOM? (~1 min)
- Active inference agent with object-centric world models
- Four Bayesian mixture models, each with a specific role
- Gradient-free: uses conjugate priors and variational inference

## Slide 4: Connection to Bayesian Inference (~1.5 min)
- Conjugate priors (NIW for Gaussians, Dirichlet for Categoricals)
- Variational inference / free energy minimization
- Dirichlet process-like model expansion (stick-breaking priors)
- Bayesian Model Reduction for pruning
- Active inference: planning as inference

## Slide 5: The Four Models (~1.5 min)
- sMM: objects from pixels (Gaussian mixture)
- iMM: object identity (clustering by color/shape)
- tMM: dynamics (switching linear systems)
- rMM: interactions (joint continuous-discrete mixture)

## Slide 6: Results — Learning Curves (~1 min)
- Show Figure 3 replication
- AXIOM vs BBF vs DreamerV3 on select games
- Highlight sample efficiency

## Slide 7: Results — Ablations (~1 min)
- No BMR, no information gain, fixed distance
- What each component contributes

## Slide 8: Takeaways (~30s)
- Bayesian structure learning enables fast adaptation
- Interpretable world models as a bonus
- Questions / discussion prompt
