# AXIOM Bayesian Analysis

A Bayesian analysis of **AXIOM: Active eXpanding Inference with Object-centric Models** (Heins et al., 2025) for CS 677 Bayesian Statistics.

AXIOM is an active inference agent that learns object-centric world models using four expanding Bayesian mixture models (sMM, iMM, tMM, rMM) and masters pixel-based games within 10,000 steps — without gradient-based optimization. This project installs the [official codebase](https://github.com/VersesTech/axiom) and conducts three focused analyses:

1. **Prior and Hyperparameter Sensitivity** — How do prior specifications and model hyperparameters affect perception, dynamics, and reward?
2. **Bayesian Model Reduction** — How does principled model selection via expected free energy compare to naive pruning?
3. **Exploration vs. Exploitation** — How does the information-gain term (KL on Dirichlet posteriors) shape learning?

## Quickstart

```bash
make setup                  # Create venv, install analysis dependencies
source .venv/bin/activate   # Activate the environment
make setup-gameworld        # Clone & install Gameworld environments
make setup-axiom            # Clone & install official AXIOM codebase
make baseline GAME=Explode  # Run a baseline agent
make test                   # Run tests
```

## Repository Layout

```
vendor/axiom/       Official AXIOM source (JAX, installed editable)
vendor/gameworld/   Official Gameworld environment suite
experiments/        Sweep runner, YAML configs for each analysis phase
analysis/           Plotting utilities, helpers, Jupyter notebooks
results/            Output CSVs, BMR logs, generated figures
tests/              Setup verification, infrastructure tests
docs/               Roadmap, paper notes, replication log, report, presentation
```

### Data Flow

```
experiments/  →  runs AXIOM with varied configs  →  writes to  →  results/
                                                                     │
analysis/     →  reads results  →  generates  →  results/figures/
```

- **`experiments/`** — Sweep scripts and YAML configs that run AXIOM (`vendor/axiom/main.py`) with different hyperparameter settings. Each run produces a CSV with per-step reward, expected utility, expected info gain, and component count.

- **`results/`** — Output of experiments, organized by phase (`prior_sensitivity/`, `bmr_ablation/`, `info_gain_sweep/`). Raw CSVs are gitignored; generated figures in `results/figures/` are tracked.

- **`analysis/`** — Code that consumes results: loads CSVs, computes statistics, generates figures. Jupyter notebooks produce the final plots for the report.

## Key Commands

| Command                        | Description                                        |
|--------------------------------|----------------------------------------------------|
| `make help`                    | List all available targets                         |
| `make baseline GAME=X`        | Run a baseline AXIOM agent on game X               |
| `make sweep-phase1`           | Run prior and hyperparameter sensitivity experiments |
| `make sweep-phase2`           | Run BMR ablation experiments                       |
| `make sweep-phase3`           | Run info-gain sweep experiments                    |
| `make figures`                | Regenerate plots from saved results                |
| `make test`                   | Run tests                                          |
| `make lint`                   | Lint analysis and experiment code                  |

All sweep commands accept `GAME=X`, `SEEDS=N`, `STEPS=N`, and `FAST=1` (default) / `FAST=0` (full runs).

## Games (Gameworld 10k)

Aviate · Bounce · Cross · Drive · Explode · Fruits · Gold · Hunt · Impact · Jump

See the [Gameworld repo](https://github.com/VersesTech/gameworld) for environment details.

## References

- Paper: `AXIOM_paper.pdf` in this repo
- Official code: https://github.com/VersesTech/axiom
- Gameworld: https://github.com/VersesTech/gameworld
