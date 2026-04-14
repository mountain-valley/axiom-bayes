# AXIOM Replication

Replication and extension of **AXIOM: Active eXpanding Inference with Object-centric Models** (Heins, Van de Maele, Tschantz et al., 2025).

AXIOM is an active inference agent that learns object-centric world models using four expanding mixture models (sMM, iMM, tMM, rMM) and masters pixel-based games within 10,000 interaction steps — without gradient-based optimization.

## Quickstart

```bash
make setup                  # Create venv, install deps
source .venv/bin/activate   # Activate the environment
make setup-gameworld        # Clone & install Gameworld environments
make train GAME=bounce      # Train on a single game
make test                   # Run unit tests
```

## Repository Layout

```
axiom/              Core library: models, inference, planning, agent
envs/               Gameworld wrappers and observation preprocessing
baselines/          BBF, DreamerV3, random agent wrappers
experiments/        Training scripts, configs, sweep orchestration
analysis/           Plotting, metrics, Jupyter notebooks for figures
results/            Output data and generated figures
tests/              Unit tests (per-module)
docs/               Paper notes, replication log, report, presentation
```

### Data Flow

```
experiments/  →  runs training  →  writes to  →  results/
                                                     
analysis/     →  reads from results  →  generates  →  results/figures/
```

- **`tests/`** — Fast unit tests for code correctness. These verify that individual modules
  are mathematically and programmatically correct (e.g., "does the NIW posterior update
  produce the right parameters?"). They don't interact with any environment and run in
  under a second via `make test`.

- **`experiments/`** — Scripts and configs that run AXIOM on games. This is where the
  actual training loop lives: initialize an agent and environment, loop for 10k steps, and
  log metrics. Per-game hyperparameters live in `experiments/configs/`.

- **`results/`** — Output of experiments. When a training run finishes, it saves reward
  arrays, model snapshots, and logged metrics here. Raw data files (`.npy`, `.pkl`, `.csv`)
  are gitignored; only generated figures in `results/figures/` are tracked.

- **`analysis/`** — Code that consumes results. Reads from `results/` to generate the
  learning curves, ablation tables, and interpretability plots (replicating Figures 3–4 and
  Table 1 from the paper). Includes reusable plotting utilities and Jupyter notebooks.

## Key Commands

| Command                        | Description                               |
|--------------------------------|-------------------------------------------|
| `make help`                    | List all available targets                |
| `make train GAME=X STEPS=N`   | Train AXIOM on game X for N steps         |
| `make sweep SEEDS=10`         | Run all 10 games across multiple seeds    |
| `make ablation`               | Run ablation variants (no BMR, no IG, …)  |
| `make test`                   | Run unit tests                            |
| `make figures`                | Regenerate plots from saved results       |
| `make lint`                   | Lint the codebase                         |

## Games (Gameworld 10k)

Aviate · Bounce · Cross · Drive · Explode · Fruits · Gold · Hunt · Impact · Jump

See the [Gameworld repo](https://github.com/VersesTech/gameworld) for environment details.

## References

- Paper: `AXIOM_paper.pdf` in this repo
- Gameworld: https://github.com/VersesTech/gameworld
