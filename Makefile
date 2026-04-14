PYTHON := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest

GAME ?= bounce
SEEDS ?= 10
STEPS ?= 10000

.PHONY: setup train sweep ablation test figures clean help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv and install project in editable mode
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo ""
	@echo "Activate with:  source .venv/bin/activate"

setup-baselines: ## Install baseline dependencies (PyTorch, etc.)
	$(PIP) install -e ".[dev,baselines]"

setup-gameworld: ## Clone and install the Gameworld environment suite
	@if [ ! -d "vendor/gameworld" ]; then \
		mkdir -p vendor && \
		git clone https://github.com/VersesTech/gameworld.git vendor/gameworld; \
	fi
	$(PIP) install -e vendor/gameworld

train: ## Train AXIOM on a single game (GAME=bounce STEPS=10000)
	$(PYTHON) experiments/run_experiment.py --game $(GAME) --steps $(STEPS)

sweep: ## Run all games x multiple seeds (SEEDS=10 STEPS=10000)
	$(PYTHON) experiments/sweep.py --seeds $(SEEDS) --steps $(STEPS)

ablation: ## Run ablation experiments
	$(PYTHON) experiments/run_ablation.py --steps $(STEPS)

test: ## Run unit tests
	$(PYTEST) -v

test-cov: ## Run tests with coverage
	$(PYTEST) --cov=axiom --cov-report=term-missing

figures: ## Regenerate plots from results/
	$(PYTHON) -m analysis.plotting

lint: ## Run linter
	.venv/bin/ruff check axiom/ envs/ baselines/ tests/

clean: ## Remove caches, temp files, and results
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache build *.egg-info
