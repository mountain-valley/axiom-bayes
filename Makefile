PYTHON := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest

GAME ?= Explode
SEEDS ?= 3
STEPS ?= 5000
AXIOM_DIR := vendor/axiom

# Fast CPU settings (override for full runs with FAST=0)
FAST ?= 1
ifeq ($(FAST),1)
FAST_ARGS := --planning_horizon 16 --planning_rollouts 16 --num_samples_per_rollout 1 --bmr_pairs 200 --bmr_samples 200
else
FAST_ARGS :=
endif

.PHONY: setup setup-locked setup-axiom setup-gameworld baseline test test-cov \
        sweep-phase1 sweep-phase2 sweep-phase3 figures lint clean help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Setup ───────────────────────────────────────────────────────────────────

setup: ## Create venv and install latest-compatible dependencies
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo ""
	@echo "Activate with:  source .venv/bin/activate"
	@echo "Then run:  make setup-gameworld && make setup-axiom"

setup-locked: ## Create venv with exact pinned versions from requirements-lock.txt
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-lock.txt
	$(PIP) install -e .
	@echo ""
	@echo "Activate with:  source .venv/bin/activate"
	@echo "Then run:  make setup-gameworld && make setup-axiom"

setup-gameworld: ## Clone and install the Gameworld environment suite
	@if [ ! -d "vendor/gameworld" ]; then \
		mkdir -p vendor && \
		git clone https://github.com/VersesTech/gameworld.git vendor/gameworld; \
	fi
	$(PIP) install -e vendor/gameworld

setup-axiom: ## Clone and install the official AXIOM codebase
	@if [ ! -d "$(AXIOM_DIR)" ]; then \
		mkdir -p vendor && \
		git clone https://github.com/VersesTech/axiom.git $(AXIOM_DIR); \
	fi
	$(PIP) install -e $(AXIOM_DIR)

# ─── Baseline ────────────────────────────────────────────────────────────────

baseline: ## Run a baseline AXIOM agent (GAME=Explode STEPS=5000)
	@mkdir -p results/baseline
	cd $(AXIOM_DIR) && WANDB_MODE=disabled $(PYTHON) main.py \
		--game $(GAME) --num_steps $(STEPS) $(FAST_ARGS)
	cp $(AXIOM_DIR)/$(shell echo $(GAME) | tr A-Z a-z).csv \
		results/baseline/$(shell echo $(GAME) | tr A-Z a-z)_baseline.csv || true
	@echo "Baseline saved to results/baseline/"

# ─── Phase 1: Prior Sensitivity ──────────────────────────────────────────────

sweep-phase1: ## Run prior sensitivity sweeps (Phase 1)
	$(PYTHON) experiments/run_sweep.py --config experiments/configs/prior_sensitivity_smm.yaml \
		--game $(GAME) --seeds $(SEEDS) --steps $(STEPS) $(if $(filter 1,$(FAST)),--fast,)
	$(PYTHON) experiments/run_sweep.py --config experiments/configs/prior_sensitivity_tmm.yaml \
		--game $(GAME) --seeds $(SEEDS) --steps $(STEPS) $(if $(filter 1,$(FAST)),--fast,)
	$(PYTHON) experiments/run_sweep.py --config experiments/configs/prior_sensitivity_rmm.yaml \
		--game $(GAME) --seeds $(SEEDS) --steps $(STEPS) $(if $(filter 1,$(FAST)),--fast,)

# ─── Phase 2: BMR ────────────────────────────────────────────────────────────

sweep-phase2: ## Run BMR ablation experiments (Phase 2)
	$(PYTHON) experiments/run_sweep.py --config experiments/configs/bmr_ablation.yaml \
		--game $(GAME) --seeds $(SEEDS) --steps $(STEPS) $(if $(filter 1,$(FAST)),--fast,)

# ─── Phase 3: Exploration vs. Exploitation ───────────────────────────────────

sweep-phase3: ## Run info-gain sweep experiments (Phase 3)
	$(PYTHON) experiments/run_sweep.py --config experiments/configs/info_gain_sweep.yaml \
		--game $(GAME) --seeds $(SEEDS) --steps $(STEPS) $(if $(filter 1,$(FAST)),--fast,)

# ─── Analysis ────────────────────────────────────────────────────────────────

figures: ## Regenerate all plots from results/
	$(PYTHON) -m analysis.plotting

# ─── Testing & Quality ───────────────────────────────────────────────────────

test: ## Run tests
	$(PYTEST) -v

test-cov: ## Run tests with coverage
	$(PYTEST) --cov=analysis --cov=experiments --cov-report=term-missing

lint: ## Run linter
	.venv/bin/ruff check analysis/ experiments/ tests/

# ─── Cleanup ─────────────────────────────────────────────────────────────────

clean: ## Remove caches and temp files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache build *.egg-info
