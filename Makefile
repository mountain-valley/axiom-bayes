ROOT_DIR := $(CURDIR)
BOOTSTRAP_PYTHON ?= /usr/bin/python3.11
PYTHON := $(ROOT_DIR)/.venv/bin/python
PIP := $(ROOT_DIR)/.venv/bin/pip
PYTEST := $(PYTHON) -m pytest

GAME ?= Explode
SEEDS ?= 3
STEPS ?= 5000
AXIOM_DIR := vendor/axiom
GAMEWORLD_GIT_URL ?= https://github.com/VersesTech/gameworld.git
AXIOM_GIT_URL ?= https://github.com/VersesTech/axiom.git
JAX_PLATFORMS ?= cpu

# Fast CPU settings (override for full runs with FAST=0)
FAST ?= 1
ifeq ($(FAST),1)
FAST_ARGS := --planning_horizon 16 --planning_rollouts 16 --num_samples_per_rollout 1 --bmr_pairs 200 --bmr_samples 200
else
FAST_ARGS :=
endif

.PHONY: setup setup-locked setup-axiom setup-axiom-gpu setup-gameworld vendor-lock baseline test test-cov \
        sweep-phase1 sweep-phase2 sweep-phase3 submit-sweep figures lint clean help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Setup ───────────────────────────────────────────────────────────────────

setup: ## Create venv and install latest-compatible dependencies
	$(BOOTSTRAP_PYTHON) -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo ""
	@echo "Activate with:  source .venv/bin/activate"
	@echo "Then run:  make setup-gameworld && make setup-axiom"

setup-locked: ## Create venv with exact pinned versions from requirements-lock.txt
	$(BOOTSTRAP_PYTHON) -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-lock.txt
	$(PIP) install -e .
	@echo ""
	@echo "Activate with:  source .venv/bin/activate"
	@echo "Then run:  make setup-gameworld && make setup-axiom"

setup-gameworld: ## Clone and install Gameworld (override clone URL: GAMEWORLD_GIT_URL=...)
	@if [ ! -d "vendor/gameworld" ]; then \
		mkdir -p vendor && \
		git clone $(GAMEWORLD_GIT_URL) vendor/gameworld; \
	fi
	$(PIP) install -e vendor/gameworld

setup-axiom: ## Clone and install AXIOM (override clone URL: AXIOM_GIT_URL=...)
	@if [ ! -d "$(AXIOM_DIR)" ]; then \
		mkdir -p vendor && \
		git clone $(AXIOM_GIT_URL) $(AXIOM_DIR); \
	fi
	$(PIP) install -e $(AXIOM_DIR)

setup-axiom-gpu: setup-axiom ## Install AXIOM with pinned GPU extras
	$(PIP) install -e "$(AXIOM_DIR)[gpu]"

vendor-lock: ## Write docs/vendor_versions.txt from current vendor/gameworld and vendor/axiom
	@{ \
		echo "# Vendor pins for reproducibility (Option A: own fork or upstream + pinned SHA)."; \
		echo "# This repo ignores vendor/ in Git; vendor/gameworld and vendor/axiom are separate"; \
		echo "# clones. After you change code there, commit and push in that repo, then run"; \
		echo "#   make vendor-lock"; \
		echo "# and commit this file in axiom-bayes."; \
		echo "#"; \
		echo "# First-time clone from your fork (only when the directory does not exist yet):"; \
		echo "#   GAMEWORLD_GIT_URL=https://github.com/<you>/gameworld.git make setup-gameworld"; \
		echo "#   AXIOM_GIT_URL=https://github.com/<you>/axiom.git make setup-axiom"; \
		echo "#"; \
		echo "# Match pins on another machine (after repos exist):"; \
		echo "#   cd vendor/gameworld && git fetch origin && git checkout <VENDOR_GAMEWORLD_COMMIT>"; \
		echo "#   cd vendor/axiom && git fetch origin && git checkout <VENDOR_AXIOM_COMMIT>"; \
		echo "# then pip install -e vendor/gameworld vendor/axiom as usual."; \
		echo ""; \
		if [ -d vendor/gameworld/.git ]; then \
			printf "VENDOR_GAMEWORLD_REMOTE=%s\n" "$$(git -C vendor/gameworld config --get remote.origin.url)"; \
			printf "VENDOR_GAMEWORLD_BRANCH=%s\n" "$$(git -C vendor/gameworld rev-parse --abbrev-ref HEAD)"; \
			printf "VENDOR_GAMEWORLD_COMMIT=%s\n" "$$(git -C vendor/gameworld rev-parse HEAD)"; \
		else \
			echo "# vendor/gameworld not present — run make setup-gameworld first"; \
		fi; \
		echo ""; \
		if [ -d $(AXIOM_DIR)/.git ]; then \
			printf "VENDOR_AXIOM_REMOTE=%s\n" "$$(git -C $(AXIOM_DIR) config --get remote.origin.url)"; \
			printf "VENDOR_AXIOM_BRANCH=%s\n" "$$(git -C $(AXIOM_DIR) rev-parse --abbrev-ref HEAD)"; \
			printf "VENDOR_AXIOM_COMMIT=%s\n" "$$(git -C $(AXIOM_DIR) rev-parse HEAD)"; \
		else \
			echo "# vendor/axiom not present — run make setup-axiom first"; \
		fi; \
	} > docs/vendor_versions.txt
	@echo "Wrote docs/vendor_versions.txt"

# ─── Baseline ────────────────────────────────────────────────────────────────

baseline: ## Run a baseline AXIOM agent (GAME=Explode STEPS=5000)
	@start=$$(date +%s); \
	mkdir -p results/baseline; \
	echo "[baseline] starting AXIOM run (this progress bar tracks env steps only)"; \
	cd $(AXIOM_DIR) && JAX_PLATFORMS=$(JAX_PLATFORMS) WANDB_MODE=disabled $(PYTHON) main.py \
		--game $(GAME) --num_steps $(STEPS) $(FAST_ARGS); \
	echo "[baseline] model run complete; finalizing output artifacts"; \
	cp $(AXIOM_DIR)/$(shell echo $(GAME) | tr A-Z a-z).csv \
		results/baseline/$(shell echo $(GAME) | tr A-Z a-z)_baseline.csv; \
	cp $(AXIOM_DIR)/$(shell echo $(GAME) | tr A-Z a-z).mp4 \
		results/baseline/$(shell echo $(GAME) | tr A-Z a-z)_baseline.mp4; \
	end=$$(date +%s); \
	elapsed=$$((end - start)); \
	printf "Baseline saved to results/baseline/ (elapsed %02d:%02d:%02d)\n" \
		$$((elapsed / 3600)) $$(((elapsed % 3600) / 60)) $$((elapsed % 60))

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

# ─── Slurm (cluster) ─────────────────────────────────────────────────────────

CONFIG ?=
submit-sweep: ## Submit a sweep config as Slurm array jobs (CONFIG=... GAME=... SEEDS=... STEPS=...)
	@if [ -z "$(CONFIG)" ]; then \
		echo "Usage: make submit-sweep CONFIG=experiments/configs/prior_sensitivity_smm.yaml"; \
		echo "  Optional: GAME=Explode SEEDS=3 STEPS=10000 FAST=0"; \
		exit 1; \
	fi
	$(PYTHON) experiments/slurm/gen_joblist.py $(CONFIG) \
		--game $(GAME) --seeds $(SEEDS) --steps $(STEPS) \
		$(if $(filter 1,$(FAST)),--fast,) > jobs.txt
	@N=$$(wc -l < jobs.txt); \
	echo "Generated $$N jobs from $(CONFIG)"; \
	echo "Submitting: sbatch --array=1-$$N ..."; \
	sbatch --array=1-$$N --export=JOBLIST=$(CURDIR)/jobs.txt \
		experiments/slurm/run_sweep_array.sh; \
	echo "Done. Monitor with: squeue -u $$USER"

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
