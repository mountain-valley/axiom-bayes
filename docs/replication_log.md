# Replication Log

Running record of progress, decisions, and findings. Append new entries at the top.

---

## 2026-04-14 — Implemented Identity Mixture Model

- Implemented `axiom/models/imm.py` with:
  - `infer_identity` using NIW posterior-predictive (multivariate Student-t) responsibilities
  - `update_params` NIW sufficient-statistic updates (`m`, `kappa`, `U`, `n`) and type mixture weights
  - `expand_if_needed` type growth based on prior-predictive novelty under the stick-breaking concentration
- Added `tests/test_imm.py` coverage for:
  - identity assignment with 2-3 distinct color/shape clusters
  - NIW posterior update correctness for a hand-computed one-point example
  - automatic type expansion when a novel object type appears
- Verified compatibility with the previously implemented sMM tests in the full suite.

## 2026-04-14 — Implemented Slot Mixture Model

- Implemented `axiom/models/smm.py` with:
  - `initialize` projection matrices `A`/`B` and variance floors from image resolution
  - `infer` E-step responsibilities via diagonal-Gaussian likelihood + log-sum-exp normalization
  - `update_params` M-step updates for per-slot position, color, extent, color variance, and mixture weights
  - `expand_if_needed` slot growth for poorly explained pixels under current active slots
- Added comprehensive tests in `tests/test_smm.py` for:
  - assignment quality on synthetic multi-rectangle RGB scenes
  - slot parameter recovery after several EM-style iterations
  - automatic slot expansion when a new object appears in-frame
- Next: implement iMM/tMM/rMM and integrate `variational.py` end-to-end.

## 2025-04-14 — Project scaffolding

- Created repo structure: `axiom/`, `envs/`, `baselines/`, `experiments/`, `analysis/`, `tests/`
- Set up `pyproject.toml` with editable install and CLI entry point
- Added Makefile with targets for setup, train, sweep, ablation, test, figures
- Created `.cursor/rules/project.mdc` for agent context
- Next: implement core mixture models (start with sMM)
