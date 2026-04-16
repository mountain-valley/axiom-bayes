# Alternate Roadmap (Time-Crunch Option C)

This is a condensed alternative to `docs/roadmap.md` for situations where runs
must fit within roughly 30 minutes on a GPU machine.

## Goal

Deliver a credible, Bayesian-focused result tied to the AXIOM paper by running
only a minimal Exploration vs. Exploitation study (Phase 3 style) on one game.

## Scope

- Keep existing supercomputer jobs running as-is.
- Run this contingency plan on a separate GPU machine.
- Use one game only: `Bounce`.
- Use a coarse info-gain sweep only: `info_gain in [0.0, 0.1, 1.0]`.
- Use `--fast` settings to meet the runtime constraint.

## Why this option

- It stays close to the AXIOM paper's "no IG" ablation (`0.0` vs `0.1`).
- It preserves the Bayesian decision-theory framing:
  - utility term vs. information-gain term,
  - exploration-to-exploitation transition.
- It avoids invasive source changes required by BMR deep-dive tasks.

## Run Budget (Target: <= 30 minutes)

- Configurations: 3 (`0.0`, `0.1`, `1.0`)
- Seeds: 3
- Total runs: 9
- Steps per run: `5000`
- Mode: `--fast` + `--no_video`

This is intentionally a coarse sweep to fit strict wall-clock limits.

## Execution Plan

### Task C1 - Launch Minimal Sweep

Run:

```bash
python experiments/run_sweep.py \
  --config experiments/configs/info_gain_sweep.yaml \
  --game Bounce \
  --steps 5000 \
  --seeds 3 \
  --fast
```

Then restrict analysis to these values only:

- `info_gain=0.0`
- `info_gain=0.1`
- `info_gain=1.0`

If needed, run explicitly by parameter values instead of full config:

```bash
python experiments/run_sweep.py \
  --param info_gain --values 0.0 0.1 1.0 \
  --game Bounce --steps 5000 --seeds 3 --fast
```

Expected outputs in:

- `results/info_gain_sweep/`

### Task C2 - Produce Three Core Figures

From `results/info_gain_sweep/`, generate:

1. Learning curves by info-gain (`0.0`, `0.1`, `1.0`)
2. Utility vs. Expected Info Gain timeseries (default `0.1` highlighted)
3. Transition-point summary (first step where utility exceeds scaled info gain)

Suggested save paths:

- `results/figures/contingency_c_reward_curves.png`
- `results/figures/contingency_c_utility_infogain.png`
- `results/figures/contingency_c_transition_points.png`

### Task C3 - Report-Ready Writeup

Write a short subsection (0.5-1 page equivalent) with:

- Question: How does epistemic bonus magnitude affect short-horizon learning?
- Method: one-game, coarse sweep, fast settings, 3 seeds.
- Result: compare `0.0` vs `0.1` replication anchor, then `1.0` extension.
- Bayesian interpretation:
  - low IG under-explores,
  - default IG balances learning and reward,
  - high IG may over-explore and delay reward harvesting.

## Interpretation Caveats

- This contingency plan prioritizes runtime over fidelity.
- `--fast` settings reduce planning/BMR quality, so absolute rewards should not
  be compared directly to full-fidelity runs.
- Treat this as a defensible interim analysis that can be superseded by
  full Phase 3 results once longer runs complete.

## Definition of Done

- 9 runs complete (`3 info_gain values x 3 seeds`).
- Three figures generated.
- Brief Bayesian interpretation drafted for class report.
- Entry appended to `docs/replication_log.md` documenting this contingency run.

