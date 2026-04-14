# Prompting Guide

How to get good results from LLM agents when working on this project.

## Context: What Agents Already Know

**Inside Cursor:** Agents automatically load `.cursor/rules/project.mdc`, which tells them
the repo layout, conventions, how to run things, paper details, and class project requirements.
You don't need to repeat any of that.

**Outside Cursor:** Paste the contents of `.cursor/rules/project.mdc` into your system
prompt, or tell the agent to read it first.

## General Principles

1. **One module per prompt.** "Implement the sMM" beats "implement AXIOM." Each mixture
   model is self-contained — use that.

2. **Quote the math.** The paper PDF is too large for agents to read. Copy the relevant
   equation and a sentence of context into your prompt instead of saying "see the paper."

3. **Point to the stub.** Every module has a file with class/method signatures and a
   docstring explaining what it does. Say "implement the methods in `axiom/models/smm.py`"
   so the agent starts from the right place.

4. **Ask for tests.** Always include "add tests to `tests/test_<module>.py` and run
   `make test`." Agents write better code when they know it will be checked.

5. **Ask for a log entry.** End implementation prompts with "append a summary to
   `docs/replication_log.md`." This builds context for future sessions.

## Prompt Templates

### Implement a module

> Implement [module name] in `axiom/models/[file].py`. Here's the math:
>
> [paste the 2-3 key equations]
>
> The class stub and method signatures are already in place. Use NumPy/SciPy.
> Add tests to `tests/test_[module].py` that verify the update on known inputs.
> Run `make test` when done. Append a summary to `docs/replication_log.md`.

### Fix a bug

> `make test` fails on `tests/test_[module].py::[test_name]`. The expected behavior
> is [describe]. Check the [E-step / M-step / update] in `axiom/[path]` against
> [equation or section reference]. Fix the bug and make all tests pass.

### Run an experiment

> Train AXIOM on [game] for [N] steps across [M] seeds. Save results to `results/`.
> Generate a learning curve using `analysis/plotting.py` and save it to
> `results/figures/`. Use `make train GAME=[game]` or the equivalent Python command.

### Write part of the report

> Draft the [section name] section of `docs/report/report.tex`. The audience is a
> CS 677 student who knows [relevant background] but not [what to explain].
> Cover: [numbered list of points]. Keep it to ~[N] page(s). Cite from
> `docs/report/references.bib`.

### Work on the presentation

> Draft slide content for Slides [N-M] in `docs/presentation/script.md`. Use
> concise bullet points (not full sentences). Focus on [motivation / Bayesian
> connections / results]. The audience is CS 677 classmates.

### Analyze results

> Load the reward data from `results/` and compute the metrics from Table 1 of the
> paper (cumulative reward, mean ± std over seeds). Use `analysis/metrics.py`.
> Generate comparison plots and save to `results/figures/`. Summarize findings
> in `docs/replication_log.md`.

## Implementation Order

Prompt agents in this sequence — each step builds on the previous:

1. **sMM** — object slots from pixels (most self-contained)
2. **iMM** — identity codes from slot features
3. **tMM** — piecewise-linear dynamics
4. **Variational inference loop** — wire sMM + iMM + tMM E/M steps
5. **rMM** — object interactions (most complex, depends on all others)
6. **Structure learning** — extend growing heuristic to all four models
7. **BMR** — Bayesian Model Reduction for rMM pruning
8. **Planning** — expected free energy action selection
9. **Agent + training loop** — full agent interacting with Gameworld
10. **Experiments + analysis** — run games, generate figures and tables
11. **Report + presentation** — write up and present results

## Tips

- **Check the replication log** (`docs/replication_log.md`) before starting a new session.
  It tells you (and the agent) what's been done and what's next.
- **Use `make test` as a gate.** Don't move to the next module until the current one's
  tests pass.
- **Keep experiments reproducible.** Always specify seeds and use the YAML configs in
  `experiments/configs/`.
- **Figures over tables.** The class project grading favors graphical results. When
  prompting for analysis, ask for plots, not just numbers.
