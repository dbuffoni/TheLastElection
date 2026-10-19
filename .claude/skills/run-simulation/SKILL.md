---
name: run-simulation
description: Run one game or a campaign of simulated players (simulated or real-model engines) and report outcomes; use when asked to simulate, play out, or generate runs.
---

# Run a simulation

1. `uv sync` once. `uv run tle check` must pass (generated params current, schemas strict, no global RNG).
2. One game: `uv run tle run --strategy gradual --scenario industrial_secular --seed 1 --n 300`
   - strategies: random, assault, gradual, populist, capture, fear, clientelist
   - scenarios: industrial_secular, rural_devout, mixed_polarised
   - override a parameter for this run only: `--set alpha_aware=0.5` (repeatable). Names come from `uv run tle params list`.
3. Many games: `uv run tle campaign run smoke` (all strategies x scenarios x 3 seeds; add `--seeds 1,2,3,4,5 --n 300` for real work).
   Campaign files are in `config/campaigns/*.toml` (designs: seeds, grid, oat, lhs).
4. Real model (needs a local server, see the `real-model-setup` skill):
   `uv run tle run --engine openai_compat --base-url http://localhost:8000/v1 --model <name> --cache runs/cache.jsonl`
5. Look at results with the `read-runs` skill. Never conclude from fewer than 3 seeds.

Telemetry lands in `runs/<run_id>/` (Arrow tables). Do not commit `runs/` (it is git-ignored).
