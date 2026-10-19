---
name: sensitivity
description: Design and run one-at-a-time, grid or Latin-hypercube parameter campaigns and read the sensitivities.
---

# Sensitivity

1. Copy a file from `config/campaigns/` and set `design = "oat" | "grid" | "lhs"`, `params = [...]` (names from `uv run tle params list`; ranges come from the registry).
2. `uv run tle campaign expand <name>` prints the number of runs: keep it under a few thousand per batch.
3. `uv run tle campaign run <name> --seeds 1,2,3,4`
4. `uv run tle campaign analyze <name> --analyzer sensitivity --metric levels_total` (also try `pillars_fallen`, `days`).
5. Check `--analyzer dispersion` first: if seed-to-seed cv is larger than the effect, add seeds.
Optional extras for Sobol/Optuna: `uv sync --extra search` (SALib, optuna), not required for the built-in designs.
