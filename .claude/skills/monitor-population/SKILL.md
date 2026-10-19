---
name: monitor-population
description: Monitor how the simulated population evolves (polarization, mobility, demography, network) against the metric registry targets; use after any run or when calibrating stages 1-2.
---

# Monitor the population

The metric registry (`config/metrics/population.toml`) declares every population metric with its spec target, stage and
family. Targets for the game outcome live separately in `config/balance/targets.toml`.

- Run with a report: `uv run tle run --strategy gradual --report` (self-contained HTML with scorecard, trajectories, panel).
- Live: `uv run tle pop watch <run_id>`; static: `uv run tle pop report <run_id>`.
- Questions: `tle pop targets <run>` (scorecard), `stability <run|campaign>` (stable/drifting/oscillating/degenerate, bistable
  across seeds), `mobility` (S7 label changes, median displacement), `identity` (scenario identity after warm-up),
  `panel` (citizen snapshots), `impulse` (spec 17.3: one decree on 20% of adults vs a control copy).
- `uv run tle metrics list|lint`: lint checks the registry against the telemetry columns (also part of `tle check`).
- Metrics rows have `phase` (`warmup` has negative days, `game` the rest); `Run.metrics.daily()` gives the per-day view.
- Before changing a parameter, look at dispersion across at least 3 seeds. Warm-up is part of the model, not a test fixture.
- Add a metric: declare it in the registry, emit it in `systems/popstats.py` or the observer, then `tle metrics lint`.
