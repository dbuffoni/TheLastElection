---
name: read-runs
description: Inspect run directories and campaign summaries (metrics, events, milestones, execution) with Polars/DuckDB; use to answer "what happened in this run".
---

# Read runs

- List: `uv run tle runs list`; one run: `uv run tle runs show <run_id> --table metrics --tail 30`
  (tables: metrics per turn, events news items, milestones pillar changes/threat phases/outcome, execution per-task engine accounting).
- Campaign table: `uv run tle campaign analyze <name> --analyzer summary|dispersion|sensitivity|stability|dominance|bands`
- In Python:
  ```python
  from tle.runs import table, sql

  table("industrial_secular-gradual-s1", "metrics", "runs")
  sql("select scenario_id, avg(approval) from metrics group by 1")  # DuckDB over all runs; run_id column added
  ```
- Open runs (still writing) are readable up to the last flushed batch (`status.json` has the heartbeat).
- Questions to ask of a run: when did capital collapse? which decree preceded a threat chain? did resistance track fear
  (expressed vs private opinion diverge)? which pillar stalled at which level and why (`decree_result == "resisted"`)?
- Population view: tables `citizens` (panel snapshots every `panel_every` days) and `bonds`; `metrics` has a `phase` column;
  `Run.metrics.daily()` for per-day series. See the monitor-population skill.
