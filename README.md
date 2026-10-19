# The Last Election

A political simulation game driven by an agentic loop of open-source language models. You are a newly elected mayor with
five years (1825 days) to dismantle the five pillars of democracy (pluralism, law, press, property, civil rights). You act
only through decrees and announcements and private messages of **at most 50 characters**. Citizens react, form ties and
factions, and the town produces events.

This repository is the Python implementation. The design specification lives in Notion
([The Last Election](https://app.notion.com/p/The-Last-Election-3e4e701bc1488137a118e7823c40f4d0)).
**Current state: a complete, tested simulation harness running on simulated engines** (no GPU or LLM needed); see
[docs/STATUS.md](docs/STATUS.md) for what is built, what is simplified and what comes next.

## Quick start (local PC)

Requirements: Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/dbuffoni/TheLastElection && cd TheLastElection
uv sync                                    # installs the whole workspace + dev tools
uv run tle check                           # health checks
uv run pytest -q -n auto                   # 45 tests, a few seconds
uv run tle run --strategy gradual          # one simulated game, telemetry in runs/
uv run tle campaign run smoke              # 7 strategies x 2 scenarios x 3 seeds, in parallel
uv run tle play                            # play the mayor yourself in the terminal
```

## Typical workflow when you start simulating and balancing

1. `uv run tle campaign run smoke --seeds 1,2,3,4,5 --n 300` : baseline outcome rates and the table of target bands.
2. Edit the bands in `config/metrics/targets.toml` (they are placeholders) to what you want the game to feel like.
3. `uv run tle campaign run sensitivity_events` then `uv run tle campaign analyze sensitivity_events --analyzer sensitivity` : which parameters matter.
4. Change parameters in `config/params/*.toml` (one registry, typed code is regenerated automatically; `uv run tle params codegen`).
5. Re-run with the same seeds and compare. `uv run tle campaign verify smoke` proves that results are reproducible.
6. When you have a GPU: follow [docs/local-setup.md](docs/local-setup.md) to point the same pipeline at a real model:
   `uv run tle run --engine openai_compat --base-url http://localhost:8000/v1 --model <name> --n 60 --cache runs/cache.jsonl`

Claude Code users: the repo ships skills (`.claude/skills/`: run-simulation, read-runs, balance, sensitivity, calibrate,
debug-determinism, add-system, real-model-setup), two agents (run-analyst, calibrator) and a hook that regenerates the typed
parameters when the registry changes. `CLAUDE.md` holds the working rules.

## Layout

| Path | What |
|---|---|
| `config/params/` | the parameter registry (one TOML per group); defaults, ranges, spec section, status |
| `config/catalogs/` | closed catalogs: trades, domains, 5 pillars x 6 levels, 43 event types, activities, scenarios |
| `config/campaigns/`, `config/metrics/` | experiment definitions and target bands |
| `packages/tle-engine` | primitives, world state, contracts, simulated engines, systems, turn loop, simulated players |
| `packages/tle-models` | real/cached engines, backends (`openai_compat`, mock), prompts |
| `packages/tle-store` | Arrow run store and the `tle.runs` Polars/DuckDB readers |
| `packages/tle-lab` | campaigns, analyzers, band loss, dominance |
| `packages/tle-cli`, `tle-game`, `tle-devtools`, `tle-testing` | the `tle` command, text client, checks, test helpers |

## Architecture in short

Three engines: **S1** (classifier/chooser over closed vocabularies), **S2** (narrator: text and tags, never numbers) and **R**
(deterministic math). Words become numbers only through the seven-level label scale. All model access goes through one
`Engine` protocol, so the simulated engine (fast, controllable noise) and a real model are interchangeable per engine.
Everything is seeded: the same seed and parameters always give the same state digest, and telemetry never influences a run.
