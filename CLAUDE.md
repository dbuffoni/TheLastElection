# The Last Election: working agreement

Political simulation: an LLM-driven town, a player-mayor with 5 years (1825 days) to dismantle five pillars of
democracy using only 50-character decrees and private messages. Spec lives in Notion
(https://app.notion.com/p/The-Last-Election-3e4e701bc1488137a118e7823c40f4d0); `docs/STATUS.md` says which parts exist.

## Architecture in one paragraph
Three engines: **S1** classifies/chooses from closed vocabularies, **S2** narrates text and tags (never numbers),
**R** is deterministic math (`tle.core`, `tle.systems`, `tle.loop`). Words become numbers only through the S7 label
scale (`tle.core.labels`). Everything an LLM does goes through the `Engine` protocol (`tle.engines.protocol`), so
`SimulatedEngine`, `RealEngine` (any backend) and `CachedEngine` are interchangeable per engine.

## Rules that must not be broken
1. **Registry-only parameters.** No numeric constant that tunes behaviour in code. Add it to `config/params/<group>.toml`,
   run `uv run tle params codegen`, use `p.<group>.<name>`. `uv run tle check` fails if `_generated.py` is stale.
2. **Determinism.** No `random`, `np.random.*`, wall clock in `core/ systems/ loop/ world/`. Randomness only via `GameRng`
   (`rng.child(*keys)` per purpose so order of sub-steps never matters). Same seed + params => same `World.digest()`.
3. **Telemetry is a pure observer.** It never draws from the game RNG; a run with telemetry off is bit-identical.
4. **50 characters.** Player text goes through `engines.errors.validate_text` at the input boundary, nowhere else.
5. **Contracts stay in the strict JSON-schema subset** (all fields required, closed enums). `tle check` enforces it.
6. **Additive state.** New `World` fields get neutral defaults; never rename/remove one.
7. **S2 never produces numbers that the model uses**, S1 never produces free text that the model uses.

## Commands
`uv sync` | `uv run pytest -q -n auto` | `uv run tle check` | `uv run tle run --strategy gradual` |
`uv run tle campaign run smoke` | `uv run tle campaign analyze smoke --analyzer sensitivity` | `uv run tle play`

## Layout
`config/` params, catalogs, campaigns, metrics (population registry), balance targets | `packages/tle-engine` core, world, systems, loop, engines, contracts |
`tle-models` real/cached engines + backends | `tle-store` run store + `tle.runs` readers | `tle-lab` campaigns, analyzers (incl. population), impulse test, HTML report, balance |
`tle-cli` the `tle` command | `tle-game` text client | `tle-devtools` checks | `tests/` | `.claude/` skills and agents.

## Working style
Small verified steps; run `uv run tle check` and the tests before committing. When a balance result looks wrong, check
dispersion across seeds before touching a parameter. Record any decision about meaning in `docs/STATUS.md`.
