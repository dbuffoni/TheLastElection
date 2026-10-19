# Implementation status

Branch of first slice: complete vertical slice, simulated engines only. "Spec x.y" refers to the Notion spec.

## Built and tested
- **Primitives P1-P14, patterns A/B/C** (`tle.core`): exact-step rates, push buffer (order independent), hysteresis gates and quantizer, centroid, pick, count, aggregate, express. `GameRng` (PCG64, serializable, `child(*keys)` streams).
- **Parameter registry**: 14 TOML groups, typed codegen (`tle.params.Params`), derived formulas, overrides with provenance; `tle params list/codegen`.
- **Catalogs**: trades, domains, 5 pillars x 6 levels (level names/requirements are DRAFT, spec 12.1), 43 event types, activities, 3 scenarios.
- **World state**: structure of arrays, bonds, factions table, influencers, pillars/threats/power fields, additive serialization, per-group digests.
- **Contracts**: 17 closed tasks (S1/S2) as Pydantic models exported to the strict JSON-schema subset; vocabulary checks.
- **Engines**: `SimulatedEngine` (all tasks, noise eps_t / failure injection eps_err, batch `choose_reaction_many`, `choose_activity_many`), `RealEngine` (validation, retries, fallback, accounting), `OpenAICompatBackend` (injectable transport), `MockBackend`, `CachedEngine` (disk cache).
- **Systems (slice v0)**: population generation (S2 -> S1 two-engine boundary), predicates, event pipeline (narrate -> extract -> choose -> push), decree handling with 50-character validation, capital, pillar level gate (theta_up(l) = theta_0 - theta_1 l / L, demotion at theta_down), two threats (3-phase chain + council vote), contagion through ties, fear, economic vote, indicators with label hysteresis, outcomes (victory / removed / term end).
- **Players**: 7 simulated strategies (random, assault, gradual, populist, capture, fear, clientelist) acting through structured decrees; human text client (`tle play`).
- **Telemetry & lab**: Arrow run store (stream while open, compacted at close, crash-readable), `tle.runs` readers, campaigns (seeds/grid/oat/lhs, process pool), analyzers (summary, dispersion, sensitivity, stability, determinism verify), target bands + band loss + dominance.
- **Harness**: CLAUDE.md, 8 skills, 2 agents, post-edit hook, `tle check`, `tools/harness_check.py`, CI workflow.
- **Tests**: primitives, label hysteresis, RNG, registry, schemas, determinism, resume, turn-length independence, failure injection, 50-char rule, real-engine retry/fallback, cache, store, campaigns, CLI.

## Session 2: population evolution monitoring
- **Demography** (`systems/demography.py`): mortality, births to couples, ageing, life transitions, slot reuse and growth; families and couples at creation.
- **Daily life and ties** (`systems/ties.py`): routine partners, tie formation/decay with homophily, ideology pull; **warm-up** (`Game(warmup=True)`, spec 17.5) runs it before day 0 with the power block off.
- **Metric registry** (`config/metrics/population.toml`, `tle.registry.metrics`): ~38 metrics with spec targets (band/min/max/drift_max/ratio/monotone, scenarios, windows). Outcome bands moved to `config/balance/targets.toml`.
- **Population statistics** (`systems/popstats.py`): Sarle bimodality, Gini, invariants; computed each turn.
- **Run store**: `citizens` panel and `bonds` tables, `checks.json`, `state/day0.npz` and `final.npz`, `phase` column.
- **Analyzers and tools** (`tle.analysis.population`, `tle pop ...`, `tle metrics ...`): drift, stability verdicts, target scorecard, mobility, flows, scenario identity, vital rates, impulse test (17.3), live watch, self-contained HTML report.
- New params: `alpha_aware`, `share_coupled_init`, `routine_partners`, `routine_step`, `tie_preference`, `circle_homophily`.
- **Calibration findings (default params, not tuned)**: stage-1 network/demography targets pass after warm-up (mean degree, stable-tie share, population, age, minors, polarization drift). Failing: mobility displacement, impulse response, saturated character share; birth rate about 2x the target.

## Simplified in this slice (decisions to review against the spec)
These are marked `slice v0` in `config/params/events.toml` (module `loop.*`).
- Decrees act on the mayor's opinion through *direct exposure* only. Indirect (published news) and contagion-gated exposure are reduced to one tie-diffusion step on expressed opinion (`_contagion`).
- Only mayor and exogenous events run the full pipeline. Faction, influencer, class and custodian events exist in the catalog but are not generated yet.
- No factions, influencers, economy of rents/transfers, community indices (tec/cul), newspaper/annual edition/almanac, private-message influencer responses (messages are a simple persuasion pull).
- Pillar resistance R_p = share of adults whose *expressed* acceptance of erosion is below -1/3; level gate uses it. Spec formulas for custodians are not implemented; threats are driven by council discontent and law-pillar resistance.
- Parameters introduced by the slice (alpha_react, alpha_ideo, alpha_aware, k_wealth, h_fear, fear_decree, persuade, k_contagion, cost_decree, cost_message, nu_news_decree) are provisional.
- **Balance is not tuned.** With defaults most aggressive strategies get the mayor removed and passive ones reach term end; the target bands in `config/balance/targets.toml` are placeholders.

## Not built
- Panel dashboard, `tle plot`, journal, `runs reindex/gc`, factions, influencers, economy, custodians, MCP server for the lab, in-process `local` model backend (see docs/local-setup.md), Optuna/SALib wrappers (extras declared, not wired), profile promotion/journal tooling, GPU-specific tuning.

## Suggested order for the next increments
1. Factions + influencers (spec 5.x) and their event generators -> turn the unused event types on.
2. Calibrate stages 1-2 on the failing population targets (see findings).
3. Economy (rents, transfers, extractive coalition), community indices.
4. Custodians and the full threat model; replace the placeholder pillar gate inputs.
5. Warm-up, newspaper and almanac (S2 text), then calibrate against real bands.
