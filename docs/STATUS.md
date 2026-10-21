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
- **Calibration findings at the end of session 2 (superseded by session 3)**: stage-1 network/demography targets pass after warm-up (mean degree, stable-tie share, population, age, minors, polarization drift). Failing: mobility displacement, impulse response, saturated character share; birth rate about 2x the target.

## Session 3: calibration of stages 1-2 (local, simulated engines, no real-model server was reachable)
Evidence: `passive` strategy (the "no player" baseline of spec 17.2), seeds 1-9 (7 survive 730 days), `tle pop impulse` seeds 1-9.
- **Bug fixed: `GameRng.child` ignored the parent's keys.** `turn_rng.child("exo")`, `child("life")`, ... returned the same stream on
  every turn (exogenous events per turn were nearly constant, daily life repeated). `child` is now keyed by the whole path.
  Every digest changes; the impulse control copies `next_event_id` so both runs share all later draws.
- **Opinion decay implemented**: `h_O` existed but was unused; opinions now relax toward their baseline `Ob` (mayor baseline 0).
- **Indirect exposure** (spec 8.1, `alpha_ind`): adults outside the involved group react like a random involved citizen, attenuated.
- **Impulse response**: `alpha_react` 0.35 -> 2.8 (range up to 4), `k_contagion` 0.01 -> 0.005; new `alpha_direct` (0.35) keeps the
  mayor-opinion pull of a decree at its old strength. Involved 0.26-0.40 (target 0.25-0.42), others 0.08-0.18 (0.07-0.15, one seed over),
  half-life 42-70 days (39-81), 9/9 seeds on the first and third measure.
- **Mobility** (decision about meaning): the metric registry evaluated "measured at two years" on the last row of a five-year run
  (`at_after_days`), and counted label changes without the S7 hysteresis (a continuous crossing at 1/6 made "median >= 1/6" and
  "<= 35% changed" contradictory). Now measured at day 730 with P7 hysteresis against the day-0 label. Dynamics: new `sigma_I`
  (idiosyncratic ideological drift, 0.13/sqrt(year)), `eps` 0.05 -> 0.07. Result: changed 21-30%, median displacement 0.18-0.21, 7/7 survivors.
- **Saturated character**: new `sd_C_init` 0.30 (was a hard-coded 0.35) and `C_max` 0.9 (events cannot push character to the bound).
  Saturated share 0.0-4.6% (was 4-14%).
- **Births**: `nu_child` 0.15 -> 0.05, crude birth rate 0.027 -> 0.010 (target 0.01); population drift -0.3%/year.
- New `passive` strategy (empty player input). Tests that played `gradual` for 20 turns use `random` (gradual is removed earlier now).
- **Outcome re-tune (stage 3 first pass)**: after the RNG fix `fear`/`populist`/`clientelist` won almost always and a passive mayor was
  removed in 33-56% of seeds. LHS search (48 points, seeds 1-6, 7 strategies, loss = distance to the four placeholder bands plus
  passive removal and runaway `fear`/`clientelist`) over `theta_0`, `theta_1`, `alpha_aware`, `g_p`, `theta_down`; best point validated
  on seeds 101-112. Final: `theta_0` 0.25 -> 0.20, `theta_1` 0.15 -> 0.16, `alpha_aware` 0.35 -> 0.47, `g_p` 0.05 -> 0.022,
  `theta_down` 0.35 -> 0.54, `theta_outcome` 0.6 -> 0.8 (earlier manual step). Validation: assault removed 0.67 (band 0.6-1.0),
  gradual victory 0.17 (0.15-0.6), populist pillars fallen 4.0 (1-4.5), random victory 0, passive removed 0.00, `fear` victory 0.33,
  `clientelist` 0.08. All four placeholder bands pass; with 12 seeds the gradual margin is thin. Mobility, impulse and saturation unchanged
  (mobility now in band on 9/9 seeds since the passive mayor survives).

## Session 4: real models and dashboard
- **Real models** (PR #3): S1 and S2 configured separately (`config/engines*.toml`, `.env`), `jev` backend (TypeSafe decision model on OpenRouter), batched
  S1 requests, deadlines, token caps. See `docs/local-setup.md`.
- **Dashboard** (`tle dash [--runs-dir runs] [--port 5006] [--live]`): packages `tle-viz` (Bokeh renderers, generic matrix, sparkline, tables) and
  `tle-dashboard` (Panel app). Views: run explorer (charts per metric group with target band, verdict frame and milestones, events around a day),
  run matrix (runs x metrics with sparklines and verdict frames), targets scorecard, outcomes per strategy/scenario, technical health (execution, warnings).
  `--live` re-reads `runs/` every two seconds. Also citizen panel (ideology scatter, histograms, mobility, class flows), network (degree distribution, tie mix), impulse view (runs the 17.3 test on demand),
  calibration status (targets x runs) and `tle plot <run> <metric>` (PNG with its regeneration command). Not yet: seed grid, parameter sweep and sensitivity matrices,
  stylized facts, journal timeline, click-through between views.
## Session 5: roadmap increments, mocks, recalibration
- **Dashboard** (core complete): run explorer, run matrix, targets, outcomes, citizen panel, network, impulse view, calibration status, technical health;
  `tle plot` for PNG export. Not yet: seed grid, sweep and sensitivity matrices, stylized facts, journal timeline, click-through.
- **Factions (5.7) and influencers (5.9)** (`systems/factions.py`, `systems/influencers.py`, `loop/actors.py`): factions are born from nascent movements (a
  demonstration tagged `faction_birth`, founders within `rho_in`, at least `N_min`), recruit through the events they take part in, hold members by commitment
  with hysteresis, drift with their members and die when weak. Influencers enter by potential and expected share, have a capital, a topic and a catchment area
  with per-axis saturation, and pull their audience. Both act through events (tone follows the actor's opinion of the mayor) and feed political capital (F) and
  the threats (H). New parameters: `nu_found`, `theta_found`, `nu_fac`, `nu_inf`, `w_z_*`, `reach_share`.
- **Custodians and active threats (5.17, 5.12)**: custodians weigh `omega_cust` in the resistance of their pillar; the council is stratified on the first axis with
  alignment bases; pressure grows with the custodians' expressed discontent and the factions' hostility, scaled by the intactness of the linked pillar; a chain
  starts with frequency `nu_T p^gamma`; the mayor is removed only if `p >= theta_outcome` and more than half of the custodians dissent. Capital deficits add to the
  pressure. Simplification: the development phase has one event, not 1-3.
- **Council discipline** (`h_O_council`): the opinion of the mayor of a councillor falls back to its alignment base faster than that of an ordinary citizen. Without
  it, shocks common to all citizens flipped the whole council together and a passive mayor was removed in 20-70% of the seeds.
- **Extractive economy (5.18)** (`systems/economy.py`): one-off transfers and persistent rents between predicates, capped by the erosion of property, concentration,
  deadweight loss, extractive coalition, extractiveness X, economic vote with transfer-driven acceptance of erosion. New strategy `rentier`; `populist` redistributes.
  Not built: transfers from events (news_feature), redistributive risk label.
- **Mocks** (`docs/mocks.md`): decree bank labelled by a real S1 (Jev), narration templates, `tle mocks label-decrees|eval|fit`, `tle run --record`, warm-up on
  simulated engines. First bootstrap: the real S1 was position-biased and weakly context-sensitive, so the mock defaults were kept (fit in `config/mocks/s1_fit.toml`).
  The Jev reaction prompt now describes the citizen in words and shuffles the options; **not yet measured against the real model** (the key reached its daily budget).
- **Full recalibration on the mocks**: one objective for the three groups of targets (population on the passive baseline, impulse test, outcome bands), 96-point LHS
  over 14 parameters, then 64 points with `theta_outcome` tied to the baseline pressure of a divided council (`theta_outcome = p_base + margin`, so that a loyal
  council with 5/12 in opposition cannot remove a quiet mayor), then 40 points on the pillar gate and the capital cost; training seeds 1-12, validation 101-124,
  hold-out 201-224. Promoted: `alpha_react` 2.36, `alpha_ind` 0.366, `k_contagion` 0.0048, `sigma_I` 0.126, `nu_inf` 0.0084, `nu_child` 0.061, `nu_T` 0.318,
  `kappa_D` 8.0, `h_O_council` 38.9, `x_star_T` 0.734, `delay_min` 4.5, `w_M` 0.70, `alpha_direct` 0.846, `theta_outcome` 0.495, `theta_0` 0.245, `theta_1` 0.147,
  `theta_down` 0.617, `alpha_aware` 0.518, `c_0` 0.225, `delta` 0.31, `x_star_K` 0.716. Result (before = the parameters of the previous pass, same seeds):

  | Target | Band | Before (val) | After (val 101-124) | After (hold-out 201-224) |
  |---|---|---|---|---|
  | assault removed | >= 0.6 | 0.04 | **0.62** | 0.50 (just under) |
  | gradual victory | 0.15-0.6 | 0.29 | 0.17 | 0.25 |
  | populist pillars fallen | 1-4.5 | 3.7 | 2.9 | 3.4 |
  | random victory | <= 0.1 | 0.00 | 0.00 | 0.00 |
  | passive removed | low (own check) | 0.04 | 0.00 | 0.08 |
  | label changed at 730 d (seeds in band) | 0.15-0.35 | 100% | 96% | 100% |
  | displacement at 730 d (seeds in band) | >= 0.167 | 91% | 96% | 100% |
  | impulse involved peak (seeds in band) | 0.25-0.42 | 100% | 67% (mean 0.27) | 67% (mean 0.28) |
  | impulse others / half-life | 0.07-0.15 / 39-81 d | in | in / 49 d | in / 49 d |
  | saturated character, births | <= 0.05, 0.01 | 0, 0.0107 | 0, 0.0109 | 0, 0.0113 |

  Still weak: assault removal sits at the edge of its band across seed sets; the impulse involved peak is at the lower edge; `fear` wins 0.6-0.8 of the games and
  `rentier` is removed 0.8-0.9 (strategies without a band).

## Simplified in this slice (decisions to review against the spec)
- Decrees act on the mayor's opinion through *direct exposure* only; indirect exposure of non-involved adults is a random involved reaction attenuated by `alpha_ind`.
- Events of classes and custodians (strike, lockout, ruling, council resolution...) exist in the catalog but are not generated yet; factions, influencers, nascent
  movements, exogenous events, threats and the mayor are.
- No community indices (tec/cul), newspaper/annual edition/almanac, private-message influencer responses (messages are a simple persuasion pull), redistributive
  risk, transfers from events.
- Parameters introduced by the slice (alpha_react, alpha_ideo, alpha_aware, alpha_direct, k_wealth, h_fear, fear_decree, persuade, k_contagion, cost_decree, cost_message,
  nu_news_decree, nu_found, nu_fac, nu_inf, sigma_I, C_max, sd_C_init) are provisional; the outcome bands were confirmed by Daniele on 2026-10-20 but remain provisional.

## Not built
- Seed grid, sweep and sensitivity matrices, stylized facts and journal views of the dashboard; `runs reindex/gc`; journal; class and custodian event generators;
  MCP server for the lab; in-process `local` model backend (see docs/local-setup.md); Optuna/SALib wrappers (extras declared, not wired); profile promotion and
  journal tooling; GPU-specific tuning.

## Suggested order for the next increments
1. Measure the new Jev reaction prompt against the real model (`--record`, `tle mocks fit`), and move the mock defaults if the real S1 becomes context-sensitive.
2. Class and custodian event generators (strike, lockout, ruling, council resolution) and 1-3 development events in the threat chains.
3. Community indices, newspaper and almanac (S2 text), transfers from events.
4. Rebalance `fear`/`rentier` and the assault band with more seeds; wire the calibration loop into `tle` (the scripts used here live outside the repo).
