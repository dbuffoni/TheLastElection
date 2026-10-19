---
name: add-system
description: Add or extend a simulation rule (system, parameter, event type, contract) following the project conventions; use to implement the next piece of the spec.
---

# Add a system

Read `docs/STATUS.md` (what exists, what is next) and the relevant Notion spec section first.
1. Parameters: add entries to `config/params/<group>.toml` (symbol, spec section, range, `status = "provisional"`), `uv run tle params codegen`.
2. State: add fields to `world/state.py` with neutral defaults (additive only); add them to `GROUPS` if they affect behaviour.
3. Rule: new module in `tle/systems/`, pure function `(world, cat, params, rng_child) -> effects`; wire it into `loop/turn.py`
   at the point the spec's turn sequence (7.3) says. Use primitives from `tle.core.prims`; pushes via `PushBuffer` so order is irrelevant.
4. New LLM task? Add contract in `contracts/models.py`, entry in `contracts/tasks.py`, a `_t_<task>` handler in `SimulatedEngine`,
   and a prompt in `tle-models` if it needs special wording. `tle check` verifies the strict subset.
5. Catalog data (event types, pillars levels, activities) lives in `config/catalogs/*.toml`.
6. Tests: unit test of the rule, determinism/resume tests still green, `uv run pytest -q -n auto`, `uv run tle check`.
7. Update `docs/STATUS.md`.
