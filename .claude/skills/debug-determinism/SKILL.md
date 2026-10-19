---
name: debug-determinism
description: Diagnose a determinism failure (same seed, different digest) or a resume mismatch.
---

# Debug determinism

1. Reproduce: `uv run tle campaign verify smoke` (re-runs and compares per-run digests).
2. `uv run tle check` (AST scan for global RNG / clock in the core).
3. Bisect by group: `w.digest_groups()` returns a hash per state group (citizens, bonds, factions, power, scalars, rng). Run two
   executions turn by turn (`Game.step`) and print the first turn and group where they diverge.
4. Usual causes: iteration over a `set`/unsorted dict feeding numbers; a draw from the main stream in a conditional branch
   (use `rng.child(...)` per purpose); float accumulation order in `np.add.at` replaced by a different reduction; state not in
   `World` (module global, cache) used by the model.
5. Resume mismatch: a field missing from `World` (not serialized) or from `GROUPS`. `tests/test_game.py::test_resume_*` guards it.
