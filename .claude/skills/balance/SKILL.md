---
name: balance
description: Check whether the simulation lands inside the target bands and find which parameters move it; use when tuning difficulty or after changing rules.
---

# Balance loop

Targets live in `config/balance/targets.toml` (bands per strategy, status `placeholder` until the designer sets them).
1. Baseline: `uv run tle campaign run smoke --seeds 1,2,3,4,5 --n 300` then read the "target bands" table (`miss` = distance outside).
2. If the baseline is dominated by one outcome (all removed / all victory), it is a structural problem, not a tuning one: inspect
   `read-runs` for the first turns before touching parameters.
3. Find levers: `uv run tle campaign run sensitivity_events` then `analyze sensitivity_events --analyzer sensitivity`.
   Rank correlation near 0 = parameter irrelevant for that metric; |rho| > 0.5 with 3+ levels = lever.
4. Move one lever, rerun with the same seeds (paired comparison). Keep the change only if the band `miss` shrinks without
   breaking another band. Use `--analyzer dominance` to ensure no strategy beats all others everywhere.
5. Record the decision (parameter, old -> new, evidence) in `docs/STATUS.md` and set the registry `status` to `calibrated`
   only after the designer approves.
Edit parameters only in `config/params/*.toml`; the hook regenerates the typed code.
