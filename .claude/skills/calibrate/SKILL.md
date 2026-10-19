---
name: calibrate
description: Calibrate free parameters against the target bands with a search (grid/LHS now, optional Optuna) and promote the result safely.
---

# Calibrate

Objective = `tle.balance.band_loss(summary_df)` (sum of squared distances outside the bands).
1. Choose <= 6 free parameters (registry `class = "free"` or `status = "provisional"` with a range).
2. LHS sweep: campaign with `design = "lhs"`, `samples = 64`, all strategies the bands mention, >= 3 seeds.
3. Per point compute `band_loss` on that point's rows (`df.filter(pl.col("point") == i)`); take the best 3 points, rerun them on
   fresh seeds (seed partitions: search seeds 1-9, validation seeds 101-109) to avoid overfitting.
4. Only the designer promotes values: edit `config/params/*.toml` `default`, set `status = "calibrated"`, run `uv run tle check`
   and the full tests, and write the evidence in `docs/STATUS.md`.
5. Optuna (optional): `uv sync --extra search`, wrap `run_campaign` + `band_loss` in an objective; keep `n_trials` small first.
