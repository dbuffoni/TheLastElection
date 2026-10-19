"""Analyzers over a campaign summary table (Polars). Each returns a DataFrame so results compose."""

from __future__ import annotations

import numpy as np
import polars as pl


def summary(df: pl.DataFrame, by=("scenario", "strategy")) -> pl.DataFrame:
    """Outcome rates and mean progress per group."""
    g = df.group_by(list(by)).agg(
        pl.len().alias("runs"),
        (pl.col("outcome") == "victory").mean().alias("victory"),
        (pl.col("outcome") == "removed").mean().alias("removed"),
        (pl.col("outcome") == "term_end").mean().alias("term_end"),
        pl.col("pillars_fallen").mean().alias("pillars_fallen"),
        pl.col("days").mean().alias("days"),
    )
    return g.sort(list(by))


def dispersion(df: pl.DataFrame, metric: str = "levels_total", by=("scenario", "strategy")) -> pl.DataFrame:
    """Seed-to-seed spread: if it is large against the effect you study, add seeds before concluding."""
    return df.group_by(list(by)).agg(
        pl.col(metric).mean().alias("mean"),
        pl.col(metric).std().alias("sd"),
        (pl.col(metric).std() / pl.col(metric).mean().abs().clip(1e-9)).alias("cv"),
    )


def sensitivity(df: pl.DataFrame, metric: str = "levels_total") -> pl.DataFrame:
    """Spearman-style rank correlation of each varied parameter (p_*) with a metric."""
    rows = []
    for c in [c for c in df.columns if c.startswith("p_")]:
        sub = df.filter(pl.col(c).is_not_null())
        if sub[c].n_unique() < 2:
            continue
        x = sub[c].rank().to_numpy()
        y = sub[metric].rank().to_numpy()
        rho = float(np.corrcoef(x, y)[0, 1]) if y.std() > 0 else 0.0
        rows.append({"param": c[2:], "rank_corr": rho, "abs": abs(rho), "levels": sub[c].n_unique()})
    return pl.DataFrame(rows).sort("abs", descending=True) if rows else pl.DataFrame()


def stability(df: pl.DataFrame) -> pl.DataFrame:
    """Share of (scenario, strategy) groups whose outcome is the same for every seed."""
    return df.group_by(["scenario", "strategy"]).agg(pl.col("outcome").n_unique().alias("distinct_outcomes"))


def verify_determinism(df_a: pl.DataFrame, df_b: pl.DataFrame) -> pl.DataFrame:
    """Compare the digests of two executions of the same campaign: any difference is a determinism bug."""
    j = df_a.select("run_id", "digest").join(
        df_b.select("run_id", pl.col("digest").alias("digest_b")), on="run_id"
    )
    return j.filter(pl.col("digest") != pl.col("digest_b"))
