"""Balance: target bands (config/metrics/targets.toml), band loss, dominance of strategies."""

from __future__ import annotations

import tomllib
from pathlib import Path

import polars as pl
from tle.registry.loader import repo_root


def load_bands(path: str | Path | None = None) -> list[dict]:
    p = Path(path) if path else repo_root() / "config" / "metrics" / "targets.toml"
    return tomllib.loads(p.read_text())["band"]


def band_report(df: pl.DataFrame, bands: list[dict] | None = None) -> pl.DataFrame:
    """One row per band: observed value, interval, distance outside it (0 if inside)."""
    rows = []
    for b in bands or load_bands():
        sub = df.filter(pl.col("strategy") == b["strategy"])
        if not len(sub):
            continue
        obs = (
            float((sub["outcome"] == b["value"]).mean())
            if b["kind"] == "rate"
            else float(sub[b["value"]].mean())
        )
        miss = max(0.0, b["lo"] - obs, obs - b["hi"])
        rows.append(
            {
                "band": b["id"],
                "strategy": b["strategy"],
                "observed": obs,
                "lo": b["lo"],
                "hi": b["hi"],
                "miss": miss,
                "ok": miss == 0.0,
                "status": b.get("status", ""),
            }
        )
    return pl.DataFrame(rows)


def band_loss(df: pl.DataFrame, bands: list[dict] | None = None) -> float:
    """Scalar objective for optimisation: sum of squared distances outside the bands."""
    rep = band_report(df, bands)
    return float((rep["miss"] ** 2).sum()) if len(rep) else float("inf")


def dominance(df: pl.DataFrame) -> pl.DataFrame:
    """Strategy x scenario matrix of mean levels dismantled: spots a strategy that beats all others everywhere."""
    m = df.group_by(["scenario", "strategy"]).agg(pl.col("levels_total").mean().alias("levels"))
    best = m.group_by("scenario").agg(pl.col("levels").max().alias("best"))
    return m.join(best, on="scenario").with_columns(
        (pl.col("levels") / pl.col("best").clip(1e-9)).alias("relative")
    )
