"""Population analyzers (spec 8.5, 10, 17): drift, stability verdicts, target evaluation, mobility, class flows,
scenario identity through the warm-up, degree and age distributions. They read a ``tle.runs.Run`` and return
plain dicts or Polars frames, so the CLI, the HTML report and the agents all see the same numbers."""

from __future__ import annotations

import numpy as np
import polars as pl
from tle.registry.metrics import MetricSpec, load_metrics
from tle.runs import Run
from tle.systems.popstats import sarle

BIMODAL = 5 / 9


# ---------------------------------------------------------------------------------------------- series tools
def window_means(days: np.ndarray, values: np.ndarray, window: float) -> tuple[float, float]:
    """Mean over the last ``window`` days and over the window before it."""
    end = days.max()
    last = values[days > end - window]
    prev = values[(days > end - 2 * window) & (days <= end - window)]
    return (
        float(np.nanmean(last)) if len(last) else np.nan,
        float(np.nanmean(prev)) if len(prev) else np.nan,
    )


def drift(days, values, window: float = 180.0, floor: float = 0.0) -> float:
    """Relative change of the mean between the last window and the one before (spec 17.2: 'drift at most 10%')."""
    a, b = window_means(np.asarray(days, float), np.asarray(values, float), window)
    if np.isnan(a) or np.isnan(b):
        return float("nan")
    scale = max(abs(b), floor, 1e-9)
    return abs(a - b) / scale


def slope_per_year(days, values) -> float:
    d, v = np.asarray(days, float), np.asarray(values, float)
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return float("nan")
    return float(np.polyfit(d[ok] / 365.0, v[ok], 1)[0])


def spectral_peak(days, values) -> tuple[float, float]:
    """(period in days, share of detrended power) of the strongest periodic component; resamples on a daily grid."""
    d, v = np.asarray(days, float), np.asarray(values, float)
    ok = np.isfinite(v)
    if ok.sum() < 16:
        return float("nan"), 0.0
    grid = np.arange(d[ok].min(), d[ok].max() + 1)
    y = np.interp(grid, d[ok], v[ok])
    y = y - np.polyval(np.polyfit(grid, y, 1), grid)
    if y.std() < 1e-12:
        return float("nan"), 0.0
    P = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1.0)
    P, f = P[1:], f[1:]
    k = int(P.argmax())
    return float(1 / f[k]), float(P[k] / P.sum())


def verdict(
    days,
    values,
    window: float = 180.0,
    tol: float = 0.10,
    lo: float = 0.0,
    hi: float = 1.0,
    floor: float = 0.0,
) -> dict:
    """stable | drifting | oscillating | degenerate for one metric of one run (bistability needs several seeds)."""
    d, v = np.asarray(days, float), np.asarray(values, float)
    v_ok = v[np.isfinite(v)]
    if len(v_ok) < 4:
        return {"verdict": "insufficient", "drift": float("nan")}
    rng_scale = max(abs(np.nanmean(v_ok)), 1e-9)
    if np.nanstd(v_ok) / rng_scale < 1e-6 and len(v_ok) > 8:
        flat = True
    else:
        flat = False
    at_limit = (
        np.isclose(v_ok[-max(2, len(v_ok) // 10) :], lo).all()
        or np.isclose(v_ok[-max(2, len(v_ok) // 10) :], hi).all()
    )
    dr = drift(d, v, window, floor)
    period, share = spectral_peak(d, v)
    amp = (np.nanpercentile(v_ok, 95) - np.nanpercentile(v_ok, 5)) / rng_scale
    if at_limit and not flat:
        kind = "degenerate"
    elif dr > tol:
        kind = "drifting"
    elif share > 0.5 and amp > 0.10 and period < (d.max() - d.min()) / 3:
        kind = "oscillating"
    else:
        kind = "stable"
    return {
        "verdict": kind,
        "drift": dr,
        "slope_per_year": slope_per_year(d, v),
        "period_days": period,
        "peak_power": share,
    }


def bistable_across_seeds(final_values) -> dict:
    """Spec 8.5: bimodal across seeds when the Sarle coefficient of the final values is above 0.555 (needs >= 8 seeds)."""
    x = np.asarray(final_values, float)
    return {"n": len(x), "sarle": sarle(x), "bistable": bool(len(x) >= 8 and sarle(x) > BIMODAL)}


# ---------------------------------------------------------------------------------------------- targets
def evaluate_targets(run: Run, specs: dict[str, MetricSpec] | None = None) -> pl.DataFrame:
    """One row per metric with a target: observed value, expectation, verdict (ok | miss | n/a)."""
    specs = specs or load_metrics()
    scenario = run.manifest.get("scenario", "")
    m = run.metrics().df
    rows = []
    for mid, s in specs.items():
        t = s.target
        if t is None or mid not in m.columns:
            continue
        if t.scenarios and scenario not in t.scenarios:
            continue
        sub = m.filter(pl.col("day") >= t.after_days) if t.after_days else m
        if not len(sub) or sub[mid].drop_nulls().len() == 0:
            rows.append(
                {
                    "metric": mid,
                    "kind": t.kind,
                    "observed": None,
                    "expected": _expect(t),
                    "verdict": "n/a",
                    "note": "run shorter than after_days",
                }
            )
            continue
        d, v = sub["day"].to_numpy(), sub[mid].to_numpy()
        k = max(1, min(len(v), 5))
        last = float(np.nanmean(v[:k] if t.at_after_days else v[-k:]))
        if t.kind == "drift_max":
            obs, ok = drift(d, v, t.window_days, s.scale_floor), None
            ok = bool(obs <= t.value) if obs == obs else None
        elif t.kind == "band":
            obs, ok = last, bool(t.lo <= last <= t.hi)
        elif t.kind == "min":
            obs, ok = last, bool(last >= t.value)
        elif t.kind == "max":
            obs, ok = last, bool(last <= t.value)
        elif t.kind == "monotone":
            obs = slope_per_year(d, v)
            ok = bool(obs >= 0) if (t.value or 1) >= 0 else bool(obs <= 0)
        else:
            obs, ok = last, None
        rows.append(
            {
                "metric": mid,
                "kind": t.kind,
                "observed": obs,
                "expected": _expect(t),
                "verdict": "n/a" if ok is None else ("ok" if ok else "miss"),
                "note": t.note,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "metric": pl.String,
            "kind": pl.String,
            "observed": pl.Float64,
            "expected": pl.String,
            "verdict": pl.String,
            "note": pl.String,
        },
    )


def _expect(t) -> str:
    if t.kind == "band":
        return f"[{t.lo}, {t.hi}]"
    if t.kind == "min":
        return f">= {t.value}"
    if t.kind == "max":
        return f"<= {t.value}"
    if t.kind == "drift_max":
        return f"drift <= {t.value} / {t.window_days:g}d"
    return t.kind


def band_loss_run(run: Run, specs=None) -> float:
    """Spec 8.6 for one run: distance outside the band divided by the band width, summed over target metrics."""
    ev = evaluate_targets(run, specs)
    loss = 0.0
    for r in ev.iter_rows(named=True):
        if r["verdict"] == "miss" and r["observed"] is not None:
            loss += 1.0
    return loss


# ---------------------------------------------------------------------------------------------- panel analyses
def _same_person(a: pl.DataFrame, b: pl.DataFrame) -> pl.DataFrame:
    return a.join(b, on=["id", "arrival_day"], suffix="_b")


def mobility(run: Run, day0: float = 0.0, day1: float | None = None) -> dict:
    """Mobility test of spec 17.3 between two panel days: share of adults whose S7 ideology label changed on at least
    one axis, median displacement (max over axes), and mean polarization drift."""
    a, b = run.panel(day0), run.panel(day1)
    if not len(a) or not len(b):
        return {}
    adult_a = a.filter(pl.col("age") >= 18)
    j = _same_person(adult_a, b)
    I = [f"I{k}" for k in (1, 2, 3)]
    lab = lambda x: np.clip(np.floor((x + 1) / 2 * 6 + 0.5), 0, 6)  # noqa: E731
    now = np.stack([j[c + "_b"].to_numpy() for c in I], 1)
    then = np.stack([j[c].to_numpy() for c in I], 1)
    pol = lambda x: 2 * np.sqrt((x**2).sum(1)) / (2 * np.sqrt(3))  # noqa: E731
    return {
        "day0": float(a["day"][0]),
        "day1": float(b["day"][0]),
        "people": len(j),
        "share_label_changed": float(np.mean((lab(now) != lab(then)).any(1))) if len(j) else float("nan"),
        "median_displacement": float(np.median(np.abs(now - then).max(1))) if len(j) else float("nan"),
        "polarization_day0": float(pol(then).mean()) if len(j) else float("nan"),
        "polarization_day1": float(pol(now).mean()) if len(j) else float("nan"),
    }


def class_flows(run: Run, day0: float = 0.0, day1: float | None = None) -> pl.DataFrame:
    """Social mobility: counts of (class at day0, class at day1) for people present at both snapshots."""
    a, b = run.panel(day0), run.panel(day1)
    if not len(a) or not len(b):
        return pl.DataFrame()
    j = _same_person(a, b)
    return (
        j.group_by(["cls", "cls_b"])
        .len()
        .sort(["cls", "cls_b"])
        .rename({"cls": "from", "cls_b": "to", "len": "citizens"})
    )


def age_distribution(run: Run, day: float | None = None, bin_years: int = 10) -> pl.DataFrame:
    p = run.panel(day)
    if not len(p):
        return pl.DataFrame()
    return (
        p.with_columns((pl.col("age") // bin_years * bin_years).cast(pl.Int32).alias("age_bin"))
        .group_by("age_bin")
        .len()
        .sort("age_bin")
    )


def degree_distribution(run: Run, day: float | None = None) -> pl.DataFrame:
    """Out-degree histogram of the graph snapshot (every tie is stored in both directions)."""
    b = run.bonds(day)
    if not len(b):
        return pl.DataFrame()
    deg = b.group_by("source").len().rename({"len": "degree"})
    return deg.group_by("degree").len().rename({"len": "citizens"}).sort("degree")


def ideology_summary(run: Run, day: float | None = None) -> pl.DataFrame:
    p = run.panel(day).filter(pl.col("age") >= 18)
    if not len(p):
        return pl.DataFrame()
    rows = []
    for k in (1, 2, 3):
        x = p[f"I{k}"].to_numpy()
        rows.append(
            {
                "day": float(p["day"][0]),
                "axis": k,
                "mean": float(x.mean()),
                "sd": float(x.std()),
                "sarle": sarle(x),
            }
        )
    return pl.DataFrame(rows)


def scenario_identity(run: Run) -> dict:
    """Does the scenario survive the warm-up (spec 17.5)? Ratios of day 0 to the freshly generated population
    (first warm-up turn) for dispersion, polarization and the Sarle coefficient."""
    w = run.warmup(
        [
            "ideology_sd_1",
            "ideology_sd_2",
            "ideology_sd_3",
            "polarization_mean",
            "bimodality_axis1",
            "ideology_mean_1",
        ]
    ).df
    m = run.metrics(
        [
            "ideology_sd_1",
            "ideology_sd_2",
            "ideology_sd_3",
            "polarization_mean",
            "bimodality_axis1",
            "ideology_mean_1",
        ]
    ).df
    if not len(w) or not len(m):
        return {}
    a, b = w.row(0, named=True), m.row(0, named=True)
    r = lambda k: float(b[k] / a[k]) if a[k] else float("nan")  # noqa: E731
    return {
        "sd_retained": {k: r(k) for k in ("ideology_sd_1", "ideology_sd_2", "ideology_sd_3")},
        "polarization_retained": r("polarization_mean"),
        "sarle_start": float(a["bimodality_axis1"]),
        "sarle_day0": float(b["bimodality_axis1"]),
        "axis1_mean_shift": float(b["ideology_mean_1"] - a["ideology_mean_1"]),
    }


def vital_rates(run: Run) -> dict:
    """Annualised crude birth / death / migration rates over the game, from the cumulative counters."""
    m = run.metrics(["pop_size", "births_total", "deaths_total", "immigrants_total", "emigrants_total"]).df
    if len(m) < 2:
        return {}
    yrs = (m["day"][-1] - m["day"][0]) / 365.0 or 1.0
    n = float(m["pop_size"].mean())
    f = lambda c: float((m[c][-1] - m[c][0]) / yrs / n)  # noqa: E731
    return {
        "birth_rate": f("births_total"),
        "death_rate": f("deaths_total"),
        "immigration": f("immigrants_total"),
        "emigration": f("emigrants_total"),
        "net_growth": float((m["pop_size"][-1] - m["pop_size"][0]) / yrs / n),
    }


def stability_table(run: Run, metrics: list[str] | None = None, window: float = 180.0) -> pl.DataFrame:
    """Stability verdict for every continuous metric of the registry in one run."""
    specs = load_metrics()
    m = run.metrics().df
    rows = []
    for mid, s in specs.items():
        if mid not in m.columns or s.family == "outcome" or mid.endswith("_total"):
            continue
        if metrics and mid not in metrics:
            continue
        v = verdict(
            m["day"].to_numpy(),
            m[mid].to_numpy(),
            window,
            tol=s.target.value if s.target and s.target.kind == "drift_max" else 0.10,
            floor=s.scale_floor,
        )
        rows.append({"metric": mid, **v})
    return pl.DataFrame(rows)
