"""Population statistics (spec 10, 17): pure functions of the world state, no side effects, no randomness.
Everything here is observation: the simulation never reads these values back (telemetry is an observer).
The ids returned by ``compute`` must match the metrics declared with ``source = "turn"`` in config/metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tle.core.labels import quantize_bipolar
from tle.core.prims import express, polarization
from tle.params import Params
from tle.world.state import BOND_TYPES, World

AXES = (1, 2, 3)


def sarle(x: np.ndarray) -> float:
    """Sarle's bimodality coefficient; above 5/9 = 0.555 suggests a bimodal distribution (uniform = 0.555)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4 or x.std() < 1e-12:
        return 0.0
    z = (x - x.mean()) / x.std()
    g = float(np.mean(z**3))
    k = float(np.mean(z**4) - 3.0)
    return (g * g + 1.0) / (k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


@dataclass
class PopRef:
    """Reference state at day 0 (after the warm-up): ideology of everyone who was there, to measure mobility."""

    I: np.ndarray
    alive: np.ndarray
    arrival: np.ndarray

    @classmethod
    def capture(cls, w: World) -> PopRef:
        return cls(w.I.copy(), w.alive.copy(), w.arrival_day.copy())


def compute(w: World, p: Params, ref: PopRef | None = None) -> dict[str, float]:
    ad = w.adults(p.demography.a_adult)
    al = w.alive.copy()
    al[w.n_used :] = False
    out: dict[str, float] = {}
    n = int(al.sum())
    nan = float("nan")
    out["pop_size"] = float(n)
    out["mean_age"] = float(w.age[al].mean()) if n else nan
    out["share_minors"] = float(np.mean(w.age[al] < p.demography.a_adult)) if n else nan
    out["share_elderly"] = float(np.mean(w.age[al] >= p.demography.a_pens)) if n else nan
    for k in ("births", "deaths", "immigrants", "emigrants", "couples"):
        out[f"{k}_total"] = float(w.stats.get(k, 0))
    na = int(ad.sum())
    if na == 0:
        return out | {k: nan for k in _ADULT_KEYS}
    I = w.I[ad]
    for j in AXES:
        out[f"ideology_mean_{j}"] = float(I[:, j - 1].mean())
        out[f"ideology_sd_{j}"] = float(I[:, j - 1].std())
    out["polarization_mean"] = float(polarization(I).mean())
    out["bimodality_axis1"] = sarle(I[:, 0])
    out["saturated_character_share"] = float(np.mean((np.abs(w.C[ad]) >= 0.999).any(axis=1)))
    E = express(w.O[ad], w.fear[ad])
    out["opinion_mean"] = float(w.O[ad, 0].mean())
    out["expressed_mean"] = float(E[:, 0].mean())
    out["opinion_gap"] = float(np.abs(w.O[ad] - E).mean())
    out["silenced_share"] = float(np.mean((w.O[ad, 0] < -1 / 3) & (E[:, 0] >= -1 / 3)))
    out["fear_mean"] = float(w.fear[ad].mean())
    out["wealth_mean"] = float(w.r[ad].mean())
    out["wealth_gini"] = gini(w.r[ad])
    for c, name in enumerate(("low", "middle", "high")):
        out[f"share_{name}"] = float(np.mean(w.cls[ad] == c))
    bn = w.bn
    out["degree_mean"] = float(bn / n) if n else nan
    out["stable_tie_share"] = float(w.bstable[:bn].mean()) if bn else 0.0
    out["tie_weight_mean"] = float(w.bw[:bn].mean()) if bn else 0.0
    for t, name in enumerate(BOND_TYPES):
        out[f"tie_share_{name}"] = float(np.mean(w.btype[:bn] == t)) if bn else 0.0
    if ref is not None:
        m = len(ref.alive)  # the table may have grown since day 0: ids below m keep their meaning
        same = ad[:m] & ref.alive & (w.arrival_day[:m] == ref.arrival)
        if same.any():
            now = quantize_bipolar(w.I[:m][same])
            then = quantize_bipolar(ref.I[same])
            out["share_label_changed"] = float(np.mean((now != then).any(axis=1)))
            out["median_displacement"] = float(np.median(np.abs(w.I[:m][same] - ref.I[same]).max(axis=1)))
        else:
            out["share_label_changed"] = out["median_displacement"] = nan
    else:
        out["share_label_changed"] = out["median_displacement"] = nan
    return out


_ADULT_KEYS = (
    [f"ideology_mean_{j}" for j in AXES]
    + [f"ideology_sd_{j}" for j in AXES]
    + [
        "polarization_mean",
        "bimodality_axis1",
        "saturated_character_share",
        "opinion_mean",
        "expressed_mean",
        "opinion_gap",
        "silenced_share",
        "fear_mean",
        "wealth_mean",
        "wealth_gini",
        "share_low",
        "share_middle",
        "share_high",
        "degree_mean",
        "stable_tie_share",
        "tie_weight_mean",
        "share_label_changed",
        "median_displacement",
    ]
    + [f"tie_share_{n}" for n in BOND_TYPES]
)


def check_invariants(w: World) -> list[str]:
    """Names of the invariants violated now (ranges, NaN, population bounds, bond integrity)."""
    bad = []
    u = slice(0, w.n_used)
    for name in ("T", "C", "I", "B", "O", "fear", "r"):
        a = getattr(w, name)[u][w.alive[u]]
        if not np.isfinite(a).all():
            bad.append(f"nan_{name}")
        lo, hi = (0.0, 1.0) if name in ("fear", "r") else (-1.0, 1.0)
        if a.size and (a.min() < lo - 1e-9 or a.max() > hi + 1e-9):
            bad.append(f"range_{name}")
    if not 0.0 <= w.K <= 1.0:
        bad.append("range_K")
    n = int(w.alive[u].sum())
    if n < 2:
        bad.append("population_extinct")
    if n > 3 * max(1, int(w.stats.get("n0", n))) and w.stats.get("n0"):
        bad.append("population_explosion")
    if w.bn:
        s, d = w.bsrc[: w.bn], w.bdst[: w.bn]
        if not (w.alive[s].all() and w.alive[d].all()):
            bad.append("bond_to_dead")
        if (s == d).any():
            bad.append("self_bond")
    return bad
