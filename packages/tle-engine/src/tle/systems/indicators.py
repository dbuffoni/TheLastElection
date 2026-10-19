"""Statistics-office indicators (spec 1.4): aggregates of expressed opinion, with labels and hysteresis."""

from __future__ import annotations

import numpy as np

from tle.core.labels import UNIPOLAR_NAMES, quantize_unipolar
from tle.core.prims import express
from tle.params import Params
from tle.world.catalogs import Catalogs
from tle.world.state import World


def compute(w: World, cat: Catalogs, p: Params) -> dict[str, float]:
    ad = w.adults(p.demography.a_adult)
    if not ad.any():
        return {}
    E = express(w.O[ad], w.fear[ad])
    mayor = E[:, 0]
    out = {
        "approval": float((mayor + 1) / 2 @ np.ones(len(mayor)) / len(mayor)),
        "support": float(np.mean(mayor > 1 / 3)),
        "opposition": float(np.mean(mayor < -1 / 3)),
        "fear": float(w.fear[ad].mean()),
        "wealth": float(w.r[ad].mean()),
        "inequality": float(np.std(w.r[ad])),
        "polarization": float(np.mean(2 * np.abs(w.I[ad, 0]))),
        "capital": float(w.K),
        "cohesion": float(w.bw[: w.bn].mean()) if w.bn else 0.0,
    }
    for i, pid in enumerate(cat.pillar_ids()):
        out[f"resistance_{pid}"] = resistance(w, p, i)
        out[f"level_{pid}"] = float(w.pillar_level[i])
    return out


def resistance(w: World, p: Params, pillar: int, ad=None) -> float:
    """R_p: share of adults whose expressed acceptance of the erosion of the pillar is below -1/3."""
    ad = w.adults(p.demography.a_adult) if ad is None else ad
    if not ad.any():
        return 0.0
    E = express(w.O[ad, 1 + pillar], w.fear[ad])
    return float(np.mean(E < -1 / 3))


def labels(w: World, ind: dict[str, float], eps: float) -> dict[str, str]:
    out = {}
    for k, v in ind.items():
        if k.startswith("level_") or k in ("wealth",) and False:
            continue
        prev = w.ind_lab.get(k)
        idx = int(quantize_unipolar(np.clip(v, 0, 1), prev, eps))
        w.ind_lab[k] = idx
        out[k] = UNIPOLAR_NAMES[idx]
    return out
