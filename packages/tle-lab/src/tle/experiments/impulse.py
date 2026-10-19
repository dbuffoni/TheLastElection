"""Impulse test (spec 17.3): a population in equilibrium receives one standard event and then nothing; the response
is the difference to a control run from the same state and the same RNG streams without the event.

Operational definition: one decree of aggressiveness 0.5 (S7 'medium') on 20% of the adults. Measured: peak shift of
opinion and of ideology in the involved and in the rest of the city, the time in which the involved shift halves,
and how many network steps the effect reaches. Targets: involved ~1 S7 step (1/3, +-25%), others ~1/9 (+-35%),
half-life ~60 days (+-35%)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from tle.core.rng import GameRng
from tle.loop.game import Game
from tle.loop.turn import PlayerInput, step
from tle.systems.reaction import run_event

TARGETS = {"involved_peak": (1 / 3, 0.25), "others_peak": (1 / 9, 0.35), "half_life_days": (60.0, 0.35)}


@dataclass
class ImpulseResult:
    days: list[float]
    involved: list[float]
    others: list[float]
    involved_ideology: list[float]
    others_ideology: list[float]
    involved_peak: float
    others_peak: float
    half_life_days: float
    hops_reached: int
    by_hop: dict[int, float] = field(default_factory=dict)

    def verdicts(self) -> dict[str, dict]:
        out = {}
        for k, (target, tol) in TARGETS.items():
            v = getattr(self, k)
            out[k] = {
                "observed": v,
                "target": target,
                "tolerance": tol,
                "ok": bool(abs(v - target) <= tol * target) if v == v else False,
            }
        return out


def _shift(a, b, ids) -> float:
    """Mean over ids of the largest absolute opinion component difference (bipolar units)."""
    if not len(ids):
        return 0.0
    return float(np.abs(a[ids] - b[ids]).max(axis=1).mean())


def _hops(w, sources: np.ndarray, limit: int = 6) -> np.ndarray:
    n = w.n_cap
    g = sp.coo_matrix((np.ones(w.bn), (w.bsrc[: w.bn], w.bdst[: w.bn])), shape=(n, n)).tocsr()
    d = dijkstra(g, unweighted=True, indices=sources, directed=False, min_only=True)
    d[~np.isfinite(d)] = limit + 1
    return d


def impulse_test(
    game: Game,
    share: float = 0.20,
    aggressiveness: str = "medium",
    days: float = 180.0,
    event_type: str = "decree_civil",
    progress=None,
) -> ImpulseResult:
    """Run the test from the current state of ``game`` (normally right after the warm-up); ``game`` is not modified."""
    data = game.save()
    treated, control = Game.load(data, game.cat, game.p), Game.load(data, game.cat, game.p)
    w = treated.w
    rng = GameRng(w.seed).child("impulse")
    ad = np.flatnonzero(w.adults(game.p.demography.a_adult))
    k = max(1, int(round(share * len(ad))))
    involved = np.sort(ad[rng.permutation(len(ad))[:k]])
    others = np.setdiff1d(ad, involved)
    run_event(
        w,
        game.p,
        treated.engines,
        rng,
        event_type=event_type,
        actor={"kind": "mayor", "name": "The mayor"},
        actor_kind="mayor",
        intensity=aggressiveness,
        involved=involved,
        hidden={
            "pillar_idx": game.cat.pillar_ids().index("civil"),
            "attacks": 1,
            "actor_sign": 1.0,
            "econ": 0.0,
        },
    )
    hop = _hops(w, involved)
    t_days, inv, oth, inv_i, oth_i = [], [], [], [], []
    by_hop_peak: dict[int, float] = {}
    t = 0.0
    while t < days:
        dt = min(game.p.time.dt_max, days - t)
        for g in (treated, control):
            step(g.w, g.cat, g.p, g.engines, PlayerInput(dt=dt), warmup=True)
        t += dt
        keep = control.w.alive[: control.w.n_cap] & treated.w.alive  # same persons
        iv, ot = involved[keep[involved]], others[keep[others]]
        t_days.append(t)
        inv.append(_shift(treated.w.O, control.w.O, iv))
        oth.append(_shift(treated.w.O, control.w.O, ot))
        inv_i.append(_shift(treated.w.I, control.w.I, iv))
        oth_i.append(_shift(treated.w.I, control.w.I, ot))
        for h in range(0, 7):
            ids = ad[(hop[ad] == h) & keep[ad]]
            if len(ids):
                by_hop_peak[h] = max(by_hop_peak.get(h, 0.0), _shift(treated.w.O, control.w.O, ids))
        if progress:
            progress(t, days)
    inv_a, oth_a = np.array(inv), np.array(oth)
    peak_i = int(inv_a.argmax())
    peak = float(inv_a[peak_i])
    half = float("nan")
    for j in range(peak_i, len(inv_a)):
        if inv_a[j] <= peak / 2:
            half = float(t_days[j] - t_days[peak_i])
            break
    noise_floor = 0.01
    reached = max([h for h, v in by_hop_peak.items() if h >= 1 and v > noise_floor], default=0)
    return ImpulseResult(t_days, inv, oth, inv_i, oth_i, peak, float(oth_a.max()), half, reached, by_hop_peak)
