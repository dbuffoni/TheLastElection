"""City council and custodians (spec 5.17): each pillar has citizens who guard it by role (councillors, magistrates,
journalists, entrepreneurs and merchants, law enforcement). They weigh ``omega_cust`` in the resistance of their pillar
and are the parties who decide the threats; capturing them (a decree on their trade or on the council) is the
referee-capture route of Levitsky and Ziblatt, which arises without any dedicated rule."""

from __future__ import annotations

import numpy as np
from tle.core.rng import GameRng
from tle.params import Params
from tle.world.catalogs import Catalogs
from tle.world.state import World


def mask(w: World, cat: Catalogs, pillar: int, ad: np.ndarray) -> np.ndarray:
    """Adult custodians of pillar index ``pillar``."""
    roles = cat.pillars[pillar].custodians
    trades = [cat.trade_ids().index(t) for t in roles if t != "councillor"]
    m = np.isin(w.trade, trades) if trades else np.zeros(w.n_cap, bool)
    if "councillor" in roles:
        m = m | (w.councillor != 0)
    return m & ad


def weights(w: World, cat: Catalogs, p: Params, pillar: int, ad: np.ndarray) -> np.ndarray:
    return np.where(mask(w, cat, pillar, ad), p.pillars.omega_cust, 1.0)


def threat_pillar(cat: Catalogs, threat: str) -> int:
    """Index of the pillar linked to the threat ("motion" or "judicial")."""
    return next(i for i, pl in enumerate(cat.pillars) if pl.threat == threat)


def seat_council(w: World, p: Params, adults: np.ndarray, rng: GameRng) -> None:
    """Council drawn among the adults, stratified on the first ideological axis so that it mirrors the city; the
    majority (a share ``q_maj`` of the seats) sits on one side of the axis, drawn at random, and starts with o^M = +1/3
    while the opposition starts with -1/3 (that is also the alignment base b^M)."""
    nc = min(p.pillars.n_council, len(adults))
    if nc == 0:
        return
    order = adults[np.argsort(w.I[adults, 0], kind="stable")]
    pick = order[np.unique(np.round(np.linspace(0, len(order) - 1, nc)).astype(int))]
    side_hi = bool(rng.random() < 0.5)
    seats = pick[np.argsort(w.I[pick, 0], kind="stable")]
    seats = seats[::-1] if side_hi else seats
    n_maj = int(np.ceil(p.pillars.q_maj * len(seats)))
    align = np.where(np.arange(len(seats)) < n_maj, 1, -1).astype(np.int8)
    w.councillor[seats] = align
    w.O[seats, 0] = align / 3.0
    w.Ob[seats, 0] = align / 3.0


def ensure_minimum(w: World, cat: Catalogs, p: Params, adults: np.ndarray, rng: GameRng) -> None:
    """Each scenario guarantees ``n_cust`` citizens per custodian trade (councillors keep their own trade)."""
    trades = {cat.trade_ids().index(t) for pl in cat.pillars for t in pl.custodians if t != "councillor"}
    free = [a for a in adults if w.councillor[a] == 0]
    for t in sorted(trades):
        have = int(np.sum(w.trade[adults] == t))
        need = p.pillars.n_cust - have
        if need > 0:
            donors = [a for a in free if w.trade[a] not in trades]
            for j in rng.permutation(len(donors))[:need]:
                w.trade[donors[j]] = t


def follow_alignment(w: World, p: Params, dt: float) -> None:
    """Party discipline: the alignment base of a councillor follows the opinion slowly (half-life ``h_b``)."""
    from tle.core.prims import relax

    c = w.councillor != 0
    if c.any():
        w.Ob[c, 0] = relax(w.Ob[c, 0], w.O[c, 0], p.vectors.h_b, dt)
