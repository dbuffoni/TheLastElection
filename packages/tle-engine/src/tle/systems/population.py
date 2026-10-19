"""Population generation (spec 1.5, 4): scenario distributions, labels through S2 then S1, classes,
trades, circles, affinity ties, council and custodians. Fully determined by (scenario, seed, params)."""

from __future__ import annotations

import numpy as np

from tle.contracts import models as m
from tle.contracts.vocab import BIPOLAR
from tle.core.rng import GameRng
from tle.engines.protocol import Engines
from tle.params import Params
from tle.world.catalogs import AGE_BANDS, Catalogs
from tle.world.state import B_AFF, N_PILLARS, World, allocate

AGE_RANGE = {"minor": (0, 17), "young": (18, 34), "adult": (35, 66), "elderly": (67, 90)}


def age_band(age: np.ndarray, a_adult: float = 18, a_pens: float = 67) -> np.ndarray:
    return np.select([age < a_adult, age < 35, age < a_pens], [0, 1, 2], 3).astype(np.int8)


def add_bonds(w: World, src, dst, btype: int, weight, stable=False, kin=0) -> None:
    """Append directed bonds (both directions are added by the caller when symmetric)."""
    src = np.atleast_1d(np.asarray(src, dtype=np.int32))
    k = len(src)
    if k == 0:
        return
    need = w.bn + k
    if need > len(w.bsrc):
        cap = max(need, 2 * len(w.bsrc))
        for name, dt in (("bsrc", np.int32), ("bdst", np.int32), ("btype", np.int8), ("bkin", np.int8),
                         ("bw", float), ("bstable", bool), ("blast", float)):
            old = getattr(w, name)
            new = np.zeros(cap, dtype=old.dtype)
            new[: len(old)] = old
            setattr(w, name, new)
    s = slice(w.bn, need)
    w.bsrc[s] = src
    w.bdst[s] = np.atleast_1d(dst)
    w.btype[s] = btype
    w.bkin[s] = kin
    w.bw[s] = weight
    w.bstable[s] = stable
    w.blast[s] = w.day
    w.bn = need


def create_world(cat: Catalogs, p: Params, engines: Engines, scenario_id: str, seed: int, n: int | None = None) -> World:
    sc = cat.scenario(scenario_id)
    n = int(n if n is not None else p.time.N)
    rng = GameRng(seed)
    w = World(seed=seed, scenario=scenario_id)
    allocate(w, n + max(20, n // 4))
    w.n_used = n
    w.alive[:n] = True

    # ages from the scenario shares
    shares = np.array([sc.age_shares.get(b, 0.0) for b in AGE_BANDS], dtype=float)
    cum = np.cumsum(shares / shares.sum())
    u = rng.random(n)
    band = np.searchsorted(cum, u, side="right").clip(0, 3)
    for b, name in enumerate(AGE_BANDS):
        lo, hi = AGE_RANGE[name]
        sel = band == b
        w.age[:n][sel] = rng.integers(lo, hi + 1, int(sel.sum()))

    # circles: small cohesive groups
    n_circ = max(1, n // 8)
    w.circle[:n] = rng.integers(0, n_circ, n)

    # labels: S2 describes, S1 classifies (the two-engine boundary)
    gen = engines.s2.call("generate_citizens", m.GenerateCitizensIn(
        scenario=scenario_id, number=n, ages=[int(a) for a in w.age[:n]], circles=[int(c) for c in w.circle[:n]],
        parents=[{} for _ in range(n)]), rng.child("gen"))
    prof_rng = rng.child("profile")
    for i in range(n):
        d = gen.citizens[i]
        w.names[i] = d.name
        out = engines.s1.call("citizen_profile", m.CitizenProfileIn(description=d.description, anchor="average citizen"), prof_rng)
        idx = [BIPOLAR.index(getattr(out, k)) for k in ("NS", "HA", "RD", "P", "SD", "CO", "ST", "I1", "I2", "I3")]
        w.T[i] = (np.array(idx[0:4]) - 3) / 3.0
        w.C[i] = (np.array(idx[4:7]) - 3) / 3.0
        w.I[i] = (np.array(idx[7:10]) - 3) / 3.0
    w.B[:] = w.I
    w.lab_I[:] = np.rint(w.I * 3 + 3).astype(np.int8)

    # wealth, class (quantiles of the scenario), trades
    r = np.clip(rng.normal(sc.wealth_mean, sc.wealth_sd, n), 0.0, 1.0)
    w.r[:n] = r
    w.r0[:n] = r
    w.r_prev[:n] = r
    order = np.argsort(r)
    cls = np.ones(n, np.int8)
    cls[order[: int(round(sc.q_low * n))]] = 0
    cls[order[n - int(round(sc.q_high * n)):]] = 2
    w.cls[:n] = cls
    tids = cat.trade_ids()
    weights = np.array([t.weight * sc.trade_multipliers.get(t.id, 1.0) for t in cat.trades], dtype=float)
    student, retiree = tids.index("student"), tids.index("retiree")
    wk = weights.copy()
    wk[[student, retiree]] = 0.0
    cumw = np.cumsum(wk / wk.sum())
    trade = np.searchsorted(cumw, rng.random(n), side="right").clip(0, len(tids) - 1)
    trade[w.age[:n] < p.demography.a_adult] = student
    trade[w.age[:n] >= p.demography.a_pens] = retiree
    w.trade[:n] = trade

    # opinions: indifferent about the mayor, wary of the erosion of each pillar
    w.O[:n, 0] = 0.0
    w.O[:n, 1:] = np.clip(-0.25 + 0.25 * rng.normal(size=(n, N_PILLARS)) - 0.1 * w.I[:n, [0]], -1, 1)
    w.Ob[:n, 1:] = w.O[:n, 1:]
    w.lab_O[:n] = np.rint(w.O[:n] * 3 + 3).astype(np.int8)
    w.fear[:n] = 0.0

    # affinity ties inside circles plus random acquaintances
    src, dst = [], []
    for c in range(n_circ):
        members = np.flatnonzero(w.circle[:n] == c)
        for a in members:
            for b in members:
                if a < b:
                    src += [a, b]
                    dst += [b, a]
    k_extra = int(sc.cohesion_degree * n / 2)
    a = rng.integers(0, n, k_extra)
    b = rng.integers(0, n, k_extra)
    keep = a != b
    src += list(np.concatenate([a[keep], b[keep]]))
    dst += list(np.concatenate([b[keep], a[keep]]))
    add_bonds(w, np.array(src), np.array(dst), B_AFF, sc.cohesion_intensity)

    # council: adults, half on the mayor's side
    adults = np.flatnonzero(w.adults(p.demography.a_adult))
    nc = min(p.pillars.n_council, len(adults))
    chosen = adults[rng.permutation(len(adults))[:nc]]
    w.councillor[chosen] = np.where(np.arange(nc) % 2 == 0, 1, -1)

    # initial state of the power block
    w.K = float(p.capital.c_0)
    w.decree_success = 0.5
    w.tec, w.cul = sc.tec, sc.cul
    w.tec_base, w.cul_base = sc.tec, sc.cul
    w.next_event_id = 1
    w.save_rng(rng)
    return w
