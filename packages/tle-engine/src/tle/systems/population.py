"""Population generation (spec 1.5, 4): scenario distributions, labels through S2 then S1, classes,
trades, circles, affinity ties, council and custodians. Fully determined by (scenario, seed, params)."""

from __future__ import annotations

import numpy as np
from tle.contracts import models as m
from tle.contracts.vocab import BIPOLAR
from tle.core.rng import GameRng
from tle.engines.protocol import Engines, call_many
from tle.params import Params
from tle.systems import custodians
from tle.world.catalogs import AGE_BANDS, Catalogs
from tle.world.state import B_AFF, B_FAM, KIN_PARENT, KIN_PARTNER, N_PILLARS, World, allocate

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
        for name, dt in (
            ("bsrc", np.int32),
            ("bdst", np.int32),
            ("btype", np.int8),
            ("bkin", np.int8),
            ("bw", float),
            ("bstable", bool),
            ("blast", float),
        ):
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


def draw_trades(cat: Catalogs, sc, n: int, rng: GameRng) -> np.ndarray:
    """Working trades drawn with the scenario weights (students and retirees are assigned by age)."""
    tids = cat.trade_ids()
    weights = np.array([t.weight * sc.trade_multipliers.get(t.id, 1.0) for t in cat.trades], dtype=float)
    weights[[tids.index("student"), tids.index("retiree")]] = 0.0
    cum = np.cumsum(weights / weights.sum())
    return np.searchsorted(cum, rng.random(n), side="right").clip(0, len(tids) - 1)


def create_world(
    cat: Catalogs, p: Params, engines: Engines, scenario_id: str, seed: int, n: int | None = None
) -> World:
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
    real_s2 = hasattr(engines.s2, "concurrency")  # a real model gets short lists and may still return fewer
    size = p.technical.B_gen if real_s2 else n
    spans = [(a, min(n, a + size)) for a in range(0, n, size)]
    parts = call_many(
        engines.s2,
        "generate_citizens",
        [
            m.GenerateCitizensIn(
                scenario=scenario_id,
                number=b - a,
                ages=[int(x) for x in w.age[a:b]],
                circles=[int(c) for c in w.circle[a:b]],
                parents=[{} for _ in range(b - a)],
            )
            for a, b in spans
        ],
        [rng.child("gen") if not real_s2 else rng.child("gen", a) for a, b in spans],
    )
    drafts = []
    for (a, b), part in zip(spans, parts):
        got = list(part.citizens)[: b - a] if part is not None else []
        drafts += got + [
            m.CitizenDraft(name=f"Citizen {a + len(got) + j}", description="T=0,0,0,0;C=0,0,0;I=0,0,0")
            for j in range(b - a - len(got))
        ]  # missing ones are neutral (13.3 fallback)
    gen = m.GenerateCitizensOut(citizens=drafts)
    prof_rng = rng.child("profile")
    par = (
        int(getattr(engines.s1, "concurrency", 1)) > 1
    )  # parallel calls need an independent stream per citizen
    outs = call_many(
        engines.s1,
        "citizen_profile",
        [
            m.CitizenProfileIn(description=gen.citizens[i].description, anchor="average citizen")
            for i in range(n)
        ],
        [prof_rng.child(i) if par else prof_rng for i in range(n)],
    )
    for i, out in enumerate(outs):
        w.names[i] = gen.citizens[i].name
        idx = [
            BIPOLAR.index(getattr(out, k))
            for k in ("NS", "HA", "RD", "P", "SD", "CO", "ST", "I1", "I2", "I3")
        ]
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
    cls[order[n - int(round(sc.q_high * n)) :]] = 2
    w.cls[:n] = cls
    tids = cat.trade_ids()
    student, retiree = tids.index("student"), tids.index("retiree")
    trade = draw_trades(cat, sc, n, rng)
    trade[w.age[:n] < p.demography.a_adult] = student
    trade[w.age[:n] >= p.demography.a_pens] = retiree
    w.trade[:n] = trade

    # opinions: indifferent about the mayor, wary of the erosion of each pillar
    w.O[:n, 0] = 0.0
    w.O[:n, 1:] = np.clip(-0.25 + 0.25 * rng.normal(size=(n, N_PILLARS)) - 0.1 * w.I[:n, [0]], -1, 1)
    w.Ob[:n, 1:] = w.O[:n, 1:]
    w.lab_O[:n] = np.rint(w.O[:n] * 3 + 3).astype(np.int8)
    w.fear[:n] = 0.0

    # families: a share of adults in fertile range form couples; minors get two parents from the couples
    fam = rng.child("families")
    single = [
        int(i)
        for i in np.flatnonzero(
            (w.age[:n] >= p.demography.a_adult)
            & (w.age[:n] <= p.demography.a_fert + p.demography.delta_a_couple)
        )
    ]
    order = [single[j] for j in fam.permutation(len(single))]
    couples = []
    for a in order:
        if w.couple[a] >= 0 or fam.random() > p.demography.share_coupled_init:
            continue
        cand = [
            b
            for b in order
            if b != a and w.couple[b] < 0 and abs(w.age[b] - w.age[a]) <= p.demography.delta_a_couple
        ]
        if not cand:
            continue
        b = cand[int(fam.integers(0, len(cand)))]
        w.couple[a], w.couple[b] = b, a
        couples.append((a, b))
    minors = np.flatnonzero(w.age[:n] < p.demography.a_adult)
    if couples:
        for i in minors:
            a, b = couples[int(fam.integers(0, len(couples)))]
            w.parent1[i], w.parent2[i] = a, b
            w.circle[i], w.cls[i] = w.circle[a], w.cls[a]
    fs, fd = [], []
    for a, b in couples:
        fs += [a, b]
        fd += [b, a]
    add_bonds(
        w, np.array(fs, dtype=int), np.array(fd, dtype=int), B_FAM, p.ties.w_fam, stable=True, kin=KIN_PARTNER
    )
    ks, kd = [], []
    for i in minors:
        for par in (w.parent1[i], w.parent2[i]):
            if par >= 0:
                ks += [i, par]
                kd += [par, i]
    add_bonds(
        w, np.array(ks, dtype=int), np.array(kd, dtype=int), B_FAM, p.ties.w_fam, stable=True, kin=KIN_PARENT
    )

    # affinity ties inside circles plus random acquaintances
    # affinity ties: cohesion_degree per citizen on average, a share of them inside the circle
    k_pairs = int(round(sc.cohesion_degree * n / 2))
    a = rng.integers(0, n, k_pairs)
    inside = rng.random(k_pairs) < p.ties.circle_homophily
    b = rng.integers(0, n, k_pairs)
    for j in np.flatnonzero(inside):  # re-draw the partner among the same circle (when it has company)
        mates = np.flatnonzero(w.circle[:n] == w.circle[a[j]])
        mates = mates[mates != a[j]]
        if len(mates):
            b[j] = mates[rng.integers(0, len(mates))]
    keep = a != b
    lo, hi = np.minimum(a[keep], b[keep]), np.maximum(a[keep], b[keep])
    _, first = np.unique(lo * n + hi, return_index=True)
    lo, hi = lo[first], hi[first]
    src = list(np.concatenate([lo, hi]))
    dst = list(np.concatenate([hi, lo]))
    add_bonds(w, np.array(src), np.array(dst), B_AFF, sc.cohesion_intensity)

    # council: adults, half on the mayor's side
    adults = np.flatnonzero(w.adults(p.demography.a_adult))
    custodians.ensure_minimum(w, cat, p, adults, rng.child("cust"))
    custodians.seat_council(w, p, adults, rng.child("council"))

    # initial state of the power block
    w.K = float(p.capital.c_0)
    w.decree_success = 0.5
    w.tec, w.cul = sc.tec, sc.cul
    w.tec_base, w.cul_base = sc.tec, sc.cul
    w.next_event_id = 1
    w.save_rng(rng)
    return w
