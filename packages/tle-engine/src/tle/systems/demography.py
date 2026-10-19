"""Demography (spec 1.6): aging, mortality that doubles every 8 years, couples, births from fertile couples,
immigration and emigration. Slots of dead or departed citizens are reused, so ids stay valid for the
telemetry of one run but a slot can host a new person (``arrival_day`` tells them apart)."""

from __future__ import annotations

import numpy as np
from tle.contracts import models as m
from tle.contracts.vocab import BIPOLAR
from tle.core.prims import count, rate
from tle.core.rng import GameRng
from tle.engines.protocol import Engines
from tle.params import Params
from tle.systems.population import add_bonds, draw_trades
from tle.world.catalogs import Catalogs
from tle.world.state import B_FAM, KIN_PARENT, KIN_PARTNER, World

_PER_CITIZEN_DEFAULTS = {
    "lab_I": 3,
    "lab_O": 3,
    "parent1": -1,
    "parent2": -1,
    "couple": -1,
    "inf_last_action": -1e9,
}


def mortality(age: np.ndarray, mu_30: float) -> np.ndarray:
    """Annual mortality: ``mu_30`` at 30 years, doubling every 8 years."""
    return mu_30 * 2.0 ** ((age - 30.0) / 8.0)


def grow(w: World, new_cap: int) -> None:
    """Enlarge every per-citizen array (additive: existing slots are untouched)."""
    old = w.n_cap
    for name in w._array_fields():
        a = getattr(w, name)
        if (
            a.ndim >= 1
            and a.shape[0] == old
            and name
            not in (
                "fac_alive",
                "fac_id",
                "fac_mu",
                "fac_S",
                "fac_birth",
                "fac_last_action",
                "fac_prev_S",
                "p_class",
                "p_cust",
                "pillar_level",
                "pillar_fall_day",
                "threat_p",
                "threat_phase",
                "threat_next",
            )
        ):
            new = np.zeros((new_cap, *a.shape[1:]), dtype=a.dtype)
            new[:old] = a
            if name in _PER_CITIZEN_DEFAULTS:
                new[old:] = _PER_CITIZEN_DEFAULTS[name]
            setattr(w, name, new)
    w.names.extend([""] * (new_cap - old))
    w.n_cap = new_cap


def alloc_slot(w: World) -> int:
    free = np.flatnonzero(~w.alive[: w.n_used])
    if len(free):
        i = int(free[0])
    else:
        if w.n_used >= w.n_cap:
            grow(w, max(w.n_cap + 16, int(w.n_cap * 1.25)))
        i = w.n_used
        w.n_used += 1
    for name in w._array_fields():
        a = getattr(w, name)
        if (
            a.ndim >= 1
            and a.shape[0] == w.n_cap
            and name
            not in (
                "fac_alive",
                "fac_id",
                "fac_mu",
                "fac_S",
                "fac_birth",
                "fac_last_action",
                "fac_prev_S",
                "p_class",
                "p_cust",
                "pillar_level",
                "pillar_fall_day",
                "threat_p",
                "threat_phase",
                "threat_next",
            )
        ):
            a[i] = _PER_CITIZEN_DEFAULTS.get(name, 0)
    return i


def drop_bonds_of(w: World, gone: np.ndarray) -> None:
    """Remove every bond that touches a citizen who left (compacts the sparse bond table)."""
    if w.bn == 0 or not len(gone):
        return
    dead = np.zeros(w.n_cap, bool)
    dead[gone] = True
    keep = ~(dead[w.bsrc[: w.bn]] | dead[w.bdst[: w.bn]])
    k = int(keep.sum())
    for name in ("bsrc", "bdst", "btype", "bkin", "bw", "bstable", "blast"):
        a = getattr(w, name)
        a[:k] = a[: w.bn][keep]
    w.bn = k
    partner_gone = np.isin(w.couple, gone) & w.alive
    w.couple[partner_gone] = -1


def _new_citizen(
    w: World,
    p: Params,
    engines: Engines,
    rng: GameRng,
    age: float,
    parents: tuple[int, int] | None,
    cls: int,
    circle: int,
    trade: int,
    wealth: float,
) -> int:
    parent_ctx = {}
    if parents is not None:
        a, b = parents
        fmt = lambda arr, i: ",".join(str(int(round(v * 3))) for v in arr[i])  # noqa: E731
        parent_ctx = {k: f"{fmt(getattr(w, k), a)}|{fmt(getattr(w, k), b)}" for k in ("T", "C", "I")}
    gen = engines.s2.call(
        "generate_citizens",
        m.GenerateCitizensIn(
            scenario=w.scenario, number=1, ages=[int(age)], circles=[circle], parents=[parent_ctx]
        ),
        rng.child("gen"),
    )
    d = gen.citizens[0]
    out = engines.s1.call(
        "citizen_profile",
        m.CitizenProfileIn(description=d.description, anchor="average citizen"),
        rng.child("profile"),
    )
    idx = [
        BIPOLAR.index(getattr(out, k)) for k in ("NS", "HA", "RD", "P", "SD", "CO", "ST", "I1", "I2", "I3")
    ]
    i = alloc_slot(w)
    w.alive[i] = True
    w.age[i] = age
    w.T[i] = (np.array(idx[0:4]) - 3) / 3.0
    w.C[i] = (np.array(idx[4:7]) - 3) / 3.0
    w.I[i] = (np.array(idx[7:10]) - 3) / 3.0
    w.B[i] = w.I[i]
    w.lab_I[i] = np.rint(w.I[i] * 3 + 3)
    w.names[i] = d.name
    w.cls[i], w.circle[i], w.trade[i] = cls, circle, trade
    w.r[i] = w.r0[i] = w.r_prev[i] = wealth
    w.O[i, 1:] = np.clip(-0.25 - 0.1 * w.I[i, 0], -1, 1)
    w.Ob[i, 1:] = w.O[i, 1:]
    w.arrival_day[i] = w.day
    if parents is not None:
        w.parent1[i], w.parent2[i] = parents
        w.cls[i] = w.cls[parents[0]]
        w.circle[i] = w.circle[parents[0]]
        w.r[i] = w.r0[i] = w.r_prev[i] = w.r[parents[0]]
        for par in parents:
            add_bonds(w, [i, par], [par, i], B_FAM, p.ties.w_fam, stable=True, kin=KIN_PARENT)
    return i


def step(w: World, cat: Catalogs, p: Params, engines: Engines, dt: float, rng: GameRng) -> dict[str, int]:
    """One demographic turn. Returns counters (births, deaths, immigrants, emigrants, new couples)."""
    yrs = dt / 365.0
    stats = dict(births=0, deaths=0, immigrants=0, emigrants=0, couples=0)
    alive = w.alive.copy()
    alive[w.n_used :] = False
    w.age[alive] += yrs
    # deaths and emigration
    idx = np.flatnonzero(alive)
    r = rng.child("vital", w.turn)
    die = r.random(len(idx)) < rate(mortality(w.age[idx], p.demography.mu_30), yrs)
    adult = w.age[idx] >= p.demography.a_adult
    leave = (~die) & adult & (r.random(len(idx)) < rate(p.demography.nu_em, yrs))
    gone = idx[die | leave]
    stats["deaths"], stats["emigrants"] = int(die.sum()), int(leave.sum())
    w.alive[gone] = False
    drop_bonds_of(w, gone)
    _life_transitions(w, cat, p, rng.child("trades", w.turn))
    # couples: unpartnered adults of fertile or near-fertile age pair with someone close in age
    rc = rng.child("couples", w.turn)
    single = np.flatnonzero(
        w.alive
        & (w.couple < 0)
        & (w.age >= p.demography.a_adult)
        & (w.age <= p.demography.a_fert + p.demography.delta_a_couple)
    )
    for _ in range(count(p.demography.nu_couple * len(single) / 2.0, yrs, rc)):
        if len(single) < 2:
            break
        a = int(single[rc.integers(0, len(single))])
        cand = single[(single != a) & (np.abs(w.age[single] - w.age[a]) <= p.demography.delta_a_couple)]
        if not len(cand):
            continue
        b = int(cand[rc.integers(0, len(cand))])
        w.couple[a], w.couple[b] = b, a
        add_bonds(w, [a, b], [b, a], B_FAM, p.ties.w_fam, stable=True, kin=KIN_PARTNER)
        single = single[(single != a) & (single != b)]
        stats["couples"] += 1
    # births: fertile couples (the younger partner not past a_fert)
    rb = rng.child("births", w.turn)
    cpl = np.flatnonzero(w.alive & (w.couple >= 0) & (np.arange(w.n_cap) < w.couple))
    for a in cpl:
        b = int(w.couple[a])
        if min(w.age[a], w.age[b]) <= p.demography.a_fert and rb.random() < rate(p.demography.nu_child, yrs):
            _new_citizen(
                w, p, engines, rb.child(int(a)), 0.0, (int(a), b), 0, 0, cat.trade_ids().index("student"), 0.0
            )
            stats["births"] += 1
    # immigration: adults like the original population
    ri = rng.child("immigration", w.turn)
    n0 = p.time.N
    for k in range(count(p.demography.nu_im * n0 / 365.0, dt, ri)):
        ad = np.flatnonzero(w.adults(p.demography.a_adult))
        if not len(ad):
            break
        tpl = int(
            ad[ri.integers(0, len(ad))]
        )  # class, wealth and trade of a random resident: the city keeps its mix
        _new_citizen(
            w,
            p,
            engines,
            ri.child(k),
            float(ri.integers(20, 50)),
            None,
            int(w.cls[tpl]),
            int(ri.integers(0, max(1, int(w.circle[: w.n_used].max()) + 1))),
            int(w.trade[tpl]),
            float(w.r[tpl]),
        )
        stats["immigrants"] += 1
    for k, v in stats.items():
        w.stats[k] = w.stats.get(k, 0) + v
    return stats


def _life_transitions(w: World, cat: Catalogs, p: Params, rng: GameRng) -> None:
    """Students who come of age draw a trade; people past retirement age become retirees."""
    tids = cat.trade_ids()
    student, retiree = tids.index("student"), tids.index("retiree")
    grads = np.flatnonzero(w.alive & (w.trade == student) & (w.age >= p.demography.a_adult))
    if len(grads):
        w.trade[grads] = draw_trades(cat, cat.scenario(w.scenario), len(grads), rng)
    w.trade[w.alive & (w.age >= p.demography.a_pens)] = retiree
