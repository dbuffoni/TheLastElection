"""Extractive economy (spec 5.18): transfers between predicates of payers and beneficiaries, persistent ones as rents with
an intensity and a life tied to the property pillar, the concentration of the flow, the deadweight loss that grows when
wealth goes to the few, the extractive coalition of those who draw wealth from power, and extractiveness X that blocks
creative destruction. Property intact means nobody can move wealth: every transfer is multiplied by ``level / L``."""

from __future__ import annotations

import numpy as np
from tle.contracts import models as m
from tle.contracts.vocab import uni_idx
from tle.core.prims import credit, gate, loss, rate, relax
from tle.params import Params
from tle.systems.predicate import match
from tle.world.catalogs import Catalogs
from tle.world.state import World


def property_open(w: World, cat: Catalogs) -> float:
    """1 - phi_property: how far the property pillar is eroded (0 intact, 1 fallen)."""
    i = cat.pillar_ids().index("property")
    return float(w.pillar_level[i]) / cat.pillars[i].L


def concentration(n_pay: int, n_ben: int) -> float:
    """c in [-1, 1]: positive from many to few (extractive), negative from few to many (redistributive)."""
    return (n_pay - n_ben) / max(1, n_pay + n_ben)


def _move(
    w: World, p: Params, payers: np.ndarray, bens: np.ndarray, k_pay: float, c: float
) -> tuple[float, np.ndarray]:
    """Levy ``k_pay`` of the wealth of the payers; the total minus the deadweight loss goes in equal shares to the
    beneficiaries. Returns (total levied, wealth change per citizen)."""
    before = w.r.copy()
    if not len(payers) or not len(bens):
        return 0.0, np.zeros(w.n_cap)
    # someone who is both payer and beneficiary receives only the balance: levy and credit are applied one after the other
    new_r = loss(w.r[payers], k_pay)
    levied = float((w.r[payers] - new_r).sum())
    w.r[payers] = new_r
    dead = p.economy.d_0 + p.economy.d_1 * max(0.0, c)
    share = levied * (1.0 - dead) / len(bens)
    w.r[bens], _surplus = credit(w.r[bens], share)
    delta = w.r - before
    w.dr_transfer += delta
    return levied, delta


def transfer(w: World, cat: Catalogs, p: Params, tr: m.TransferOut, creator: str = "mayor") -> dict | None:
    """A transfer from a decree or an event: one-off (applied now) or persistent (a rent that is applied every turn)."""
    payers = np.flatnonzero(match(w, cat, p, tr.payers) & w.adults(p.demography.a_adult))
    bens = np.flatnonzero(match(w, cat, p, tr.beneficiaries) & w.adults(p.demography.a_adult))
    a = uni_idx(tr.intensity) / 6.0
    openness = property_open(w, cat)
    if a <= 0 or openness <= 0 or not len(payers) or not len(bens):
        return None
    c = concentration(len(payers), len(bens))
    if tr.duration == "persistent":
        rent = {
            "payers": tr.payers.model_dump(),
            "beneficiaries": tr.beneficiaries.model_dump(),
            "I": a,
            "l0": openness,
            "creator": creator,
            "c": c,
            "day": w.day,
        }
        w.rents.append(rent)
        w.stats["rents_created"] = w.stats.get("rents_created", 0) + 1
        return rent
    levied, delta = _move(w, p, payers, bens, p.economy.eta_tr * a * openness, c)
    _credit_coalition(w, p, creator, c, delta)
    w.stats["transfers_one_off"] = w.stats.get("transfers_one_off", 0) + 1
    return {"one_off": True, "levied": levied, "c": c}


def revoke(w: World, cat: Catalogs, p: Params, pred: m.Predicate) -> int:
    """Revocation: zero the intensity of the rents whose beneficiaries satisfy the predicate."""
    hit = match(w, cat, p, pred)
    n = 0
    for r in w.rents:
        ben = match(w, cat, p, m.Predicate(**r["beneficiaries"]))
        if r["I"] > 0 and (ben & hit).any():
            r["I"] = 0.0
            n += 1
    return n


def _credit_coalition(w: World, p: Params, creator: str, c: float, delta: np.ndarray) -> None:
    """Credits from extractive rents (c > 0) created by the mayor feed the accumulator of the receivers."""
    if creator == "mayor" and c > 0:
        gain_ = np.maximum(delta, 0.0)
        w.y_rent = np.clip(w.y_rent + p.economy.kappa_y * gain_, 0.0, 1.0)
        w.y_last_active = np.where(gain_ > 0, w.day, w.y_last_active)


def step(w: World, cat: Catalogs, p: Params, dt: float) -> None:
    """Persistent rents act every turn; rents decay when the property pillar is demoted below its level at creation and
    close below ``theta_rent``; the coalition accumulator decays after its grace; extractiveness is smoothed."""
    w.dr_transfer = np.zeros(w.n_cap)
    openness = property_open(w, cat)
    gains = extractive_gain = 0.0
    wealth_before = w.r.copy()
    for r in w.rents:
        if r["I"] <= 0:
            continue
        if openness < r["l0"]:
            r["I"] = float(relax(r["I"], 0.0, p.economy.h_rent, dt, 0.0, 1.0))
        payers = np.flatnonzero(match(w, cat, p, m.Predicate(**r["payers"])) & w.adults(p.demography.a_adult))
        bens = np.flatnonzero(
            match(w, cat, p, m.Predicate(**r["beneficiaries"])) & w.adults(p.demography.a_adult)
        )
        c = concentration(len(payers), len(bens))
        r["c"] = c
        k_pay = float(rate(p.economy.nu_tr * r["I"], dt)) * openness
        _, delta = _move(w, p, payers, bens, k_pay, c)
        pos = float(np.maximum(delta, 0.0).sum())
        gains += pos
        if c > 0:
            extractive_gain += pos
        _credit_coalition(w, p, r["creator"], c, delta)
    w.rents = [r for r in w.rents if r["I"] >= p.economy.theta_rent]
    # coalition: accumulator decay after the grace period, entry/exit with hysteresis
    idle = (w.day - w.y_last_active) > p.economy.G_y
    w.y_rent = np.where(idle, relax(w.y_rent, 0.0, p.economy.h_y, dt, 0.0, 1.0), w.y_rent)
    on, _ = gate(w.coalition, w.y_rent, p.economy.theta_y_in, p.economy.theta_y_out, 0.0, w.day)
    w.coalition = on & w.alive
    # extractiveness: share of the wealth gains of the turn that comes from extractive rents
    total_gain = float(np.maximum(w.r - wealth_before, 0.0).sum()) + 1e-9
    w.extractiveness = float(
        relax(
            w.extractiveness,
            min(1.0, extractive_gain / total_gain) if gains > 0 else 0.0,
            p.labels.h_agg,
            dt,
            0.0,
            1.0,
        )
    )
