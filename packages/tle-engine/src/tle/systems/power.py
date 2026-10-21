"""Power block (spec 5): political capital, the five pillars with a level gate, and the two threats
(no-confidence motion and judicial) with three-phase chains and a council vote. Slice v0: custodians
are reduced to the council and the resistance of the population."""

from __future__ import annotations

import numpy as np
from tle.core.prims import express, rate, relax
from tle.core.rng import GameRng
from tle.params import Params
from tle.systems import custodians, indicators
from tle.world.catalogs import Catalogs
from tle.world.state import World

MOTION, JUDICIAL = 0, 1
PHASE_EVENTS = (
    ("motion_filed", "motion_debate", "motion_vote"),
    ("judicial_investigation", "judicial_hearing", "judicial_verdict"),
)


def theta_up(p: Params, level: int, n_levels: int) -> float:
    """Acceptance gate for climbing from ``level``: harder the further the erosion has gone (5.11)."""
    return p.pillars.theta_0 - p.pillars.theta_1 * level / n_levels


def capital_step(w: World, p: Params, dt: float, support: float) -> None:
    """Pattern B: capital grows toward 1 at the income g_0 scaled by the mayor's support."""
    support = float(
        np.clip(support + p.capital.w_F * w.stats.get("fac_F", 0.0), 0.0, 1.0)
    )  # faction balance (5.8)
    income = p.capital.g_0 * (0.25 + 0.75 * support)
    w.K = float(w.K + (1.0 - w.K) * rate(income, dt))


def try_pillar_step(w: World, cat: Catalogs, p: Params, pillar: int, intensity_idx: int) -> tuple[str, int]:
    """Gate a decree against a pillar after the population reacted. Returns (outcome, level)."""
    pil = cat.pillars[pillar]
    level = int(w.pillar_level[pillar])
    if level >= len(pil.levels):
        return "already_fallen", level
    nxt = pil.levels[level]
    if intensity_idx < nxt.min_intensity:
        return "too_mild", level
    R = indicators.resistance(w, p, pillar, cat=cat)
    if R < theta_up(p, level, len(pil.levels)):
        w.pillar_level[pillar] = level + 1
        if level + 1 == len(pil.levels):
            w.pillar_fall_day[pillar] = w.day
        return "advanced", level + 1
    return "resisted", level


def demotion_step(w: World, cat: Catalogs, p: Params, decreed_pillar: int | None) -> list[int]:
    """A pillar that was not touched this turn slides back when resistance stays above theta_down."""
    down = []
    for i in range(len(cat.pillars)):
        if i == decreed_pillar or w.pillar_level[i] == 0:
            continue
        if w.pillar_level[i] >= len(cat.pillars[i].levels) and w.pillar_fall_day[i] > 0:
            continue  # fallen pillars stay fallen
        if indicators.resistance(w, p, i, cat=cat) >= p.pillars.theta_down:
            w.pillar_level[i] -= 1
            down.append(i)
    return down


def threats_step(w: World, cat: Catalogs, p: Params, dt: float, rng: GameRng) -> list[dict]:
    """Active threats (5.12): the pressure of each threat grows with the expressed discontent of its custodians (the
    council for the motion, the magistrates for the judicial route) and the hostility of factions, scaled by the
    intactness of the linked pillar; a chain of three phases starts with frequency nu_T p^gamma; at the outcome the
    mayor is removed only if the pressure reached theta_outcome AND more than half of the custodians dissent."""
    out: list[dict] = []
    ad = w.adults(p.demography.a_adult)
    E = express(w.O, w.fear)
    hostility = w.stats.get("fac_H", 0.0)
    for t, name in ((MOTION, "motion"), (JUDICIAL, "judicial")):
        pi = custodians.threat_pillar(cat, name)
        cust = custodians.mask(w, cat, pi, ad) & w.alive
        dissent = float(np.mean(E[cust, 0] <= -1 / 3)) if cust.any() else 0.0
        phi = (
            1.0 - float(w.pillar_level[pi]) / cat.pillars[pi].L
        )  # an eroded pillar protects less (and weighs less)
        g = p.threats.g_T * phi * (p.threats.w_M * dissent + p.threats.w_H * hostility)
        pt = w.threat_p[t]
        pt = pt + (1.0 - pt) * float(rate(g, dt))
        w.threat_p[t] = float(relax(pt, 0.0, p.threats.h_pT, dt, 0.0, 1.0))
        if w.threat_phase[t] == 0:
            if cust.any() and rng.child("trigger", t).random() < float(
                rate(p.threats.nu_T * w.threat_p[t] ** p.threats.gamma_T, dt)
            ):
                w.threat_phase[t] = 1
                w.threat_next[t] = (
                    w.day
                    + p.threats.delay_min
                    + (p.threats.delay_max - p.threats.delay_min) * float(rng.random())
                )
                out.append({"threat": t, "event": PHASE_EVENTS[t][0], "phase": 1})
        elif w.day >= w.threat_next[t]:
            ph = int(w.threat_phase[t])
            if ph < 3:
                w.threat_phase[t] = ph + 1
                w.threat_next[t] = (
                    w.day
                    + p.threats.delay_min
                    + (p.threats.delay_max - p.threats.delay_min) * float(rng.random())
                )
                out.append({"threat": t, "event": PHASE_EVENTS[t][ph], "phase": ph + 1})
            else:
                removed = bool(cust.any() and w.threat_p[t] >= p.threats.theta_outcome and dissent > 0.5)
                out.append(
                    {
                        "threat": t,
                        "event": PHASE_EVENTS[t][2],
                        "phase": 3,
                        "vote_against": dissent,
                        "removed": removed,
                    }
                )
                w.threat_phase[t] = 0
                if removed:
                    w.outcome, w.outcome_day = "removed", w.day
                else:
                    w.threat_p[t] *= 0.5  # dismissed
                    w.K = max(0.0, w.K - 0.15)
    w.stats["threat_events"] = w.stats.get("threat_events", 0) + len(out)
    return out
