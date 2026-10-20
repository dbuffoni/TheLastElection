"""Events started by factions, influencers and nascent movements (spec 6): each acts with a frequency proportional to its
strength or capital, picks among the event types its situation admits, and the pipeline runs like for any other actor."""

from __future__ import annotations

import numpy as np
from tle.contracts.vocab import uni_name
from tle.core.labels import quantize_bipolar
from tle.core.prims import express
from tle.core.rng import GameRng
from tle.params import Params
from tle.systems import factions
from tle.systems.reaction import EventResult, run_event
from tle.world.catalogs import Catalogs
from tle.world.state import World


def _levels_ok(w: World, cat: Catalogs, pillar: str) -> bool:
    i = cat.pillar_ids().index(pillar)
    return int(w.pillar_level[i]) < cat.pillars[i].L


def admitted(w: World, cat: Catalogs, kind: str, tokens: set[str], opinion: float = 0.0) -> list:
    """Event types of the actor kind whose preconditions all hold (unknown preconditions never hold) and whose tone fits
    the actor: a hostile act needs an actor who does not approve of the mayor, a friendly one an actor who does not oppose."""
    out = []
    for et in cat.event_types:
        if kind not in et.actors:
            continue
        if (et.hostility > 0.25 and opinion > 0.0) or (et.hostility < -0.25 and opinion < 0.0):
            continue
        ok = True
        for r in et.requires:
            if r in tokens:
                continue
            if r.startswith("pillar:") and r.split(":", 1)[1] in cat.pillar_ids():
                continue
            ok = False
            break
        if ok:
            out.append(et)
    return out


def _world_tokens(w: World, cat: Catalogs) -> set[str]:
    t = set()
    if _levels_ok(w, cat, "law"):
        t.add("law_below_fall")
    if _levels_ok(w, cat, "press"):
        t.add("press_free")
    return t


def _pick(rng: GameRng, types: list):
    return types[rng.choice_index([e.weight for e in types])] if types else None


def _fire(
    w, cat, p, engines, rng, et, actor, kind, strength, ideology, ad, slot=None, found=False
) -> EventResult | None:
    pillar_idx = None
    if et.target == "pillar" and et.pillar and et.pillar in cat.pillar_ids():
        pillar_idx = cat.pillar_ids().index(et.pillar)
    sign = -1.0 if et.hostility > 0 else 1.0
    return run_event(
        w,
        p,
        engines,
        rng,
        event_type=et.id,
        actor=actor,
        actor_kind=kind,
        intensity=uni_name(int(np.clip(round(1 + 4 * strength), 1, 5))),
        involved=np.flatnonzero(ad),
        hidden={
            "actor_sign": sign,
            "pillar_idx": pillar_idx,
            "attacks": et.attacks,
            "econ": et.econ,
            "actor_dir": ideology,
        },
        can_found=found,
        actor_ideology=quantize_bipolar(ideology) if ideology is not None else None,
        faction_slot=slot,
    )


def generate(w: World, cat: Catalogs, p: Params, engines, rng: GameRng, dt: float) -> list[EventResult]:
    """Faction, influencer and nascent-movement events of one turn."""
    out: list[EventResult] = []
    ad = w.adults(p.demography.a_adult)
    if not ad.any():
        return out
    E = express(w.O, w.fear)
    wt = _world_tokens(w, cat)
    # factions
    for s in factions.alive_slots(w):
        r = rng.child("fac", int(w.fac_id[s]), w.turn)
        if r.random() >= 1.0 - np.exp(-p.factions.nu_fac * w.fac_S[s] * dt):
            continue
        v = factions.member_opinion(w, p, int(s), E)
        tokens = set(wt)
        S = w.fac_S[s]
        if S >= 0.6:
            tokens.add("strong")
        if S < 0.4 and S <= w.fac_prev_S[s]:
            tokens.add("weak_declining")
        if v < -1 / 3:
            tokens.add("hostile")
        if v > 1 / 3:
            tokens.add("friendly")
        mem = np.flatnonzero(w.fac_member[:, s])
        if len(mem) > 1 and float(w.I[mem].std(axis=0).mean()) > 0.35:
            tokens.add("high_variance")
        et = _pick(r, admitted(w, cat, "faction", tokens, v))
        if et is None:
            continue
        res = _fire(
            w,
            cat,
            p,
            engines,
            r,
            et,
            {"kind": "faction", "name": w.fac_name[s]},
            "faction",
            S,
            w.fac_mu[s],
            ad,
            int(s),
            et.can_found,
        )
        if res:
            out.append(res)
    # influencers
    for a in np.flatnonzero(w.inf_on & w.alive):
        r = rng.child("inf", int(a), w.turn)
        if r.random() >= 1.0 - np.exp(-p.influencers.nu_inf * w.inf_K[a] * dt):
            continue
        tokens = set(wt)
        o = float(E[a, 0])
        if o < -1 / 3:
            tokens.add("hostile")
        if o > 1 / 3:
            tokens.add("friendly")
        et = _pick(r, admitted(w, cat, "influencer", tokens, o))
        if et is None:
            continue
        res = _fire(
            w,
            cat,
            p,
            engines,
            r,
            et,
            {"kind": "influencer", "name": w.names[a] or f"Citizen {a}"},
            "influencer",
            float(w.inf_K[a]),
            w.I[a],
            ad,
        )
        if res:
            out.append(res)
    # nascent movements: discontent makes a demonstration that can found a faction
    opp = float(np.mean(E[ad, 0] < -1 / 3))
    if opp > p.factions.theta_found and (~w.fac_alive).any():
        r = rng.child("nascent", w.turn)
        if r.random() < 1.0 - np.exp(
            -p.factions.nu_found * (opp - p.factions.theta_found) / (1.0 - p.factions.theta_found) * dt
        ):
            et = cat.event_type("demonstration")
            seed_i = np.flatnonzero(ad & (E[:, 0] < -1 / 3))
            ideology = w.I[seed_i].mean(axis=0) if len(seed_i) else np.zeros(3)
            res = _fire(
                w,
                cat,
                p,
                engines,
                r,
                et,
                {"kind": "faction", "name": "A nascent movement"},
                "faction",
                0.5,
                ideology,
                ad,
                None,
                True,
            )
            if res:
                out.append(res)
    return out
