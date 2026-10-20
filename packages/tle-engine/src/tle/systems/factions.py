"""Factions (spec 5.7): born from an event tagged ``faction_birth``, recruit through the events they take part in,
hold members by commitment with hysteresis, drift slowly with their members and die when weak or small. They act on
the game only through political capital (the faction balance F) and the threats (the hostility H)."""

from __future__ import annotations

import numpy as np
from tle.contracts import models as m
from tle.core.prims import centroid, dist, gain, lam, loss, rate, relax
from tle.core.rng import GameRng
from tle.params import Params
from tle.world.state import F_CAP, World


def alive_slots(w: World) -> np.ndarray:
    return np.flatnonzero(w.fac_alive)


def kill(w: World, s: int) -> None:
    """Death: the pull is zeroed; members' vectors return through decay, without jumps."""
    w.fac_alive[s] = False
    w.fac_S[s] = 0.0
    w.fac_m[:, s] = 0.0
    w.fac_member[:, s] = False
    w.fac_last_active[:, s] = 0.0
    w.stats["faction_deaths"] = w.stats.get("faction_deaths", 0) + 1


def found(
    w: World,
    p: Params,
    engines,
    rng: GameRng,
    *,
    news,
    option: int,
    involved: np.ndarray,
    choice: np.ndarray,
    F,
) -> int | None:
    """Birth at the seed vector ``t`` of the adherence option: founders are the involved who chose it and are within
    ``rho_in`` of ``t``; fewer than ``N_min`` and no faction is born."""
    free = np.flatnonzero(~w.fac_alive)
    if not len(free):
        return None
    cand = involved[choice == option]
    if not len(cand):
        return None
    t = np.where(np.isnan(F.t_I[option]), w.I[cand].mean(axis=0), F.t_I[option])
    cand = cand[dist(w.I[cand], t) <= p.factions.rho_in]
    if len(cand) < p.factions.N_min:
        return None
    s = int(free[0])
    named = engines.s2.call(
        "name_faction",
        m.NameFactionIn(seed_labels={"direction": ",".join(f"{v:+.1f}" for v in t)}, news=news),
        rng.child("name", int(w.next_fac_id)),
    )
    w.fac_alive[s] = True
    w.fac_id[s] = w.next_fac_id
    w.next_fac_id += 1
    w.fac_mu[s] = t
    w.fac_S[s] = p.factions.S_0
    w.fac_prev_S[s] = p.factions.S_0
    w.fac_birth[s] = w.day
    w.fac_last_action[s] = w.day
    w.fac_name[s] = named.name if named is not None else f"Movement {w.fac_id[s]}"
    w.fac_m[:, s] = 0.0
    w.fac_member[:, s] = False
    w.fac_m[cand, s] = p.factions.theta_m_in
    w.fac_member[cand, s] = True
    w.fac_last_active[cand, s] = w.day
    w.stats["faction_births"] = w.stats.get("faction_births", 0) + 1
    return s


def on_event(w: World, p: Params, slot: int, ids: np.ndarray, choice: np.ndarray, F) -> None:
    """An event of the faction (as actor): commitment gain through the stance toward it (direct path), and a hit on
    its strength from the mean stance of those involved."""
    if not len(ids):
        return
    v = np.nan_to_num(F.stance_actor[choice], nan=0.0)
    d = dist(w.I[ids], w.fac_mu[slot])
    g = p.factions.g_m * np.maximum(0.0, v) * np.maximum(0.0, 1.0 - d / p.factions.rho_out)
    w.fac_m[ids, slot] = gain(w.fac_m[ids, slot], g)
    w.fac_last_active[ids, slot] = w.day
    vbar = float(v.mean())
    w.fac_S[slot] = float(
        gain(w.fac_S[slot], p.factions.g_S * vbar)
        if vbar >= 0
        else loss(w.fac_S[slot], -p.factions.g_S * vbar)
    )
    w.fac_last_action[slot] = w.day


def step(w: World, p: Params, dt: float) -> None:
    """Commitment decay after the grace period, membership with hysteresis, centroid inertia, strength decay, death,
    and the conflict rule (whoever belongs to two conflicting factions stays with the one of greater commitment)."""
    n = w.n_used
    alive = w.alive.copy()
    for s in alive_slots(w):
        m_ = w.fac_m[:, s]
        idle = (w.day - w.fac_last_active[:, s]) > p.factions.G_m
        m_ = np.where(idle, relax(m_, 0.0, p.factions.h_m, dt, 0.0, 1.0), m_)
        w.fac_m[:, s] = np.where(alive, m_, 0.0)
        mem = w.fac_member[:, s]
        w.fac_member[:, s] = alive & ((mem & (m_ >= p.factions.theta_m_out)) | (m_ >= p.factions.theta_m_in))
        members = np.flatnonzero(w.fac_member[:, s] & (np.arange(w.n_cap) < n))
        if len(members):
            target = centroid(w.I[members], w.fac_m[members, s])
            w.fac_mu[s] = relax(w.fac_mu[s], target, p.factions.h_c, dt)
        w.fac_prev_S[s] = w.fac_S[s]
        w.fac_S[s] = float(relax(w.fac_S[s], 0.0, p.factions.h_S, dt, 0.0, 1.0))
        if w.fac_S[s] < p.factions.theta_death or len(members) < p.factions.N_min:
            kill(w, int(s))
    sl = alive_slots(w)
    for a in sl:
        for b in sl:
            if (
                a < b
                and w.fac_alive[a]
                and w.fac_alive[b]
                and dist(w.fac_mu[a], w.fac_mu[b]) > p.factions.rho_conf
            ):
                both = w.fac_member[:, a] & w.fac_member[:, b]
                if both.any():
                    keep_a = w.fac_m[:, a] >= w.fac_m[:, b]
                    w.fac_member[both & keep_a, b] = False
                    w.fac_member[both & ~keep_a, a] = False


def pull(w: World, p: Params, dt: float) -> None:
    """Members are pulled toward the centroid as political nodes of the extended graph (5.5)."""
    k0 = float(rate(lam(p.ties.h_tr), dt))
    for s in alive_slots(w):
        wt = p.factions.beta_fac * np.where(w.fac_member[:, s], w.fac_m[:, s], 0.0)
        k = k0 * wt / (wt + 1.0)
        w.I = np.clip(w.I + k[:, None] * (w.fac_mu[s][None, :] - w.I), -1.0, 1.0)


def member_opinion(w: World, p: Params, s: int, E: np.ndarray) -> float:
    """Mean expressed opinion of the mayor among the adult members of faction ``s``."""
    mem = w.fac_member[:, s] & w.adults(p.demography.a_adult)
    return float(E[mem, 0].mean()) if mem.any() else 0.0


def balance(w: World, p: Params, E: np.ndarray) -> tuple[float, float]:
    """(F, H): signed faction balance on political capital and hostility feeding the threats. A faction counts with its
    strength and its share of the adults; its sign is the members' opinion of the mayor."""
    ad = w.adults(p.demography.a_adult)
    na = max(1, int(ad.sum()))
    f = h = 0.0
    for s in alive_slots(w):
        share = float((w.fac_member[:, s] & ad).sum()) / na
        v = member_opinion(w, p, int(s), E)
        f += w.fac_S[s] * share * v
        h += w.fac_S[s] * share * max(0.0, -v)
    return float(np.clip(f * 3.0, -1.0, 1.0)), float(np.clip(h * 3.0, 0.0, 1.0))


assert F_CAP >= 1
