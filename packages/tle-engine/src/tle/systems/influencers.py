"""Influencers (spec 5.9): citizens whose potential (centrality, temperament P, wealth rank) is high enough become
opinion leaders with a capital, a topic (the axes where they are most extreme) and a catchment area; citizens have a
capacity of one per axis, so overlapping topics split their audience. They pull their audience toward their ideology."""

from __future__ import annotations

import numpy as np
from tle.core.prims import dist, lam, rate, relax
from tle.params import Params
from tle.world.state import World


def topic(I: np.ndarray) -> np.ndarray:
    """omega_k = |I_k| / sum |I_h| (equal weights if the ideology is null)."""
    a = np.abs(I)
    s = a.sum(axis=-1, keepdims=True)
    return np.where(s > 1e-9, a / np.maximum(s, 1e-9), np.full_like(a, 1 / 3))


def potential(w: World, p: Params, ad: np.ndarray) -> np.ndarray:
    """z = 0.5 c + 0.3 pi + 0.2 r, with c the weighted degree normalized by the population maximum."""
    deg = np.zeros(w.n_cap)
    if w.bn:
        np.add.at(deg, w.bdst[: w.bn], w.bw[: w.bn])
    c = deg / deg.max() if deg.max() > 0 else deg
    pi = (w.T[:, 3] + 1.0) / 2.0
    z = p.influencers.w_z_c * c + p.influencers.w_z_pi * pi + p.influencers.w_z_r * w.r
    return np.where(ad, z, -1.0)


def catchment(
    w: World, p: Params, ad: np.ndarray, I_a: np.ndarray, K: float
) -> tuple[np.ndarray, np.ndarray]:
    """(strength pi_ai, topic omega_a) of one influencer over every citizen: 0 outside the catchment area."""
    om = topic(I_a)
    rho = p.influencers.rho_0 + p.influencers.rho_1 * K
    d = dist(w.I, I_a[None, :], W=om)
    pi_ = np.where(ad & (d <= rho), 1.0 - d / rho, 0.0)
    return pi_, om


def effective_shares(w: World, p: Params, ad: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """e_ai = sum_k omega_ak pi_ai / max(1, L_ik) for every influencer in ``ids`` (rows) over all citizens (columns)."""
    if not len(ids):
        return np.zeros((0, w.n_cap))
    pis, oms = [], []
    for a in ids:
        pi_, om = catchment(w, p, ad, w.I[a], w.inf_K[a])
        pis.append(pi_)
        oms.append(om)
    pis, oms = np.array(pis), np.array(oms)  # (A, N), (A, 3)
    load = np.einsum("ak,ai->ik", oms, pis)  # (N, 3)
    return np.einsum("ak,ai,ik->ai", oms, pis, 1.0 / np.maximum(1.0, load))


def step(w: World, p: Params, dt: float) -> None:
    ad = w.adults(p.demography.a_adult)
    n_ad = max(1, int(ad.sum()))
    ids = np.flatnonzero(w.inf_on & w.alive)
    # capital: Pattern B with income proportional to reach, decay toward 0
    if len(ids):
        e = effective_shares(w, p, ad, ids)
        reach = np.minimum(1.0, e.sum(axis=1) / (p.influencers.reach_share * n_ad))
        income = p.influencers.g_inf * reach
        K = w.inf_K[ids]
        K = K + (1.0 - K) * rate(income, dt)
        w.inf_K[ids] = relax(K, 0.0, p.influencers.h_inf, dt, 0.0, 1.0)
        out = (w.inf_K[ids] < p.influencers.theta_K_out) & ((w.day - w.inf_since[ids]) >= p.influencers.D_inf)
        gone = ids[out | ~ad[ids]]
        w.inf_on[gone] = False
        w.inf_K[gone] = 0.0
    # entry: the best candidate per turn, if its potential and its expected share clear the thresholds
    z = potential(w, p, ad)
    z[w.inf_on] = -1.0
    best = int(np.argmax(z))
    if z[best] >= p.influencers.theta_z:
        pi_, om = catchment(w, p, ad, w.I[best], p.influencers.K_inf_0)
        cur = np.flatnonzero(w.inf_on & w.alive)
        load = (
            np.einsum(
                "ak,ai->ik",
                np.array([topic(w.I[a]) for a in cur]),
                np.array([catchment(w, p, ad, w.I[a], w.inf_K[a])[0] for a in cur]),
            )
            if len(cur)
            else np.zeros((w.n_cap, 3))
        )
        exp_share = (
            float((om[None, :] * pi_[:, None] / np.maximum(1.0, load + om[None, :] * pi_[:, None])).sum())
            / n_ad
        )
        if exp_share >= p.influencers.theta_share:
            w.inf_on[best] = True
            w.inf_K[best] = p.influencers.K_inf_0
            w.inf_since[best] = w.day
            w.stats["influencer_entries"] = w.stats.get("influencer_entries", 0) + 1


def pull(w: World, p: Params, dt: float) -> None:
    """The audience of each influencer is pulled toward their ideology on the axes of the topic (others masked)."""
    ids = np.flatnonzero(w.inf_on & w.alive)
    if not len(ids):
        return
    ad = w.adults(p.demography.a_adult)
    k0 = float(rate(lam(p.ties.h_tr), dt))
    e = effective_shares(w, p, ad, ids)
    new = w.I.copy()
    for row, a in zip(e, ids):
        om = topic(w.I[a])
        wt = p.influencers.beta_inf * row
        k = k0 * wt / (wt + 1.0)
        mask = (om / om.max())[None, :]
        new = new + k[:, None] * mask * (w.I[a][None, :] - w.I)
        new[a] = w.I[a]
    w.I = np.clip(new, -1.0, 1.0)


def opinion_of_mayor(w: World, a: int, E: np.ndarray) -> float:
    return float(E[a, 0])
