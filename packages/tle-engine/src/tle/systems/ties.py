"""Daily life and ties (spec 5.1, 5.14): a share of the citizens each turn pick an activity (S1), meet others doing the
same, gain or create ties, drift ideologically toward the activity's target, and ties decay without contact, stabilize
with hysteresis and leave the network below ``theta_w_out``. Routine is an event like any other: pushes are computed on
the start-of-turn state and applied once."""

from __future__ import annotations

import numpy as np
from tle.core.prims import PushBuffer, gate, lam, rate
from tle.core.rng import GameRng
from tle.engines.protocol import Engines
from tle.params import Params
from tle.systems.population import add_bonds
from tle.systems.reaction import label_context
from tle.world.catalogs import Catalogs
from tle.world.state import B_AFF, B_FAM, B_PRO, World

BETA = ("beta_fam", "beta_aff", "beta_pro", "beta_riv", "beta_pol")  # indexed by bond type


def _bond_keys(w: World) -> np.ndarray:
    return w.bsrc[: w.bn].astype(np.int64) * w.n_cap + w.bdst[: w.bn]


def _has_tie(sorted_keys: np.ndarray, keys: np.ndarray) -> np.ndarray:
    if not len(sorted_keys):
        return np.zeros(len(keys))
    pos = np.minimum(np.searchsorted(sorted_keys, keys), len(sorted_keys) - 1)
    return (sorted_keys[pos] == keys).astype(float)


def daily_life(
    w: World, cat: Catalogs, p: Params, engines: Engines, dt: float, rng: GameRng
) -> dict[str, int]:
    """Sample citizens, choose activities, make contacts, update ties and ideology. Returns counters."""
    stats = {"participants": 0, "contacts": 0, "new_ties": 0}
    alive = np.flatnonzero(w.alive[: w.n_used])
    share = 1.0 - (1.0 - p.time.phi) ** (dt / p.time.dt_max)
    r = rng.child("daily", w.turn)
    ids = alive[r.random(len(alive)) < share]
    if len(ids) < 2:
        return stats
    ctx = label_context(w, p, ids)
    choice = engines.s1.choose_activity_many(ctx, r.child("choose"))
    acts = cat.activities
    stats["participants"] = len(ids)
    sorted_keys = np.sort(_bond_keys(w))
    src, dst, kinds = [], [], []
    push_idx, push_t, push_k = [], [], []
    for a, act in enumerate(acts):
        members = ids[choice == a]
        if len(members) < 1:
            continue
        # ideological drift toward the design target (masked axes stay untouched); the same step for everyone
        t_I = np.array(act.t_I if hasattr(act, "t_I") else [np.nan] * 3, dtype=float)
        if not np.isnan(t_I).all():
            push_idx.append(members)
            push_t.append(np.broadcast_to(t_I, (len(members), 3)))
            push_k.append(np.full(len(members), p.ties.routine_step * act.social_weight))
        if len(members) < 2:
            continue
        # who is met: anyone in town, acquaintances far likelier than strangers; a professional activity (work)
        # only brings together people of the same trade, a crowd-sized circle keeps strangers out (B_bond)
        k = p.ties.routine_partners
        for m_i in members:
            pool = alive[alive != m_i]
            if act.professional:
                pool = pool[w.trade[pool] == w.trade[m_i]]
            elif len(members) > p.ties.B_bond:
                pool = pool[w.circle[pool] == w.circle[m_i]]
            if not len(pool):
                continue
            has = _has_tie(sorted_keys, m_i * w.n_cap + pool)
            score = np.log(1.0 + p.ties.tie_preference * has) + r.gumbel(len(pool))
            pick = pool[np.argsort(-score)[:k]]
            src += [m_i] * len(pick)
            dst += list(pick)
            kinds += [B_PRO if act.professional else B_AFF] * len(pick)
    if push_idx:
        buf = PushBuffer(w.n_cap, 3)
        for i, t, k in zip(push_idx, push_t, push_k):
            buf.add(i, t, k)
        w.I = buf.apply(w.I)
    if not src:
        return stats
    a, b, kd = np.array(src, np.int64), np.array(dst, np.int64), np.array(kinds, np.int8)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo * w.n_cap + hi
    _, first = np.unique(key, return_index=True)
    lo, hi, kd = lo[first], hi[first], kd[first]
    stats["contacts"] = len(lo)
    # directed bonds in both directions
    fa, fb = np.concatenate([lo, hi]), np.concatenate([hi, lo])
    fk = np.concatenate([kd, kd])
    existing = _bond_keys(w)
    order = np.argsort(existing)
    sorted_keys = existing[order]
    want = fa * w.n_cap + fb
    pos = np.searchsorted(sorted_keys, want)
    pos = np.minimum(pos, max(len(sorted_keys) - 1, 0))
    hit = (sorted_keys[pos] == want) if len(sorted_keys) else np.zeros(len(want), bool)
    bi = order[pos[hit]] if len(sorted_keys) else np.array([], int)
    w.bw[bi] = np.minimum(1.0, w.bw[bi] + p.ties.g_w * (1.0 - w.bw[bi]))
    w.blast[bi] = w.day
    new = ~hit
    if new.any():
        for t in (B_AFF, B_PRO):
            sel = new & (fk == t)
            if sel.any():
                add_bonds(w, fa[sel], fb[sel], t, p.ties.w_0)
        stats["new_ties"] = int(new.sum() // 2)
    return stats


def tie_dynamics(w: World, p: Params, dt: float) -> int:
    """Decay without contact (after the grace period G_w), stability with hysteresis, exit below theta_w_out.
    Family ties never decay or leave. Returns the number of ties dropped."""
    n = w.bn
    if n == 0:
        return 0
    fam = w.btype[:n] == B_FAM
    idle = (w.day - w.blast[:n]) > p.ties.G_w
    decaying = idle & ~fam
    k = float(rate(lam(p.ties.h_w), dt))
    w.bw[:n] = np.where(decaying, w.bw[:n] * (1.0 - k), w.bw[:n])
    st, _ = gate(w.bstable[:n], w.bw[:n], p.ties.theta_stab_in, p.ties.theta_stab_out, np.zeros(n), w.day)
    w.bstable[:n] = st | fam
    drop = (
        idle & (~w.bstable[:n]) & (w.bw[:n] < p.ties.theta_w_out) & ~fam
    )  # only a tie that has gone quiet can leave
    if drop.any():
        keep = ~drop
        k2 = int(keep.sum())
        for name in ("bsrc", "bdst", "btype", "bkin", "bw", "bstable", "blast"):
            arr = getattr(w, name)
            arr[:k2] = arr[:n][keep]
        w.bn = k2
    return int(drop.sum())


def ideology_pull(w: World, p: Params, dt: float) -> None:
    """Continuous pull toward the ideology of one's ties (spec 5.5): weight beta(type) x intensity, half-life h_tr."""
    n = w.bn
    if n == 0:
        return
    beta = np.array([getattr(p.ties, b) for b in BETA])
    wt = beta[w.btype[:n]] * w.bw[:n]
    s, d = w.bsrc[:n], w.bdst[:n]
    num = np.zeros_like(w.I)
    den = np.zeros(w.n_cap)
    np.add.at(num, d, wt[:, None] * w.I[s])
    np.add.at(den, d, wt)
    target = np.where(den[:, None] > 1e-9, num / np.maximum(den[:, None], 1e-9), w.I)
    k = float(rate(lam(p.ties.h_tr), dt)) * (den / (den + 1.0))[:, None]
    w.I = np.clip(w.I + k * (target - w.I), -1.0, 1.0)
