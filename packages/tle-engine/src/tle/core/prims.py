"""The 14 primitives of the mathematical core (spec 3.3) plus a vectorized push buffer.

Everything here is pure: arguments in, values out. Randomness comes only from an explicit ``GameRng``.
"""

from __future__ import annotations

import math

import numpy as np

from .rng import GameRng

INF = math.inf


# P3 / helpers -------------------------------------------------------------------------------
def rate(rho, dt):
    """P3: daily rate -> step over ``dt`` days, exact for any turn length."""
    return 1.0 - np.exp(-np.asarray(rho, dtype=float) * dt)


def lam(h):
    """Half-life (days) -> decay rate. Infinite half-life means no decay."""
    h = np.asarray(h, dtype=float)
    return np.where(np.isinf(h), 0.0, np.log(2.0) / np.where(np.isinf(h), 1.0, h))


def lam_s(h: float) -> float:
    return 0.0 if math.isinf(h) else math.log(2.0) / h


# P1 -----------------------------------------------------------------------------------------
def dist(x, y, W=None):
    """Weighted Euclidean distance normalized to [0, 1] (spec 3.2). ``x``/``y`` may be batched on the
    first axes; the last axis is the vector dimension."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x.shape[-1] if x.ndim else 1
    Wv = np.ones(d) if W is None else np.asarray(W, dtype=float)
    num = np.sqrt(np.sum(Wv * (x - y) ** 2, axis=-1))
    den = 2.0 * np.sqrt(np.sum(Wv))
    return num / den if den > 0 else np.zeros_like(num)


def polarization(I):
    """pi_i = 2 d(I_i, 0) in [0, 1]."""
    I = np.asarray(I, dtype=float)
    return 2.0 * dist(I, np.zeros_like(I))


# P2 / P2b ------------------------------------------------------------------------------------
def pull(x, t, k, lo=-1.0, hi=1.0):
    """P2: attraction toward ``t`` (k > 0) or repulsion (k < 0); ``k`` may be a vector (mask)."""
    return np.clip(x + k * (t - x), lo, hi)


def pull_many(x, pushes, lo=-1.0, hi=1.0):
    """P2b for a single state vector: ``pushes = [(t, k), ...]`` computed on the start-of-turn ``x``
    and applied once, normalised among themselves."""
    if not pushes:
        return np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    num = sum(np.asarray(k) * (np.asarray(t) - x) for t, k in pushes)
    den = max(1.0, float(sum(np.max(np.abs(np.asarray(k))) for _, k in pushes)))
    return np.clip(x + num / den, lo, hi)


def pull_many_decay(x, pushes, base, k_dec, lo=-1.0, hi=1.0):
    """P2b with the decay toward ``base`` applied outside the normaliser."""
    moved = pull_many(x, pushes, lo, hi) if pushes else np.asarray(x, dtype=float)
    return np.clip(moved + k_dec * (base - np.asarray(x, dtype=float)), lo, hi)


class PushBuffer:
    """Vectorized P2b for a whole (n, d) state array.

    ``add(idx, target, k)`` records pushes for the citizens in ``idx``; ``target`` is (m, d) (NaN =
    masked dimension, which gets k = 0) or (d,), ``k`` is (m,) or (m, d). ``apply`` evaluates every
    citizen on the start-of-turn state: pushes normalise among themselves (denominator
    ``max(1, sum of per-push max |k|)``) and the decay toward the base is added outside the
    normaliser. The result does not depend on the order in which pushes were added.
    """

    def __init__(self, n: int, d: int):
        self.n, self.d = n, d
        self._items: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.count = 0

    def add(self, idx, target, k) -> None:
        idx = np.asarray(idx, dtype=int)
        if idx.size == 0:
            return
        target = np.asarray(target, dtype=float)
        if target.ndim == 1:
            target = np.broadcast_to(target, (idx.size, self.d))
        k = np.asarray(k, dtype=float)
        if k.ndim == 0:
            k = np.full(idx.size, float(k))
        if k.ndim == 1:
            k = k[:, None] * np.ones((1, self.d))
        mask = np.isnan(target)
        self._items.append((idx, np.where(mask, 0.0, target), np.where(mask, 0.0, k)))
        self.count += idx.size

    def apply(self, x, base=None, k_dec=None, lo=-1.0, hi=1.0):
        x = np.asarray(x, dtype=float)
        num = np.zeros_like(x)
        den = np.zeros(self.n)
        for idx, tt, kk in self._items:
            np.add.at(num, idx, kk * (tt - x[idx]))
            np.add.at(den, idx, np.abs(kk).max(axis=1))
        out = x + num / np.maximum(1.0, den)[:, None]
        if base is not None and k_dec is not None:
            kd = np.asarray(k_dec, dtype=float)
            if kd.ndim == 1:
                kd = kd[:, None]
            out = out + kd * (np.asarray(base, dtype=float) - x)
        return np.clip(out, lo, hi)


# P4 / P5 ------------------------------------------------------------------------------------
def relax(x, base, h, dt, lo=-1.0, hi=1.0):
    """P4: decay as an attraction toward the base (also an exponential moving average)."""
    return pull(x, base, rate(lam(h), dt), lo, hi)


def gain(x, k):
    return pull(x, 1.0, k, 0.0, 1.0)


def loss(x, k):
    return pull(x, 0.0, k, 0.0, 1.0)


def spend(x, c):
    """P5: returns (new value, deficit)."""
    x = np.asarray(x, dtype=float)
    c = np.asarray(c, dtype=float)
    return np.where(c <= x, x - c, 0.0), np.where(c <= x, 0.0, c - x)


def credit(x, c):
    """P5: returns (new value, surplus lost beyond the cap)."""
    x = np.asarray(x, dtype=float)
    c = np.asarray(c, dtype=float)
    return np.minimum(1.0, x + c), np.maximum(0.0, x + c - 1.0)


def equilibrium(g, lam_):
    """Equilibrium of an accumulator with income ``g`` and decay ``lam_``: g / (g + lam)."""
    return g / (g + lam_)


def income_for_equilibrium(h, x_star):
    """Maximum income that gives equilibrium ``x_star`` with half-life ``h``: g = lam * x / (1 - x)."""
    return lam_s(h) * x_star / (1.0 - x_star)


# P6 -----------------------------------------------------------------------------------------
def gate(on, v, th_in, th_out, since, now, dwell=0.0):
    """Threshold with hysteresis and minimum dwell time. Works on scalars or arrays; returns (on, since)."""
    on = np.asarray(on, dtype=bool)
    v = np.asarray(v, dtype=float)
    since = np.asarray(since, dtype=float)
    turn_on = (~on) & (v >= th_in)
    turn_off = on & (v < th_out) & ((now - since) >= dwell)
    new_on = (on | turn_on) & ~turn_off
    new_since = np.where(turn_on | turn_off, now, since)
    return new_on, new_since


# P9 -----------------------------------------------------------------------------------------
def centroid(X, m):
    X = np.asarray(X, dtype=float)
    m = np.asarray(m, dtype=float)
    s = m.sum()
    if s <= 0:
        return X.mean(axis=0) if len(X) else np.zeros(X.shape[-1] if X.ndim > 1 else 1)
    return (m[:, None] * X).sum(axis=0) / s


# P10 / P11 / P13 -----------------------------------------------------------------------------
def trigger(nu, dt, rng: GameRng, size=None):
    """P10: stochastic trigger with probability ``rate(nu, dt)``; ``nu`` is a frequency per day."""
    p = rate(nu, dt)
    if size is None and np.ndim(p) == 0:
        return bool(rng.random() < p)
    return rng.random(np.shape(p) if size is None else size) < p


def pick(items, weights, n, rng: GameRng):
    """P11: weighted draw without replacement."""
    items, weights, out = list(items), [float(w) for w in weights], []
    for _ in range(min(n, len(items))):
        j = rng.choice_index(weights)
        out.append(items.pop(j))
        weights.pop(j)
    return out


def pick_topk(ranked, rng: GameRng, weights=(0.6, 0.3, 0.1)):
    k = min(len(ranked), len(weights))
    return pick(ranked[:k], weights[:k], 1, rng)[0]


def count(nu, dt, rng: GameRng) -> int:
    """P13: number of events of an aggregate process with stochastic rounding."""
    x = float(nu) * float(dt)
    n = int(math.floor(x))
    return n + (1 if rng.random() < x - n else 0)


# P12 / P14 ----------------------------------------------------------------------------------
def aggregate(values, weights):
    """P12: weighted mean."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    s = weights.sum()
    return float((weights * values).sum() / s) if s > 0 else 0.0


def express(x, fear):
    """P14: negative components are attenuated by (1 - fear); support is never attenuated."""
    x = np.asarray(x, dtype=float)
    f = np.asarray(fear, dtype=float)
    if f.ndim < x.ndim:
        f = f.reshape(f.shape + (1,) * (x.ndim - f.ndim))
    return np.where(x >= 0, x, (1.0 - f) * x)
