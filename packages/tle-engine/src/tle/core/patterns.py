"""The three generic patterns (spec 3.4): state vector (A), scalar accumulator (B), group (C)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import prims


def vector_step(x, pushes, base, h_decay, dt, lo=-1.0, hi=1.0):
    """Pattern A: one pull_many (P2b) with the decay toward the base applied outside the normaliser."""
    k_dec = prims.rate(prims.lam(h_decay), dt)
    return prims.pull_many_decay(x, pushes, base, k_dec, lo, hi)


@dataclass
class Accumulator:
    """Pattern B: scalar in [0, 1], capped growth, decay after a grace period, expenses, thresholds."""

    value: float = 0.0
    h_decay: float = float("inf")
    base: float = 0.0
    grace: float = 0.0
    last_active: float = 0.0
    contrib: dict = field(default_factory=dict)  # event id -> (contribution, day), for parent events (6.6)

    def step(self, income_rate: float, dt: float, now: float, active: bool = True) -> None:
        if active:
            self.last_active = now
        idle = max(0.0, (now - self.last_active) - self.grace)
        pushes = [
            (1.0, float(prims.rate(income_rate, dt))),
            (self.base, float(prims.rate(prims.lam_s(self.h_decay), min(dt, idle)))),
        ]
        self.value = float(prims.pull_many(self.value, pushes, 0.0, 1.0))

    def hit(self, k: float, ev=None, now: float = 0.0) -> None:
        """Point event: k > 0 gain, k < 0 loss (a null effect never moves toward the centre)."""
        old = self.value
        self.value = float(prims.gain(self.value, k) if k > 0 else prims.loss(self.value, -k))
        if ev is not None:
            self.contrib[ev] = (abs(self.value - old), now)

    def spend(self, c: float) -> float:
        """Spend ``c``; returns the deficit (0 if covered)."""
        v, deficit = prims.spend(self.value, c)
        self.value = float(v)
        return float(deficit)


@dataclass
class Group:
    """Pattern C: centroid, weighted members, membership with hysteresis, optional strength."""

    center: np.ndarray
    members: dict = field(default_factory=dict)  # id -> weight
    strength: Accumulator | None = None

    def recenter(self, X, h_c: float, dt: float) -> None:
        if not self.members:
            return
        ids = list(self.members)
        target = prims.centroid(np.asarray(X)[ids], [self.members[i] for i in ids])
        self.center = prims.relax(self.center, target, h_c, dt)

    def inside(self, x, on: bool, rho_in: float, rho_out: float, W=None) -> bool:
        d = float(prims.dist(x, self.center, W))
        return d <= rho_out if on else d <= rho_in
