"""Explicit, serializable game RNG. Global ``random`` and ``numpy.random`` module calls are banned
outside this file (checked by ``tle check``)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def _key_to_int(key: Any) -> int:
    if isinstance(key, (int, np.integer)):
        return int(key) & 0xFFFFFFFFFFFFFFFF
    h = hashlib.blake2b(str(key).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little")


class GameRng:
    """Game random source. The main stream is advanced explicitly; ``child`` derives independent,
    order-free streams (used per event so permuting events of a turn cannot change results)."""

    def __init__(self, seed: int):
        self.seed = int(seed)
        self._gen = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([self.seed & 0xFFFFFFFF, 0xA11CE]))
        )

    # -- state -----------------------------------------------------------------------------
    def get_state(self) -> str:
        return json.dumps({"seed": self.seed, "bg": self._gen.bit_generator.state})

    def set_state(self, state: str) -> None:
        d = json.loads(state)
        self.seed = int(d["seed"])
        self._gen.bit_generator.state = d["bg"]

    @classmethod
    def from_state(cls, state: str) -> GameRng:
        r = cls(0)
        r.set_state(state)
        return r

    def child(self, *keys: Any) -> GameRng:
        """Independent stream keyed by ``(seed, keys)``; does not touch the main stream."""
        r = GameRng.__new__(GameRng)
        r.seed = self.seed
        ss = np.random.SeedSequence(
            [
                self.seed & 0xFFFFFFFF,
                0xC41D,
                *[_key_to_int(k) & 0xFFFFFFFF for k in keys],
                *[(_key_to_int(k) >> 32) & 0xFFFFFFFF for k in keys],
            ]
        )
        r._gen = np.random.Generator(np.random.PCG64(ss))
        return r

    # -- draws -----------------------------------------------------------------------------
    def random(self, size=None):
        return self._gen.random(size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        return self._gen.normal(loc, scale, size)

    def integers(self, low, high=None, size=None):
        return self._gen.integers(low, high, size)

    def beta(self, a, b, size=None):
        return self._gen.beta(a, b, size)

    def gumbel(self, size=None):
        return self._gen.gumbel(0.0, 1.0, size)

    def permutation(self, n):
        return self._gen.permutation(n)

    def choice_index(self, weights) -> int:
        w = np.asarray(weights, dtype=float)
        s = w.sum()
        if s <= 0:
            return int(self._gen.integers(0, len(w)))
        return int(np.searchsorted(np.cumsum(w) / s, self._gen.random(), side="right").clip(0, len(w) - 1))

    def poisson(self, lam, size=None):
        return self._gen.poisson(lam, size)
