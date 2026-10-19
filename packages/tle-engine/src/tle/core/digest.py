"""State digest (spec 5.2): a hash of the state arrays, per group, so a divergence between two runs
is localized to the turn and to the state group that diverged first."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def hash_arrays(items: list[tuple[str, Any]]) -> str:
    h = hashlib.blake2b(digest_size=8)
    for name, v in items:
        h.update(name.encode())
        if isinstance(v, np.ndarray):
            a = np.ascontiguousarray(v)
            h.update(str(a.dtype).encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        else:
            h.update(repr(v).encode())
    return h.hexdigest()


def combine(groups: dict[str, str]) -> str:
    h = hashlib.blake2b(digest_size=8)
    for k in sorted(groups):
        h.update(k.encode())
        h.update(groups[k].encode())
    return h.hexdigest()
