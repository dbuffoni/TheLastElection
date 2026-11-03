"""Distance between citizens and a cache of the answers of a real S1 model.

A citizen is, for the model, a row of S7 labels (``LabelContext``). Two citizens whose labels are (nearly) the same would get
(nearly) the same answer to the same question, so one model call can serve both: when a citizen is closer than ``threshold`` to
one that was already answered (in this batch or earlier, for the same question), the answer is copied instead of asked again.

Distance in [0, 1]: the mean over all features of the normalised gap (ordinal labels: |a-b| / 6, age band and class on their own
range) or of the mismatch (trade, circle, group membership: 0 or 1). 0 = identical labels. ``threshold = 0`` keeps only exact
duplicates; ``None`` / negative switches the cache off.

Deterministic: citizens are visited in index order, the nearest earlier answer wins (ties: the lowest index), so the same game gives
the same answers. The model only ever sees the representatives; the other citizens are never asked.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np


def features(ctx) -> tuple[np.ndarray, np.ndarray]:
    """(ordinal (m, a) scaled to [0, 1], categorical (m, c) integer codes) of a ``LabelContext``."""
    ords = [
        ctx.age_band / 3.0,
        ctx.social_class / 2.0,
        ctx.fear / 6.0,
        ctx.risk / 6.0,
    ]
    cols = [np.asarray(o, dtype=float).reshape(len(ctx), -1) for o in ords]
    for a in (ctx.T, ctx.C, ctx.I, ctx.O):
        cols.append(np.asarray(a, dtype=float).reshape(len(ctx), -1) / 6.0)
    if ctx.social is not None:
        cols.append(np.asarray(ctx.social, dtype=float).reshape(len(ctx), -1) / 6.0)
    cats = [ctx.trade, ctx.circle] + ([ctx.member] if ctx.member is not None else [])
    cat = np.stack([np.asarray(c).astype(int).reshape(len(ctx)) for c in cats], axis=1)
    return np.concatenate(cols, axis=1), cat


def distances(a_ord, a_cat, b_ord, b_cat) -> np.ndarray:
    """Distances of one citizen (a) to many (b)."""
    n = a_ord.shape[-1] + a_cat.shape[-1]
    return (np.abs(b_ord - a_ord).sum(axis=1) + (b_cat != a_cat).sum(axis=1)) / n


class SimilarityCache:
    def __init__(self, threshold: float | None, capacity: int = 2000):
        self.threshold = threshold
        self.capacity = capacity
        self._store: OrderedDict[str, tuple[list, list]] = (
            OrderedDict()
        )  # key -> (rows [(ord, cat)], answers)
        self.hits = 0
        self.asked = 0

    @property
    def on(self) -> bool:
        return self.threshold is not None and self.threshold >= 0

    def resolve(self, key: str, ctx, ask: Callable[[list[int]], list[Any]]) -> list[Any]:
        """Answers for every citizen of ``ctx``. ``ask(indices)`` asks the model about those citizens only (in that order)."""
        m = len(ctx)
        if not self.on or m == 0:
            return ask(list(range(m)))
        f_ord, f_cat = features(ctx)
        rows, answers = self._store.get(key, ([], []))
        rows, answers = list(rows), list(answers)  # earlier answers for the same question
        n_old = len(rows)
        asked_idx: list[int] = []
        source: list[
            tuple[str, int]
        ] = []  # per citizen: ("rep", position in asked_idx) or ("old", position in answers)
        for i in range(m):
            best, best_d = None, None
            if rows:
                d = distances(
                    f_ord[i], f_cat[i], np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])
                )
                j = int(np.argmin(d))
                if d[j] <= self.threshold:
                    best, best_d = j, float(d[j])
            if best is None:
                asked_idx.append(i)
                rows.append((f_ord[i], f_cat[i]))
                source.append(("rep", len(asked_idx) - 1))
            else:
                source.append(("rep", best - n_old) if best >= n_old else ("old", best))
        fresh = ask(asked_idx) if asked_idx else []
        out = []
        for kind, pos in source:
            out.append(fresh[pos] if kind == "rep" else answers[pos])
        self.asked += len(asked_idx)
        self.hits += m - len(asked_idx)
        new_rows = rows[n_old:]
        self._store[key] = (rows[:n_old] + new_rows, answers + list(fresh))
        self._store.move_to_end(key)
        while sum(len(v[1]) for v in self._store.values()) > self.capacity and len(self._store) > 1:
            self._store.popitem(last=False)
        r, a = self._store[key]
        if len(a) > self.capacity:
            self._store[key] = (r[-self.capacity :], a[-self.capacity :])
        return out
