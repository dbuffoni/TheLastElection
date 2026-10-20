"""Evaluate a closed-vocabulary ``Predicate`` (spec 4.2) over the population: who is involved."""

from __future__ import annotations

import numpy as np
from tle.contracts.models import Predicate
from tle.params import Params
from tle.systems.population import age_band
from tle.world.catalogs import AGE_BANDS, CLASSES, Catalogs
from tle.world.state import World


def match(w: World, cat: Catalogs, p: Params, pred: Predicate) -> np.ndarray:
    """Boolean mask over the citizen table (alive adults are the default audience)."""
    n = w.n_cap
    mask = w.alive.copy()
    if pred.age_band != "any":
        mask &= age_band(w.age, p.demography.a_adult, p.demography.a_pens) == AGE_BANDS.index(pred.age_band)
    else:
        mask &= w.age >= p.demography.a_adult
    if pred.social_class != "any":
        mask &= w.cls == CLASSES.index(pred.social_class)
    if pred.trade != "any":
        mask &= w.trade == cat.trade_ids().index(pred.trade)
    if (
        pred.circle.isdigit()
    ):  # a free-text field: anything but a circle number means "any" (a model may invent values)
        mask &= w.circle == int(pred.circle)
    if pred.faction == "member":
        mask &= w.fac_member.any(axis=1)
    elif pred.faction == "non-member":
        mask &= ~w.fac_member.any(axis=1)
    if pred.influencer == "influencer":
        mask &= w.inf_on
    elif pred.influencer == "non-influencer":
        mask &= ~w.inf_on
    axis, _, side = pred.ideology_extreme.partition(":")
    if axis in ("1", "2", "3") and side in ("high", "low"):
        v = w.I[:, int(axis) - 1]
        mask &= (v >= 2 / 3) if side == "high" else (v <= -2 / 3)
    if pred.role == "councillor":
        mask &= w.councillor != 0
    return mask & (np.arange(n) < w.n_used)
