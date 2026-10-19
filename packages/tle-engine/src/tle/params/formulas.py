"""Functions available to derived-parameter formulas in the registry (spec 7.1)."""

from __future__ import annotations

import math

from tle.core.prims import income_for_equilibrium, lam_s


def lam(h):
    return lam_s(h)


FORMULA_NAMESPACE = {
    "income_for_equilibrium": income_for_equilibrium,
    "lam": lam,
    "ceil": math.ceil,
    "floor": math.floor,
    "max": max,
    "min": min,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
}
