"""Closed label vocabularies of the contracts (spec 2.3). Names, not integers, travel to the models."""

from __future__ import annotations

from typing import Literal

import numpy as np

UNIPOLAR = ("null", "very low", "low", "medium", "high", "very high", "maximum")
STANCE = ("opposition", "dissent", "mild dissent", "indifference", "mild support", "support", "adherence")
BIPOLAR = (
    "very negative",
    "negative",
    "slightly negative",
    "neutral",
    "slightly positive",
    "positive",
    "very positive",
)
MASK = "no effect"

Unipolar = Literal["null", "very low", "low", "medium", "high", "very high", "maximum"]
Stance = Literal[
    "opposition", "dissent", "mild dissent", "indifference", "mild support", "support", "adherence"
]
StanceOrMask = Literal[
    "opposition",
    "dissent",
    "mild dissent",
    "indifference",
    "mild support",
    "support",
    "adherence",
    "no effect",
]
Bipolar = Literal[
    "very negative",
    "negative",
    "slightly negative",
    "neutral",
    "slightly positive",
    "positive",
    "very positive",
]
BipolarOrMask = Literal[
    "very negative",
    "negative",
    "slightly negative",
    "neutral",
    "slightly positive",
    "positive",
    "very positive",
    "no effect",
]

_UNI = {n: i for i, n in enumerate(UNIPOLAR)}
_STA = {n: i for i, n in enumerate(STANCE)}
_BIP = {n: i for i, n in enumerate(BIPOLAR)}


def uni_idx(name: str) -> int:
    return _UNI[name]


def uni_name(idx) -> str:
    return UNIPOLAR[int(np.clip(idx, 0, 6))]


def stance_value(name: str) -> float:
    """Lookup (P8): stance label -> bipolar value; the mask is NaN."""
    return float("nan") if name == MASK else (_STA[name] - 3) / 3.0


def stance_name(value: float) -> str:
    if value != value:
        return MASK
    return STANCE[int(np.clip(round(value * 3) + 3, 0, 6))]


def bip_value_of(name: str) -> float:
    return float("nan") if name == MASK else (_BIP[name] - 3) / 3.0


def bip_name(value: float) -> str:
    if value != value:
        return MASK
    return BIPOLAR[int(np.clip(round(value * 3) + 3, 0, 6))]


def uni_value_of(name: str) -> float:
    return _UNI[name] / 6.0
