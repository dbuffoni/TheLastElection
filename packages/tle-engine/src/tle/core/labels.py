"""S7 label scale: one ordered seven-level scale, one quantizer, one lookup (spec 2.3, P7, P8).

Labels are stored as integer indices ``u`` in 0..6 (``u = k + 3`` with ``k`` in -3..3).
Bipolar reading is ``(u - 3) / 3`` in [-1, 1]; unipolar reading is ``u / 6`` in [0, 1].
"""

from __future__ import annotations

import numpy as np

N_LEVELS = 7
NEUTRAL = 3

UNIPOLAR_NAMES = ("null", "very low", "low", "medium", "high", "very high", "maximum")
STANCE_NAMES = (
    "opposition",
    "dissent",
    "mild dissent",
    "indifference",
    "mild support",
    "support",
    "adherence",
)
NO_EFFECT = "no effect"  # mask value for extractions


def bipolar_names(negative: str, positive: str) -> tuple[str, ...]:
    """Names of a bipolar axis: three degrees per pole plus neutral."""
    return (
        f"very {negative}",
        negative,
        f"mildly {negative}",
        "neutral",
        f"mildly {positive}",
        positive,
        f"very {positive}",
    )


def bip_value(idx):
    """Bipolar lookup (P8): index 0..6 -> -1..1."""
    return (np.asarray(idx, dtype=float) - NEUTRAL) / 3.0


def uni_value(idx):
    """Unipolar lookup (P8): index 0..6 -> 0..1."""
    return np.asarray(idx, dtype=float) / 6.0


def quantize01(x, prev_idx=None, eps: float = 0.05):
    """P7 on a value already in [0, 1]: nearest level, with hysteresis around the previous level.

    The label changes only if the value leaves the interval of the current level by more than ``eps``.
    """
    x = np.asarray(x, dtype=float)
    nearest = np.clip(np.floor(x * 6.0 + 0.5), 0, 6).astype(np.int8)
    if prev_idx is None:
        return nearest
    prev = np.asarray(prev_idx).astype(np.int8)
    lo = np.clip((prev - 0.5) / 6.0, 0.0, 1.0)
    hi = np.clip((prev + 0.5) / 6.0, 0.0, 1.0)
    stay = (x >= lo - eps) & (x <= hi + eps)
    return np.where(stay, prev, nearest).astype(np.int8)


def quantize_bipolar(v, prev_idx=None, eps: float = 0.05):
    """P7 for values in [-1, 1]: brought into [0, 1] with (v + 1) / 2 so one quantizer serves both."""
    return quantize01((np.asarray(v, dtype=float) + 1.0) / 2.0, prev_idx, eps)


def quantize_unipolar(v, prev_idx=None, eps: float = 0.05):
    return quantize01(v, prev_idx, eps)


def round_bipolar(v):
    """Plain nearest-level snapping of bipolar values (no hysteresis), used at generation."""
    return bip_value(quantize_bipolar(v))


def trend_value(value, moving_avg, scale: float = 2.0):
    """Trend of an indicator: difference with its moving average, rescaled into [-1, 1]."""
    return np.clip((np.asarray(value, dtype=float) - np.asarray(moving_avg, dtype=float)) * scale, -1.0, 1.0)


def label_name(idx: int, kind: str = "unipolar") -> str:
    idx = int(idx)
    if kind == "stance":
        return STANCE_NAMES[idx]
    return UNIPOLAR_NAMES[idx]
