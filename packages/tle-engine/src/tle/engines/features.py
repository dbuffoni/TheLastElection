"""Numeric reading (P8) of extracted option features, and the reverse (labels from numbers)."""

from __future__ import annotations

import numpy as np

from tle.contracts.models import OptionFeatureOut
from tle.contracts.vocab import bip_value_of, stance_value, uni_value_of

from .protocol import OptionFeatures


def features_to_arrays(feats: list[OptionFeatureOut]) -> OptionFeatures:
    K = len(feats)
    t_I = np.full((K, 3), np.nan)
    t_C = np.full((K, 3), np.nan)
    e_r = np.full(K, np.nan)
    F = np.zeros(K)
    stance = np.full((K, 8), np.nan)
    for k, f in enumerate(feats):
        t_I[k] = [bip_value_of(f.t_I1), bip_value_of(f.t_I2), bip_value_of(f.t_I3)]
        t_C[k] = [bip_value_of(f.t_C1), bip_value_of(f.t_C2), bip_value_of(f.t_C3)]
        e_r[k] = bip_value_of(f.e_r)
        F[k] = uni_value_of(f.F)
        stance[k] = [
            stance_value(f.stance_mayor), stance_value(f.stance_pillar_1), stance_value(f.stance_pillar_2),
            stance_value(f.stance_pillar_3), stance_value(f.stance_pillar_4), stance_value(f.stance_pillar_5),
            stance_value(f.stance_actor), stance_value(f.stance_target),
        ]
    return OptionFeatures(t_I=t_I, t_C=t_C, e_r=e_r, F=F, stance=stance)
