"""Metric registry (spec: Technical Implementation 6). Every metric is declared once in ``config/metrics/*.toml``;
the same declaration drives telemetry columns, analyzers, reports and gates."""

from __future__ import annotations

import tomllib
from collections import OrderedDict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .loader import repo_root

Family = Literal["continuous", "outcome", "threshold", "saturation", "ratio", "execution", "distribution"]
Source = Literal["turn", "panel", "events", "campaign"]


class Target(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["band", "min", "max", "drift_max", "ratio", "monotone"]
    value: float | None = None
    lo: float | None = None
    hi: float | None = None
    window_days: float = 180.0
    scenarios: list[str] = []  # empty = every scenario
    after_days: float = 0.0  # measure from this day (e.g. mobility after two years)
    note: str = ""


class MetricSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    title: str
    spec: list[str] = []
    family: Family = "continuous"
    source: Source = "turn"
    unit: str = ""
    introduced: int = 1
    stage: int = 1
    group: str = "population"  # report panel: population | ideology | opinion | wealth | network | technical
    target: Target | None = None
    scale_floor: float = (
        0.0  # drift is relative to max(|level|, floor): bipolar means near 0 need one S7 step (0.333)
    )
    chart: str = "timeseries"
    aggregate: str = "median_p10_p90"
    description: str = ""


def load_metrics(root: Path | None = None) -> OrderedDict[str, MetricSpec]:
    root = root or repo_root()
    out: OrderedDict[str, MetricSpec] = OrderedDict()
    for f in sorted((root / "config" / "metrics").glob("*.toml")):
        for mid, body in tomllib.loads(f.read_text()).items():
            if mid in out:
                raise ValueError(f"metric '{mid}' declared twice ({f.name})")
            out[mid] = MetricSpec(id=mid, **body)
    return out
