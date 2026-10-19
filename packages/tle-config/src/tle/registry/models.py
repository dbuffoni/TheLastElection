"""Typed entries of the parameter registry (spec 7.1)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Status = Literal["draft", "provisional", "free", "calibrated", "frozen"]
ParamClass = Literal["fixed", "derived", "free"]
Scale = Literal["linear", "log", "int", "choice"]


class ParamSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    group: str
    symbol: str
    spec: str
    description: str
    default: float | int
    range: tuple[float, float] | None = None
    scale: Scale = "linear"
    cls: ParamClass = Field(alias="class")
    stage: int = 0
    introduced: int = 1
    module: str = ""
    status: Status = "draft"
    formula: str | None = None
    rationale: str = ""

    @property
    def is_int(self) -> bool:
        return self.scale == "int" or isinstance(self.default, int) and not isinstance(self.default, bool)
