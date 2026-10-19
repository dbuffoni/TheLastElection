"""Reading and validating ``config/``: parameter registry, profiles and layering (spec 7)."""

from __future__ import annotations

import tomllib
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tle.params.formulas import FORMULA_NAMESPACE

from .models import ParamSpec


def find_root(start: Path | None = None) -> Path:
    """Repository root: the nearest parent containing ``config/params``."""
    p = (start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "config" / "params").is_dir():
            return cand
    raise FileNotFoundError(
        "config/params not found above the current directory; run from inside the repository or set TLE_ROOT"
    )


def repo_root() -> Path:
    import os

    env = os.environ.get("TLE_ROOT")
    return find_root(Path(env)) if env else find_root()


@dataclass
class Registry:
    specs: OrderedDict[str, ParamSpec] = field(default_factory=OrderedDict)

    def __getitem__(self, k: str) -> ParamSpec:
        return self.specs[k]

    def __contains__(self, k: str) -> bool:
        return k in self.specs

    def __iter__(self):
        return iter(self.specs.values())

    def groups(self) -> OrderedDict[str, list[ParamSpec]]:
        g: OrderedDict[str, list[ParamSpec]] = OrderedDict()
        for s in self.specs.values():
            g.setdefault(s.group, []).append(s)
        return g

    def defaults(self) -> dict[str, Any]:
        return {k: s.default for k, s in self.specs.items()}

    def free(self) -> list[ParamSpec]:
        return [s for s in self.specs.values() if s.status == "free"]


def load_registry(root: Path | None = None) -> Registry:
    root = root or repo_root()
    reg = Registry()
    for f in sorted((root / "config" / "params").glob("*.toml")):
        group = f.stem
        data = tomllib.loads(f.read_text())
        for pid, body in data.items():
            if pid in reg.specs:
                raise ValueError(f"{f}: duplicate parameter id '{pid}'")
            reg.specs[pid] = ParamSpec.model_validate({"id": pid, "group": group, **body})
    return reg


# the order of groups in the generated model: a stable, human-meaningful one
GROUP_ORDER = [
    "time",
    "labels",
    "vectors",
    "ties",
    "factions",
    "capital",
    "influencers",
    "community",
    "economy",
    "pillars",
    "threats",
    "events",
    "demography",
    "technical",
]


def ordered_specs(reg: Registry) -> list[ParamSpec]:
    rank = {g: i for i, g in enumerate(GROUP_ORDER)}
    return sorted(reg.specs.values(), key=lambda s: (rank.get(s.group, 99), list(reg.specs).index(s.id)))


def resolve_derived(reg: Registry, values: dict[str, Any]) -> dict[str, Any]:
    """Evaluate derived parameters from their formulas (repeated passes, so order does not matter)."""
    out = dict(values)
    pending = [s for s in reg if s.cls == "derived" and s.formula]
    for _ in range(len(pending) + 1):
        left = []
        for s in pending:
            try:
                v = eval(s.formula, {"__builtins__": {}}, {**FORMULA_NAMESPACE, **out})  # noqa: S307
            except NameError:
                left.append(s)
                continue
            out[s.id] = int(v) if s.is_int else float(v)
        pending = left
        if not pending:
            break
    if pending:
        raise ValueError(f"cannot resolve derived parameters: {[s.id for s in pending]}")
    return out


# layering (spec 7.3) ----------------------------------------------------------------------
LAYERS = ("registry", "scenario", "profile", "campaign", "cli")


@dataclass
class ResolvedParams:
    values: dict[str, Any]
    origin: dict[str, str]  # param id -> layer name (profile entries carry the profile id)

    def to_json(self) -> dict[str, Any]:
        return {k: {"value": self.values[k], "layer": self.origin[k]} for k in self.values}


def load_profile_chain(profile: str | None, root: Path | None = None) -> list[tuple[str, dict[str, Any]]]:
    """Profiles declare ``extends = "inc-(N-1)"`` and hold only values calibrated in their increment."""
    root = root or repo_root()
    chain: list[tuple[str, dict[str, Any]]] = []
    name = profile
    seen = set()
    while name:
        if name in seen:
            raise ValueError(f"profile cycle at {name}")
        seen.add(name)
        path = root / "config" / "profiles" / f"{name}.toml"
        if not path.exists():
            raise FileNotFoundError(f"profile '{name}' not found at {path}")
        data = tomllib.loads(path.read_text())
        chain.append((name, dict(data.get("params", {}))))
        name = data.get("extends")
    chain.reverse()
    return chain


def resolve_params(
    reg: Registry,
    *,
    scenario_overlay: dict[str, Any] | None = None,
    profile: str | None = None,
    campaign: dict[str, Any] | None = None,
    cli: dict[str, Any] | None = None,
    root: Path | None = None,
) -> ResolvedParams:
    values: dict[str, Any] = reg.defaults()
    origin = {k: "registry" for k in values}

    def apply(layer: str, d: dict[str, Any]):
        for k, v in d.items():
            if k not in reg:
                raise KeyError(f"unknown parameter '{k}' in layer '{layer}'; see `tle params ls`")
            if reg[k].cls == "derived":
                raise ValueError(f"'{k}' is derived (formula: {reg[k].formula}); set its inputs instead")
            values[k] = int(v) if reg[k].is_int else float(v)
            origin[k] = layer

    apply("scenario", scenario_overlay or {})
    for pname, pvals in load_profile_chain(profile, root) if profile else []:
        apply(f"profile:{pname}", pvals)
    apply("campaign", campaign or {})
    apply("cli", cli or {})
    values = resolve_derived(reg, values)
    for s in reg:
        origin.setdefault(s.id, "registry")
    return ResolvedParams(values, origin)
