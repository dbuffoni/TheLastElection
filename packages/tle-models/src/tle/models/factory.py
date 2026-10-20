"""Per-engine model selection (spec Technical Implementation 3.6): S1 (the Jev-style classifier) and S2 (the narrator)
are configured independently, each either ``simulated`` or ``openai_compat`` (vLLM, llama.cpp, Ollama, LM Studio, a
hosted API, or a local server hosting Jev). Settings come from an optional TOML file and are overridden by CLI flags."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from tle.engines.protocol import Engines
from tle.engines.simulated import SimulatedEngine

from .backends import OpenAICompatBackend
from .cache import CachedEngine
from .real import RealEngine

KINDS = ("simulated", "openai_compat")


@dataclass(frozen=True)
class EngineSpec:
    kind: str = "simulated"
    base_url: str = "http://localhost:8000/v1"
    model: str = "local-model"
    api_key_env: str = ""  # name of the environment variable holding the key (never the key itself)
    structured: bool = (
        True  # False for servers without strict json_schema output (the validator and retries still apply)
    )
    timeout: float = 120.0
    temperature: float = 0.0


def load_specs(path: Path | None = None) -> dict[str, EngineSpec]:
    """``[s1]`` and ``[s2]`` tables of an engines file; a missing file or table means simulated."""
    out = {"s1": EngineSpec(), "s2": EngineSpec()}
    if path is None:
        return out
    data = tomllib.loads(Path(path).read_text())
    for k in out:
        if k in data:
            out[k] = EngineSpec(**data[k])
    return out


def _check(name: str, spec: EngineSpec) -> None:
    if spec.kind not in KINDS:
        raise ValueError(f"{name}: engine kind '{spec.kind}' must be one of {KINDS}")


def build_engines(
    cat, p, specs: dict[str, EngineSpec], cache: Path | None = None
) -> tuple[Engines, dict[str, dict]]:
    """Engines for a game plus a description of each (kind/model/url) for the run manifest. A cache file is shared by
    the real engines (keys include the task and its input, so S1 and S2 never collide)."""
    sim = SimulatedEngine(cat, p)
    built, info = {}, {}
    for name in ("s1", "s2"):
        spec = specs[name]
        _check(name, spec)
        if spec.kind == "simulated":
            built[name] = SimulatedEngine(cat, p) if name == "s2" else sim
            info[name] = {"kind": "simulated"}
            continue
        be = OpenAICompatBackend(
            base_url=spec.base_url,
            model=spec.model,
            api_key=os.environ.get(spec.api_key_env, "not-needed") if spec.api_key_env else "not-needed",
            timeout=spec.timeout,
            structured=spec.structured,
            name=f"openai_compat:{name}",
        )
        real = RealEngine(
            be, cat.vocab(), fallback=sim, n_retry=p.technical.n_retry, temperature=spec.temperature
        )
        built[name] = CachedEngine(real, cache) if cache else real
        info[name] = {"kind": spec.kind, "model": spec.model, "base_url": spec.base_url}
    return Engines(built["s1"], built["s2"]), info


def with_overrides(specs: dict[str, EngineSpec], **kw) -> dict[str, EngineSpec]:
    """``kw`` keys are ``s1_model``, ``s2_base_url``, ``s1_kind``... ; None values are ignored."""
    out = dict(specs)
    for key, v in kw.items():
        if v is None:
            continue
        name, field = key.split("_", 1)
        out[name] = replace(out[name], **{field: v})
    return out
