"""Which Liquid model serves which role, and how the server is run. Settings come from ``config/liquid.toml`` (optional),
then environment variables, then CLI flags."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    role: str  # "s1" | "s2"
    kind: str  # "decision" (d1: one forward pass, no tokens) | "chat" (LFM instruct) | "mock" (no weights, for tests)
    repo: str  # HuggingFace repo id, or a local directory
    alias: str  # the name clients put in the request's "model" field
    approx_gb: float  # weights in bf16, to warn before loading on a small GPU
    trust_remote_code: bool = False
    revision: str | None = None


# S1 is Liquid's decision model d1 (the same "System One" interface the game already speaks through `kind = "jev"`);
# S2 is the non-thinking LFM2.5 instruct model. Smaller / larger alternatives of the same families are listed in README.
DEFAULT_SPECS = {
    "s1": ModelSpec("s1", "decision", "LiquidAI/d1-2B", "liquid-s1", 3.9, trust_remote_code=True),
    "s2": ModelSpec("s2", "chat", "LiquidAI/LFM2.5-1.2B-Instruct", "liquid-s2", 2.4),
}


@dataclass(frozen=True)
class LiquidConfig:
    specs: dict[str, ModelSpec] = field(default_factory=lambda: dict(DEFAULT_SPECS))
    roles: tuple[str, ...] = ("s1", "s2")  # roles loaded by `serve` (one machine can serve just one)
    device: str = "auto"  # auto | cuda | rocm | cpu
    dtype: str = "auto"  # auto | bf16 | fp16 | fp32
    host: str = "127.0.0.1"  # 0.0.0.0 to serve a desktop to another machine
    port: int = 8100
    api_key: str = ""  # if set, requests need `Authorization: Bearer <key>`
    max_batch: int = 16  # requests fused into one forward / generate call
    max_wait_ms: int = 15  # how long the first request of a batch waits for company
    max_new_tokens: int = 1024  # cap when a request does not give max_tokens
    lazy_load: bool = (
        False  # load a role when its first request arrives (saves memory when only one role is used)
    )


def load_config(path: Path | None = None, **overrides) -> LiquidConfig:
    cfg = LiquidConfig()
    p = Path(path) if path else Path("config/liquid.toml")
    if p.is_file():
        data = tomllib.loads(p.read_text())
        specs = dict(cfg.specs)
        for role in ("s1", "s2"):
            if role in data:
                specs[role] = replace(specs[role], **data[role])
        top = {k: v for k, v in data.items() if k not in ("s1", "s2")}
        if "roles" in top:
            top["roles"] = tuple(top["roles"])
        cfg = replace(cfg, specs=specs, **top)
    env = {
        "device": os.environ.get("TLE_LIQUID_DEVICE"),
        "host": os.environ.get("TLE_LIQUID_HOST"),
        "api_key": os.environ.get("TLE_LIQUID_API_KEY"),
        "port": int(os.environ["TLE_LIQUID_PORT"]) if os.environ.get("TLE_LIQUID_PORT") else None,
    }
    cfg = replace(cfg, **{k: v for k, v in env.items() if v})
    specs = dict(cfg.specs)
    for role in ("s1", "s2"):  # TLE_LIQUID_S1_REPO=/path/to/dir swaps the weights without touching files
        if os.environ.get(f"TLE_LIQUID_{role.upper()}_REPO"):
            specs[role] = replace(specs[role], repo=os.environ[f"TLE_LIQUID_{role.upper()}_REPO"])
    cfg = replace(cfg, specs=specs)
    kw = {k: v for k, v in overrides.items() if v is not None}
    if "roles" in kw:
        kw["roles"] = tuple(kw["roles"])
    return replace(cfg, **kw)


def fetch(spec: ModelSpec, local_dir: str | None = None) -> str:
    """Download the weights (resumable, cached by huggingface_hub) and return the local path. A local directory is
    returned as it is. Needs network access to huggingface.co."""
    if Path(spec.repo).is_dir():
        return spec.repo
    from huggingface_hub import snapshot_download

    return snapshot_download(
        spec.repo,
        revision=spec.revision,
        local_dir=local_dir,
        allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt", "*.model", "*.jinja", "tokenizer*"],
    )
