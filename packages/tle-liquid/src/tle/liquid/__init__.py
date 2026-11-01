"""tle.liquid: Liquid AI models (d1 decision model for S1, LFM instruct for S2) behind a batched, vendor-neutral engine.
This package never imports the game; the game reaches it only over HTTP through its existing engine settings."""

from .config import DEFAULT_SPECS, LiquidConfig, ModelSpec, load_config
from .engine import LiquidEngine

__all__ = ["DEFAULT_SPECS", "LiquidConfig", "LiquidEngine", "ModelSpec", "load_config"]
