"""Load the closed catalogs from ``config/catalogs/*.toml`` into engine objects (spec 4.3)."""

from __future__ import annotations

import tomllib
from pathlib import Path

from tle.world.catalogs import Catalogs

from tle.registry.loader import repo_root

CATALOG_FILES = ("trades", "domains", "pillars", "event_types", "activities", "scenarios")


def load_catalogs(root: Path | None = None) -> Catalogs:
    root = root or repo_root()
    merged: dict = {}
    for name in CATALOG_FILES:
        path = root / "config" / "catalogs" / f"{name}.toml"
        if not path.exists():
            raise FileNotFoundError(f"missing catalog file {path}")
        merged.update(tomllib.loads(path.read_text()))
    return Catalogs.model_validate(merged)
