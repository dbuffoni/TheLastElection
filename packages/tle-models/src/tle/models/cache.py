"""Response cache (spec 3.6): same task and same input -> same answer, stored on disk as JSON lines.
Makes real-model runs resumable and repeatable, and lets a campaign replay an expensive run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel
from tle.contracts.tasks import TASKS
from tle.core.rng import GameRng


class CachedEngine:
    def __init__(self, inner, path: Path | None = None, name: str = "cached"):
        self.inner, self.name, self.path = inner, name, path
        self.store: dict[str, str] = {}
        self.hits = self.misses = 0
        if path and path.exists():
            for line in path.read_text().splitlines():
                k, v = json.loads(line)
                self.store[k] = v

    @staticmethod
    def key(task: str, inp: BaseModel) -> str:
        return hashlib.blake2b(
            f"{task}|{json.dumps(inp.model_dump(), sort_keys=True)}".encode(), digest_size=16
        ).hexdigest()

    def call(self, task: str, inp: BaseModel, rng: GameRng):
        k = self.key(task, inp)
        if k in self.store:
            self.hits += 1
            return TASKS[task].output_model.model_validate_json(self.store[k])
        self.misses += 1
        out = self.inner.call(task, inp, rng)
        if out is not None:
            self.store[k] = out.model_dump_json()
            if self.path:
                with self.path.open("a") as f:
                    f.write(json.dumps([k, self.store[k]]) + "\n")
        return out

    def __getattr__(self, item):  # batched choices and accounting go straight to the inner engine
        return getattr(self.inner, item)
