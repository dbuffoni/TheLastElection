"""Recording of a real S1 (spec 15, mocks): every per-event choice of reactions is written as one JSON line with the
labels of the citizens, the actor, the extracted features of the options and the choices, and every classified decree
with its structure. ``tle mocks fit`` fits the simulated engine against these records; they are also the bootstrap
data of the decree bank. The recorder is a pure observer: it changes nothing of what the engine answers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _l(a):
    return np.asarray(a).tolist()


class RecordingEngine:
    def __init__(self, inner, path: Path, name: str = "recording"):
        self.inner, self.path, self.name = inner, Path(path), name
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, rec: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def call(self, task, inp, rng):
        out = self.inner.call(task, inp, rng)
        if task == "classify_decree" and out is not None and getattr(inp, "_structure", None) is None:
            self._write({"kind": "decree", "text": inp.text, "out": json.loads(out.model_dump_json())})
        return out

    def choose_reaction_many(self, ctx, actor, news, options, features, rng):
        choice = self.inner.choose_reaction_many(ctx, actor, news, options, features, rng)
        self._write(
            {
                "kind": "reaction",
                "actor": {
                    "kind": actor.kind,
                    "ideology": None if actor.ideology is None else _l(actor.ideology),
                },
                "event_type": getattr(news, "event_type", ""),
                "ctx": {
                    "age_band": _l(ctx.age_band),
                    "social_class": _l(ctx.social_class),
                    "T": _l(ctx.T),
                    "C": _l(ctx.C),
                    "I": _l(ctx.I),
                    "O": _l(ctx.O),
                    "fear": _l(ctx.fear),
                    "member": None if ctx.member is None else _l(ctx.member),
                    "social": None if ctx.social is None else _l(ctx.social),
                },
                "stance": np.where(np.isnan(features.stance), None, features.stance).tolist(),
                "choice": _l(choice),
            }
        )
        return choice

    def __getattr__(self, item):
        return getattr(self.inner, item)
