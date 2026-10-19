"""Simulated players (spec 15.5): they act through already structured decrees, so the text classifier
is bypassed, but the 50-character rule still applies to the label text they carry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from tle.contracts import models as m
from tle.contracts.vocab import MASK, uni_name
from tle.core.rng import GameRng
from tle.systems.predicate import match
from tle.world.catalogs import Catalogs
from tle.world.state import World

from .turn import PlayerInput

PILLAR_DOMAIN = {"pluralism": "institutions", "law": "institutions", "press": "press", "property": "economy", "civil": "security"}


def structured(pillar: str, agg: int, domain: str | None = None, econ: str = MASK, text: str | None = None, **pred) -> PlayerInput:
    base = dict(age_band="any", social_class="any", trade="any", circle="any", faction="any", influencer="any",
                ideology_extreme="any", role="any")
    base.update(pred)
    cd = m.ClassifyDecreeOut(
        pillar=pillar, domain=domain or PILLAR_DOMAIN.get(pillar, "worship"), aggressiveness=uni_name(agg),
        predicate=m.Predicate(**base), tec_effect=MASK, cul_effect=MASK, econ_effect=econ, threat="none",
        transfer=None, revocation=None)
    label = (text or f"{pillar} level {agg}")[:50]
    return PlayerInput(decree=label, structure=cd)


@dataclass
class Strategy:
    name: str
    act: Callable[[World, Catalogs, GameRng], PlayerInput]

    def __call__(self, w: World, cat: Catalogs, rng: GameRng) -> PlayerInput:
        return self.act(w, cat, rng.child("strategy", w.turn))


def _next_pillar(w: World, cat: Catalogs, order: list[str]):
    for pid in order:
        i = cat.pillar_ids().index(pid)
        if w.pillar_level[i] < len(cat.pillars[i].levels):
            return pid, i
    return None, None


def _random(w, cat, rng):
    pid = cat.pillar_ids()[int(rng.integers(0, 5))] if rng.random() < 0.7 else "none"
    return structured(pid, int(rng.integers(1, 6)))


def _assault(w, cat, rng):
    pid, i = _next_pillar(w, cat, ["pluralism", "law", "press", "civil", "property"])
    if pid is None:
        return PlayerInput()
    return structured(pid, min(6, int(w.pillar_level[i]) + 2), text=f"assault {pid}")


def _gradual(w, cat, rng):
    if w.K < 0.35:
        return PlayerInput()
    pid, i = _next_pillar(w, cat, ["press", "property", "civil", "law", "pluralism"])
    if pid is None:
        return PlayerInput()
    return structured(pid, int(w.pillar_level[i]) + 1, text=f"gradual {pid}")


def _populist(w, cat, rng):
    if w.turn % 3 != 2 or w.K < 0.3:
        return structured("none", 1, domain="worship", econ="positive", text="Public festival and bonus for all")
    return _gradual(w, cat, rng)


def _capture(w, cat, rng):
    pid, i = _next_pillar(w, cat, ["pluralism", "law", "press", "civil", "property"])
    if pid is None or w.K < 0.3:
        return PlayerInput()
    return structured(pid, int(w.pillar_level[i]) + 1, text=f"capture {pid}", role="councillor" if pid == "pluralism" else "any")


def _fear(w, cat, rng):
    pid, i = _next_pillar(w, cat, ["civil", "law", "press", "pluralism", "property"])
    if pid is None:
        return PlayerInput()
    return structured(pid, min(6, int(w.pillar_level[i]) + 2), text=f"crackdown {pid}")


def _clientelist(w, cat, rng):
    if w.turn % 2 == 0:
        return structured("none", 2, domain="economy", econ="positive", social_class="low", text="Grants for the poor")
    return _gradual(w, cat, rng)


STRATEGIES: dict[str, Strategy] = {
    s.name: s for s in [
        Strategy("random", _random), Strategy("assault", _assault), Strategy("gradual", _gradual),
        Strategy("populist", _populist), Strategy("capture", _capture), Strategy("fear", _fear),
        Strategy("clientelist", _clientelist),
    ]
}
