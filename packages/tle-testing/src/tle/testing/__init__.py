"""Shared test helpers: small fast games, failure injection, determinism assertions."""

from __future__ import annotations

from tle.catalogs.loader import load_catalogs
from tle.engines.protocol import Engines
from tle.engines.simulated import SimulatedEngine
from tle.loop.game import Game
from tle.params import default_params

_CAT = None


def catalogs():
    global _CAT
    if _CAT is None:
        _CAT = load_catalogs()
    return _CAT


def small_game(
    seed: int = 1,
    n: int = 80,
    scenario: str = "industrial_secular",
    overrides: dict | None = None,
    perfect: bool = False,
) -> Game:
    cat = catalogs()
    p = default_params().with_overrides(overrides) if overrides else default_params()
    eng = Engines(SimulatedEngine(cat, p, perfect=perfect), SimulatedEngine(cat, p, perfect=perfect))
    return Game(cat, p, eng, scenario=scenario, seed=seed, n=n)


def inject_failures(game: Game, eps_err: float) -> Game:
    """Failure injection (eps_err): every engine call can fail and fall back (spec 15.4)."""
    for e in (game.engines.s1, game.engines.s2):
        e.eps_err = eps_err
    return game
