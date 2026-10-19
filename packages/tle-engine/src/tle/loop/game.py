"""The game object: a world plus its catalogs, parameters and engines; the only public entry point
for a human or simulated player."""

from __future__ import annotations

from tle.engines.protocol import Engines
from tle.engines.simulated import SimulatedEngine
from tle.params import Params, default_params
from tle.systems.population import create_world
from tle.world.catalogs import Catalogs
from tle.world.state import World

from .turn import PlayerInput, TurnPackage, step


class Game:
    def __init__(
        self,
        cat: Catalogs,
        params: Params | None = None,
        engines: Engines | None = None,
        scenario: str = "industrial_secular",
        seed: int = 0,
        n: int | None = None,
        world: World | None = None,
    ):
        self.cat = cat
        self.p = params or default_params()
        self.engines = engines or Engines(SimulatedEngine(cat, self.p), SimulatedEngine(cat, self.p))
        self.w = world or create_world(cat, self.p, self.engines, scenario, seed, n)

    @property
    def over(self) -> bool:
        return bool(self.w.outcome)

    def step(self, inp: PlayerInput | None = None) -> TurnPackage:
        if self.over:
            raise RuntimeError(f"the game is over: {self.w.outcome}")
        return step(self.w, self.cat, self.p, self.engines, inp)

    def save(self) -> bytes:
        return self.w.to_bytes()

    @classmethod
    def load(
        cls, data: bytes, cat: Catalogs, params: Params | None = None, engines: Engines | None = None
    ) -> Game:
        return cls(cat, params, engines, world=World.from_bytes(data))


def play(game: Game, strategy, max_turns: int | None = None, observers=()) -> World:
    """Run a game to its end with a simulated player; ``observers(game, pkg)`` see every turn."""
    rng = game.w.rng.child("player")
    n = 0
    while not game.over and (max_turns is None or n < max_turns):
        inp = strategy(game.w, game.cat, rng) if strategy is not None else PlayerInput()
        pkg = game.step(inp)
        for ob in observers:
            ob(game, pkg)
        n += 1
    return game.w
