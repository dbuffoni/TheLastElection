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
        warmup: bool = True,
        observers=(),
    ):
        self.cat = cat
        self.p = params or default_params()
        self.engines = engines or Engines(SimulatedEngine(cat, self.p), SimulatedEngine(cat, self.p))
        if world is None:
            world = create_world(cat, self.p, self.engines, scenario, seed, n)
            self.w = world
            if warmup and self.p.time.T_warm > 0:
                self.run_warmup(observers)
        self.w = world

    def run_warmup(self, observers=()) -> None:
        """T_warm days without a mayor (spec 17.5): exogenous and routine events only, so that ties and opinions are
        already in equilibrium at day 0. Observers see the warm-up with negative days; the clock then restarts."""
        w = self.w
        t_end = float(self.p.time.T_warm)
        while w.day < t_end:
            pkg = step(
                w,
                self.cat,
                self.p,
                self.engines,
                PlayerInput(dt=min(self.p.time.dt_max, t_end - w.day)),
                warmup=True,
            )
            pkg.day -= t_end
            pkg.phase = "warmup"
            for ob in observers:
                ob(self, pkg)
        w.day, w.turn, w.dt_prev = 0.0, 0, 1.0
        w.K, w.decree_success = float(self.p.capital.c_0), 0.5
        w.threat_p[:] = 0
        w.threat_phase[:] = 0
        w.pillar_level[:] = 0
        w.events.clear()
        w.stats = {}
        w.r_prev = w.r.copy()
        w.warmup_done = True
        w.save_rng(w.rng)

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
    for ob in observers:
        getattr(ob, "on_start", lambda g: None)(game)
    n = 0
    while not game.over and (max_turns is None or n < max_turns):
        inp = strategy(game.w, game.cat, rng) if strategy is not None else PlayerInput()
        pkg = game.step(inp)
        for ob in observers:
            ob(game, pkg)
        n += 1
    return game.w
