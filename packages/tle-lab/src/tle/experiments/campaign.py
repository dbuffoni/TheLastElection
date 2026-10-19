"""Campaigns (spec: laboratory): a TOML file expands into a list of runs (design), runs execute in a process
pool, each run writes its telemetry, and a summary table is written next to them. Pure functions of the
campaign file, the registry and the code version: re-running gives the same table."""

from __future__ import annotations

import itertools
import json
import tomllib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl


@dataclass
class Campaign:
    name: str
    scenarios: list[str]
    strategies: list[str]
    seeds: list[int]
    n: int = 300
    design: str = "seeds"  # seeds | grid | oat | lhs
    params: list[str] = field(default_factory=list)
    steps: int = 3
    samples: int = 16
    grid: dict[str, list[float]] = field(default_factory=dict)
    fixed: dict[str, float] = field(default_factory=dict)  # overrides applied to every run
    max_turns: int | None = None
    engine: str = "simulated"


def load_campaign(path: str | Path) -> Campaign:
    d = tomllib.loads(Path(path).read_text())
    c = d["campaign"]
    return Campaign(**c, grid=d.get("grid", {}), fixed=d.get("fixed", {}))


def _ranges() -> dict[str, tuple[float, float]]:
    from tle.registry.loader import load_registry

    reg = load_registry()
    return {k: tuple(s.range) for k, s in reg.specs.items() if getattr(s, "range", None)}


def expand(c: Campaign) -> list[dict]:
    """The list of run specs (scenario, strategy, seed, overrides, label)."""
    points: list[dict[str, float]] = [{}]
    if c.design == "grid":
        keys = list(c.grid)
        points = [dict(zip(keys, vals)) for vals in itertools.product(*(c.grid[k] for k in keys))]
    elif c.design in ("oat", "lhs"):
        rng_ = _ranges()
        if c.design == "oat":
            points = [{}]
            for k in c.params:
                lo, hi = rng_[k]
                points += [{k: float(v)} for v in np.linspace(lo, hi, c.steps)]
        else:
            r = np.random.default_rng(0)
            u = (
                np.argsort(r.random((c.samples, len(c.params))), axis=0)
                + r.random((c.samples, len(c.params)))
            ) / c.samples
            points = [
                {k: float(rng_[k][0] + u[i, j] * (rng_[k][1] - rng_[k][0])) for j, k in enumerate(c.params)}
                for i in range(c.samples)
            ]
    specs = []
    for pi, pt in enumerate(points):
        for sc, st, seed in itertools.product(c.scenarios, c.strategies, c.seeds):
            specs.append(
                {
                    "campaign": c.name,
                    "scenario": sc,
                    "strategy": st,
                    "seed": seed,
                    "n": c.n,
                    "overrides": {**c.fixed, **pt},
                    "point": pi,
                    "max_turns": c.max_turns,
                    "engine": c.engine,
                }
            )
    return specs


def run_one(spec: dict, out_dir: str | None = None) -> dict:
    """Run one game with a simulated player and return its per-run summary row."""
    from tle.catalogs.loader import load_catalogs
    from tle.engines.protocol import Engines
    from tle.engines.simulated import SimulatedEngine
    from tle.loop.game import Game, play
    from tle.loop.strategies import STRATEGIES
    from tle.params import default_params, to_flat
    from tle.telemetry import RunStore, TelemetryObserver

    cat = load_catalogs()
    p = default_params().with_overrides(spec["overrides"]) if spec["overrides"] else default_params()
    eng = Engines(SimulatedEngine(cat, p), SimulatedEngine(cat, p))
    obs = []
    store = None
    run_id = f"{spec['campaign']}-p{spec['point']:03d}-{spec['scenario']}-{spec['strategy']}-s{spec['seed']}"
    if out_dir:
        store = RunStore(
            Path(out_dir),
            run_id,
            {
                "scenario": spec["scenario"],
                "seed": spec["seed"],
                "label": spec["strategy"],
                "overrides": spec["overrides"],
                "n": spec["n"],
            },
            to_flat(p),
        )
        obs.append(TelemetryObserver(store))
    game = Game(cat, p, eng, scenario=spec["scenario"], seed=spec["seed"], n=spec["n"], observers=obs)
    w = play(game, STRATEGIES[spec["strategy"]], spec["max_turns"], obs)
    if store:
        store.close(w.outcome)
    levels = w.pillar_level.astype(int)
    fallen = sum(int(levels[i] >= len(game.cat.pillars[i].levels)) for i in range(len(levels)))
    return {
        "run_id": run_id,
        "campaign": spec["campaign"],
        "point": spec["point"],
        "scenario": spec["scenario"],
        "strategy": spec["strategy"],
        "seed": spec["seed"],
        "outcome": w.outcome or "unfinished",
        "days": float(w.day),
        "turns": int(w.turn),
        "pillars_fallen": fallen,
        "levels_total": int(levels.sum()),
        "capital": float(w.K),
        "digest": w.digest(),
        **{f"p_{k}": float(v) for k, v in spec["overrides"].items()},
    }


def _worker(args):
    return run_one(*args)


def run_campaign(
    c: Campaign, out_root: str | Path = "runs", workers: int | None = None, telemetry: bool = True
) -> pl.DataFrame:
    out = Path(out_root) / f"campaign-{c.name}"
    out.mkdir(parents=True, exist_ok=True)
    specs = expand(c)
    jobs = [(s, str(out / "runs") if telemetry else None) for s in specs]
    if workers == 1 or len(jobs) == 1:
        rows = [_worker(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_worker, jobs, chunksize=max(1, len(jobs) // 64)))
    df = pl.DataFrame(rows)
    df.write_csv(out / "summary.csv")
    (out / "campaign.json").write_text(
        json.dumps({"name": c.name, "design": c.design, "runs": len(rows)}, indent=2)
    )
    return df
