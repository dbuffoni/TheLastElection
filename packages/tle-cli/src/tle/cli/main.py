"""The ``tle`` command line: run games, campaigns, analyses and checks."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, help="The Last Election: simulation tooling.")
campaign_app = typer.Typer(no_args_is_help=True, help="Campaigns: expand, run, analyse.")
params_app = typer.Typer(no_args_is_help=True, help="Parameter registry.")
runs_app = typer.Typer(no_args_is_help=True, help="Inspect run directories.")
app.add_typer(campaign_app, name="campaign")
app.add_typer(params_app, name="params")
app.add_typer(runs_app, name="runs")
console = Console()


def _show(df, title: str = "", max_rows: int = 60) -> None:
    t = Table(title=title)
    for c in df.columns:
        t.add_column(c)
    for row in df.head(max_rows).iter_rows():
        t.add_row(*[f"{v:.3f}" if isinstance(v, float) else str(v) for v in row])
    console.print(t)


@app.command()
def run(
    strategy: str = "gradual",
    scenario: str = "industrial_secular",
    seed: int = 1,
    n: int = 300,
    runs_dir: Path = Path("runs"),
    telemetry: bool = True,
    engine: str = typer.Option("simulated", help="simulated | openai_compat"),
    base_url: str = typer.Option("http://localhost:8000/v1", help="openai_compat endpoint"),
    model: str = typer.Option("local-model", help="openai_compat model name"),
    cache: Path | None = typer.Option(None, help="response cache file for real engines"),
    override: list[str] = typer.Option([], "--set", help="parameter override name=value (repeatable)"),
):
    """Play one game with a simulated player and write its telemetry."""
    from tle.catalogs.loader import load_catalogs
    from tle.engines.protocol import Engines
    from tle.engines.simulated import SimulatedEngine
    from tle.loop.game import Game, play
    from tle.loop.strategies import STRATEGIES
    from tle.params import default_params, to_flat
    from tle.telemetry import RunStore, TelemetryObserver

    cat = load_catalogs()
    ov = {k: float(v) for k, v in (o.split("=") for o in override)}
    p = default_params().with_overrides(ov) if ov else default_params()
    sim = SimulatedEngine(cat, p)
    if engine == "simulated":
        eng = Engines(sim, SimulatedEngine(cat, p))
    else:
        from tle.models import CachedEngine, OpenAICompatBackend, RealEngine

        be = OpenAICompatBackend(base_url=base_url, model=model)
        real = RealEngine(be, cat.vocab(), fallback=sim, n_retry=p.technical.n_retry)
        eng = Engines(
            CachedEngine(real, cache) if cache else real, CachedEngine(real, cache) if cache else real
        )
    game = Game(cat, p, eng, scenario=scenario, seed=seed, n=n)
    obs, store = [], None
    if telemetry:
        run_id = f"{scenario}-{strategy}-s{seed}"
        store = RunStore(
            runs_dir,
            run_id,
            {
                "scenario": scenario,
                "seed": seed,
                "label": strategy,
                "overrides": ov,
                "n": n,
                "engine": engine,
            },
            to_flat(p),
        )
        obs.append(TelemetryObserver(store))
    w = play(game, STRATEGIES[strategy], None, obs)
    if store:
        store.close(w.outcome)
    console.print(
        f"[bold]{strategy}[/] on {scenario} seed {seed}: [bold]{w.outcome}[/] after {w.day:.0f} days, "
        f"pillars {w.pillar_level.tolist()}, digest {w.digest()[:12]}"
    )


@app.command()
def play(scenario: str = "industrial_secular", seed: int = 0, n: int = 200):
    """Play the game yourself in the terminal."""
    from tle.game import main

    main(scenario, seed, n)


@campaign_app.command("list")
def campaign_list():
    for f in sorted(Path("config/campaigns").glob("*.toml")):
        console.print(f.stem)


@campaign_app.command("expand")
def campaign_expand(name: str):
    """Show how many runs a campaign file expands to."""
    from tle.experiments import expand, load_campaign

    specs = expand(load_campaign(Path("config/campaigns") / f"{name}.toml"))
    console.print(f"{len(specs)} runs; first: {json.dumps(specs[0])}")


@campaign_app.command("run")
def campaign_run(
    name: str,
    runs_dir: Path = Path("runs"),
    workers: int = typer.Option(0, help="0 = all cores"),
    telemetry: bool = True,
    seeds: str = typer.Option("", help="override seeds, e.g. 1,2,3"),
    n: int = typer.Option(0, help="override population"),
):
    """Run a campaign in a process pool and print the summary and the band report."""
    from tle.analysis import summary
    from tle.balance import band_report
    from tle.experiments import load_campaign, run_campaign

    c = load_campaign(Path("config/campaigns") / f"{name}.toml")
    if seeds:
        c.seeds = [int(s) for s in seeds.split(",")]
    if n:
        c.n = n
    df = run_campaign(c, runs_dir, workers or None, telemetry)
    _show(summary(df), f"{name}: summary")
    rep = band_report(df)
    if len(rep):
        _show(rep, "target bands")


@campaign_app.command("analyze")
def campaign_analyze(
    name: str, analyzer: str = "summary", metric: str = "levels_total", runs_dir: Path = Path("runs")
):
    """Analyse a finished campaign: summary | dispersion | sensitivity | stability | dominance | bands."""
    import polars as pl
    from tle import analysis, balance

    df = pl.read_csv(runs_dir / f"campaign-{name}" / "summary.csv")
    fn = {
        "summary": analysis.summary,
        "stability": analysis.stability,
        "dominance": balance.dominance,
        "bands": balance.band_report,
        "dispersion": lambda d: analysis.dispersion(d, metric),
        "sensitivity": lambda d: analysis.sensitivity(d, metric),
    }[analyzer]
    _show(fn(df), f"{name}: {analyzer}")


@campaign_app.command("verify")
def campaign_verify(name: str, runs_dir: Path = Path("runs")):
    """Re-run a campaign and compare state digests with the stored summary (determinism check)."""
    import polars as pl
    from tle.analysis import verify_determinism
    from tle.experiments import load_campaign, run_campaign

    a = pl.read_csv(runs_dir / f"campaign-{name}" / "summary.csv")
    c = load_campaign(Path("config/campaigns") / f"{name}.toml")
    b = run_campaign(c, Path(runs_dir) / "_verify", telemetry=False)
    bad = verify_determinism(a, b)
    console.print(
        "[green]deterministic: all digests match[/]" if not len(bad) else f"[red]{len(bad)} runs differ[/]"
    )
    if len(bad):
        _show(bad)
        raise typer.Exit(1)


@params_app.command("list")
def params_list(group: str = ""):
    from tle.registry.loader import load_registry

    reg = load_registry()
    t = Table()
    for c in ("name", "group", "default", "range", "status", "description"):
        t.add_column(c)
    for k, s in reg.specs.items():
        if group and s.group != group:
            continue
        t.add_row(k, s.group, str(s.default), str(getattr(s, "range", "")), s.status, s.description[:60])
    console.print(t)


@params_app.command("codegen")
def params_codegen(check: bool = False):
    """Regenerate tle.params._generated from the TOML registry (or --check that it is current)."""
    from tle.registry import codegen

    if check:
        if not codegen.is_current():
            console.print("[red]_generated.py is stale: run `tle params codegen`[/]")
            raise typer.Exit(1)
        console.print("[green]generated parameters are current[/]")
    else:
        console.print(codegen.write())


@runs_app.command("list")
def runs_list(runs_dir: Path = Path("runs")):
    import polars as pl
    from tle.runs import list_runs

    rows = list_runs(runs_dir)
    _show(
        pl.DataFrame(rows).select(
            [
                c
                for c in ("run_id", "scenario", "label", "seed", "state", "turn", "day", "outcome")
                if rows and c in rows[0]
            ]
        )
        if rows
        else pl.DataFrame(),
        "runs",
    )


@runs_app.command("show")
def runs_show(
    run: str,
    table: str = "metrics",
    tail: int = 15,
    runs_dir: Path = Path("runs"),
    cols: str = "turn,day,capital,approval,fear,opposition,decree_result",
):
    from tle.runs import table as tbl

    df = tbl(run, table, runs_dir)
    keep = [c for c in cols.split(",") if c in df.columns] if table == "metrics" and cols else df.columns
    df = df.select(keep)
    _show(df.tail(tail), f"{run}/{table}")


@app.command()
def check():
    """Fast project health checks: generated params current, schemas strict, catalogs load, boundary rules."""
    from tle.devtools import run_checks

    ok = run_checks(console)
    raise typer.Exit(0 if ok else 1)


if __name__ == "__main__":
    app()
