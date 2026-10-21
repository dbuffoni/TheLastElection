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
pop_app = typer.Typer(
    no_args_is_help=True, help="Population monitoring: report, watch, stability, mobility, impulse."
)
metrics_app = typer.Typer(no_args_is_help=True, help="Metric registry (config/metrics).")
app.add_typer(runs_app, name="runs")
app.add_typer(pop_app, name="pop")
app.add_typer(metrics_app, name="metrics")
console = Console()


def _show(df, title: str = "", max_rows: int = 60) -> None:
    t = Table(title=title)
    for c in df.columns:
        t.add_column(c)
    for row in df.head(max_rows).iter_rows():
        t.add_row(*[f"{v:.3f}" if isinstance(v, float) else str(v) for v in row])
    console.print(t)


mocks_app = typer.Typer(help="Mocks of S1 and S2: fit against a real model, decree bank.")
app.add_typer(mocks_app, name="mocks")


@mocks_app.command("fit")
def mocks_fit(
    records: list[Path],
    out: Path = Path("config/mocks/s1_fit.toml"),
):
    """Fit lambda_HA, tau_sim and w_actor_stance of the simulated S1 on the choices recorded from a real S1."""
    from tle.mocks import fit, load_reactions, write_toml

    recs = load_reactions(records)
    if not recs:
        raise typer.BadParameter("no reaction records found (run with --record against a real S1)")
    res = fit(recs)
    write_toml(res, out)
    t = Table(title=f"fit on {res['n_choices']} real choices in {res['events']} events")
    for c in ("", "log-lik", "argmax agreement", "lambda_HA", "tau_sim", "w_actor_stance", "beta_support"):
        t.add_column(c)
    for k in ("before", "after"):
        r = res[k]
        t.add_row(
            k,
            f"{r['loglik']:.1f}",
            f"{r['agreement']:.0%}",
            *[f"{r['params'][x]:.3f}" for x in ("lambda_HA", "tau_sim", "w_actor_stance", "beta_support")],
        )
    console.print(t)
    console.print(f"uniform choice would score {res['uniform_loglik']:.1f}; written to {out}")


@mocks_app.command("eval")
def mocks_eval(bank: Path = Path("config/mocks/decree_bank.toml")):
    """Agreement of the simulated decree classifier with the real S1 labels of the bank (rules alone and with the bank)."""
    import tomllib

    from tle.mocks import evaluate_classifier

    entries = tomllib.loads(bank.read_text())["decrees"]
    t = Table(title=f"simulated classifier vs real S1 on {len(entries)} decrees")
    for c in ("mode", "pillar", "domain", "aggressiveness ±1", "transfer detected", "unclassifiable"):
        t.add_column(c)
    for name, kw in (
        ("rules only", dict(use_bank=False)),
        ("rules + bank (leave one out)", dict(use_bank=True)),
    ):
        r = evaluate_classifier(entries, **kw)
        t.add_row(
            name,
            f"{r['pillar']:.0%}",
            f"{r['domain']:.0%}",
            f"{r['aggressiveness_within_1']:.0%}",
            f"{r['transfer_detected']:.0%}",
            str(r["unclassifiable"]),
        )
    console.print(t)


@mocks_app.command("label-decrees")
def mocks_label(
    battery: Path = Path("config/mocks/decree_battery.txt"),
    engines_file: Path = Path("config/engines.openrouter.toml"),
    out: Path = Path("config/mocks/decree_bank.toml"),
):
    """Label the decrees of the battery with the real S1 and write the bank the simulated classifier learns from."""
    from tle.catalogs.loader import load_catalogs
    from tle.contracts import models as m
    from tle.core.rng import GameRng
    from tle.engines.protocol import call_many
    from tle.mocks import bank_entry, read_battery, write_bank
    from tle.models import build_engines, load_dotenv, load_specs
    from tle.params import default_params

    load_dotenv()
    cat, p = load_catalogs(), default_params()
    eng, info = build_engines(cat, p, load_specs(engines_file))
    texts = read_battery(battery)
    outs = call_many(
        eng.s1,
        "classify_decree",
        [m.ClassifyDecreeIn(text=t) for t in texts],
        [GameRng(i) for i in range(len(texts))],
    )
    entries = [bank_entry(t, o) for t, o in zip(texts, outs) if o is not None]
    write_bank(entries, out)
    console.print(f"{len(entries)}/{len(texts)} decrees labelled by {info['s1']} -> {out}")


@app.command("dash")
def dash(
    runs_dir: Path = Path("runs"),
    port: int = 5006,
    live: bool = typer.Option(
        False, "--live", help="watch runs/ and refresh every two seconds (open runs included)"
    ),
    show: bool = typer.Option(True, help="open the browser"),
):
    """Open the dashboard (Panel app over the run store). Needs the tle-dashboard package."""
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "tle.dashboard", "--runs-dir", str(runs_dir), "--port", str(port)]
    if live:
        cmd.append("--live")
    if not show:
        cmd.append("--no-show")
    console.print(f"dashboard on http://localhost:{port}  (Ctrl-C to stop)")
    raise typer.Exit(subprocess.call(cmd))


@app.command("plot")
def plot(
    run: str,
    metric: str,
    out: Path = Path("plots"),
    runs_dir: Path = Path("runs"),
):
    """Save a metric chart of one run as a PNG (target band, verdict frame, regeneration command in the footer)."""
    from tle.analysis import population as pop
    from tle.registry.metrics import load_metrics
    from tle.runs import Run
    from tle.viz import metric_png

    r = Run(runs_dir / run)
    spec = load_metrics().get(metric)
    ver = {x["metric"]: x["verdict"] for x in pop.evaluate_targets(r).iter_rows(named=True)}
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{run}-{metric}.png"
    cmd = f"uv run tle plot {run} {metric} --runs-dir {runs_dir}"
    console.print(metric_png(r.metrics().df, metric, path, spec, ver.get(metric), cmd))


@app.command()
def run(
    strategy: str = "gradual",
    scenario: str = "industrial_secular",
    seed: int = 1,
    n: int = 300,
    runs_dir: Path = Path("runs"),
    telemetry: bool = True,
    engine: str = typer.Option("simulated", help="simulated | openai_compat (both engines)"),
    base_url: str = typer.Option("http://localhost:8000/v1", help="openai_compat endpoint (both engines)"),
    model: str = typer.Option("local-model", help="openai_compat model name (both engines)"),
    engines_file: Path | None = typer.Option(
        None, "--engines", help="TOML with one table per engine, s1 and s2 (see config/engines.toml)"
    ),
    record: Path | None = typer.Option(
        None, help="write the S1 choices and decrees of a real model as JSON lines (mock fit)"
    ),
    warmup_real: bool = typer.Option(
        False, help="run the warm-up on the real models too (slow); default: simulated"
    ),
    max_turns: int | None = typer.Option(None, help="stop after this many turns (trials with real models)"),
    s1_engine: str | None = typer.Option(None, help="S1 kind: simulated | openai_compat | jev"),
    s1_base_url: str | None = typer.Option(None, help="S1 endpoint, e.g. a local Jev server"),
    s1_model: str | None = typer.Option(None, help="S1 model name"),
    s2_engine: str | None = typer.Option(None, help="S2 kind: simulated | openai_compat | jev"),
    s2_base_url: str | None = typer.Option(None, help="S2 endpoint"),
    s2_model: str | None = typer.Option(None, help="S2 model name"),
    cache: Path | None = typer.Option(None, help="response cache file for real engines"),
    report: bool = typer.Option(False, help="also write the HTML population report"),
    override: list[str] = typer.Option([], "--set", help="parameter override name=value (repeatable)"),
):
    """Play one game with a simulated player and write its telemetry."""
    from tle.catalogs.loader import load_catalogs
    from tle.loop.game import Game, play
    from tle.loop.strategies import STRATEGIES
    from tle.params import default_params, to_flat
    from tle.telemetry import RunStore, TelemetryObserver

    cat = load_catalogs()
    ov = {k: float(v) for k, v in (o.split("=") for o in override)}
    p = default_params().with_overrides(ov) if ov else default_params()
    from tle.engines.protocol import Engines
    from tle.engines.simulated import SimulatedEngine
    from tle.models import build_engines, load_dotenv, load_specs, with_overrides

    load_dotenv()
    specs = load_specs(engines_file)
    if engine != "simulated":  # legacy flags: one server for both engines
        specs = with_overrides(
            specs,
            s1_kind=engine,
            s2_kind=engine,
            s1_base_url=base_url,
            s2_base_url=base_url,
            s1_model=model,
            s2_model=model,
        )
    specs = with_overrides(
        specs,
        s1_kind=s1_engine,
        s1_base_url=s1_base_url,
        s1_model=s1_model,
        s2_kind=s2_engine,
        s2_base_url=s2_base_url,
        s2_model=s2_model,
    )
    eng, engine_info = build_engines(cat, p, specs, cache, record)
    engine = "simulated" if all(v["kind"] == "simulated" for v in engine_info.values()) else "real"
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
                "engines": engine_info,
            },
            to_flat(p),
        )
        obs.append(TelemetryObserver(store))
    warm = None
    if (
        engine != "simulated" and not warmup_real
    ):  # the warm-up only settles ties and opinions: simulated engines do
        warm = Engines(SimulatedEngine(cat, p), SimulatedEngine(cat, p))
    game = Game(
        cat, p, eng, scenario=scenario, seed=seed, n=n, observers=obs, warmup_engines=warm
    )  # observers also record the warm-up
    w = play(game, STRATEGIES[strategy], max_turns, obs)
    if store:
        store.close(w.outcome)
        if report:
            from tle.report import write_report
            from tle.runs import open_run

            console.print(
                f"population report: {write_report(open_run(store.dir), store.dir / 'population.html')}"
            )
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


@app.command("journal")
def journal_cmd(
    run: str,
    min_importance: int = typer.Option(2, help="0 to 3: show entries at least this important"),
    kind: str = typer.Option(
        "", help="comma-separated kinds: decree,message,event,pillar,threat,mood,outcome"
    ),
    find: str = typer.Option("", help="only entries containing this word"),
    tail: int = typer.Option(40, help="last N entries (0 for all)"),
    runs_dir: Path = Path("runs"),
):
    """Read the diary of a run: what happened, day by day."""
    from tle.runs import open_run

    df = open_run(run, runs_dir).journal(min_importance, [k for k in kind.split(",") if k])
    if find and len(df):
        import polars as pl

        f = find.lower()
        df = df.filter(
            pl.col("headline").str.to_lowercase().str.contains(f, literal=True)
            | pl.col("detail").str.to_lowercase().str.contains(f, literal=True)
        )
    if not len(df):
        typer.echo("no diary entries (older runs have none)")
        return
    for r in (df.tail(tail) if tail else df).iter_rows(named=True):
        typer.echo(f"day {r['day']:>6.0f}  {r['kind']:<8} {'*' * max(1, r['importance']):<3} {r['headline']}")
        if r["detail"]:
            typer.echo(f"{'':>20}{r['detail']}")


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


_RUNS = typer.Option(Path("runs"), "--runs-dir", help="folder with run directories")
SPARK = "▁▂▃▄▅▆▇█"


def _spark(vals, width: int = 28) -> str:
    v = [x for x in vals if x is not None and x == x]
    if len(v) < 2:
        return ""
    if len(v) > width:
        step = len(v) / width
        v = [v[int(i * step)] for i in range(width)]
    lo, hi = min(v), max(v)
    return "".join(SPARK[int((x - lo) / (hi - lo) * 7)] if hi > lo else SPARK[0] for x in v)


def _open(run: str, runs_dir: Path):
    from tle.runs import open_run

    return open_run(run, runs_dir)


@pop_app.command("report")
def pop_report(
    run: str,
    out: Path = typer.Option(None, help="output .html (default: <run>/population.html)"),
    runs_dir: Path = _RUNS,
):
    """Write a self-contained HTML report of the population's evolution (open it in any browser)."""
    from tle.report import write_report

    r = _open(run, runs_dir)
    path = write_report(r, out or r.path / "population.html")
    console.print(f"report written: {path}")


@pop_app.command("watch")
def pop_watch(
    run: str,
    follow: bool = typer.Option(False, "--follow", "-f", help="refresh while the run is open"),
    every: float = 2.0,
    group: str = typer.Option("", help="population|ideology|opinion|wealth|network"),
    runs_dir: Path = _RUNS,
):
    """Live table of the population metrics of a run (works on open runs): last value, trend, sparkline, target verdict."""
    import time

    from rich.live import Live
    from tle.analysis import population as pa
    from tle.registry.metrics import load_metrics

    specs = load_metrics()
    r = _open(run, runs_dir)

    def render():
        m = r.metrics().df
        st = r.status
        t = Table(
            title=f"{r.id}  ·  {st.get('state')}  ·  day {st.get('day', 0):.0f}  ·  turn {st.get('turn')}  ·  {st.get('outcome') or 'playing'}"
        )
        for c in ("metric", "last", "Δ30d", "trend", "target", "verdict"):
            t.add_column(c)
        if not len(m):
            return t
        ev = (
            {x["metric"]: x for x in pa.evaluate_targets(r, specs).iter_rows(named=True)}
            if st.get("state") == "finished" or len(m) > 40
            else {}
        )
        for mid, s in specs.items():
            if mid not in m.columns or (group and s.group != group):
                continue
            col = m[mid].to_list()
            last = col[-1]
            ago = m.filter(m["day"] <= m["day"][-1] - 30)[mid].to_list()
            d30 = (last - ago[-1]) if ago and last is not None and ago[-1] is not None else None
            e = ev.get(mid)
            t.add_row(
                mid,
                f"{last:.3f}" if last is not None else "–",
                f"{d30:+.3f}" if d30 is not None else "",
                _spark(col),
                e["expected"] if e else "",
                ("[green]ok[/]" if e["verdict"] == "ok" else "[red]miss[/]" if e["verdict"] == "miss" else "")
                if e
                else "",
            )
        chk = r.checks()
        if not chk.get("ok", True):
            t.caption = "[red]invariants violated: " + ", ".join(chk["first_violation_day"]) + "[/]"
        return t

    if not follow:
        console.print(render())
        return
    with Live(render(), console=console, refresh_per_second=2) as live:
        while True:
            live.update(render())
            if r.status.get("state") == "finished":
                break
            time.sleep(every)


@pop_app.command("stability")
def pop_stability(run: str, window: float = 180.0, runs_dir: Path = _RUNS):
    """Verdict per metric: stable | drifting | oscillating | degenerate (spec 8.5)."""
    from tle.analysis import population as pa

    _show(pa.stability_table(_open(run, runs_dir), window=window), "stability")


@pop_app.command("targets")
def pop_targets(run: str, runs_dir: Path = _RUNS):
    """Evaluate the registry targets (drift, bands, minimums) on one run."""
    from tle.analysis import population as pa

    _show(pa.evaluate_targets(_open(run, runs_dir)), "targets")


@pop_app.command("mobility")
def pop_mobility(run: str, day0: float = 0.0, day1: float = typer.Option(None), runs_dir: Path = _RUNS):
    """Mobility test of spec 17.3 between two panel days, with class flows."""
    from tle.analysis import population as pa

    r = _open(run, runs_dir)
    console.print(pa.mobility(r, day0, day1))
    _show(pa.class_flows(r, day0, day1), "class flows")


@pop_app.command("identity")
def pop_identity(run: str, runs_dir: Path = _RUNS):
    """Does the scenario survive the warm-up? (spec 17.5): dispersion, polarization and Sarle retained."""
    from tle.analysis import population as pa

    console.print(pa.scenario_identity(_open(run, runs_dir)))


@pop_app.command("panel")
def pop_panel(run: str, day: float = typer.Option(None), limit: int = 20, runs_dir: Path = _RUNS):
    """Print the citizen panel at the snapshot closest to a day (use `tle.runs` in Python for analysis)."""
    p = _open(run, runs_dir).panel(day)
    console.print(f"{len(p)} citizens at day {p['day'][0]:.0f}" if len(p) else "no panel")
    cols = ["id", "age", "cls", "wealth", "I1", "I2", "I3", "op_mayor", "ex_mayor", "fear"]
    _show(p.select([c for c in cols if c in p.columns]).head(limit), "citizen panel")


@pop_app.command("impulse")
def pop_impulse(
    scenario: str = "industrial_secular",
    seed: int = 1,
    n: int = 300,
    days: float = 180.0,
    share: float = 0.2,
    aggressiveness: str = "medium",
    override: list[str] = typer.Option([], "--set", help="parameter override name=value (repeatable)"),
):
    """Impulse test (spec 17.3): one standard decree on 20% of the adults against a control run from the same state."""
    from tle.catalogs.loader import load_catalogs
    from tle.experiments import impulse_test
    from tle.loop.game import Game
    from tle.params import default_params

    ov = {k: float(v) for k, v in (o.split("=") for o in override)}
    p = default_params().with_overrides(ov) if ov else default_params()
    g = Game(load_catalogs(), p, scenario=scenario, seed=seed, n=n)
    res = impulse_test(g, share, aggressiveness, days)
    t = Table(title=f"impulse test, {scenario}, seed {seed}")
    for c in ("measure", "observed", "target", "tolerance", "verdict"):
        t.add_column(c)
    for k, v in res.verdicts().items():
        t.add_row(
            k,
            f"{v['observed']:.3f}",
            f"{v['target']:.3f}",
            f"±{v['tolerance']:.0%}",
            "[green]ok[/]" if v["ok"] else "[red]miss[/]",
        )
    console.print(t)
    console.print(
        f"network steps reached: {res.hops_reached}   peak opinion shift by hop: "
        + ", ".join(f"{h}:{v:.3f}" for h, v in sorted(res.by_hop.items()))
    )


@metrics_app.command("list")
def metrics_list(group: str = ""):
    from tle.registry.metrics import load_metrics

    t = Table()
    for c in ("id", "group", "source", "stage", "target", "title"):
        t.add_column(c)
    for k, s in load_metrics().items():
        if group and s.group != group:
            continue
        t.add_row(k, s.group, s.source, str(s.stage), s.target.kind if s.target else "", s.title)
    console.print(t)


@metrics_app.command("lint")
def metrics_lint():
    """Fail if a metric is computed but not registered, registered but never written, or its target is malformed."""
    from tle.devtools import metrics_lint_errors

    errs = metrics_lint_errors()
    for e in errs:
        console.print(f"[red]FAIL[/] {e}")
    if not errs:
        console.print("[green]metric registry consistent with telemetry[/]")
    raise typer.Exit(1 if errs else 0)


if __name__ == "__main__":
    app()
