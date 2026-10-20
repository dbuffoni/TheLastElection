from __future__ import annotations

import html
from pathlib import Path

import panel as pn
import polars as pl
from tle.analysis import population as pop
from tle.registry.metrics import load_metrics
from tle.runs import Run, list_runs
from tle.viz import (
    badge,
    bar_figure,
    curves_figure,
    histogram_figure,
    matrix_html,
    metric_figure,
    scatter_figure,
    sparkline_svg,
    table_html,
    target_band,
)

LIVE_MS = 2000
GROUPS = {
    "population": "Demography",
    "ideology": "Ideology and polarization",
    "opinion": "Opinion, fear and silence",
    "wealth": "Wealth and classes",
    "network": "Social network",
}
MATRIX_METRICS = [
    "pop_size",
    "polarization_mean",
    "saturated_character_share",
    "share_label_changed",
    "median_displacement",
    "degree_mean",
    "opinion_mean",
    "fear_mean",
]


def _verdicts(run: Run) -> dict[str, str]:
    try:
        ev = pop.evaluate_targets(run)
        return {r["metric"]: r["verdict"] for r in ev.iter_rows(named=True)}
    except Exception:
        return {}


def _milestones(run: Run) -> pl.DataFrame | None:
    from tle.runs import table

    df = table(run.path, "milestones")
    return df if len(df) and "what" in df.columns else None


def build_app(runs_dir: str | Path = "runs", live: bool = False):
    """The dashboard as a Panel layout; ``live`` re-reads the runs every two seconds (open streams included)."""
    pn.extension(sizing_mode="stretch_width")
    root = Path(runs_dir)
    specs = load_metrics()

    def run_ids() -> list[str]:
        return [r["run_id"] for r in list_runs(root)] if root.exists() else []

    ids = run_ids()
    run_sel = pn.widgets.Select(label="Run", options=ids, value=ids[-1] if ids else None)
    group_sel = pn.widgets.Select(label="Group", options={v: k for k, v in GROUPS.items()}, value="ideology")
    warm = pn.widgets.Checkbox(label="include warm-up", value=False)
    day_sel = pn.widgets.FloatInput(label="Inspect day", value=0.0, step=7.0, width=140)
    badge_pane = pn.pane.HTML("", height=40)
    tick = pn.widgets.IntInput(
        label="tick", value=0, visible=False
    )  # bumped by the live loop and the refresh button
    refresh = pn.widgets.Button(label="Refresh", button_type="default", width=90)
    refresh.on_click(lambda _e: setattr(tick, "value", tick.value + 1))

    def load(run_id: str | None) -> Run | None:
        return Run(root / run_id) if run_id and (root / run_id).exists() else None

    def status_html(run: Run | None) -> str:
        if run is None:
            return "<i>no runs yet</i>"
        try:
            st = run.status
            return (
                f"{badge(st.get('state', '?'), 'ok' if st.get('state') == 'finished' else 'drift')} "
                f"day {st.get('day', 0):.0f} &middot; outcome {html.escape(str(st.get('outcome') or '-'))}"
            )
        except Exception:
            return "<i>status unavailable</i>"

    # ------------------------------------------------------------------ run explorer
    def explorer(run_id, group, include_warm, day, _tick=0):
        run = load(run_id)
        if run is None:
            return pn.pane.HTML("<p>No run selected. Start one with <code>tle run</code>.</p>")
        badge_pane.object = status_html(run)
        m = run.metrics(phase=None if include_warm else "game").df
        ver = _verdicts(run)
        ms = _milestones(run)
        figs = [
            metric_figure(m, mid, s, ver.get(mid), ms)
            for mid, s in specs.items()
            if s.group == group and mid in m.columns
        ]
        grid = pn.GridBox(*[pn.pane.Bokeh(f, sizing_mode="fixed") for f in figs], ncols=3)
        from tle.runs import table

        ev = table(run.path, "events")
        if len(ev) and "day" in ev.columns:
            ev = ev.filter((pl.col("day") >= day - 7) & (pl.col("day") <= day + 7))
        ev = ev.select(
            [c for c in ("day", "type", "title", "intensity", "actor", "involved") if c in ev.columns]
        )
        return pn.Column(
            grid, pn.pane.HTML(f"<h4>Events around day {day:.0f} (±7)</h4>"), pn.pane.HTML(table_html(ev, 60))
        )

    explorer_view = pn.Column(
        pn.Row(run_sel, group_sel, warm, day_sel, refresh, badge_pane),
        pn.bind(explorer, run_sel, group_sel, warm, day_sel, tick),
    )

    # ------------------------------------------------------------------ run matrix
    def matrix(_tick=0):
        runs = run_ids()
        if not runs:
            return pn.pane.HTML("<p>No runs.</p>")
        cache: dict[str, tuple] = {}
        for rid in runs[-40:]:
            r = load(rid)
            try:
                cache[rid] = (r.metrics().df, _verdicts(r))
            except Exception:
                cache[rid] = (pl.DataFrame(), {})

        def cell(rid, mid):
            df, ver = cache[rid]
            if not len(df) or mid not in df.columns:
                return None
            s = specs.get(mid)
            lo, hi = target_band(s.target if s else None)
            last = df[mid].drop_nulls()
            val = f"{last[-1]:.3g}" if len(last) else "-"
            return {
                "html": sparkline_svg(df["day"].to_list(), df[mid].to_list(), (lo, hi))
                + f"<br><small>{val}</small>",
                "verdict": ver.get(mid) if ver.get(mid) not in (None, "n/a") else None,
            }

        return pn.pane.HTML(matrix_html(list(cache), MATRIX_METRICS, cell, "run"), height=700)

    matrix_view = pn.Column(
        pn.pane.Markdown(
            "Rows are runs, columns metrics; each cell shows the trajectory, the target band and a verdict frame."
        ),
        pn.bind(matrix, tick),
    )

    # ------------------------------------------------------------------ targets
    def scorecard(run_id, _tick=0):
        run = load(run_id)
        if run is None:
            return pn.pane.HTML("<p>No run selected.</p>")
        ev = pop.evaluate_targets(run)
        return pn.pane.HTML(
            table_html(ev.select("metric", "kind", "observed", "expected", "verdict", "note"), 100, "verdict")
        )

    scorecard_view = pn.Column(run_sel, pn.bind(scorecard, run_sel, tick))

    # ------------------------------------------------------------------ campaign / outcomes
    def outcomes(_tick=0):
        rows = list_runs(root) if root.exists() else []
        if not rows:
            return pn.pane.HTML("<p>No runs.</p>")
        df = pl.DataFrame(
            [
                {
                    "strategy": r.get("label"),
                    "scenario": r.get("scenario"),
                    "outcome": r.get("outcome") or "running",
                    "day": r.get("day"),
                }
                for r in rows
            ]
        )
        agg = (
            df.group_by("scenario", "strategy")
            .agg(
                pl.len().alias("runs"),
                (pl.col("outcome") == "victory").mean().alias("victory"),
                (pl.col("outcome") == "removed").mean().alias("removed"),
                (pl.col("outcome") == "term_end").mean().alias("term_end"),
                pl.col("day").median().alias("median_day"),
            )
            .sort("scenario", "strategy")
        )
        return pn.pane.HTML(table_html(agg))

    outcome_view = pn.Column(refresh, pn.bind(outcomes, tick))

    # ------------------------------------------------------------------ technical health
    def health(run_id, _tick=0):
        run = load(run_id)
        if run is None:
            return pn.pane.HTML("<p>No run selected.</p>")
        from tle.runs import table

        ex = table(run.path, "execution")
        parts = ["<h4>Execution per task</h4>", table_html(ex.tail(60))]
        w = run.path / "warnings.jsonl"
        if w.exists():
            parts += [
                "<h4>Warnings (malformed answers)</h4>",
                f"<pre>{html.escape(w.read_text()[-4000:])}</pre>",
            ]
        return pn.pane.HTML("".join(parts))

    health_view = pn.Column(run_sel, pn.bind(health, run_sel, tick))

    # ------------------------------------------------------------------ citizen panel
    snap_sel = pn.widgets.Select(label="Snapshot day", options=[], width=160)

    def panel_days(run_id, _tick=0):
        run = load(run_id)
        days = run.panel_days() if run else []
        snap_sel.options = days
        if days and snap_sel.value not in days:
            snap_sel.value = days[len(days) // 2]
        return pn.pane.HTML(f"<i>{len(days)} snapshots</i>")

    def citizens(run_id, day, _tick=0):
        run = load(run_id)
        if run is None or day is None:
            return pn.pane.HTML("<p>No panel snapshots (the run has no citizens table yet).</p>")
        p = run.panel(day)
        p = p.filter(pl.col("age") >= 18)
        first = run.panel(run.panel_days()[0]).filter(pl.col("age") >= 18)
        figs = [
            scatter_figure(p, "I1", "I2", f"ideology, day {day:.0f}"),
            histogram_figure(p["I1"].to_list(), "axis 1 (economic), now"),
            histogram_figure(first["I1"].to_list(), "axis 1, first snapshot"),
            histogram_figure(p["wealth"].to_list(), "wealth", lo=0.0, hi=1.0),
        ]
        try:
            mob = pop.mobility(run, day0=run.panel_days()[0], day1=day)
            flows = pop.class_flows(run, day0=run.panel_days()[0], day1=day)
        except Exception:
            mob, flows = {}, pl.DataFrame()
        mob_df = pl.DataFrame([{k: v for k, v in mob.items()}]) if mob else pl.DataFrame()
        return pn.Column(
            pn.GridBox(*[pn.pane.Bokeh(f, sizing_mode="fixed") for f in figs], ncols=2),
            pn.pane.HTML(
                "<h4>Mobility since the first snapshot</h4>"
                + table_html(mob_df)
                + "<h4>Class flows</h4>"
                + table_html(flows)
            ),
        )

    citizen_view = pn.Column(
        pn.Row(run_sel, snap_sel, pn.bind(panel_days, run_sel, tick)),
        pn.bind(citizens, run_sel, snap_sel, tick),
    )

    # ------------------------------------------------------------------ network
    def network(run_id, day, _tick=0):
        run = load(run_id)
        if run is None:
            return pn.pane.HTML("<p>No run selected.</p>")
        m = run.metrics().df
        deg = pop.degree_distribution(run, day) if day is not None else pl.DataFrame()
        figs = []
        if len(deg):
            figs.append(bar_figure(deg["degree"].to_list(), deg["citizens"].to_list(), "degree distribution"))
        for mid in ("degree_mean", "stable_tie_share", "tie_weight_mean"):
            if mid in m.columns:
                figs.append(metric_figure(m, mid, specs.get(mid), _verdicts(run).get(mid)))
        mix = [c for c in m.columns if c.startswith("tie_share_")]
        if mix:
            figs.append(
                curves_figure(
                    m["day"].to_list(),
                    {c.removeprefix("tie_share_"): m[c].to_list() for c in mix},
                    title="tie type mix",
                )
            )
        return pn.GridBox(*[pn.pane.Bokeh(f, sizing_mode="fixed") for f in figs], ncols=3)

    network_view = pn.Column(pn.Row(run_sel, snap_sel), pn.bind(network, run_sel, snap_sel, tick))

    # ------------------------------------------------------------------ impulse view (17.3)
    imp_seed = pn.widgets.IntInput(label="Seed", value=1, width=80)
    imp_agg = pn.widgets.Select(
        label="Aggressiveness", options=["low", "medium", "high"], value="medium", width=140
    )
    imp_btn = pn.widgets.Button(label="Run impulse test", button_type="primary", width=150)
    imp_out = pn.Column(
        pn.pane.HTML(
            "<p>Press the button: one standard decree on 20% of the adults against a control copy (about a minute).</p>"
        )
    )

    def run_impulse(_e):
        from tle.catalogs.loader import load_catalogs
        from tle.experiments import impulse_test
        from tle.loop.game import Game

        imp_out[:] = [pn.pane.HTML("<p>running...</p>")]
        res = impulse_test(
            Game(load_catalogs(), scenario="industrial_secular", seed=imp_seed.value, n=300),
            0.2,
            imp_agg.value,
            180.0,
        )
        fig = curves_figure(
            res.days,
            {"involved": res.involved, "others": res.others},
            {
                "half-life": res.half_life_days + res.days[res.involved.index(max(res.involved))]
                if res.half_life_days == res.half_life_days
                else float("nan")
            },
            "opinion shift, treated minus control",
        )
        rows = pl.DataFrame(
            [
                {
                    "measure": k,
                    "observed": v["observed"],
                    "target": v["target"],
                    "tolerance": v["tolerance"],
                    "verdict": "ok" if v["ok"] else "miss",
                }
                for k, v in res.verdicts().items()
            ]
        )
        imp_out[:] = [pn.pane.Bokeh(fig, sizing_mode="fixed"), pn.pane.HTML(table_html(rows, 10, "verdict"))]

    imp_btn.on_click(run_impulse)
    impulse_view = pn.Column(pn.Row(imp_seed, imp_agg, imp_btn), imp_out)

    # ------------------------------------------------------------------ calibration status
    def calibration(_tick=0):
        runs = run_ids()[-12:]
        if not runs:
            return pn.pane.HTML("<p>No runs.</p>")
        ev = {}
        for rid in runs:
            r = load(rid)
            try:
                ev[rid] = {x["metric"]: x for x in pop.evaluate_targets(r).iter_rows(named=True)}
            except Exception:
                ev[rid] = {}
        mets = [m for m, s in specs.items() if s.target is not None]

        def cell(mid, rid):
            x = ev.get(rid, {}).get(mid)
            if x is None:
                return None
            v = x["observed"]
            return {"html": badge("n/a" if v is None else f"{v:.3g}", x["verdict"]), "verdict": None}

        return pn.pane.HTML(matrix_html(mets, runs, cell, "target metric"), height=800)

    calibration_view = pn.Column(
        pn.pane.Markdown(
            "Rows are the metrics with a target, columns the latest runs; the badge is the observed value."
        ),
        refresh,
        pn.bind(calibration, tick),
    )

    tabs = pn.Tabs(
        ("Run explorer", explorer_view),
        ("Run matrix", matrix_view),
        ("Targets", scorecard_view),
        ("Outcomes", outcome_view),
        ("Citizen panel", citizen_view),
        ("Network", network_view),
        ("Impulse", impulse_view),
        ("Calibration status", calibration_view),
        ("Technical health", health_view),
        dynamic=True,
    )
    app = pn.Column(pn.pane.Markdown("# The Last Election: dashboard"), tabs)

    if live:

        def live_tick():
            new = run_ids()
            if new != run_sel.options:
                run_sel.options = new
                if run_sel.value is None and new:
                    run_sel.value = new[-1]
            tick.value += 1  # re-reads the open streams

        pn.state.add_periodic_callback(live_tick, LIVE_MS)
    return app
