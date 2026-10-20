from __future__ import annotations

import html
from pathlib import Path

import panel as pn
import polars as pl
from tle.analysis import population as pop
from tle.registry.metrics import load_metrics
from tle.runs import Run, list_runs
from tle.viz import badge, matrix_html, metric_figure, sparkline_svg, table_html, target_band

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
        pn.Row(run_sel, group_sel, warm, day_sel, badge_pane),
        pn.bind(explorer, run_sel, group_sel, warm, day_sel),
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
        pn.bind(matrix),
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

    scorecard_view = pn.Column(run_sel, pn.bind(scorecard, run_sel))

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

    outcome_view = pn.Column(pn.bind(outcomes))

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

    health_view = pn.Column(run_sel, pn.bind(health, run_sel))

    tabs = pn.Tabs(
        ("Run explorer", explorer_view),
        ("Run matrix", matrix_view),
        ("Targets", scorecard_view),
        ("Outcomes", outcome_view),
        ("Technical health", health_view),
        dynamic=True,
    )
    app = pn.Column(pn.pane.Markdown("# The Last Election: dashboard"), tabs)

    if live:

        def tick():
            new = run_ids()
            if new != run_sel.options:
                run_sel.options = new
                if run_sel.value is None and new:
                    run_sel.value = new[-1]
            run_sel.param.trigger("value")  # re-reads the open streams
            badge_pane.object = status_html(load(run_sel.value))

        pn.state.add_periodic_callback(tick, LIVE_MS)
    return app
