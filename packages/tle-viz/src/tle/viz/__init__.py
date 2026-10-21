"""Renderers shared by the dashboard and the PNG export (spec 10): every chart carries the target band, the milestone
annotations and a verdict colour; matrices are built by one generic function. They draw result objects (Polars frames,
target tables) and compute nothing of their own."""

from __future__ import annotations

import html
import math
from collections.abc import Callable, Sequence

import numpy as np
import polars as pl
from bokeh.models import BoxAnnotation, ColumnDataSource, HoverTool, Span
from bokeh.plotting import figure

VERDICT_COLORS = {"ok": "#2a9d5c", "drift": "#e0a100", "miss": "#d1495b", "n/a": "#9aa0a6"}
LINE = "#3b6ea5"
JOURNAL_COLORS = {
    "decree": "#7b3fa0",
    "message": "#b07cc6",
    "event": "#3b6ea5",
    "pillar": "#d1495b",
    "threat": "#e0701a",
    "mood": "#6b7280",
    "outcome": "#111111",
}


def verdict_color(v: str | None) -> str:
    return VERDICT_COLORS.get(v or "n/a", VERDICT_COLORS["n/a"])


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def target_band(target) -> tuple[float | None, float | None]:
    """(low, high) of the area to colour for a registry target, None where open or not an area."""
    if target is None:
        return None, None
    if target.kind == "band":
        return target.lo, target.hi
    if target.kind == "min":
        return target.value, None
    if target.kind == "max":
        return None, target.value
    return None, None


def metric_figure(
    metrics: pl.DataFrame,
    metric: str,
    spec=None,
    verdict: str | None = None,
    milestones: pl.DataFrame | None = None,
    width: int = 420,
    height: int = 230,
    journal: pl.DataFrame | None = None,
):
    """Time series of one metric with its target band, milestone lines and a verdict-coloured frame. ``journal`` (diary
    entries) adds a dot per entry on the curve, coloured by kind, with the entry text on hover."""
    color = verdict_color(verdict)
    title = (spec.title if spec is not None else metric)[:60]
    p = figure(
        width=width,
        height=height,
        title=title,
        x_axis_label="day",
        tools="pan,wheel_zoom,box_zoom,reset,hover",
        tooltips=[("day", "@x{0}"), (metric, "@y{0.000}")],
        outline_line_color=color,
        outline_line_width=3 if verdict else 1,
        toolbar_location="above",
    )
    p.title.text_font_size = "12px"
    if metric in metrics.columns and len(metrics):
        d = metrics["day"].to_list()
        y = [v if _finite(v) else float("nan") for v in metrics[metric].to_list()]
        p.line(d, y, line_width=2, color=LINE)
        lo, hi = target_band(spec.target if spec is not None else None)
        if lo is not None or hi is not None:
            p.add_layout(BoxAnnotation(bottom=lo, top=hi, fill_color=VERDICT_COLORS["ok"], fill_alpha=0.12))
        after = getattr(getattr(spec, "target", None), "after_days", 0) if spec is not None else 0
        if after:
            p.add_layout(Span(location=after, dimension="height", line_color="#888", line_dash="dotted"))
    if milestones is not None and len(milestones):
        for day, what in zip(milestones["day"].to_list(), milestones["what"].to_list()):
            col = "#d1495b" if what in ("removed", "motion_vote") else "#8e7dbe"
            p.add_layout(Span(location=day, dimension="height", line_color=col, line_alpha=0.5, line_width=1))
    if journal is not None and len(journal) and metric in metrics.columns and len(metrics):
        journal_markers(p, metrics["day"].to_list(), metrics[metric].to_list(), journal)
    return p


def journal_markers(p, days: Sequence[float], values: Sequence[float], journal: pl.DataFrame) -> None:
    """Dots on a curve at the day of each diary entry (value interpolated), with a hover that reads the entry."""
    pts = [(d, v) for d, v in zip(days, values) if _finite(v)]
    if len(pts) < 2:
        return
    xs, ys = zip(*pts)
    j = journal.filter((pl.col("day") >= xs[0]) & (pl.col("day") <= xs[-1]))
    if not len(j):
        return
    day = j["day"].to_list()
    src = ColumnDataSource(
        {
            "day": day,
            "y": [float(np.interp(d, xs, ys)) for d in day],
            "kind": j["kind"].to_list(),
            "headline": [h[:90] for h in j["headline"].to_list()],
            "color": [JOURNAL_COLORS.get(k, "#888") for k in j["kind"].to_list()],
            "size": [4 + 2 * i for i in j["importance"].to_list()],
        }
    )
    r = p.scatter(
        "day", "y", source=src, size="size", fill_color="color", line_color="white", line_width=0.5, alpha=0.9
    )
    p.add_tools(
        HoverTool(renderers=[r], tooltips=[("day", "@day{0}"), ("kind", "@kind"), ("diary", "@headline")])
    )


def journal_timeline_figure(journal: pl.DataFrame, width: int = 900, height: int = 260):
    """One row per entry kind, one dot per entry (size = importance), hover shows the text: the shape of the story."""
    kinds = [k for k in JOURNAL_COLORS if len(journal) and k in journal["kind"].to_list()] or list(
        JOURNAL_COLORS
    )
    p = figure(
        width=width,
        height=height,
        y_range=list(reversed(kinds)),
        x_axis_label="day",
        tools="pan,wheel_zoom,box_zoom,reset",
        toolbar_location="above",
        title="Diary timeline",
    )
    p.title.text_font_size = "12px"
    if len(journal):
        src = ColumnDataSource(
            {
                "day": journal["day"].to_list(),
                "kind": journal["kind"].to_list(),
                "headline": [h[:90] for h in journal["headline"].to_list()],
                "color": [JOURNAL_COLORS.get(k, "#888") for k in journal["kind"].to_list()],
                "size": [4 + 2 * i for i in journal["importance"].to_list()],
            }
        )
        r = p.scatter(
            "day", "kind", source=src, size="size", fill_color="color", line_color="white", alpha=0.85
        )
        p.add_tools(
            HoverTool(renderers=[r], tooltips=[("day", "@day{0}"), ("kind", "@kind"), ("diary", "@headline")])
        )
    return p


def journal_table_html(journal: pl.DataFrame, max_rows: int = 300) -> str:
    """The diary as a readable table: day, kind, weight, headline with its narration underneath."""
    if not len(journal):
        return f"{_CSS}<p><i>no entries with these filters</i></p>"
    rows = []
    for r in journal.sort("day").head(max_rows).iter_rows(named=True):
        color = JOURNAL_COLORS.get(r["kind"], "#888")
        rows.append(
            f'<tr><td>{r["day"]:.0f}</td><td><span class="tle-b" style="background:{color}">{html.escape(r["kind"])}</span></td>'
            f"<td>{'&#9679;' * max(1, r['importance'])}</td><td><b>{html.escape(r['headline'])}</b>"
            f"<br><small>{html.escape(r['detail'])}</small></td></tr>"
        )
    return f'{_CSS}<table class="tle-t"><tr><th>day</th><th>kind</th><th>weight</th><th>entry</th></tr>{"".join(rows)}</table>'


def sparkline_svg(
    days: Sequence[float], values: Sequence[float], band=(None, None), color: str = LINE, w=110, h=28
) -> str:
    pts = [(d, v) for d, v in zip(days, values) if _finite(v)]
    if len(pts) < 2:
        return f'<svg width="{w}" height="{h}"></svg>'
    x0, x1 = pts[0][0], pts[-1][0]
    ys = [v for _, v in pts] + [b for b in band if _finite(b)]
    y0, y1 = min(ys), max(ys)
    sx = lambda x: 2 + (w - 4) * (x - x0) / ((x1 - x0) or 1)  # noqa: E731
    sy = lambda y: h - 2 - (h - 4) * (y - y0) / ((y1 - y0) or 1)  # noqa: E731
    path = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts[:: max(1, len(pts) // 60)])
    area = ""
    if _finite(band[0]) or _finite(band[1]):
        top = sy(band[1]) if _finite(band[1]) else 0
        bot = sy(band[0]) if _finite(band[0]) else h
        area = f'<rect x="0" y="{top:.1f}" width="{w}" height="{max(0.0, bot - top):.1f}" fill="#2a9d5c" opacity="0.15"/>'
    return f'<svg width="{w}" height="{h}">{area}<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{path}"/></svg>'


_CSS = (
    "<style>.tle-t{border-collapse:collapse;font:12px system-ui,sans-serif}.tle-t th,.tle-t td{border:1px solid #d0d4da;"
    "padding:3px 6px;text-align:left;vertical-align:middle}.tle-t th{background:#f2f4f7;position:sticky;top:0}"
    ".tle-t td.c{text-align:center}.tle-b{display:inline-block;padding:1px 7px;border-radius:9px;color:#fff;font-size:11px}</style>"
)


def badge(text: str, verdict: str | None) -> str:
    return f'<span class="tle-b" style="background:{verdict_color(verdict)}">{html.escape(str(text))}</span>'


def matrix_html(
    rows: Sequence[str], cols: Sequence[str], cell: Callable[[str, str], dict | None], corner: str = ""
) -> str:
    """Generic matrix (spec 10.4): ``cell(row, col)`` returns ``{"html": ..., "verdict": ...}`` or None."""
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = []
    for r in rows:
        tds = []
        for c in cols:
            x = cell(r, c)
            if x is None:
                tds.append('<td class="c" style="background:#fafafa"></td>')
            else:
                border = (
                    f"box-shadow: inset 0 0 0 2px {verdict_color(x.get('verdict'))};"
                    if x.get("verdict")
                    else ""
                )
                tds.append(f'<td class="c" style="{border}">{x.get("html", "")}</td>')
        body.append(f"<tr><th>{html.escape(r)}</th>{''.join(tds)}</tr>")
    return f'{_CSS}<table class="tle-t"><tr><th>{html.escape(corner)}</th>{head}</tr>{"".join(body)}</table>'


def table_html(df: pl.DataFrame, max_rows: int = 200, verdict_col: str | None = None) -> str:
    if not len(df):
        return f"{_CSS}<p><i>no data</i></p>"
    head = "".join(f"<th>{html.escape(c)}</th>" for c in df.columns)
    rows = []
    for r in df.head(max_rows).iter_rows(named=True):
        tds = []
        for c in df.columns:
            v = r[c]
            if c == verdict_col:
                tds.append(f"<td>{badge(v, v)}</td>")
            else:
                s = f"{v:.4g}" if isinstance(v, float) else ("" if v is None else str(v))
                tds.append(f"<td>{html.escape(s)}</td>")
        rows.append(f"<tr>{''.join(tds)}</tr>")
    return f'{_CSS}<table class="tle-t"><tr>{head}</tr>{"".join(rows)}</table>'


CLASS_COLORS = ["#3b6ea5", "#e0a100", "#d1495b"]  # low, middle, high


def scatter_figure(panel: pl.DataFrame, x: str = "I1", y: str = "I2", title: str = "", width=420, height=380):
    """Ideology scatter of one snapshot, coloured by class (citizen panel, spec 10.3)."""
    p = figure(
        width=width,
        height=height,
        title=title,
        x_range=(-1.05, 1.05),
        y_range=(-1.05, 1.05),
        x_axis_label=x,
        y_axis_label=y,
    )
    for c, col in enumerate(CLASS_COLORS):
        d = panel.filter(pl.col("cls") == c)
        if len(d):
            p.scatter(
                d[x].to_list(),
                d[y].to_list(),
                size=5,
                color=col,
                alpha=0.6,
                legend_label=["low", "middle", "high"][c],
            )
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    return p


def histogram_figure(
    values: Sequence[float],
    title: str = "",
    bins: int = 24,
    lo: float = -1.0,
    hi: float = 1.0,
    width=420,
    height=230,
):
    import numpy as np

    h, e = np.histogram([v for v in values if _finite(v)], bins=bins, range=(lo, hi))
    p = figure(width=width, height=height, title=title, tools="pan,wheel_zoom,reset")
    p.quad(
        top=h.tolist(),
        bottom=0,
        left=e[:-1].tolist(),
        right=e[1:].tolist(),
        fill_color=LINE,
        line_color="white",
        alpha=0.85,
    )
    return p


def bar_figure(labels: Sequence, values: Sequence[float], title: str = "", width=420, height=230):
    cats = [str(x) for x in labels]
    p = figure(x_range=cats, width=width, height=height, title=title, tools="")
    p.vbar(x=cats, top=list(values), width=0.8, color=LINE)
    return p


def curves_figure(
    days: Sequence[float],
    series: dict[str, Sequence[float]],
    marks: dict[str, float] | None = None,
    title: str = "",
    width=520,
    height=300,
):
    """Several curves over days with vertical marks (e.g. the half-life of the impulse test)."""
    p = figure(width=width, height=height, title=title, x_axis_label="day since the impulse")
    palette = ["#3b6ea5", "#d1495b", "#2a9d5c", "#8e7dbe"]
    for (name, ys), col in zip(series.items(), palette):
        p.line(
            list(days),
            [v if _finite(v) else float("nan") for v in ys],
            color=col,
            line_width=2,
            legend_label=name,
        )
    for name, x in (marks or {}).items():
        if _finite(x):
            p.add_layout(Span(location=x, dimension="height", line_color="#888", line_dash="dashed"))
    p.legend.location = "top_right"
    return p


def metric_png(
    metrics: pl.DataFrame, metric: str, path, spec=None, verdict: str | None = None, command: str = ""
) -> str:
    """The same chart as ``metric_figure`` as a PNG, with its regeneration command in the footer (spec 10.4)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=130)
    if metric in metrics.columns:
        ax.plot(metrics["day"].to_list(), metrics[metric].to_list(), color=LINE, lw=1.8)
    lo, hi = target_band(spec.target if spec is not None else None)
    if lo is not None or hi is not None:
        ax.axhspan(
            lo if lo is not None else ax.get_ylim()[0],
            hi if hi is not None else ax.get_ylim()[1],
            color=VERDICT_COLORS["ok"],
            alpha=0.12,
        )
    ax.set_title(spec.title if spec is not None else metric, fontsize=10)
    ax.set_xlabel("day")
    for s in ax.spines.values():
        s.set_color(verdict_color(verdict))
        s.set_linewidth(2.2 if verdict else 0.8)
    if command:
        fig.text(0.01, 0.01, command, fontsize=6, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return str(path)
