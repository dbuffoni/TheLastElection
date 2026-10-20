"""Renderers shared by the dashboard and the PNG export (spec 10): every chart carries the target band, the milestone
annotations and a verdict colour; matrices are built by one generic function. They draw result objects (Polars frames,
target tables) and compute nothing of their own."""

from __future__ import annotations

import html
import math
from collections.abc import Callable, Sequence

import polars as pl
from bokeh.models import BoxAnnotation, Span
from bokeh.plotting import figure

VERDICT_COLORS = {"ok": "#2a9d5c", "drift": "#e0a100", "miss": "#d1495b", "n/a": "#9aa0a6"}
LINE = "#3b6ea5"


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
):
    """Time series of one metric with its target band, milestone lines and a verdict-coloured frame."""
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
    return p


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
