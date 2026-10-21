from bokeh.models import BoxAnnotation
from tle.loop.game import play
from tle.loop.strategies import STRATEGIES
from tle.registry.metrics import load_metrics
from tle.runs import Run
from tle.telemetry import RunStore, TelemetryObserver
from tle.testing import small_game
from tle.viz import matrix_html, metric_figure, sparkline_svg, table_html, target_band, verdict_color


def _run(tmp_path):
    g = small_game(3)
    store = RunStore(
        tmp_path, "r1", {"scenario": "industrial_secular", "seed": 3, "label": "random"}, {"a": 1}
    )
    play(g, STRATEGIES["random"], 12, [TelemetryObserver(store)])
    store.close()
    return Run(tmp_path / "r1")


def test_metric_figure_draws_the_target_band_and_verdict_frame(tmp_path):
    run = _run(tmp_path)
    specs = load_metrics()
    s = specs["degree_mean"] if specs["degree_mean"].target else next(x for x in specs.values() if x.target)
    fig = metric_figure(run.metrics().df, s.id, s, "miss")
    assert fig.outline_line_color == verdict_color("miss")
    lo, hi = target_band(s.target)
    has_band = lo is not None or hi is not None
    assert has_band == any(isinstance(r, BoxAnnotation) for r in fig.center)


def test_generic_matrix_and_tables_render_html():
    html = matrix_html(
        ["a", "b"], ["x", "y"], lambda r, c: {"html": f"{r}{c}", "verdict": "ok"} if r == "a" else None, "run"
    )
    assert "<table" in html and "ax" in html and verdict_color("ok") in html
    assert "polyline" in sparkline_svg([0, 1, 2], [1.0, 2.0, 1.5], (1.0, 2.0))
    import polars as pl

    assert "miss" in table_html(pl.DataFrame({"metric": ["m"], "verdict": ["miss"]}), verdict_col="verdict")


def test_dashboard_app_builds_over_a_run_directory(tmp_path):
    from tle.dashboard import build_app

    _run(tmp_path)
    app = build_app(tmp_path, live=False)
    tabs = app[1]
    assert list(tabs._names) == [
        "Run explorer",
        "Diary",
        "Run matrix",
        "Targets",
        "Outcomes",
        "Citizen panel",
        "Network",
        "Impulse",
        "Calibration status",
        "Technical health",
    ]
    for tab in tabs:  # every view renders over a real run without error
        tab.get_root()
    empty = build_app(tmp_path / "none")  # no runs yet: the app still opens
    empty.get_root()


def test_png_export_and_plot_command(tmp_path):
    from tle.cli.main import app
    from typer.testing import CliRunner

    _run(tmp_path)
    res = CliRunner().invoke(
        app, ["plot", "r1", "pop_size", "--runs-dir", str(tmp_path), "--out", str(tmp_path / "p")]
    )
    assert res.exit_code == 0, res.output
    png = tmp_path / "p" / "r1-pop_size.png"
    assert png.read_bytes()[:4] == b"\x89PNG"


def test_journal_is_recorded_readable_and_filterable(tmp_path):
    run = _run(tmp_path)
    j = run.journal()
    assert len(j) and set(j["kind"].unique()) <= {
        "decree",
        "message",
        "event",
        "pillar",
        "threat",
        "mood",
        "outcome",
    }
    assert j["headline"].str.len_chars().min() > 0
    assert len(run.journal(min_importance=3)) <= len(j)
    assert set(run.journal(kinds=["event"])["kind"].unique()) <= {"event"}


def test_journal_does_not_change_the_run_and_marks_charts(tmp_path):
    from tle.viz import journal_table_html, journal_timeline_figure

    g_off = small_game(3)
    play(g_off, STRATEGIES["random"], 12, [])
    g_on = small_game(3)
    store = RunStore(
        tmp_path, "r2", {"scenario": "industrial_secular", "seed": 3, "label": "random"}, {"a": 1}
    )
    play(g_on, STRATEGIES["random"], 12, [TelemetryObserver(store)])
    store.close()
    assert g_on.w.digest() == g_off.w.digest()  # the diary is a pure observer
    run = _run(tmp_path)
    j = run.journal()
    m = run.metrics().df
    fig = metric_figure(m, "pop_size", None, None, journal=j)
    assert any(getattr(r.glyph, "fill_color", None) == "color" for r in fig.renderers if hasattr(r, "glyph"))
    assert "<table" in journal_table_html(j) and journal_timeline_figure(j) is not None


def test_journal_command(tmp_path):
    from tle.cli.main import app
    from typer.testing import CliRunner

    _run(tmp_path)
    res = CliRunner().invoke(
        app, ["journal", "r1", "--runs-dir", str(tmp_path), "--min-importance", "0", "--tail", "0"]
    )
    assert res.exit_code == 0, res.output
    assert "day" in res.output
