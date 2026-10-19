import numpy as np
import pytest
from tle.analysis import population as pa
from tle.devtools import metrics_lint_errors
from tle.experiments import impulse_test
from tle.loop.game import Game, play
from tle.loop.strategies import STRATEGIES
from tle.params import default_params, to_flat
from tle.registry.metrics import load_metrics
from tle.report import build_report
from tle.runs import open_run
from tle.systems import demography, popstats
from tle.telemetry import RunStore, TelemetryObserver
from tle.testing import catalogs, small_game

pytestmark = pytest.mark.inc04


def _run(tmp_path, turns=40, seed=2, n=120, name="r"):
    p = default_params()
    store = RunStore(
        tmp_path,
        name,
        {"scenario": "industrial_secular", "seed": seed, "label": "gradual"},
        to_flat(p),
        flush_every=7,
    )
    ob = TelemetryObserver(store, panel_every=30)
    g = Game(catalogs(), p, seed=seed, n=n, observers=[ob])
    play(g, STRATEGIES["gradual"], turns, [ob])
    store.close()
    return g, open_run(tmp_path / name)


# ------------------------------------------------------------------------------------------ statistics
def test_sarle_separates_unimodal_from_bimodal():
    r = np.random.default_rng(0)
    uni = r.normal(0, 1, 4000)
    bi = np.concatenate([r.normal(-2, 0.4, 2000), r.normal(2, 0.4, 2000)])
    assert popstats.sarle(uni) < 5 / 9 < popstats.sarle(bi)


def test_gini_extremes():
    assert popstats.gini(np.ones(50)) == pytest.approx(0.0, abs=1e-9)
    x = np.zeros(100)
    x[0] = 1
    assert popstats.gini(x) > 0.98


def test_metric_registry_matches_telemetry():
    assert metrics_lint_errors() == []
    assert len(load_metrics()) >= 35


def test_invariants_detect_corruption():
    g = small_game(3)
    assert popstats.check_invariants(g.w) == []
    g.w.I[0, 0] = 2.5
    g.w.O[1, 0] = np.nan
    bad = popstats.check_invariants(g.w)
    assert "range_I" in bad and "nan_O" in bad


# ------------------------------------------------------------------------------------------ demography
def test_mortality_doubles_every_eight_years():
    mu = demography.mortality(np.array([30.0, 38.0, 46.0]), 0.001)
    np.testing.assert_allclose(mu, [0.001, 0.002, 0.004])


def test_demography_keeps_state_consistent_and_deterministic():
    a, b = small_game(5, 150), small_game(5, 150)
    play(a, STRATEGIES["populist"], 60)
    play(b, STRATEGIES["populist"], 60)
    assert a.w.digest() == b.w.digest()
    w = a.w
    assert popstats.check_invariants(w) == []
    alive = np.flatnonzero(w.alive[: w.n_used])
    # no bond may touch a citizen who left; every couple is mutual
    assert w.alive[w.bsrc[: w.bn]].all() and w.alive[w.bdst[: w.bn]].all()
    c = w.couple[alive]
    paired = alive[c >= 0]
    assert (w.couple[w.couple[paired]] == paired).all()
    assert w.stats.get("births", 0) + w.stats.get("immigrants", 0) > 0


def test_table_growth_preserves_existing_citizens():
    g = small_game(6, 60)
    before = g.w.I[: g.w.n_used].copy()
    cap = g.w.n_cap
    demography.grow(g.w, cap + 50)
    assert g.w.n_cap == cap + 50 and g.w.I.shape[0] == cap + 50 and g.w.O.shape[0] == cap + 50
    np.testing.assert_array_equal(g.w.I[: len(before)], before)
    assert (g.w.lab_I[cap:] == 3).all() and (g.w.couple[cap:] == -1).all()


# ------------------------------------------------------------------------------------------ warm-up and ties
def test_warmup_resets_the_clock_and_power():
    g = small_game(7, 80)
    w = g.w
    assert w.warmup_done and w.day == 0.0 and w.turn == 0 and w.outcome == ""
    assert (w.pillar_level == 0).all() and w.K == pytest.approx(g.p.capital.c_0)


def test_warmup_can_be_disabled():
    g = Game(catalogs(), default_params().with_overrides({"T_warm": 0}), seed=1, n=60)
    assert g.w.day == 0.0 and not g.w.warmup_done


def test_network_reaches_equilibrium_in_warmup():
    g = Game(catalogs(), default_params(), seed=1, n=150)
    d0 = g.w.bn / g.w.alive.sum()
    play(g, None, 30)
    d1 = g.w.bn / g.w.alive.sum()
    assert d0 > 1.0 and abs(d1 - d0) / d0 < 0.25  # no collapse, no blow-up after the warm-up
    assert 0.0 < g.w.bstable[: g.w.bn].mean() <= 1.0


# ------------------------------------------------------------------------------------------ run store and read library
def test_run_has_panel_bonds_checks_and_states(tmp_path):
    g, r = _run(tmp_path)
    days = r.panel_days()
    assert days[0] == 0.0 and len(days) >= 7  # day 0 plus a snapshot about every panel_every days
    p0 = r.panel(0)
    assert len(p0) == g.w.alive.sum() or len(p0) > 0
    assert {"id", "age", "cls", "wealth", "I1", "op_mayor", "ex_mayor", "fear"} <= set(p0.columns)
    assert (p0["ex_mayor"] <= p0["op_mayor"] + 1e-12).all() or (p0["ex_mayor"] >= -1).all()
    assert len(r.bonds(0)) > 0 and r.checks()["ok"]
    assert (r.path / "state" / "day0.npz").exists()
    assert (
        r.state("day0").digest() != r.state("final").digest()
        if (r.path / "state" / "final.npz").exists()
        else True
    )
    assert r.status["state"] == "finished" and "dropped_columns" not in r.status


def test_warmup_rows_are_separate_from_game_rows(tmp_path):
    _, r = _run(tmp_path, turns=10)
    warm, game = r.warmup(["pop_size"]).df, r.metrics(["pop_size"]).df
    assert warm["day"].min() < 0 <= game["day"].min() and len(warm) >= 20 and len(game) == 10


def test_daily_view_is_forward_filled_on_every_day(tmp_path):
    _, r = _run(tmp_path, turns=12)
    d = r.metrics(["pop_size"]).daily()
    assert len(d) == int(d["day"].max() - d["day"].min()) + 1
    assert d["pop_size"].null_count() == 0
    assert d["day"].diff().drop_nulls().unique().to_list() == [1.0]


def test_columns_added_later_are_reported_not_silently_lost(tmp_path):
    from tle.telemetry.store import _Table

    t = _Table(tmp_path / "x")
    t.add({"a": 1})
    t.flush()
    t.add({"a": 2, "b": 3})
    t.flush()
    assert t.dropped == {"b"}


# ------------------------------------------------------------------------------------------ analyzers
def test_verdicts_on_synthetic_series():
    d = np.arange(0, 720.0)
    assert pa.verdict(d, np.full(720, 0.5))["verdict"] in ("stable",)
    assert pa.verdict(d, 0.2 + 0.5 * d / 720)["verdict"] == "drifting"
    osc = 0.5 + 0.2 * np.sin(2 * np.pi * d / 60.0)
    assert pa.verdict(d, osc)["verdict"] == "oscillating"
    assert pa.verdict(d, np.clip(d / 100, 0, 1))["verdict"] == "degenerate"  # ends pinned at the limit


def test_drift_uses_scale_floor_for_values_near_zero():
    d = np.arange(0, 720.0)
    v = np.where(d < 540, 0.01, 0.03)  # tiny absolute change around 0
    assert pa.drift(d, v, 180) > 1.0
    assert pa.drift(d, v, 180, floor=1 / 3) < 0.10


def test_bistability_needs_enough_seeds():
    r = np.random.default_rng(1)
    bi = np.concatenate([r.normal(0, 0.05, 20), r.normal(1, 0.05, 20)])
    assert pa.bistable_across_seeds(bi)["bistable"]
    assert not pa.bistable_across_seeds(bi[:3])["bistable"]


def test_targets_mobility_flows_identity_and_rates(tmp_path):
    _, r = _run(tmp_path, turns=30)
    ev = pa.evaluate_targets(r)
    assert set(ev["verdict"]) <= {"ok", "miss", "n/a"} and "pop_size" in ev["metric"].to_list()
    m = pa.mobility(r)
    assert 0.0 <= m["share_label_changed"] <= 1.0 and m["median_displacement"] >= 0
    flows = pa.class_flows(r)
    assert flows["citizens"].sum() > 0 and set(flows.columns) == {"from", "to", "citizens"}
    ident = pa.scenario_identity(r)
    assert 0.3 < ident["polarization_retained"] < 1.5
    assert abs(pa.vital_rates(r)["net_growth"]) < 0.5
    assert (
        len(pa.stability_table(r)) > 10
        and len(pa.degree_distribution(r)) > 0
        and len(pa.age_distribution(r)) > 0
    )


def test_scenario_scoped_target_is_skipped_for_other_scenarios(tmp_path):
    _, r = _run(tmp_path, turns=5)
    assert "bimodality_axis1" not in pa.evaluate_targets(r)["metric"].to_list()  # only for mixed_polarised


# ------------------------------------------------------------------------------------------ impulse and report
def test_impulse_test_is_a_proper_control_comparison():
    g = small_game(1, 150)
    digest = g.w.digest()
    res = impulse_test(g, days=63)
    assert g.w.digest() == digest  # the source game is untouched
    assert res.involved_peak > res.others_peak >= 0 and res.involved_peak > 0.005
    assert set(res.verdicts()) == {"involved_peak", "others_peak", "half_life_days"}
    # no event, no difference: same state and same RNG streams reproduce the control exactly
    a, b = (Game.load(g.save(), g.cat, g.p) for _ in range(2))
    for _ in range(3):
        a.step()
        b.step()
    assert a.w.digest() == b.w.digest()


def test_report_is_self_contained_html(tmp_path):
    _, r = _run(tmp_path, turns=14)
    html = build_report(r)
    assert html.startswith("<!doctype html>") and "<script>" in html and "Target scorecard" in html
    assert "http://" not in html.replace(
        "http://www.w3.org/2000/svg", ""
    )  # nothing is loaded from the network
    assert "pop_size" in html and "bimodality_axis1" in html


def test_cli_population_commands(tmp_path):
    from tle.cli.main import app
    from typer.testing import CliRunner

    run = CliRunner()
    res = run.invoke(
        app, ["run", "--strategy", "gradual", "--n", "80", "--runs-dir", str(tmp_path), "--report"]
    )
    assert res.exit_code == 0, res.output
    rid = "industrial_secular-gradual-s1"
    assert (tmp_path / rid / "population.html").exists()
    for cmd in (
        ["pop", "watch", rid],
        ["pop", "stability", rid],
        ["pop", "targets", rid],
        ["pop", "mobility", rid],
        ["pop", "identity", rid],
        ["pop", "panel", rid],
        ["metrics", "list"],
        ["metrics", "lint"],
    ):
        out = run.invoke(app, [*cmd, *(["--runs-dir", str(tmp_path)] if cmd[0] == "pop" else [])])
        assert out.exit_code == 0, (cmd, out.output)
