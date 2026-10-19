from tle.cli.main import app
from tle.devtools import boundary_violations
from typer.testing import CliRunner


def test_deterministic_core_has_no_global_rng_or_clock():
    assert boundary_violations() == []


def test_cli_check_and_params_and_run(tmp_path):
    r = CliRunner()
    assert r.invoke(app, ["check"]).exit_code == 0
    assert r.invoke(app, ["params", "list", "--group", "events"]).exit_code == 0
    res = r.invoke(app, ["run", "--strategy", "random", "--n", "60", "--runs-dir", str(tmp_path)])
    assert res.exit_code == 0 and "random" in res.output
    assert r.invoke(app, ["runs", "list", "--runs-dir", str(tmp_path)]).exit_code == 0


def test_text_game_client_enforces_limit():
    from tle.game import main

    out = []
    lines = iter(["d " + "x" * 60, "s", "q"])
    main(n=60, input_fn=lambda _: next(lines), print_fn=out.append)
    assert any("Too long" in o for o in out)
