import numpy as np
import pytest
from tle.engines.errors import TextTooLong, validate_text
from tle.loop.game import Game, play
from tle.loop.strategies import STRATEGIES
from tle.loop.turn import PlayerInput
from tle.testing import catalogs, inject_failures, small_game
from tle.world.state import World

pytestmark = [pytest.mark.inc10]


def test_text_limit_is_50_characters():
    assert validate_text("x" * 50) == "x" * 50
    with pytest.raises(TextTooLong):
        validate_text("x" * 51)
    with pytest.raises(ValueError):
        validate_text("   ")
    assert validate_text("  spaced   out  ") == "spaced out"


def test_game_rejects_long_decree_and_message():
    g = small_game()
    with pytest.raises(TextTooLong):
        g.step(PlayerInput(decree="Ban " + "the opposition " * 5))
    with pytest.raises(TextTooLong):
        g.step(PlayerInput(messages=[(0, "y" * 51)]))


def test_unclassifiable_decree_is_free_and_returned():
    g = small_game()
    k0 = g.w.K
    pkg = g.step(PlayerInput(decree="xyzzy plugh"))
    assert pkg.rejected and pkg.decree is None
    assert g.w.K >= k0  # no charge (capital only grows)


def test_free_text_decree_runs_the_pipeline():
    g = small_game(seed=2)
    pkg = g.step(PlayerInput(decree="Restrict the press and journalists"))
    assert pkg.decree["pillar"] == "press" and pkg.decree["executed"]
    assert any(n.event_type == "decree_press" for n in pkg.news)


def test_same_seed_same_digest_and_different_seed_differs():
    a = play(small_game(7), STRATEGIES["gradual"], 25).digest()
    b = play(small_game(7), STRATEGIES["gradual"], 25).digest()
    c = play(small_game(8), STRATEGIES["gradual"], 25).digest()
    assert a == b and a != c


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_every_strategy_terminates_with_valid_state(name):
    g = small_game(3, 60)
    w = play(g, STRATEGIES[name])
    assert w.outcome in {"victory", "removed", "term_end"}
    assert np.all(np.abs(w.O[: w.n_used]) <= 1 + 1e-9) and np.all((w.fear >= 0) & (w.fear <= 1))
    assert 0 <= w.K <= 1 and w.pillar_level.min() >= 0


def test_resume_from_saved_state_is_bit_identical():
    g = small_game(11)
    play(g, STRATEGIES["gradual"], 12)
    data = g.save()
    play(g, STRATEGIES["gradual"], 12)
    h = Game.load(data, catalogs())
    play(h, STRATEGIES["gradual"], 12)
    assert g.w.digest() == h.w.digest()


def test_world_serialisation_roundtrip():
    g = small_game(4)
    play(g, STRATEGIES["populist"], 8)
    assert World.from_bytes(g.w.to_bytes()).digest() == g.w.digest()


def test_turn_length_independence_of_decay():
    """With no player action and no events, two half turns equal one whole turn for the decay terms."""
    a = small_game(5, overrides={"nu_exo": 0.0})
    b = small_game(5, overrides={"nu_exo": 0.0})
    a.w.fear[:] = 0.8
    b.w.fear[:] = 0.8
    a.step(PlayerInput(dt=14.0))
    b.step(PlayerInput(dt=7.0))
    b.step(PlayerInput(dt=7.0))
    np.testing.assert_allclose(a.w.fear, b.w.fear, atol=1e-9)


def test_failure_injection_never_breaks_the_game():
    g = inject_failures(small_game(6), 0.5)
    w = play(g, STRATEGIES["assault"], 30)
    assert w.turn > 0  # fallbacks keep the pipeline running


def test_fear_suppresses_expressed_resistance():
    from tle.core.prims import express

    g = small_game(9)
    g.w.O[:, 1] = -0.9
    g.w.fear[:] = 0.0
    r0 = float(np.mean(express(g.w.O[:, 1], g.w.fear) < -1 / 3))
    g.w.fear[:] = 0.9
    r1 = float(np.mean(express(g.w.O[:, 1], g.w.fear) < -1 / 3))
    assert r0 == 1.0 and r1 == 0.0
