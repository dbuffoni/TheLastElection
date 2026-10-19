import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tle.core import labels, prims
from tle.core.rng import GameRng

pytestmark = pytest.mark.inc01


@given(rho=st.floats(0.001, 1), dt=st.floats(0.1, 30), n=st.integers(2, 12))
def test_rate_is_exact_for_any_step(rho, dt, n):
    """P3: n small steps equal one big step (turn length independence)."""
    x = 1.0
    for _ in range(n):
        x *= 1 - prims.rate(rho, dt / n)
    assert x == pytest.approx(1 - prims.rate(rho, dt), rel=1e-9)


def test_relax_converges_and_halves():
    assert prims.relax(1.0, 0.0, 10.0, 10.0) == pytest.approx(0.5)  # one half-life
    assert prims.relax(0.3, 0.3, 10.0, 99.0) == pytest.approx(0.3)


def test_pull_many_is_order_independent():
    pushes = [(1.0, 0.3), (-1.0, 0.2), (0.5, 0.4)]
    a = prims.pull_many(0.0, pushes)
    b = prims.pull_many(0.0, pushes[::-1])
    assert a == pytest.approx(b)


def test_pushbuffer_order_independent_and_masked():
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 10, 40)
    tgt = rng.uniform(-1, 1, (40, 3))
    tgt[::5, 1] = np.nan  # mask: no push on this component
    k = rng.uniform(0, 0.5, 40)
    x = rng.uniform(-1, 1, (10, 3))
    outs = []
    for perm in (np.arange(40), rng.permutation(40)):
        buf = prims.PushBuffer(10, 3)
        for j in perm:
            buf.add(idx[j : j + 1], tgt[j : j + 1], k[j])
        outs.append(buf.apply(x))
    np.testing.assert_allclose(outs[0], outs[1], atol=1e-12)
    assert np.all(np.abs(outs[0]) <= 1)


def test_express_attenuates_only_negatives():
    x = np.array([-0.9, 0.9])
    np.testing.assert_allclose(prims.express(x, 0.5), [-0.45, 0.9])
    np.testing.assert_allclose(prims.express(x, 0.0), x)


def test_gate_hysteresis():
    on, since = prims.gate(False, 0.5, 0.6, 0.4, 0.0, 1.0)
    assert not on
    on, since = prims.gate(on, 0.65, 0.6, 0.4, since, 2.0)
    assert on
    on, since = prims.gate(on, 0.5, 0.6, 0.4, since, 3.0)
    assert on  # inside the band: stays
    on, _ = prims.gate(on, 0.3, 0.6, 0.4, since, 4.0)
    assert not on


def test_quantizer_hysteresis_and_lookup():
    assert int(labels.quantize_bipolar(0.0)) == 3
    assert int(labels.quantize_bipolar(1.0)) == 6 and int(labels.quantize_bipolar(-1.0)) == 0
    # just past a boundary but within eps of the previous level: the label does not flicker
    boundary = (3.5 / 6) * 2 - 1
    assert int(labels.quantize_bipolar(boundary + 0.02, 3, 0.05)) == 3
    assert int(labels.quantize_bipolar(boundary + 0.2, 3, 0.05)) == 4
    assert labels.bip_value(6) == 1.0 and labels.uni_value(3) == 0.5


def test_dist_is_normalised():
    assert prims.dist(np.ones(3), -np.ones(3)) == pytest.approx(1.0)
    assert prims.dist(np.zeros(3), np.zeros(3)) == 0.0


def test_rng_state_roundtrip_and_children():
    r = GameRng(5)
    r.random(3)
    s = r.get_state()
    a = r.random(5)
    r2 = GameRng.from_state(s)
    np.testing.assert_array_equal(a, r2.random(5))
    # children are independent of how much the parent has been used and of creation order
    c1 = GameRng(5).child("a", 1).random(3)
    p = GameRng(5)
    p.random(100)
    p.child("b")
    np.testing.assert_array_equal(c1, p.child("a", 1).random(3))
    assert not np.array_equal(c1, GameRng(5).child("a", 2).random(3))


def test_count_stochastic_rounding_mean():
    r = GameRng(1)
    m = np.mean([prims.count(0.3, 1.0, r) for _ in range(4000)])
    assert m == pytest.approx(0.3, abs=0.03)
