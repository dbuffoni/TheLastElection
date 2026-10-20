import numpy as np
from tle.contracts import models as m
from tle.loop.strategies import transfer_of
from tle.systems import custodians, economy
from tle.testing import catalogs, small_game


def _g(seed=6, n=200):
    g = small_game(seed, n)
    return g.w, g.p, catalogs()


def test_property_intact_blocks_every_transfer_and_erosion_opens_it():
    w, p, cat = _g()
    tr = transfer_of({"social_class": "high"}, {"social_class": "low"}, 4, persistent=False)
    assert economy.transfer(w, cat, p, tr) is None  # nobody can move wealth with property intact
    w.pillar_level[cat.pillar_ids().index("property")] = 3
    before = w.r.copy()
    out = economy.transfer(w, cat, p, tr)
    assert out is not None and out["c"] < 0  # from few to many: redistributive
    ad = w.adults(p.demography.a_adult)
    moved = w.r - before
    assert moved[ad].sum() < 0  # deadweight loss: wealth is conserved except for the loss
    assert moved[ad & (w.cls == 0)].sum() > 0 and moved[ad & (w.cls == 2)].sum() < 0


def test_extractive_rent_builds_a_coalition_and_dies_when_property_is_restored():
    w, p, cat = _g(7)
    w.pillar_level[cat.pillar_ids().index("property")] = 4
    tr = transfer_of({}, {"role": "councillor"}, 4)
    assert economy.transfer(w, cat, p, tr, "mayor")["c"] > 0
    for _ in range(30):
        w.day += 7
        economy.step(w, cat, p, 7.0)
    assert w.coalition.any() and w.extractiveness > 0
    w.pillar_level[cat.pillar_ids().index("property")] = 1  # property restored below the level at creation
    for _ in range(80):
        w.day += 7
        economy.step(w, cat, p, 7.0)
    assert not w.rents  # the rent decayed and closed


def test_revocation_zeroes_the_rents_whose_beneficiaries_match():
    w, p, cat = _g(8)
    w.pillar_level[cat.pillar_ids().index("property")] = 4
    economy.transfer(w, cat, p, transfer_of({}, {"role": "councillor"}, 3), "mayor")
    assert (
        economy.revoke(
            w, cat, p, m.Predicate(**{**transfer_of({}, {"role": "councillor"}).beneficiaries.model_dump()})
        )
        == 1
    )
    assert w.rents[0]["I"] == 0.0


def test_council_mirrors_the_city_and_custodians_weigh_more():
    w, p, cat = _g(9, 250)
    ad = w.adults(p.demography.a_adult)
    seats = np.flatnonzero(w.councillor != 0)
    assert len(seats) == p.pillars.n_council
    assert abs((w.councillor[seats] > 0).mean() - p.pillars.q_maj) < 0.1
    assert (np.sign(w.Ob[seats, 0]) == w.councillor[seats]).all() and (
        np.abs(w.Ob[seats, 0]) > 0.2
    ).all()  # alignment base
    pi = cat.pillar_ids().index("law")
    for t in ("magistrate",):
        assert (w.trade[ad] == cat.trade_ids().index(t)).sum() >= p.pillars.n_cust
    wt = custodians.weights(w, cat, p, pi, ad)
    assert wt.max() == p.pillars.omega_cust and wt.min() == 1.0
