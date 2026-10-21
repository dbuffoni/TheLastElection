import numpy as np
from tle.contracts import models as m
from tle.core.prims import dist
from tle.core.rng import GameRng
from tle.engines.protocol import OptionFeatures
from tle.loop import actors
from tle.params import default_params
from tle.systems import factions, influencers
from tle.testing import catalogs, small_game


def _game(seed=4, n=120):
    return small_game(seed, n)


def test_faction_is_born_from_founders_near_the_seed_and_needs_n_min():
    g = _game()
    w, p = g.w, g.p
    ad = np.flatnonzero(w.adults(p.demography.a_adult))
    t = w.I[ad].mean(axis=0)
    F = OptionFeatures(
        t_I=np.tile(t, (2, 1)),
        t_C=np.full((2, 3), np.nan),
        e_r=np.zeros(2),
        F=np.ones(2),
        stance=np.zeros((2, 8)),
    )
    news = m.NewsItem(
        event_id=1,
        title="t",
        text="x",
        event_type="demonstration",
        intensity="medium",
        actor_kind="faction",
        target="mayor",
        tags=[],
    )
    few = factions.found(
        w, p, g.engines, GameRng(1), news=news, option=1, involved=ad[:3], choice=np.ones(3, dtype=int), F=F
    )
    assert few is None  # fewer than N_min founders: no faction
    s = factions.found(
        w, p, g.engines, GameRng(1), news=news, option=1, involved=ad, choice=np.ones(len(ad), dtype=int), F=F
    )
    assert s is not None and w.fac_alive[s] and w.fac_member[:, s].sum() >= p.factions.N_min
    assert (dist(w.I[w.fac_member[:, s]], w.fac_mu[s]) <= p.factions.rho_in + 1e-9).all()


def test_faction_dies_when_its_strength_fades_and_members_return_without_jumps():
    g = _game()
    w, p = g.w, g.p
    ad = np.flatnonzero(w.adults(p.demography.a_adult))[:20]
    w.fac_alive[0] = True
    w.fac_S[0] = 0.5
    w.fac_mu[0] = w.I[ad].mean(axis=0)
    w.fac_m[ad, 0] = p.factions.theta_m_in
    w.fac_member[ad, 0] = True
    before = w.I.copy()
    factions.pull(w, p, 7.0)
    assert np.abs(w.I - before).max() < 0.2  # a pull, not a jump
    for _ in range(40):
        w.day += 7
        factions.step(w, p, 7.0)
    assert not w.fac_alive[0] and not w.fac_member[:, 0].any()


def test_influencer_entry_catchment_and_saturation():
    g = _game(2, 150)
    w, p = g.w, g.p
    ad = w.adults(p.demography.a_adult)
    a = int(np.argmax(influencers.potential(w, p, ad)))
    w.inf_on[a], w.inf_K[a], w.inf_since[a] = True, p.influencers.K_inf_0, w.day
    e1 = influencers.effective_shares(w, p, ad, np.array([a]))
    assert 0 <= e1.min() and e1.max() <= 1 + 1e-9
    b = int(np.argsort(influencers.potential(w, p, ad))[-2])
    w.I[b] = w.I[a]  # same topic and ideology: the audience is split
    w.inf_on[b], w.inf_K[b] = True, p.influencers.K_inf_0
    e2 = influencers.effective_shares(w, p, ad, np.array([a, b]))
    assert e2.sum(axis=0).max() <= 3.0 + 1e-9  # capacity 1 per axis, three axes
    pis = [influencers.catchment(w, p, ad, w.I[x], w.inf_K[x])[0] for x in (a, b)]
    assert (e2 <= np.array(pis) + 1e-9).all()  # saturation can only reduce the audience


def test_actor_events_respect_tone_and_requirements():
    cat = catalogs()
    w = _game().w
    types = actors.admitted(
        w, cat, "influencer", {"hostile"}, opinion=0.5
    )  # approves of the mayor: no hostile act
    assert all(t.hostility <= 0.25 for t in types)
    assert "demonstration" not in [
        t.id for t in actors.admitted(w, cat, "faction", set())
    ]  # needs strong and hostile


def test_politics_run_is_deterministic():
    from tle.loop.game import play
    from tle.loop.strategies import STRATEGIES

    d = []
    for _ in range(2):
        g = _game(5, 100)
        play(g, STRATEGIES["passive"], 30)
        d.append(g.w.digest())
    assert d[0] == d[1] and default_params().factions.N_min >= 2
