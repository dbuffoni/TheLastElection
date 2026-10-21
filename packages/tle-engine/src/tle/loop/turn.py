"""One turn of the game (spec 7.3, slice v0): player input -> decree pipeline -> exogenous events ->
social contagion and decay -> power block -> indicators -> outcome. Every random draw comes from child
streams keyed by (turn, purpose), so the result does not depend on the order of the sub-steps."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from tle.contracts import models as m
from tle.contracts.vocab import bip_value_of, uni_idx, uni_name
from tle.core.prims import count, express, rate, relax
from tle.core.rng import GameRng
from tle.engines.errors import UnclassifiableDecree, validate_text
from tle.engines.protocol import Engines
from tle.loop import actors
from tle.params import Params
from tle.systems import custodians, demography, economy, factions, indicators, influencers, power, ties
from tle.systems.predicate import match
from tle.systems.reaction import EventResult, run_event
from tle.world.catalogs import Catalogs
from tle.world.state import World

DOMAIN_EVENT = {"economy": "decree_economy", "worship": "decree_culture", "security": "decree_security"}


@dataclass
class PlayerInput:
    decree: str | None = None
    structure: m.ClassifyDecreeOut | None = None  # simulated players pass an already structured decree (15.5)
    messages: list[tuple[int, str]] = field(default_factory=list)  # (citizen id, text)
    dt: float | None = None


@dataclass
class TurnPackage:
    turn: int
    day: float
    dt: float
    news: list[EventResult] = field(default_factory=list)
    decree: dict | None = None
    indicators: dict[str, float] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    pillars: dict[str, int] = field(default_factory=dict)
    threats: list[dict] = field(default_factory=list)
    capital: float = 0.0
    warnings: list[dict] = field(default_factory=list)
    outcome: str = ""
    demography: dict = field(default_factory=dict)
    phase: str = "game"
    rejected: str | None = None  # a decree that could not be classified goes back at no cost (13.1)


def classify(engines: Engines, inp: PlayerInput, rng: GameRng) -> m.ClassifyDecreeOut:
    text = validate_text(inp.decree)
    cin = m.ClassifyDecreeIn(text=text)
    cin._structure = inp.structure
    out = engines.s1.call("classify_decree", cin, rng)
    if out is None:
        raise UnclassifiableDecree("the decree could not be classified; please rephrase it")
    return out


def step(
    w: World, cat: Catalogs, p: Params, engines: Engines, inp: PlayerInput | None = None, warmup: bool = False
) -> TurnPackage:
    inp = inp or PlayerInput()
    rng = w.rng
    dt = float(inp.dt if inp.dt is not None else p.time.dt_max)
    turn_rng = rng.child("turn", w.turn)
    pkg = TurnPackage(turn=w.turn, day=w.day, dt=dt)
    decreed_pillar: int | None = None
    ad = w.adults(p.demography.a_adult)
    pil_ids = cat.pillar_ids()

    # 1. private messages (<= 50 characters each): a small pull on one citizen's opinion of the mayor
    for cid, text in inp.messages:
        validate_text(text)
        if 0 <= cid < w.n_used and w.alive[cid] and w.K >= p.events.cost_message:
            w.K -= p.events.cost_message
            w.O[cid, 0] += p.events.persuade * (0.6 - w.O[cid, 0]) * (1.0 if w.O[cid, 0] > -0.5 else 0.3)
            w.last_player_day = w.day

    # 2. the decree
    if inp.decree is not None:
        try:
            cd = classify(engines, inp, turn_rng.child("classify"))
        except UnclassifiableDecree as e:
            pkg.rejected = str(e)
            cd = None
        if cd is not None:
            pkg.decree = _handle_decree(w, cat, p, engines, turn_rng, cd, inp.decree, pkg)
            if cd.pillar != "none" and pkg.decree.get("executed"):
                decreed_pillar = pil_ids.index(cd.pillar)

    # 3. exogenous events
    for k in range(count(p.events.nu_exo, dt, turn_rng.child("n_exo"))):
        r = _exogenous(w, cat, p, engines, turn_rng.child("exo", k), ad)
        if r:
            pkg.news.append(r)

    for r_ in actors.generate(w, cat, p, engines, turn_rng.child("actors"), dt):
        pkg.news.append(r_)

    # 4. contagion through ties, ideology drift, fear decay, wealth relaxation
    _contagion(w, p, dt)
    life = ties.daily_life(w, cat, p, engines, dt, turn_rng.child("life"))
    ties.ideology_pull(w, p, dt)
    factions.pull(w, p, dt)
    influencers.pull(w, p, dt)
    economy.step(w, cat, p, dt)
    custodians.follow_alignment(w, p, dt)
    factions.step(w, p, dt)
    influencers.step(w, p, dt)
    w.stats["ties_dropped"] = w.stats.get("ties_dropped", 0) + ties.tie_dynamics(w, p, dt)
    w.I = relax(w.I, w.B, p.vectors.h_I, dt)  # ideology fades toward its base ...
    w.B = relax(
        w.B, w.I, p.vectors.h_b, dt
    )  # ... and the base itself follows slowly: lived events leave a scar
    drift = turn_rng.child("drift").normal(size=w.I.shape) * (p.vectors.sigma_I * (dt / 365.0) ** 0.5)
    drift[~ad] = 0.0
    w.I = np.clip(w.I + drift, -1.0, 1.0)  # personal experience: idiosyncratic ideological drift ...
    w.B = np.clip(w.B + drift, -1.0, 1.0)  # ... which moves the base along, so it persists
    w.O = relax(w.O, w.Ob, p.vectors.h_O, dt)  # opinions fade toward the personal baseline (h_O)
    c_ = (
        w.councillor != 0
    )  # party discipline: councillors fall back to their alignment faster than ordinary citizens
    w.O[c_, 0] = relax(w.O[c_, 0], w.Ob[c_, 0], p.vectors.h_O_council, dt)
    w.fear = relax(w.fear, 0.0, p.events.h_fear, dt, 0.0, 1.0)
    w.r = relax(w.r, w.r0, 365.0, dt, 0.0, 1.0)
    # economic vote (5.18): the change in wealth of the turn, from any source, pushes the opinion of the mayor toward +1
    # if positive and -1 if negative (k = kappa_ec |dr|); the part due to transfers also pushes the acceptance of the
    # erosion of property and rule of law: who gains from extraction accepts the rules that allow it, who pays defends them
    dr = w.r - w.r_prev
    k = np.clip(p.economy.kappa_ec * np.abs(dr), 0.0, 1.0)
    w.O[:, 0] = np.clip(w.O[:, 0] + k * (np.sign(dr) - w.O[:, 0]), -1.0, 1.0)
    kt = np.clip(p.economy.kappa_ec * np.abs(w.dr_transfer), 0.0, 1.0)
    for pid in ("property", "law"):
        j = 1 + cat.pillar_ids().index(pid)
        w.O[:, j] = np.clip(w.O[:, j] + kt * (np.sign(w.dr_transfer) - w.O[:, j]), -1.0, 1.0)
    w.r_prev = w.r.copy()

    demo = demography.step(w, cat, p, engines, dt, turn_rng.child("demography"))
    ad = w.adults(p.demography.a_adult)

    # 5. time advances; power block
    w.day += dt
    w.turn += 1
    w.dt_prev = dt
    ind = indicators.compute(w, cat, p)
    w.stats["fac_F"], w.stats["fac_H"] = factions.balance(w, p, express(w.O, w.fear))
    if not warmup:
        power.capital_step(w, p, dt, ind.get("support", 0.0))
        pkg.threats = power.threats_step(w, cat, p, dt, turn_rng.child("threats"))
        for ev in pkg.threats:
            pkg.news.append(
                EventResult(-1, ev["event"], ev["event"].replace("_", " "), "", "high", "threat", 0)
            )
        power.demotion_step(w, cat, p, decreed_pillar)

    # 6. outcome
    done = all(w.pillar_level[i] >= len(cat.pillars[i].levels) for i in range(len(cat.pillars)))
    if warmup:
        done = False
    if not w.outcome and done:
        w.outcome, w.outcome_day = "victory", w.day
    if not warmup and not w.outcome and w.day >= p.time.horizon_days:
        w.outcome, w.outcome_day = "term_end", w.day

    # 7. report
    ind = indicators.compute(w, cat, p)
    pkg.indicators = ind
    pkg.labels = indicators.labels(w, ind, p.labels.eps)
    pkg.pillars = {pid: int(w.pillar_level[i]) for i, pid in enumerate(pil_ids)}
    pkg.capital = float(w.K)
    pkg.demography = demo | life
    pkg.outcome = w.outcome
    pkg.warnings = engines.drain_warnings()
    w.save_rng(rng)
    return pkg


def _handle_decree(w, cat, p, engines, rng, cd: m.ClassifyDecreeOut, text: str, pkg: TurnPackage) -> dict:
    agg = uni_idx(cd.aggressiveness)
    cost = p.events.cost_decree * (0.25 + agg / 6.0)
    info = {
        "text": text,
        "pillar": cd.pillar,
        "domain": cd.domain,
        "aggressiveness": cd.aggressiveness,
        "cost": round(cost, 4),
        "executed": False,
    }
    if cost > w.K:
        # the decree passes anyway (5.8): capital drops to 0 and the deficit adds to the pressure of a threat, the judicial
        # route if it hits the rule of law and the no-confidence motion otherwise
        deficit = cost - w.K
        w.K = 0.0
        t = power.JUDICIAL if cd.pillar == "law" else power.MOTION
        w.threat_p[t] = float(w.threat_p[t] + (1.0 - w.threat_p[t]) * min(1.0, p.capital.kappa_D * deficit))
        w.stats["decree_deficit"] = w.stats.get("decree_deficit", 0) + 1
    else:
        w.K -= cost
    w.last_player_day = w.day
    info["executed"] = True
    involved = np.flatnonzero(match(w, cat, p, cd.predicate))
    event_type = (
        f"decree_{cd.pillar}" if cd.pillar != "none" else DOMAIN_EVENT.get(cd.domain, "decree_announce")
    )
    econ = bip_value_of(cd.econ_effect)
    hidden = {
        "pillar_idx": cat.pillar_ids().index(cd.pillar) if cd.pillar != "none" else -1,
        "attacks": 1 if cd.pillar != "none" else 0,
        "actor_sign": 1.0,
        "econ": 0.0 if econ != econ else econ,
    }
    # fear: a decree frightens in proportion to its aggressiveness (everyone it reaches)
    if cd.pillar != "none":
        w.fear[involved] = np.clip(w.fear[involved] + p.events.fear_decree * (agg / 6.0) ** 2, 0.0, 1.0)
    res = run_event(
        w,
        p,
        engines,
        rng,
        event_type=event_type,
        actor={"kind": "mayor", "name": "The mayor"},
        actor_kind="mayor",
        intensity=cd.aggressiveness,
        involved=involved,
        hidden=hidden,
    )
    if res:
        pkg.news.append(res)
    # direct exposure: citizens judge the mayor by what the decree does to what they value
    if len(involved):
        if cd.pillar != "none":
            pi = cat.pillar_ids().index(cd.pillar)
            stance = np.tanh(
                3.0 * (w.O[involved, 1 + pi] + 0.25)
            )  # who accepts the erosion likes it, others do not
            target = np.where(stance >= 0, 1.0, -1.0)
            k = np.abs(stance) * p.events.alpha_direct * (0.3 + agg / 6.0)
        else:
            target = np.ones(len(involved))
            k = np.full(len(involved), p.events.persuade * (0.5 + agg / 6.0))
        w.O[involved, 0] = np.clip(w.O[involved, 0] + k * (target - w.O[involved, 0]), -1.0, 1.0)
    if cd.pillar != "none" and len(
        involved
    ):  # awareness: an attack on a pillar is noticed, whatever the reactions
        pi = cat.pillar_ids().index(cd.pillar)
        ai = 1 + pi
        w.O[involved, ai] = np.clip(
            w.O[involved, ai] + p.events.alpha_aware * (agg / 6.0) * (-1.0 - w.O[involved, ai]), -1, 1
        )
    if cd.transfer is not None:
        tr = economy.transfer(w, cat, p, cd.transfer, "mayor")
        info["transfer"] = "rent" if tr and "I" in tr else ("one_off" if tr else "blocked")
    if cd.revocation is not None:
        info["revoked"] = economy.revoke(w, cat, p, cd.revocation)
    if cd.pillar != "none":
        pi = cat.pillar_ids().index(cd.pillar)
        outcome, level = power.try_pillar_step(w, cat, p, pi, agg)
        info.update(result=outcome, level=level)
        if outcome == "resisted":
            w.K = max(0.0, w.K - p.events.cost_decree * 0.5)  # a failed push costs standing
            w.stats["pillar_resisted"] = w.stats.get("pillar_resisted", 0) + 1
    else:
        info["result"] = "announced"
    return info


def _exogenous(w, cat, p, engines, rng, ad) -> EventResult | None:
    types = [e for e in cat.event_types if "exogenous" in e.actors]
    # extractiveness blocks creative destruction: it reduces the frequency of discoveries by (1 - X) (5.18)
    et = types[
        rng.choice_index(
            [e.weight * ((1.0 - w.extractiveness) if e.category == "discovery" else 1.0) for e in types]
        )
    ]
    intensity = uni_name(int(rng.integers(1, 6)))
    kind = et.predicate_kinds[rng.integers(0, len(et.predicate_kinds))] if et.predicate_kinds else "all"
    mask = ad.copy()
    if kind == "circle":
        mask &= w.circle == int(rng.integers(0, max(1, int(w.circle[: w.n_used].max()) + 1)))
    elif kind == "class":
        mask &= w.cls == int(rng.integers(0, 3))
    elif kind == "trade":
        mask &= w.trade == int(rng.integers(0, len(cat.trades) - 2))
    elif kind == "age_band":
        mask &= (w.age < 35) if rng.random() < 0.5 else (w.age >= 35)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    return run_event(
        w,
        p,
        engines,
        rng,
        event_type=et.id,
        actor={"kind": "exogenous", "name": et.id.replace("_", " ").title()},
        actor_kind="exogenous",
        intensity=intensity,
        involved=np.flatnonzero(mask),
        hidden={"actor_sign": sign, "pillar_idx": None, "attacks": 0, "econ": et.econ},
    )


def _contagion(w: World, p: Params, dt: float) -> None:
    """Opinions diffuse through ties on what people *say*, weighted by tie strength (indirect exposure)."""
    if w.bn == 0:
        return
    s, d, bw = w.bsrc[: w.bn], w.bdst[: w.bn], w.bw[: w.bn]
    E = express(w.O, w.fear)
    num = np.zeros_like(w.O)
    den = np.zeros(w.n_cap)
    np.add.at(num, d, bw[:, None] * E[s])
    np.add.at(den, d, bw)
    nb = np.where(den[:, None] > 0, num / np.maximum(den[:, None], 1e-9), w.O)
    k = float(rate(p.events.k_contagion, dt)) * np.minimum(1.0, den)[:, None]
    w.O = np.clip(w.O + k * (nb - w.O), -1.0, 1.0)
