"""The event pipeline (spec 6, 7.3 steps): S2 narrates a seed into a news item with reaction options, S1
extracts the numeric features of each option, S1 chooses a reaction for every involved citizen, and the
choices become synchronous pushes on opinion, ideology and wealth. Order independent by construction:
all pushes of one event are computed on the start-of-event state and applied once (P2b)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from tle.contracts import models as m
from tle.contracts.vocab import uni_idx
from tle.core.labels import quantize_bipolar, quantize_unipolar
from tle.core.prims import PushBuffer, express
from tle.core.rng import GameRng
from tle.engines.features import features_to_arrays
from tle.engines.protocol import ActorCtx, Engines, LabelContext, call_many
from tle.params import Params
from tle.systems import factions
from tle.systems.population import age_band
from tle.world.state import World


@dataclass
class EventResult:
    event_id: int
    event_type: str
    title: str
    text: str
    intensity: str
    actor_kind: str
    involved: int
    tags: list[str] = field(default_factory=list)
    option_counts: list[int] = field(default_factory=list)
    published: bool = True


def label_context(w: World, p: Params, ids: np.ndarray) -> LabelContext:
    eps = p.labels.eps
    O = quantize_bipolar(w.O[ids], w.lab_O[ids], eps)
    w.lab_O[ids] = O
    return LabelContext(
        ids=ids,
        age_band=age_band(w.age[ids], p.demography.a_adult, p.demography.a_pens),
        social_class=w.cls[ids].astype(np.int8),
        trade=w.trade[ids],
        circle=w.circle[ids],
        T=quantize_bipolar(w.T[ids]),
        C=quantize_bipolar(w.C[ids]),
        I=quantize_bipolar(w.I[ids]),
        O=O,
        fear=quantize_unipolar(w.fear[ids]),
        risk=np.where(w.cls[ids] == 2, quantize_unipolar(np.clip(w.r[ids], 0, 1)), 0).astype(np.int8),
    )


def run_event(
    w: World,
    p: Params,
    engines: Engines,
    rng: GameRng,
    *,
    event_type: str,
    actor: dict[str, str],
    actor_kind: str,
    intensity: str,
    involved: np.ndarray,
    hidden: dict,
    target: dict | None = None,
    parent: int | None = None,
    can_found: bool = False,
    actor_ideology: np.ndarray | None = None,
    faction_slot: int | None = None,
) -> EventResult | None:
    """Run one event end to end and apply its effects to ``w``. Returns None if S2 fails (13.3)."""
    eid = w.next_event_id
    w.next_event_id += 1
    seed = m.EventSeed(
        event_id=eid,
        day=int(w.day),
        parent=parent,
        parent_context="",
        actor=actor,
        event_type=event_type,
        target=target or {},
        intensity=intensity,
        involved_group=f"{len(involved)} citizens",
        characters=[],
        can_found=can_found,
        n_options=p.events.K_opt,
    )
    inp = m.NarrateEventIn(seed=seed)
    inp._hidden = hidden
    out = engines.s2.call("narrate_event", inp, rng.child("narrate", eid))
    if out is None:
        return None
    news = m.NewsItem(
        event_id=eid,
        title=out.title,
        text=out.text,
        event_type=event_type,
        intensity=intensity,
        actor_kind=actor_kind,
        target=str((target or {}).get("kind", "none")),
        tags=out.tags,
    )
    # S1: extract the numeric features of every option
    ins = []
    for k, opt in enumerate(out.options):
        oi = m.OptionFeatureIn(news=news, option_text=opt.text)
        oi._annotation = out._annotation[k] if out._annotation else None
        ins.append(oi)
    feats = call_many(engines.s1, "option_feature", ins, [rng.child("feat", eid, k) for k in range(len(ins))])
    F = features_to_arrays(feats)
    res = EventResult(
        eid, event_type, out.title, out.text, intensity, actor_kind, int(len(involved)), list(out.tags)
    )
    if len(involved) == 0:
        w.events.append(
            {"id": eid, "day": w.day, "type": event_type, "title": out.title, "intensity": intensity}
        )
        return res
    # S1: choose a reaction; beyond B_S1 involved, choose for a sample and let the rest copy the nearest sampled citizen
    creng = rng.child("choose", eid)
    B = p.technical.B_S1
    if len(involved) > B:
        sample = involved[creng.permutation(len(involved))[:B]]
    else:
        sample = involved
    ctx = label_context(w, p, sample)
    choice_s = engines.s1.choose_reaction_many(
        ctx, ActorCtx(kind=actor_kind, ideology=actor_ideology), news, list(out.options), F, creng
    )
    if len(sample) < len(involved):
        rest = np.setdiff1d(involved, sample)
        ctx_r = label_context(w, p, rest)
        d = np.abs(ctx_r.O[:, None, :].astype(float) - ctx.O[None, :, :].astype(float)).sum(axis=2)
        choice_r = choice_s[d.argmin(axis=1)]
        ids = np.concatenate([sample, rest])
        choice = np.concatenate([choice_s, choice_r])
    else:
        ids, choice = sample, choice_s
    apply_reactions(w, p, ids, choice, F, float(uni_idx(intensity)) / 6.0)
    if can_found and "faction_birth" in out.tags:
        adh = [k for k, o in enumerate(out.options) if o.adherence]
        if adh:
            factions.found(w, p, engines, rng, news=news, option=adh[0], involved=ids, choice=choice, F=F)
    if faction_slot is not None and w.fac_alive[faction_slot]:
        factions.on_event(w, p, faction_slot, ids, choice, F)
    # indirect exposure (spec 8.1): adults outside the involved group hear of it through the news and react like a
    # random involved citizen, attenuated by alpha_ind
    rest_ad = np.setdiff1d(np.flatnonzero(w.adults(p.demography.a_adult)), ids)
    if len(rest_ad):
        irng = rng.child("indirect", eid)
        pick = choice[irng.integers(0, len(choice), size=len(rest_ad))]
        apply_reactions(w, p, rest_ad, pick, F, float(uni_idx(intensity)) / 6.0, scale=p.vectors.alpha_ind)
    res.option_counts = np.bincount(choice, minlength=F.K).tolist()
    w.events.append(
        {
            "id": eid,
            "day": w.day,
            "type": event_type,
            "title": out.title,
            "intensity": intensity,
            "involved": int(len(ids)),
        }
    )
    del w.events[: -p.events.n_log * 10]
    return res


def apply_reactions(
    w: World, p: Params, ids: np.ndarray, choice: np.ndarray, F, intensity01: float, scale: float = 1.0
) -> None:
    """Synchronous pushes (P2b): opinion toward the stance targets, ideology toward the extracted direction,
    wealth by the economic effect. ``F.F`` (strength of each option) scales the pull."""
    strength = np.clip(0.35 + 0.65 * F.F[choice], 0.0, 1.0) * max(intensity01, 0.15) * scale
    buf = PushBuffer(w.n_cap, 6)
    buf.add(ids, F.t_O[choice], p.events.alpha_react * strength)
    w.O = buf.apply(w.O)
    bi = PushBuffer(w.n_cap, 3)
    bi.add(ids, F.t_I[choice], p.events.alpha_ideo * strength)
    w.I = bi.apply(w.I)
    bc = PushBuffer(w.n_cap, 3)
    bc.add(ids, F.t_C[choice], p.events.alpha_ideo * 0.5 * strength)
    cap = np.maximum(
        p.vectors.C_max, np.abs(w.C)
    )  # events never push a trait past C_max (nor out of its extreme)
    w.C = np.clip(bc.apply(w.C), -cap, cap)
    e = F.e_r[choice]
    ok = ~np.isnan(e)
    if ok.any():
        w.r[ids[ok]] = np.clip(w.r[ids[ok]] + p.events.k_wealth * e[ok] * strength[ok], 0.0, 1.0)


def expressed(w: World) -> np.ndarray:
    """P14: what citizens say, as opposed to what they think."""
    return express(w.O, w.fear)
