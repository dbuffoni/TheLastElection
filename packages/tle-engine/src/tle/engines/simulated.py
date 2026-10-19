"""Simulated S1 and S2 engines (spec 15): deterministic rules with noise controlled by the game RNG that
respect the same contracts as the real models. They see only labels (fidelity principle 15.2)."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel
from tle.contracts import models as m
from tle.contracts.tasks import TASKS
from tle.contracts.vocab import (
    BIPOLAR,
    MASK,
    STANCE,
    bip_name,
    stance_name,
    uni_idx,
    uni_name,
)
from tle.core.labels import bip_value, uni_value
from tle.core.prims import dist
from tle.core.rng import GameRng
from tle.params import Params
from tle.world.catalogs import Catalogs

from .errors import UnclassifiableDecree
from .protocol import ActorCtx, ExecutionRecord, LabelContext, OptionFeatures

DIRECTIONS = {
    "religious": (1.0, 0.0, 0.0),
    "secular": (-1.0, 0.0, 0.0),
    "statist": (0.0, 1.0, 0.0),
    "market": (0.0, -1.0, 0.0),
    "progressive": (0.0, 0.0, -1.0),
    "conservative": (0.0, 0.0, 1.0),
}
STANCE_LEVELS = (-3, -2, -1, 0, 1, 2, 3, -2, 2, 0)  # seven levels plus three repetitions with other strength
FIRST_NAMES = (
    "Anna",
    "Marco",
    "Lucia",
    "Paolo",
    "Elena",
    "Luca",
    "Sara",
    "Giorgio",
    "Marta",
    "Enzo",
    "Irene",
    "Carlo",
    "Livia",
    "Dario",
    "Nora",
    "Piero",
    "Ada",
    "Bruno",
    "Clara",
    "Fabio",
)
SURNAMES = (
    "Rossi",
    "Bianchi",
    "Conti",
    "Greco",
    "Fabbri",
    "Moretti",
    "Riva",
    "Serra",
    "Villa",
    "Neri",
    "Gallo",
    "Fontana",
    "Costa",
    "Marino",
    "Ferri",
    "Bruno",
)


def _any_pred(**kw: str) -> m.Predicate:
    base = dict(
        age_band="any",
        social_class="any",
        trade="any",
        circle="any",
        faction="any",
        influencer="any",
        ideology_extreme="any",
        role="any",
    )
    base.update(kw)
    return m.Predicate(**base)


PILLAR_KW = {
    "press": ("press", "newspaper", "journalist", "media", "gazette", "broadcast", "radio", "news"),
    "law": ("judge", "court", "magistrate", "law ", "trial", "justice", "tribunal"),
    "pluralism": ("election", "council", "opposition", "party", "vote", "candidate", "ballot"),
    "property": (
        "tax",
        "property",
        "business",
        "market",
        "shop",
        "seize",
        "land",
        "levy",
        "licen",
        "tariff",
        "firm",
    ),
    "civil": (
        "police",
        "curfew",
        "arrest",
        "protest",
        "assembly",
        "patrol",
        "surveil",
        "militia",
        "security",
        "march",
        "rally",
    ),
}
PILLAR_DOMAIN = {
    "pluralism": "institutions",
    "law": "institutions",
    "press": "press",
    "property": "economy",
    "civil": "security",
}
HIGH_KW = (
    "ban",
    "close",
    "abolish",
    "arrest",
    "seize",
    "dissolve",
    "outlaw",
    "shut",
    "jail",
    "forbid",
    "purge",
)
MED_KW = (
    "restrict",
    "limit",
    "suspend",
    "cut",
    "fine",
    "require",
    "permit",
    "freeze",
    "tax",
    "levy",
    "control",
)
LOW_KW = ("review", "inspect", "ask", "urge", "audit", "study", "invite", "request")
POS_ECON = ("subsid", "bonus", "grant", "free", "relief", "reward", "pay raise", "build")
NEG_ECON = ("tax", "fine", "levy", "seize", "tariff")
PRED_KW = [
    (("journalist",), dict(trade="journalist")),
    (("judge", "magistrate"), dict(trade="magistrate")),
    (("merchant", "shop"), dict(trade="merchant")),
    (("councillor", "council"), dict(role="councillor")),
    (("police",), dict(trade="law_enforcement")),
    (("rich", "wealthy", "elite"), dict(social_class="high")),
    (("poor",), dict(social_class="low")),
    (("youth", "young", "student"), dict(age_band="young")),
    (("elder", "pensioner"), dict(age_band="elderly")),
    (("farmer",), dict(trade="farmer")),
    (("worker",), dict(trade="worker")),
    (("priest", "clergy", "cleric"), dict(trade="cleric")),
    (("entrepreneur", "business owner"), dict(trade="entrepreneur")),
]


def shift_label(idx, rng: GameRng, eps: float, lo: int = 0, hi: int = 6):
    """Ordinal noise (15.4): with probability eps a label shifts one step up or down."""
    idx = np.asarray(idx)
    if eps <= 0:
        return idx
    hit = rng.random(idx.shape) < eps
    step = np.where(rng.random(idx.shape) < 0.5, -1, 1)
    return np.clip(np.where(hit, idx + step, idx), lo, hi)


def activity_weights(ctx: LabelContext, n_act: int, act_index: dict[str, int]) -> np.ndarray:
    """Design affinity table (15.3): weights of the daily-activity choice, from labels only."""
    mcount = len(ctx)
    W = np.zeros((mcount, n_act))
    T = bip_value(ctx.T)
    C = bip_value(ctx.C)
    I = bip_value(ctx.I)
    O = bip_value(ctx.O)
    fear = uni_value(ctx.fear)
    young = (ctx.age_band == 1).astype(float)
    active = (ctx.age_band >= 1) & (ctx.age_band <= 2)  # young and adult work; elderly are retired
    W[:, act_index["work"]] = np.where(active, 1.0, 0.0)
    W[:, act_index["temple"]] = 0.10 + 0.40 * (I[:, 0] + 1) / 2
    W[:, act_index["study"]] = 0.05 + 0.35 * (T[:, 0] + 1) / 2 + 0.30 * young
    W[:, act_index["associations"]] = 0.10 + 0.40 * (C[:, 1] + 1) / 2
    W[:, act_index["leisure"]] = 0.30
    disc = np.maximum(0.0, -O[:, 0])
    W[:, act_index["protest"]] = 1.2 * disc * (1 - fear) * (T[:, 0] + 1) / 2
    return W


class SimulatedEngine:
    """Implements every task of the closed list with a deterministic rule (15.3) and controlled noise (15.4)."""

    def __init__(self, catalogs: Catalogs, params: Params, name: str = "simulated", perfect: bool = False):
        self.cat = catalogs
        self.p = params
        self.name = name
        self.perfect = perfect
        self.eps_t = 0.0 if perfect else params.technical.eps_t
        self.eps_err = 0.0 if perfect else params.technical.eps_err
        self.tau = 0.0 if perfect else params.technical.tau_sim
        self.lam_ha = params.technical.lambda_HA
        self.n_retry = params.technical.n_retry
        self._exec: dict[str, ExecutionRecord] = {}
        self._warn: list[dict[str, Any]] = []
        self._faction_counter = 0
        self.act_index = {a: i for i, a in enumerate(catalogs.activity_ids())}
        self.vocab = catalogs.vocab()

    # ------------------------------------------------------------------ accounting
    def _rec(self, task: str) -> ExecutionRecord:
        return self._exec.setdefault(task, ExecutionRecord(task=task))

    def drain_execution(self) -> list[ExecutionRecord]:
        out = list(self._exec.values())
        self._exec = {}
        return out

    def drain_warnings(self) -> list[dict[str, Any]]:
        out, self._warn = self._warn, []
        return out

    def _attempt(self, task: str, rng: GameRng) -> tuple[bool, int]:
        """Failure injection (eps_err): returns (succeeded, attempts used)."""
        rec = self._rec(task)
        rec.calls += 1
        for a in range(1, self.n_retry + 2):
            rec.attempts += 1
            if self.eps_err <= 0 or rng.random() >= self.eps_err:
                return True, a
        rec.fallbacks += 1
        return False, self.n_retry + 1

    def _warn_fallback(self, task: str, inp: Any) -> None:
        self._warn.append({"task": task, "input": str(inp)[:200], "response": "<malformed>"})

    # ------------------------------------------------------------------ generic call
    def call(self, task: str, inp: BaseModel, rng: GameRng) -> BaseModel | None:
        spec = TASKS[task]
        handler = getattr(self, f"_t_{task}")
        ok, _ = self._attempt(task, rng)
        if not ok:
            self._warn_fallback(task, inp)
            return getattr(self, f"_fb_{task}", lambda i, r: None)(inp, rng)
        out = handler(inp, rng)
        assert out is None or isinstance(out, spec.output_model), f"{task} returned {type(out)}"
        return out

    # ------------------------------------------------------------------ S2: generate_citizens / S1: citizen_profile
    def _t_generate_citizens(self, inp: m.GenerateCitizensIn, rng: GameRng) -> m.GenerateCitizensOut:
        sc = self.cat.scenario(inp.scenario)
        drafts = []
        for i in range(inp.number):
            if inp.parents and inp.parents[i]:
                par = inp.parents[i]
                T = _mean_shift(par["T"], rng)
                C = _mean_shift(par["C"], rng)
                I = _mean_shift(par["I"], rng)
            else:
                T = np.clip(np.rint(rng.normal(0, 0.35, 4) * 3), -3, 3).astype(int)
                C = np.clip(np.rint(rng.normal(0, self.p.vectors.sd_C_init, 3) * 3), -3, 3).astype(int)
                if sc.bimodal:
                    mean1 = sc.axis1_mean if rng.random() < 0.5 else -sc.axis1_mean
                else:
                    mean1 = sc.axis1_mean
                I = np.empty(3)
                I[0] = rng.normal(mean1, sc.axis1_sd)
                I[1:] = rng.normal(0, 0.35, 2)
                I = np.clip(np.rint(np.clip(I, -1, 1) * 3), -3, 3).astype(int)
            name = f"{FIRST_NAMES[int(rng.integers(0, len(FIRST_NAMES)))]} {SURNAMES[int(rng.integers(0, len(SURNAMES)))]}"
            desc = (
                "T=" + ",".join(map(str, T)) + ";C=" + ",".join(map(str, C)) + ";I=" + ",".join(map(str, I))
            )
            drafts.append(m.CitizenDraft(name=name, description=desc))
        return m.GenerateCitizensOut(citizens=drafts)

    def _fb_generate_citizens(self, inp: m.GenerateCitizensIn, rng: GameRng) -> m.GenerateCitizensOut:
        return m.GenerateCitizensOut(
            citizens=[
                m.CitizenDraft(name=f"Citizen {i}", description="T=0,0,0,0;C=0,0,0;I=0,0,0")
                for i in range(inp.number)
            ]
        )

    def _t_citizen_profile(self, inp: m.CitizenProfileIn, rng: GameRng) -> m.CitizenProfileOut:
        parts = dict(kv.split("=") for kv in inp.description.split(";"))
        T = [int(x) + 3 for x in parts["T"].split(",")]
        C = [int(x) + 3 for x in parts["C"].split(",")]
        I = [int(x) + 3 for x in parts["I"].split(",")]
        allv = shift_label(np.array(T + C + I), rng, self.eps_t)
        names = [BIPOLAR[int(v)] for v in allv]
        trade = self.cat.trade_ids()[0]
        return m.CitizenProfileOut(
            NS=names[0],
            HA=names[1],
            RD=names[2],
            P=names[3],
            SD=names[4],
            CO=names[5],
            ST=names[6],
            I1=names[7],
            I2=names[8],
            I3=names[9],
            trade=trade,
        )

    def _fb_citizen_profile(self, inp, rng) -> m.CitizenProfileOut:
        z = "neutral"
        return m.CitizenProfileOut(
            NS=z, HA=z, RD=z, P=z, SD=z, CO=z, ST=z, I1=z, I2=z, I3=z, trade=self.cat.trade_ids()[0]
        )

    # ------------------------------------------------------------------ S1: classify_decree
    def _t_classify_decree(self, inp: m.ClassifyDecreeIn, rng: GameRng) -> m.ClassifyDecreeOut:
        if inp._structure is not None:
            return inp._structure
        return self.classify_text(inp.text)

    def _fb_classify_decree(self, inp, rng):
        raise UnclassifiableDecree("the decree could not be classified; please rephrase it")

    def classify_text(self, text: str) -> m.ClassifyDecreeOut:
        """Keyword classifier for free text: a development convenience of the simulated S1 (a real S1 reads
        the text itself). Structured decrees of the simulated player never pass through here."""
        t = " " + text.lower() + " "
        pillar = "none"
        score = 0
        for pid, kws in PILLAR_KW.items():
            s = sum(1 for k in kws if k in t)
            if s > score:
                pillar, score = pid, s
        announce = any(k in t for k in ("announce", "thank", "celebrate", "festival", "welcome", "honor"))
        econ_pos = any(k in t for k in POS_ECON)
        econ_neg = any(k in t for k in NEG_ECON)
        if pillar == "none" and not (announce or econ_pos or econ_neg or "all" in t or "build" in t):
            raise UnclassifiableDecree("the decree could not be classified; please rephrase it")
        agg = 2
        if any(k in t for k in LOW_KW):
            agg = 1
        if any(k in t for k in MED_KW):
            agg = 3
        if any(k in t for k in HIGH_KW):
            agg = 5
        if " all " in t or " every " in t or "total" in t:
            agg = min(6, agg + 1)
        if pillar == "none":
            agg = min(agg, 2)
        pred: dict[str, str] = {}
        for kws, d in PRED_KW:
            if any(k in t for k in kws):
                pred.update(d)
        econ = MASK
        if econ_pos and not econ_neg:
            econ = bip_name(0.5)
        elif econ_neg and not econ_pos:
            econ = bip_name(-0.33)
        transfer = None
        if "tax" in t or "levy" in t:
            transfer = m.TransferOut(
                payers=_any_pred(**(pred or {"social_class": "high"})),
                beneficiaries=_any_pred(social_class="low"),
                intensity=uni_name(max(1, agg)),
                duration="one_off",
            )
        domain = PILLAR_DOMAIN.get(
            pillar, "economy" if (econ_pos or econ_neg) else "worship" if announce else "institutions"
        )
        return m.ClassifyDecreeOut(
            pillar=pillar,
            domain=domain,
            aggressiveness=uni_name(agg),
            predicate=_any_pred(**pred),
            tec_effect=MASK,
            cul_effect=bip_name(0.33) if announce else MASK,
            econ_effect=econ,
            threat="none",
            transfer=transfer,
            revocation=None,
        )

    # ------------------------------------------------------------------ S1: rank_event_types
    def _t_rank_event_types(self, inp: m.RankEventTypesIn, rng: GameRng) -> m.RankEventTypesOut:
        stance = float(inp.actor_profile.get("stance_value", "0"))  # actor's stance toward the mayor, -1..1
        strength = float(inp.actor_profile.get("strength_value", "0.5"))
        scores = []
        for eid in inp.admitted:
            et = self.cat.event_type(eid)
            hostile_pref = -stance  # hostile actors prefer confrontational types
            s = et.weight * (1.0 + 0.8 * et.hostility * hostile_pref) + 0.3 * strength * abs(et.hostility)
            scores.append(s + 1e-6 * rng.random())  # ties resolved by the RNG
        order = [inp.admitted[i] for i in np.argsort(-np.array(scores))]
        return m.RankEventTypesOut(order=order)

    def _fb_rank_event_types(self, inp: m.RankEventTypesIn, rng: GameRng) -> m.RankEventTypesOut:
        perm = rng.permutation(len(inp.admitted))
        return m.RankEventTypesOut(order=[inp.admitted[i] for i in perm])

    # ------------------------------------------------------------------ narrate_event (S2)
    def _t_narrate_event(self, inp: m.NarrateEventIn, rng: GameRng) -> m.NarrateEventOut:
        seed = inp.seed
        hid = inp._hidden or {}
        et = self.cat.event_type(seed.event_type)
        K = self.p.events.K_opt
        levels = [STANCE_LEVELS[i % len(STANCE_LEVELS)] for i in range(K)]
        inten = uni_idx(seed.intensity)
        actor_kind = seed.actor.get("kind", "exogenous")
        sign = float(hid.get("actor_sign", 1.0))  # sign of the actor's stance toward the mayor
        pillar_idx = hid.get("pillar_idx", None)
        attacks = int(hid.get("attacks", et.attacks))
        econ = float(hid.get("econ", et.econ))
        dirv = hid.get("actor_dir")
        if et.direction in DIRECTIONS:
            dir_I = np.array(DIRECTIONS[et.direction])
        elif et.direction == "actor" and dirv is not None:
            dir_I = np.asarray(dirv, dtype=float)
        else:
            dir_I = np.zeros(3)
        dir_C = np.asarray(et.t_C, dtype=float)
        feats: list[m.OptionFeatureOut] = []
        options: list[m.OptionOut] = []
        tags: list[str] = []
        if inten >= 4:
            tags.append("determinant")
        if et.polarizing:
            tags.append("polarizing")
        if actor_kind == "exogenous":
            tags.append("exogenous")
        birth = bool(seed.can_found and et.can_found and rng.random() < 0.35)
        if birth:
            tags.append("faction_birth")
        adherence_done = False
        for k, s in enumerate(levels):
            sv = s / 3.0
            r = rng.random()
            shift = -1 if r < 0.25 else (1 if r > 0.75 else 0)
            if k >= 7:
                shift = -1
            Fk = int(np.clip(inten + shift, 0, 6))
            if actor_kind in ("exogenous",):
                st_actor = MASK
            else:
                st_actor = stance_name(sv)
            if actor_kind == "mayor":
                mayor_v = sv
            elif actor_kind == "exogenous":
                mayor_v = sv * sign if et.target == "mayor" else float("nan")
            elif actor_kind in ("threat",):
                mayor_v = -sv
            else:
                mayor_v = sv * (1.0 if sign >= 0 else -1.0)
            st_pillars = [MASK] * 5
            if pillar_idx is not None and pillar_idx >= 0 and attacks != 0:
                acc = mayor_v if attacks > 0 else -sv  # support for a defender is opposition to erosion
                st_pillars[pillar_idx] = stance_name(acc) if acc == acc else MASK
            tI = [bip_name(sv * d) if abs(d) >= 1 / 6 else MASK for d in dir_I]
            tC = [bip_name(sv * d) if d == d and abs(d) >= 1 / 6 else MASK for d in dir_C]
            adherent = bool(birth and not adherence_done and s == 3)
            adherence_done = adherence_done or adherent
            feats.append(
                m.OptionFeatureOut(
                    t_I1=tI[0],
                    t_I2=tI[1],
                    t_I3=tI[2],
                    t_C1=tC[0],
                    t_C2=tC[1],
                    t_C3=tC[2],
                    e_r=bip_name(econ) if econ != 0 else MASK,
                    F=uni_name(Fk),
                    stance_mayor=stance_name(mayor_v) if mayor_v == mayor_v else MASK,
                    stance_pillar_1=st_pillars[0],
                    stance_pillar_2=st_pillars[1],
                    stance_pillar_3=st_pillars[2],
                    stance_pillar_4=st_pillars[3],
                    stance_pillar_5=st_pillars[4],
                    stance_actor=st_actor,
                    stance_target=MASK,
                )
            )
            options.append(
                m.OptionOut(text=f"[{STANCE[s + 3]}] {et.id.replace('_', ' ')}", adherence=adherent)
            )
        title = f"{seed.actor.get('name', actor_kind.title())}: {et.id.replace('_', ' ')}"
        out = m.NarrateEventOut(
            title=title, text=f"{title}, {seed.intensity} intensity.", tags=tags, options=options
        )
        out._annotation = feats
        return out

    def _fb_narrate_event(self, inp, rng):
        return None  # the event does not happen (13.3)

    # ------------------------------------------------------------------ S1 extraction
    def _t_news_feature(self, inp: m.NewsFeatureIn, rng: GameRng) -> m.NewsFeatureOut:
        idx = shift_label(np.array(uni_idx(inp.news.intensity)), rng, self.eps_t)
        return m.NewsFeatureOut(intensity=uni_name(int(idx)), transfer=None)

    def _fb_news_feature(self, inp, rng):
        return m.NewsFeatureOut(intensity=inp.news.intensity, transfer=None)

    def _t_option_feature(self, inp: m.OptionFeatureIn, rng: GameRng) -> m.OptionFeatureOut:
        ann: m.OptionFeatureOut = inp._annotation
        if ann is None or self.eps_t <= 0:
            return ann
        d = ann.model_dump()
        for key, val in d.items():
            if key == "F":
                d[key] = uni_name(int(shift_label(np.array(uni_idx(val)), rng, self.eps_t)))
            elif val != MASK:
                names = STANCE if key.startswith("stance") else BIPOLAR
                d[key] = names[int(shift_label(np.array(names.index(val)), rng, self.eps_t))]
        return m.OptionFeatureOut(**d)

    def _fb_option_feature(self, inp, rng):
        z = MASK
        return m.OptionFeatureOut(
            t_I1=z,
            t_I2=z,
            t_I3=z,
            t_C1=z,
            t_C2=z,
            t_C3=z,
            e_r=z,
            F="null",
            stance_mayor=z,
            stance_pillar_1=z,
            stance_pillar_2=z,
            stance_pillar_3=z,
            stance_pillar_4=z,
            stance_pillar_5=z,
            stance_actor=z,
            stance_target=z,
        )

    # ------------------------------------------------------------------ pillar_level
    def _t_pillar_level(self, inp: m.PillarLevelIn, rng: GameRng) -> m.PillarLevelOut:
        pil = self.cat.pillar(inp.pillar)
        ids = [lv.id for lv in pil.levels]
        cur = 0 if inp.current_level == f"{pil.id}_0" else ids.index(inp.current_level) + 1
        et = self.cat.event_type(inp.news.event_type)
        inten = uni_idx(inp.news.intensity)
        cand = cur
        if et.attacks > 0 or et.id == "pillar_fall":
            for k, lv in enumerate(pil.levels, start=1):
                if et.id in lv.event_types and inten >= lv.min_intensity:
                    cand = max(cand, k)
            if et.id == "pillar_fall":
                cand = pil.L
        elif et.attacks < 0:  # custodians defend: reactive demotion, deeper for intense events
            cand = max(0, cur - (1 if inten < 5 else 2))
        cand = int(np.clip(shift_label(np.array(cand), rng, self.eps_t, 0, pil.L), 0, pil.L))
        return m.PillarLevelOut(level=f"{pil.id}_{cand}")

    def _fb_pillar_level(self, inp, rng):
        return m.PillarLevelOut(level=inp.current_level)

    # ------------------------------------------------------------------ S2 service texts
    def _t_name_faction(self, inp: m.NameFactionIn, rng: GameRng) -> m.NameFactionOut:
        self._faction_counter += 1
        return m.NameFactionOut(
            name=f"Movement {self._faction_counter}", label=f"movement-{self._faction_counter}"
        )

    def _fb_name_faction(self, inp, rng):
        return m.NameFactionOut(name="Movement", label="movement")

    def _t_message_responses(self, inp: m.MessageResponsesIn, rng: GameRng) -> m.MessageResponsesOut:
        return m.MessageResponsesOut(options=["accept", "ignore", "reject"])

    _fb_message_responses = _t_message_responses

    def _t_newspaper_articles(self, inp: m.NewspaperArticlesIn, rng: GameRng) -> m.NewspaperArticlesOut:
        return m.NewspaperArticlesOut(articles=[m.Article(title=i.title, text=i.text) for i in inp.items])

    _fb_newspaper_articles = _t_newspaper_articles

    def _t_annual_edition(self, inp: m.TextIn, rng: GameRng) -> m.TextOut:
        return m.TextOut(
            text=" | ".join(f"day {e.get('day', '?')}: {e.get('title', '')}" for e in inp.log[:10])
        )

    _t_almanac = _t_annual_edition
    _fb_annual_edition = _t_annual_edition
    _fb_almanac = _t_annual_edition

    def _t_suggest_decree(self, inp: m.SuggestDecreeIn, rng: GameRng) -> m.SuggestDecreeOut:
        ideas = {
            "press": "Newspapers must submit stories for review",
            "law": "Judges answer to the mayor's office",
            "pluralism": "Postpone the next council election",
            "property": "Shops need a licence the mayor can revoke",
            "civil": "Public gatherings need a permit",
        }
        order = sorted(ideas, key=lambda p: inp.pillars.get(p, "medium"))
        return m.SuggestDecreeOut(decrees=[ideas[p] for p in order[:3]])

    def _fb_suggest_decree(self, inp, rng):
        return m.SuggestDecreeOut(decrees=[])

    def _t_choose_influencer_response(
        self, inp: m.ChooseInfluencerResponseIn, rng: GameRng
    ) -> m.ChooseInfluencerResponseOut:
        o_mayor = float(inp.context.get("opinion_mayor", 0.0))
        pref = {"accept": 1.0, "ignore": 0.0, "reject": -1.0}
        best = max(
            inp.options, key=lambda o: -abs(pref.get(o.id, 0.0) - o_mayor) + self.tau * float(rng.gumbel())
        )
        return m.ChooseInfluencerResponseOut(option=best.id)

    def _fb_choose_influencer_response(self, inp, rng):
        return m.ChooseInfluencerResponseOut(option="ignore")

    # ------------------------------------------------------------------ S1: choose_reaction (batch)
    def choose_reaction_many(
        self,
        ctx: LabelContext,
        actor: ActorCtx,
        news: Any,
        options: list[Any],
        features: OptionFeatures,
        rng: GameRng,
    ) -> np.ndarray:
        """Utility model of 15.3: u = -d(s_hat, s_k) - lambda_HA * max(0, HA) * e_k + tau * Gumbel."""
        mcount = len(ctx)
        rec = self._rec("choose_reaction")
        rec.calls += mcount
        if mcount == 0:
            return np.zeros(0, dtype=int)
        S = np.concatenate(
            [features.stance[:, [6]], features.stance[:, [0]], features.stance[:, 1:6]], axis=1
        )  # (K,7): actor, mayor, pillars
        O_hat = bip_value(ctx.O)  # (m,6)
        mayor_hat = O_hat[:, 0]
        if actor.kind == "mayor" or actor.kind in ("threat", "system"):
            actor_hat = mayor_hat
        elif actor.kind in ("faction", "influencer") and actor.ideology is not None:
            d = dist(bip_value(ctx.I), bip_value(actor.ideology)[None, :])
            actor_hat = np.clip(1.0 - 4.0 * d, -1.0, 1.0)
        elif actor.kind in ("class", "custodian") and ctx.member is not None:
            actor_hat = np.where(ctx.member, 1.0 / 3.0, 0.0)
        else:
            actor_hat = np.zeros(mcount)
        s_hat = np.concatenate([actor_hat[:, None], O_hat], axis=1)  # (m,7)
        present = ~np.isnan(S)  # (K,7)
        npresent = np.maximum(1, present.sum(axis=1))  # (K,)
        diff = np.where(present[None, :, :], (s_hat[:, None, :] - np.nan_to_num(S)[None, :, :]) ** 2, 0.0)
        d = np.sqrt(diff.sum(axis=2)) / (2.0 * np.sqrt(npresent))[None, :]  # (m,K)
        e = np.max(np.where(present, np.abs(S), 0.0), axis=1)
        HA = np.maximum(0.0, bip_value(ctx.T[:, 1]))
        U = -d - self.lam_ha * HA[:, None] * e[None, :]
        if self.tau > 0:
            U = U + self.tau * rng.gumbel(U.shape)
        choice = U.argmax(axis=1)
        # contract failures: retries, then fallback to the option with stance toward the mayor closest to indifference
        if self.eps_err > 0:
            fails = rng.random((mcount, self.n_retry + 1)) < self.eps_err
            first_ok = np.where(~fails, np.arange(self.n_retry + 1)[None, :], 99).min(axis=1)
            fb = first_ok == 99
            rec.attempts += int(np.where(fb, self.n_retry + 1, first_ok + 1).sum())
            if fb.any():
                rec.fallbacks += int(fb.sum())
                mv = np.abs(np.nan_to_num(features.stance[:, 0], nan=9.0))
                choice = np.where(fb, int(mv.argmin()), choice)
                self._warn.append(
                    {
                        "task": "choose_reaction",
                        "input": f"{int(fb.sum())} citizens",
                        "response": "<malformed>",
                    }
                )
        else:
            rec.attempts += mcount
        return choice

    # ------------------------------------------------------------------ S1: choose_activity (batch)
    def choose_activity_many(self, ctx: LabelContext, rng: GameRng) -> np.ndarray:
        mcount = len(ctx)
        rec = self._rec("choose_activity")
        rec.calls += mcount
        rec.attempts += mcount
        if mcount == 0:
            return np.zeros(0, dtype=int)
        A = len(self.act_index)
        W = activity_weights(ctx, A, self.act_index)
        g = rng.gumbel(W.shape)
        choice = (np.log(np.maximum(W, 1e-12)) + g).argmax(axis=1)
        if self.eps_t > 0:  # a catalog id is replaced with a neighboring id
            hit = rng.random(mcount) < self.eps_t
            step = np.where(rng.random(mcount) < 0.5, -1, 1)
            choice = np.where(hit, (choice + step) % A, choice)
        if self.eps_err > 0:
            fails = rng.random((mcount, self.n_retry + 1)) < self.eps_err
            fb = fails.all(axis=1)
            if fb.any():
                rec.fallbacks += int(fb.sum())
                choice = np.where(fb, self.act_index["work"], choice)
        return choice


def _mean_shift(s: str, rng: GameRng) -> np.ndarray:
    """Newborn labels: mean of the parents plus a shift of one step with probability 1/3 (15.3)."""
    vals = np.array([[int(x) for x in part.split(",")] for part in s.split("|")], dtype=float)
    mean = np.rint(vals.mean(axis=0)).astype(int)
    shift = np.where(rng.random(mean.shape) < 1 / 3, np.where(rng.random(mean.shape) < 0.5, -1, 1), 0)
    return np.clip(mean + shift, -3, 3)
