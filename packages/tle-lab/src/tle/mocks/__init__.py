"""Fit of the simulated engines against a real model (spec 15, 17): the records of a real S1 (see
``tle.models.recording``) are replayed through the utility model of the simulated engine, and its parameters are the ones
that maximize the likelihood of the real choices. Calibration logic lives here, in the lab, not in the engine."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from tle.catalogs.loader import load_catalogs
from tle.engines.protocol import ActorCtx, LabelContext, OptionFeatures
from tle.engines.simulated import SimulatedEngine
from tle.params import default_params

FIT_PARAMS = ("lambda_HA", "tau_sim", "w_actor_stance", "beta_support", "support_gate", "w_pillar_stance")
# structural priors keep the mechanisms alive: aversion to extremes and sensitivity to the actor never fit to zero
BOUNDS = {
    "lambda_HA": (0.05, 1.5),
    "tau_sim": (0.02, 3.0),
    "w_actor_stance": (0.5, 4.0),
    "beta_support": (0.0, 3.0),
    "support_gate": (0.0, 1.0),
    "w_pillar_stance": (0.25, 10.0),
}


def load_reactions(paths: list[Path | str]) -> list[dict]:
    out = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            if r.get("kind") == "reaction":
                out.append(r)
    return out


def _ctx(rec: dict) -> tuple[LabelContext, ActorCtx, OptionFeatures, np.ndarray]:
    c = rec["ctx"]
    m_ = len(c["age_band"])
    ctx = LabelContext(
        ids=np.arange(m_),
        age_band=np.array(c["age_band"]),
        social_class=np.array(c["social_class"]),
        trade=np.zeros(m_, int),
        circle=np.zeros(m_, int),
        T=np.array(c["T"]),
        C=np.array(c["C"]),
        I=np.array(c["I"]),
        O=np.array(c["O"]),
        fear=np.array(c["fear"]),
        risk=np.zeros(m_, int),
        member=None if c["member"] is None else np.array(c["member"], bool),
        social=None if c.get("social") is None else np.array(c["social"]),
    )
    st = np.array([[np.nan if v is None else v for v in row] for row in rec["stance"]], dtype=float)
    K = len(st)
    feats = OptionFeatures(
        t_I=np.full((K, 3), np.nan), t_C=np.full((K, 3), np.nan), e_r=np.zeros(K), F=np.ones(K), stance=st
    )
    ideo = rec["actor"]["ideology"]
    actor = ActorCtx(kind=rec["actor"]["kind"], ideology=None if ideo is None else np.array(ideo))
    return ctx, actor, feats, np.array(rec["choice"])


def log_likelihood(records: list[dict], theta: dict[str, float], cat=None) -> tuple[float, int, float]:
    """(sum of log P(real choice), number of choices, agreement of the argmax with the real choice)."""
    cat = cat or load_catalogs()
    p = default_params().with_overrides(theta)
    eng = SimulatedEngine(cat, p, perfect=True)
    ll, n, agree = 0.0, 0, 0
    for rec in records:
        ctx, actor, feats, choice = _ctx(rec)
        U = eng.choice_utilities(ctx, actor, feats, params=p) / max(theta["tau_sim"], 1e-6)
        U = U - U.max(axis=1, keepdims=True)
        logp = U - np.log(np.exp(U).sum(axis=1, keepdims=True))
        ll += float(logp[np.arange(len(choice)), choice].sum())
        agree += int((U.argmax(axis=1) == choice).sum())
        n += len(choice)
    return ll, n, agree / max(n, 1)


def fit(records: list[dict]) -> dict:
    cat = load_catalogs()
    base = default_params()
    x0 = np.array([getattr(base.technical, k) for k in FIT_PARAMS], dtype=float)
    lo = np.array([BOUNDS[k][0] for k in FIT_PARAMS])
    hi = np.array([BOUNDS[k][1] for k in FIT_PARAMS])

    def theta(x):
        return {k: float(np.clip(v, lo[i], hi[i])) for i, (k, v) in enumerate(zip(FIT_PARAMS, x))}

    ll0, n, agree0 = log_likelihood(records, theta(x0), cat)
    res = minimize(
        lambda x: -log_likelihood(records, theta(x), cat)[0],
        x0,
        method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 200},
    )
    best = theta(res.x)
    ll1, _, agree1 = log_likelihood(records, best, cat)
    K = np.mean([len(r["stance"]) for r in records])
    return {
        "n_choices": n,
        "events": len(records),
        "uniform_loglik": -n * float(np.log(K)),
        "before": {"loglik": ll0, "agreement": agree0, "params": theta(x0)},
        "after": {"loglik": ll1, "agreement": agree1, "params": best},
    }


def write_toml(result: dict, path: Path | str) -> None:
    lines = [
        "# Fit of the simulated S1 against a real model (tle mocks fit). Promote by editing config/params/technical.toml.",
        f"n_choices = {result['n_choices']}",
        f"events = {result['events']}",
        f"loglik_before = {result['before']['loglik']:.2f}",
        f"loglik_after = {result['after']['loglik']:.2f}",
        f"loglik_uniform = {result['uniform_loglik']:.2f}",
        f"agreement_before = {result['before']['agreement']:.3f}",
        f"agreement_after = {result['after']['agreement']:.3f}",
        "",
        "[params]",
    ]
    lines += [f"{k} = {v:.4f}" for k, v in result["after"]["params"].items()]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------------------------- decree bank
def read_battery(path: Path | str) -> list[str]:
    return [x.strip() for x in Path(path).read_text().splitlines() if x.strip() and not x.startswith("#")]


def _nonany(pred) -> dict:
    return {k: v for k, v in pred.model_dump().items() if v != "any"}


def bank_entry(text: str, out) -> dict:
    """One labelled decree as a plain dict (what the simulated classifier needs to reproduce the answer)."""
    e = {
        "text": text,
        "pillar": out.pillar,
        "domain": out.domain,
        "aggressiveness": out.aggressiveness,
        "tec_effect": out.tec_effect,
        "cul_effect": out.cul_effect,
        "econ_effect": out.econ_effect,
        "predicate": _nonany(out.predicate),
    }
    if out.transfer is not None:
        e["transfer"] = {
            "payers": _nonany(out.transfer.payers),
            "beneficiaries": _nonany(out.transfer.beneficiaries),
            "intensity": out.transfer.intensity,
            "duration": out.transfer.duration,
        }
    return e


def write_bank(entries: list[dict], path: Path | str) -> None:
    def toml_val(v):
        if isinstance(v, dict):
            return "{ " + ", ".join(f"{k} = {toml_val(x)}" for k, x in v.items()) + " }" if v else "{}"
        return json.dumps(v)

    lines = [
        "# Decrees labelled by a real S1 (tle mocks label-decrees). Bootstrap data of the simulated classifier.",
        "",
    ]
    for e in entries:
        lines.append("[[decrees]]")
        lines += [f"{k} = {toml_val(v)}" for k, v in e.items()]
        lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))


def evaluate_classifier(entries: list[dict], leave_one_out: bool = True, use_bank: bool = True) -> dict:
    """Agreement of the simulated classifier with the labels of the real S1 on the bank itself: exact match of pillar and
    domain, aggressiveness within one step, same transfer duration; with the bank, each decree is held out in turn."""
    from tle.contracts.vocab import uni_idx
    from tle.core.labels import UNIPOLAR_NAMES  # noqa: F401  (documents the scale used for aggressiveness)

    cat = load_catalogs()
    eng = SimulatedEngine(cat, default_params(), perfect=True)
    n = ok_p = ok_d = ok_a = ok_t = skipped = 0
    for i, e in enumerate(entries):
        bank = (
            [x for j, x in enumerate(entries) if j != i]
            if (use_bank and leave_one_out)
            else ([] if not use_bank else entries)
        )
        try:
            out = eng.classify_text(e["text"], bank)
        except Exception:
            skipped += 1
            n += 1  # a refused decree counts as a miss
            continue
        n += 1
        ok_p += out.pillar == e["pillar"]
        ok_d += out.domain == e["domain"]
        ok_a += abs(uni_idx(out.aggressiveness) - uni_idx(e["aggressiveness"])) <= 1
        ok_t += (out.transfer is not None) == ("transfer" in e)
    return {
        "n": n,
        "unclassifiable": skipped,
        "pillar": ok_p / max(n, 1),
        "domain": ok_d / max(n, 1),
        "aggressiveness_within_1": ok_a / max(n, 1),
        "transfer_detected": ok_t / max(n, 1),
    }
