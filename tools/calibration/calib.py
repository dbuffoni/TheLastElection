"""Joint calibration on the mocks: population (passive baseline), impulse test and outcome bands in one objective."""

import json
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from concurrent.futures import ProcessPoolExecutor

from sweep import one

STRATS = ["assault", "gradual", "populist", "random", "passive", "fear", "clientelist", "capture", "rentier"]


def impulse_job(args):
    ov, seed = args
    from tle.catalogs.loader import load_catalogs
    from tle.experiments import impulse_test
    from tle.loop.game import Game
    from tle.params import default_params

    p = default_params().with_overrides(ov) if ov else default_params()
    r = impulse_test(
        Game(load_catalogs(), p, scenario="industrial_secular", seed=seed, n=300), 0.2, "medium", 180.0
    )
    return r.involved_peak, r.others_peak, r.half_life_days


def to_overrides(raw):
    """theta_outcome sits above the baseline pressure of a loyal-but-divided council (5/12 in opposition), by `margin`."""
    ov = {k: v for k, v in raw.items() if k != "margin"}
    if "margin" in raw:
        f = ov["w_M"] * 0.42
        G = ov["x_star_T"] / (1.0 - ov["x_star_T"])
        ov["theta_outcome"] = float(f * G / (f * G + 1.0) + raw["margin"])
    return ov


def dist(v, lo, hi):
    if v != v:
        return 1.0
    return max(0.0, lo - v) / max(hi - lo, 1e-9) if v < lo else max(0.0, v - hi) / max(hi - lo, 1e-9)


def evaluate(ov, seeds, ex, impulse_seeds=(1, 2, 3)):
    jobs = [(ov, s, st) for st in STRATS for s in seeds]
    futs = [ex.submit(one, j) for j in jobs]
    ifuts = [ex.submit(impulse_job, (ov, s)) for s in impulse_seeds]
    res = {}
    for (o, s, st), f in zip(jobs, futs):
        res.setdefault(st, []).append(f.result())
    imp = np.array([f.result() for f in ifuts])
    out = {}
    for st in STRATS:
        rs = res[st]
        n = len(rs)
        out[st] = {
            "V": sum(r["outcome"] == "victory" for r in rs) / n,
            "R": sum(r["outcome"] == "removed" for r in rs) / n,
            "fallen": float(np.mean([r["fallen"] for r in rs])),
        }
    pas = [r for r in res["passive"] if "lab" in r]
    out["pop"] = {
        k: float(np.mean([r[k] for r in pas])) if pas else float("nan")
        for k in ("lab", "disp", "sat", "birth", "popdrift")
    }
    out["pop"]["lab_in"] = float(np.mean([0.15 <= r["lab"] <= 0.35 for r in pas])) if pas else 0.0
    out["pop"]["disp_in"] = float(np.mean([r["disp"] >= 0.1667 for r in pas])) if pas else 0.0
    out["imp"] = {
        "involved": float(imp[:, 0].mean()),
        "others": float(imp[:, 1].mean()),
        "half": float(np.nanmean(imp[:, 2])),
        "involved_in": float(np.mean([0.25 <= x <= 0.417 for x in imp[:, 0]])),
        "others_in": float(np.mean([0.072 <= x <= 0.15 for x in imp[:, 1]])),
    }
    return out


def loss(o):
    l = 0.0
    p, i = o["pop"], o["imp"]
    l += 1.0 - p["lab_in"] + 1.0 - p["disp_in"]
    l += (
        20 * max(0.0, p["sat"] - 0.05) + 40 * abs(p["birth"] - 0.01) + 2 * max(0.0, abs(p["popdrift"]) - 0.01)
    )
    l += (1.0 - i["involved_in"]) + (1.0 - i["others_in"]) + dist(i["half"], 39, 81)
    l += 2 * max(0.0, 0.6 - o["assault"]["R"]) / 0.4
    l += (
        dist(o["gradual"]["V"], 0.15, 0.6) * 2
        + dist(o["populist"]["fallen"], 1.0, 4.5)
        + 3 * max(0.0, o["random"]["V"] - 0.1)
    )
    l += 4.0 * max(0.0, o["passive"]["R"] - 0.1)
    for st in ("fear", "clientelist", "capture", "rentier"):
        l += 0.5 * max(0.0, o[st]["V"] - 0.75)
    return float(l)


RANGES = {
    "alpha_react": (2.0, 3.5),
    "alpha_ind": (0.2, 0.5),
    "k_contagion": (0.003, 0.012),
    "sigma_I": (0.08, 0.15),
    "nu_inf": (0.004, 0.02),
    "nu_child": (0.05, 0.07),
    "nu_T": (0.2, 0.6),
    "margin": (0.0, 0.15),
    "kappa_D": (3.0, 8.0),
    "h_O_council": (15.0, 70.0),
    "x_star_T": (0.7, 0.95),
    "delay_min": (3.0, 7.0),
    "w_M": (0.7, 1.0),
    "alpha_direct": (0.45, 0.9),
}

if __name__ == "__main__":
    n, a, b = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    seeds = range(a, b)
    rng = np.random.default_rng(11)
    u = np.stack([(rng.permutation(n) + rng.random(n)) / n for _ in RANGES], 1)
    out = []
    with ProcessPoolExecutor(8) as ex:
        base = evaluate({}, seeds, ex)
        print("BASELINE loss %.3f" % loss(base), json.dumps(base)[:1500], flush=True)
        for i in range(n):
            ov = {k: lo + u[i, j] * (hi - lo) for j, (k, (lo, hi)) in enumerate(RANGES.items())}
            o = evaluate(to_overrides(ov), seeds, ex)
            l = loss(o)
            out.append((l, ov, o))
            print("%d %.3f" % (i, l), flush=True)
    out.sort(key=lambda x: x[0])
    json.dump(
        [{"loss": l, "ov": ov, "res": o} for l, ov, o in out[:8]],
        open("tools/calibration/out/calib_top.json", "w"),
        indent=1,
    )
    for l, ov, o in out[:5]:
        print("TOP %.3f" % l, ",".join(f"{k}={v:.4g}" for k, v in ov.items()))
