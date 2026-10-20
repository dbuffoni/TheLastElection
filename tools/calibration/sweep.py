import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tle.catalogs.loader import load_catalogs
from tle.core.labels import quantize_bipolar
from tle.loop.game import Game
from tle.loop.strategies import STRATEGIES
from tle.params import default_params


def one(args):
    ov, seed, strat = args
    cat = load_catalogs()
    p = default_params().with_overrides(ov) if ov else default_params()
    g = Game(cat, p, scenario="industrial_secular", seed=seed, n=300)
    w = g.w
    ad0 = w.adults(p.demography.a_adult).copy()
    I0 = w.I.copy()
    arr0 = w.arrival_day.copy()
    m = len(ad0)
    pop0 = w.alive.sum()
    age0 = w.age[w.alive].mean()
    out = {}
    rng = w.rng.child("player")
    st = STRATEGIES[strat]
    while not g.over:
        g.step(st(w, cat, rng))
        if "mob" not in out and w.day >= 730:
            ad = w.adults(p.demography.a_adult)
            same = ad[:m] & ad0 & (w.arrival_day[:m] == arr0)
            then = quantize_bipolar(I0[same])
            now = quantize_bipolar(w.I[:m][same], then, p.labels.eps)
            out["lab"] = float(np.mean((now != then).any(axis=1)))
            out["disp"] = float(np.median(np.abs(w.I[:m][same] - I0[same]).max(axis=1)))
            out["mob"] = 1
    ad = w.adults(p.demography.a_adult)
    out["sat"] = float(np.mean((np.abs(w.C[ad]) >= 0.999).any(axis=1)))
    yrs = w.day / 365
    n = w.alive.sum()
    out["birth"] = w.stats.get("births", 0) / yrs / n
    out["death"] = w.stats.get("deaths", 0) / yrs / n
    out["popdrift"] = (n - pop0) / pop0 / yrs
    out["agedrift"] = (w.age[w.alive].mean() - age0) / age0 / yrs
    out["outcome"] = w.outcome
    out["fallen"] = int((w.pillar_level >= 6).sum())
    return out


def run(ov, seeds=range(1, 6), strat="passive"):
    with ProcessPoolExecutor(8) as ex:
        rs = list(ex.map(one, [(ov, s, strat) for s in seeds]))
    keys = ["lab", "disp", "sat", "birth", "death", "popdrift", "agedrift"]
    return {k: [round(r.get(k, float("nan")), 3) for r in rs] for k in keys}, [r["outcome"] for r in rs]


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        ov = {k: float(v) for k, v in (x.split("=") for x in spec.split(",") if x)}
        res, oc = run(ov)
        print("##", spec or "baseline", set(oc))
        for k, v in res.items():
            print(f"  {k:9s}", v, "mean=%.3f" % np.mean(v))
