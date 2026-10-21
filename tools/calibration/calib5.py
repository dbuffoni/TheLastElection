"""Wide search after the mocks of S1 were fitted on a real model (actor stance and support dominate the choices): the
frequency of the events of factions, influencers and nascent movements joins the free parameters."""
import sys, json, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import calib
from calib import *

calib.RANGES.clear()
calib.RANGES.update({
    "alpha_react": (1.2, 3.2), "alpha_ind": (0.15, 0.5), "k_contagion": (0.003, 0.05), "sigma_I": (0.07, 0.15),
    "nu_inf": (0.002, 0.012), "nu_fac": (0.02, 0.2), "nu_found": (0.01, 0.1), "nu_child": (0.055, 0.067),
    "nu_T": (0.15, 0.6), "margin": (0.0, 0.15), "kappa_D": (3.0, 9.0), "h_O_council": (15.0, 70.0),
    "x_star_T": (0.65, 0.95), "w_M": (0.7, 1.0), "alpha_direct": (0.45, 0.9), "alpha_aware": (0.4, 0.65),
    "gamma_social": (0.0, 0.5), "news_share": (0.05, 0.3), "rho_0": (0.06, 0.15), "h_O": (30.0, 120.0),
})
n = int(sys.argv[1]); seeds = range(1, 13)
rng = np.random.default_rng(53)
u = np.stack([(rng.permutation(n) + rng.random(n)) / n for _ in calib.RANGES], 1)
out = []
OUT = __import__("pathlib").Path(__file__).parent / "out"
with ProcessPoolExecutor(8) as ex:
    for i in range(n):
        raw = {k: lo + u[i, j] * (hi - lo) for j, (k, (lo, hi)) in enumerate(calib.RANGES.items())}
        ov = to_overrides(raw)
        o = evaluate(ov, seeds, ex); out.append((loss(o), raw, ov, o)); print(i, "%.3f" % out[-1][0], flush=True)
    out.sort(key=lambda x: x[0])
    json.dump([{"loss": l, "raw": r, "ov": ov, "res": o} for l, r, ov, o in out[:8]], open(OUT / "calib5_top.json", "w"), indent=1)
    for l, r, ov, o in out[:3]:
        v = evaluate(ov, range(101, 125), ex, impulse_seeds=(4, 5, 6))
        print("VAL train=%.3f val=%.3f" % (l, loss(v)), ",".join(f"{k}={x:.4g}" for k, x in ov.items()), flush=True)
        print("VALRES", json.dumps(v), flush=True)
