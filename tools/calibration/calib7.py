"""Second search after calib5: the pillar gate, the awareness of attacks, the cost of decrees and the threats, around the best
point of calib5 (population and impulse already in band)."""
import sys, json, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from calib import *

OUT = __import__("pathlib").Path(__file__).parent / "out"
BASE = json.load(open(OUT / "calib6_top.json"))[1]["ov"]
R6 = {
    "theta_0": (0.18, 0.3), "theta_1": (0.08, 0.2), "theta_down": (0.4, 0.7), "alpha_aware": (0.45, 0.62), "c_0": (0.2, 0.4),
    "delta": (0.3, 0.7), "x_star_K": (0.75, 0.95), "alpha_direct": (0.8, 1.2), "nu_T": (0.3, 0.6), "margin": (0.0, 0.1),
    "omega_cust": (4.0, 9.0), "h_O_council": (20.0, 60.0),
}
n = int(sys.argv[1]); seeds = range(1, 13)
rng = np.random.default_rng(67)
u = np.stack([(rng.permutation(n) + rng.random(n)) / n for _ in R6], 1)
out = []
with ProcessPoolExecutor(8) as ex:
    for i in range(n):
        raw = {k: lo + u[i, j] * (hi - lo) for j, (k, (lo, hi)) in enumerate(R6.items())}
        ov = dict(BASE)
        ov.update({k: v for k, v in raw.items() if k != "margin"})
        w_M, xs = ov["w_M"], ov["x_star_T"]
        f = w_M * 0.42; G = xs / (1 - xs)
        ov["theta_outcome"] = float(f * G / (f * G + 1) + raw["margin"])
        o = evaluate(ov, seeds, ex); out.append((loss(o), ov, o)); print(i, "%.3f" % out[-1][0], flush=True)
    out.sort(key=lambda x: x[0])
    json.dump([{"loss": l, "ov": ov, "res": o} for l, ov, o in out[:8]], open(OUT / "calib7_top.json", "w"), indent=1)
    for l, ov, o in out[:3]:
        v = evaluate(ov, range(101, 125), ex, impulse_seeds=(4, 5, 6))
        print("VAL train=%.3f val=%.3f" % (l, loss(v)), ",".join(f"{k}={x:.4g}" for k, x in ov.items()), flush=True)
        print("VALRES", json.dumps(v), flush=True)
