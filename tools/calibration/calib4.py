import json
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from calib import *

top = json.load(open("tools/calibration/out/calib3_top.json"))[0]
BASE = top["ov"]
R2 = {
    "theta_0": (0.2, 0.32),
    "theta_1": (0.1, 0.2),
    "theta_down": (0.4, 0.65),
    "alpha_aware": (0.4, 0.6),
    "c_0": (0.2, 0.4),
    "delta": (0.3, 0.7),
    "x_star_K": (0.6, 0.9),
}
n = int(sys.argv[1])
seeds = range(1, 13)
rng = np.random.default_rng(41)
u = np.stack([(rng.permutation(n) + rng.random(n)) / n for _ in R2], 1)
out = []
with ProcessPoolExecutor(8) as ex:
    for i in range(n):
        ov = dict(BASE)
        ov.update({k: lo + u[i, j] * (hi - lo) for j, (k, (lo, hi)) in enumerate(R2.items())})
        o = evaluate(ov, seeds, ex)
        out.append((loss(o), ov, o))
        print(i, "%.3f" % out[-1][0], flush=True)
    out.sort(key=lambda x: x[0])
    json.dump(
        [{"loss": l, "ov": ov, "res": o} for l, ov, o in out[:8]],
        open("tools/calibration/out/calib4_top.json", "w"),
        indent=1,
    )
    for l, ov, o in out[:3]:
        v = evaluate(ov, range(101, 125), ex, impulse_seeds=(4, 5, 6))
        print(
            "VAL train=%.3f val=%.3f" % (l, loss(v)),
            ",".join(f"{k}={x:.4g}" for k, x in ov.items()),
            flush=True,
        )
        print("VALRES", json.dumps(v), flush=True)
