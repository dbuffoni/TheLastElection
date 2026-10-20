import json
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from calib import *

n = int(sys.argv[1])
seeds = range(1, 13)
rng = np.random.default_rng(31)
u = np.stack([(rng.permutation(n) + rng.random(n)) / n for _ in RANGES], 1)
out = []
with ProcessPoolExecutor(8) as ex:
    for i in range(n):
        raw = {k: lo + u[i, j] * (hi - lo) for j, (k, (lo, hi)) in enumerate(RANGES.items())}
        ov = to_overrides(raw)
        o = evaluate(ov, seeds, ex)
        out.append((loss(o), raw, ov, o))
        print(i, "%.3f" % out[-1][0], flush=True)
    out.sort(key=lambda x: x[0])
    json.dump(
        [{"loss": l, "raw": r, "ov": ov, "res": o} for l, r, ov, o in out[:8]],
        open("tools/calibration/out/calib3_top.json", "w"),
        indent=1,
    )
    for l, r, ov, o in out[:3]:
        v = evaluate(ov, range(101, 125), ex, impulse_seeds=(4, 5, 6))
        print(
            "VAL train=%.3f val=%.3f" % (l, loss(v)),
            ",".join(f"{k}={x:.4g}" for k, x in ov.items()),
            flush=True,
        )
        print("VALRES", json.dumps({k: v[k] for k in v}), flush=True)
