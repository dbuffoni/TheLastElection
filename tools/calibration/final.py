import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from calib import *

before = {
    "alpha_react": 2.8,
    "alpha_ind": 0.3,
    "k_contagion": 0.005,
    "sigma_I": 0.115,
    "nu_inf": 0.008,
    "nu_child": 0.06,
    "nu_T": 0.295,
    "kappa_D": 5.622,
    "h_O_council": 15.0,
    "x_star_T": 0.739,
    "delay_min": 7.0,
    "w_M": 0.826,
    "alpha_direct": 0.61,
    "theta_outcome": 0.5,
    "theta_0": 0.255,
    "theta_1": 0.16,
    "theta_down": 0.54,
    "alpha_aware": 0.579,
    "c_0": 0.318,
    "delta": 0.5,
    "x_star_K": 0.8,
}
res = {}
with ProcessPoolExecutor(8) as ex:
    for name, ov in (("before", before), ("after", {})):
        for label, seeds in (("val", range(101, 125)), ("holdout", range(201, 225))):
            r = evaluate(ov, seeds, ex, impulse_seeds=(4, 5, 6) if label == "val" else (7, 8, 9))
            res[f"{name}-{label}"] = {"loss": loss(r), **r}
            print(name, label, "loss %.3f" % loss(r), flush=True)
json.dump(res, open("tools/calibration/out/final.json", "w"), indent=1)
