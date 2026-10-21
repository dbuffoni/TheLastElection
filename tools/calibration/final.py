"""Before/after table on the validation and hold-out seeds. `before` = the parameters promoted by the previous calibration pass
(run on the current code), `after` = the registry defaults."""
import sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from calib import *

before = {"alpha_react": 2.36, "alpha_ind": 0.366, "k_contagion": 0.00479, "sigma_I": 0.126, "nu_inf": 0.00842, "nu_child": 0.061, "nu_T": 0.318,
          "kappa_D": 7.995, "h_O_council": 38.9, "x_star_T": 0.7338, "delay_min": 4.515, "w_M": 0.7016, "alpha_direct": 0.846, "theta_outcome": 0.4948,
          "theta_0": 0.2449, "theta_1": 0.1466, "theta_down": 0.6165, "alpha_aware": 0.5184, "c_0": 0.2247, "delta": 0.3122, "x_star_K": 0.7159,
          "nu_fac": 0.2, "nu_found": 0.1, "gamma_social": 0.0, "news_share": 0.1, "rho_0": 0.15, "h_O": 60.0, "kappa_ec": 2.0, "cost_decree": 0.15}
if __name__ == "__main__":
    res = {}
    with ProcessPoolExecutor(8) as ex:
        for name, ov in (("before", before), ("after", {})):
            for label, seeds, imps in (("val", range(101, 125), (4, 5, 6)), ("holdout", range(201, 225), (7, 8, 9))):
                r = evaluate(ov, seeds, ex, impulse_seeds=imps)
                res[f"{name}-{label}"] = {"loss": loss(r), **r}
                print(name, label, "loss %.2f" % loss(r), flush=True)
    json.dump(res, open(__import__("pathlib").Path(__file__).parent / "out" / "final.json", "w"), indent=1)
