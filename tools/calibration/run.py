"""One command for the whole calibration on the mocks (about 25 minutes on 8 cores):

    uv run python tools/calibration/run.py [--points 48] [--seeds 12] [--apply]

1. wide LHS over the free parameters (population, impulse, threats, awareness) with `theta_outcome` tied to the baseline
   pressure of a divided council, on the training seeds 1..seeds;
2. local grid over the pillar gate and the cost of decrees (`theta_0`, `delta`, `cost_decree`) around the best point;
3. table on the validation (101-124) and hold-out (201-224) seeds; with --apply the winning values become registry defaults
   (config/params/*.toml, ranges widened when needed) and `tle params codegen` must be run.
Everything is written to tools/calibration/out/run.json. The objective is calib.loss."""

import argparse, glob, itertools, json, re, sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
from calib import ProcessPoolExecutor, evaluate, loss, to_overrides  # noqa: E402

RANGES = {
    "alpha_react": (1.2, 3.2), "alpha_ind": (0.15, 0.5), "k_contagion": (0.003, 0.012), "sigma_I": (0.08, 0.15),
    "nu_inf": (0.002, 0.012), "nu_fac": (0.02, 0.2), "nu_found": (0.01, 0.1), "nu_child": (0.055, 0.067),
    "nu_T": (0.15, 0.6), "margin": (0.0, 0.15), "kappa_D": (3.0, 9.0), "h_O_council": (15.0, 70.0),
    "x_star_T": (0.65, 0.95), "w_M": (0.7, 1.0), "alpha_direct": (0.45, 1.2), "alpha_aware": (0.4, 0.65),
    "news_share": (0.05, 0.3), "h_O": (30.0, 120.0),
}
GRID = {"theta_0": (0.24, 0.27, 0.285), "delta": (0.3, 0.5, 0.6), "cost_decree": (0.15, 0.18, 0.22)}


def table(name, r):
    p, i = r["pop"], r["imp"]
    return (f"{name:9s} loss {loss(r):.2f} | assault R {r['assault']['R']:.2f}  gradual V {r['gradual']['V']:.2f}  populist {r['populist']['fallen']:.1f}  "
            f"random V {r['random']['V']:.2f}  passive R {r['passive']['R']:.2f} | label in {p['lab_in']:.0%} disp in {p['disp_in']:.0%} "
            f"sat {p['sat']:.3f} birth {p['birth']:.4f} | impulse {i['involved']:.2f}/{i['others']:.3f}/{i['half']:.0f}d")


def apply_defaults(ov):
    for k, v in ov.items():
        done = False
        for f in glob.glob(str(HERE.parent.parent / "config" / "params" / "*.toml")):
            s = open(f).read()
            m = re.search(rf"^\[{k}\]", s, re.M)
            if not m:
                continue
            i = m.start(); j = s.index("default", i); e = s.index("\n", j)
            s = s[:j] + f"default     = {round(v, 4)}" + s[e:]
            mm = re.compile(r"range\s*=\s*\[\s*([^,\]]+),\s*([^\]]+)\]").search(s, i); nxt = s.find("\n[", i + 1)
            if mm and (nxt < 0 or mm.start() < nxt):
                lo, hi = float(mm.group(1)), float(mm.group(2))
                if v < lo or v > hi:
                    s = s[:mm.start()] + f"range       = [{min(lo, round(v * .5, 4))}, {max(hi, round(v * 1.5, 4))}]" + s[mm.end():]
            open(f, "w").write(s); done = True
        assert done, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=12, help="training seeds 1..N")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    seeds = range(1, a.seeds + 1)
    rng = np.random.default_rng(97)
    u = np.stack([(rng.permutation(a.points) + rng.random(a.points)) / a.points for _ in RANGES], 1)
    out = {"stage1": [], "stage2": []}
    with ProcessPoolExecutor(a.workers) as ex:
        best = None
        for n in range(a.points):
            raw = {k: lo + u[n, j] * (hi - lo) for j, (k, (lo, hi)) in enumerate(RANGES.items())}
            ov = to_overrides(raw); r = evaluate(ov, seeds, ex); l = loss(r)
            out["stage1"].append({"loss": l, "ov": ov})
            if best is None or l < best[0]:
                best = (l, ov)
            print(f"[1/3] point {n + 1}/{a.points} loss {l:.2f} (best {best[0]:.2f})", flush=True)
        for combo in itertools.product(*GRID.values()):
            ov = dict(best[1]); ov.update(dict(zip(GRID, combo)))
            r = evaluate(ov, seeds, ex); l = loss(r)
            out["stage2"].append({"loss": l, "ov": ov})
            if l < best[0]:
                best = (l, ov)
            print(f"[2/3] {dict(zip(GRID, combo))} loss {l:.2f} (best {best[0]:.2f})", flush=True)
        ov = best[1]
        out["best"] = ov
        for name, ss, imp in (("validation", range(101, 125), (4, 5, 6)), ("hold-out", range(201, 225), (7, 8, 9))):
            r = evaluate(ov, ss, ex, impulse_seeds=imp); out[name] = r
            print("[3/3]", table(name, r), flush=True)
    (HERE / "out").mkdir(exist_ok=True)
    json.dump(out, open(HERE / "out" / "run.json", "w"), indent=1)
    if a.apply:
        apply_defaults(ov)
        print("defaults written to config/params; now: uv run tle params codegen && uv run tle check && uv run pytest")


if __name__ == "__main__":
    main()
