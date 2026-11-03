"""Worker of ``tle-liquid calibrate``: loads one model, then times each configuration it is given and prints one JSON line per
result. It runs in its own process because a configuration that is too big can hang the GPU and kill the process; the driver
then restarts a worker for the configurations that remain. ``python -m tle.liquid.calib_worker --role s1 --device cpu ...``"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

DATA = Path(__file__).resolve().parents[3] / "data"
S1_TASKS = (
    "option",
    "activity",
)  # what runs every turn; the setup tasks (classify / extract) run once per game
S2_MAX_TOKENS = 96  # enough to exercise generation without making every measurement long


def task_of(r: dict) -> str:
    names = {k.split(":", 1)[-1] for k in r["questions"]}
    if "option" in names:
        return "option"
    if "activity" in names:
        return "activity"
    return "classify" if len(names) == 11 else "extract"


def group(singles: list[dict]) -> dict:
    """G single-citizen requests -> one request with G citizens (the shape of ``jev._build_batch``). Options that the real
    batch shares are repeated here, so a grouped request is slightly larger than the real one: a safe overestimate."""
    if len(singles) == 1:
        return {"state": singles[0]["state"], "questions": singles[0]["questions"]}
    questions = {}
    for j, s in enumerate(singles):
        for k, q in s["questions"].items():
            q2 = copy.deepcopy(q)
            q2["instructions"] += f" Answer for the item at index {j} of state.items."
            questions[f"i{j}:{k}"] = q2
    return {"state": {"shared": {}, "items": [s["state"] for s in singles]}, "questions": questions}


def s1_requests(pool: dict[str, list[dict]], task: str, g: int, r: int, offset: int) -> list[dict]:
    rows = pool[task]
    return [group([rows[(offset + i * g + j) % len(rows)] for j in range(g)]) for i in range(r)]


def run_config(rt, role: str, g: int, r: int, pool, offset: int = 0) -> dict:
    t0 = time.perf_counter()
    if role == "s1":
        n = 0
        for task in S1_TASKS:
            out = rt.run_batch(s1_requests(pool, task, g, r, offset))
            if any(isinstance(o, BaseException) for o in out):
                raise RuntimeError("a request failed")
            n += g * r
        unit = "citizens/s"
    else:
        reqs = []
        for i in range(r):
            q = copy.deepcopy(pool[(offset + i) % len(pool)])
            q["max_tokens"] = min(int(q.get("max_tokens") or S2_MAX_TOKENS), S2_MAX_TOKENS)
            reqs.append(q)
        out = rt.run_batch(reqs)
        if any(isinstance(o, BaseException) for o in out):
            raise RuntimeError("a request failed")
        n, unit = r, "requests/s"
    secs = time.perf_counter() - t0
    return {"G": g, "R": r, "ok": True, "secs": round(secs, 2), "score": round(n / secs, 3), "unit": unit}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["s1", "s2"], required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--mode", default="real", choices=["real", "mock"])
    ap.add_argument("--configs", required=True, help="JSON [[G,R],...]")
    ap.add_argument("--data", default=str(DATA))
    a = ap.parse_args(argv)

    from .config import load_config
    from .engine import LiquidEngine

    data = Path(a.data)
    if a.role == "s1":
        pool: dict = {t: [] for t in ("option", "activity", "classify", "extract")}
        for line in (data / "s1_requests.jsonl").read_text().splitlines():
            r = json.loads(line)
            pool[task_of(r)].append(r)
    else:
        pool = [json.loads(x) for x in (data / "s2_requests.jsonl").read_text().splitlines()]
    eng = LiquidEngine(load_config(None, device=a.device, roles=[a.role]), mode=a.mode)
    rt = eng.runtimes[a.role]
    print("READY", flush=True)
    fail = os.environ.get("TLE_LIQUID_CALIB_FAIL")  # tests: "G>=5" makes the worker die like a GPU hang
    for i, (g, r) in enumerate(json.loads(a.configs)):
        print("START " + json.dumps([g, r]), flush=True)
        if fail and eval(fail, {}, {"G": g, "R": r}):  # noqa: S307 (test hook, env controlled by the developer)
            os._exit(134)
        try:
            if i == 0 and a.role == "s1":
                run_config(rt, a.role, 1, 1, pool)  # warm-up: kernels, caches
            res = run_config(rt, a.role, g, r, pool, offset=i * 7)
        except Exception as e:
            res = {"G": g, "R": r, "ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
        print("RESULT " + json.dumps(res), flush=True)
    eng.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
