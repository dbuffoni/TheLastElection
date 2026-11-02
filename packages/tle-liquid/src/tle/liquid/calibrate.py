"""``tle-liquid calibrate``: finds the best G (citizens per request, S1 only) and R (requests fused per model call) for THIS
machine and saves them to ``.tle-liquid/calibration.json``, which ``tle-liquid up`` reads.

Workload: real requests of the game, shipped in ``packages/tle-liquid/data/`` (so a fresh clone can calibrate without
playing). For every role the layout of ``up`` is used (same device, same CPU threads). Each configuration is timed in a worker
process; a configuration that hangs the GPU kills only that worker and every bigger configuration is skipped. The best one is
then confirmed three times: a configuration that merely got lucky once is dropped for the next best.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

CALIBRATION_FILE = Path(".tle-liquid/calibration.json")
GRIDS = {  # (G values, R values) by device kind and effort
    ("gpu", "full"): ((1, 2, 5, 10), (1, 2, 4, 8, 16)),
    ("gpu", "quick"): ((1, 5), (1, 4, 8)),
    ("cpu", "full"): ((1, 5), (1, 2, 4)),
    ("cpu", "quick"): ((1,), (1, 4)),
}
MAX_CITIZENS = 64  # per measurement: G * R above this adds time, not information
PLATEAU = 0.95  # configurations within 5% of the best count as equal: the smaller (safer) one wins
CONFIRMATIONS = 3


def grid(kind: str, effort: str, role: str) -> list[tuple[int, int]]:
    gs, rs = GRIDS[(kind, effort)]
    if role == "s2":
        gs = (1,)
    out = [(g, r) for g in gs for r in rs if g * r <= MAX_CITIZENS]
    return sorted(out, key=lambda c: (c[0] * c[1], c[0]))  # small first: a hang then rules out what is bigger


def candidates(rows: list[dict]) -> list[dict]:
    """Successful rows in the order they should be tried: first the ones within PLATEAU of the best score, smallest G then R
    first (less risk for the same speed), then the rest by score."""
    ok = [r for r in rows if r.get("ok")]
    if not ok:
        return []
    best = max(r["score"] for r in ok)
    near = sorted((r for r in ok if r["score"] >= PLATEAU * best), key=lambda r: (r["G"], r["R"]))
    rest = sorted((r for r in ok if r not in near), key=lambda r: -r["score"])
    return near + rest


def skipped_after(fail: tuple[int, int], todo: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Configurations at least as big as a failed one in both G and R."""
    return [c for c in todo if c[0] >= fail[0] and c[1] >= fail[1]]


class Worker:
    """One worker process: feeds it configurations, yields its output lines, kills it when it goes quiet."""

    def __init__(self, role: str, device: str, configs, env: dict, mode: str):
        argv = [
            sys.executable, "-m", "tle.liquid.calib_worker", "--role", role, "--device", device,
            "--mode", mode, "--configs", json.dumps([list(c) for c in configs]),
        ]  # fmt: skip
        self.p = subprocess.Popen(
            argv, env={**os.environ, **env}, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.p.stdout:
            self.q.put(line.strip())
        self.q.put(None)

    def line(self, timeout: float):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return "TIMEOUT"

    def kill(self):
        if self.p.poll() is None:
            self.p.kill()
        self.p.wait()


def measure(role, device, configs, env, mode, timeout, log) -> list[dict]:
    """Time ``configs`` in order; returns one row per configuration tried (failed ones with ok=False). After a failure the
    remaining configurations at least as big are skipped and a fresh worker takes the rest."""
    rows: list[dict] = []
    todo = list(configs)
    while todo:
        w = Worker(role, device, todo, env, mode)
        failed = None
        cur = None
        try:
            ln = w.line(timeout * 4)  # model loading
            if ln != "READY":
                log(f"   worker did not start ({ln})")
                return rows
            for _ in list(todo):
                ln = w.line(timeout)
                while ln and ln.startswith("START "):
                    cur = tuple(json.loads(ln[6:]))
                    ln = w.line(timeout)
                if ln and ln.startswith("RESULT "):
                    row = json.loads(ln[7:])
                    rows.append(row)
                    todo.remove((row["G"], row["R"]))
                    log(
                        f"   G={row['G']:<2} R={row['R']:<2} "
                        + (
                            f"{row['score']:>7.2f} {row['unit']}"
                            if row["ok"]
                            else "failed: " + row.get("error", "")
                        )
                    )
                else:  # None = process died (a GPU hang), TIMEOUT = silent for too long
                    failed = cur
                    break
        finally:
            w.kill()
        if failed is None:
            break
        rows.append(
            {"G": failed[0], "R": failed[1], "ok": False, "error": "worker died or timed out (GPU hang?)"}
        )
        gone = skipped_after(failed, todo)
        log(f"   G={failed[0]:<2} R={failed[1]:<2} HANG -> skipping {len(gone)} bigger configuration(s)")
        todo = [c for c in todo if c not in gone]
        time.sleep(5)  # let the driver recover
    return rows


def confirm(role, device, cand, env, mode, timeout, log) -> dict | None:
    cfg = (cand["G"], cand["R"])
    rows = measure(role, device, [cfg] * CONFIRMATIONS, env, mode, timeout, lambda m: None)
    got = [r for r in rows if r.get("ok")]
    if len(got) < CONFIRMATIONS:
        log(f"   G={cfg[0]} R={cfg[1]}: confirmed {len(got)}/{CONFIRMATIONS}, discarded")
        return None
    score = sum(r["score"] for r in got) / len(got)
    log(f"   G={cfg[0]} R={cfg[1]}: confirmed {CONFIRMATIONS}/{CONFIRMATIONS}, {score:.2f} {cand['unit']}")
    return {**cand, "score": round(score, 3)}


def calibrate_role(role, device, kind, effort, env, mode, timeout, log) -> dict:
    log(f"[{role}] on {device}: measuring")
    configs = grid(kind, effort, role)
    rows = measure(role, device, configs, env, mode, timeout, log)
    chosen = None
    for cand in candidates(rows)[:4]:
        chosen = confirm(role, device, cand, env, mode, timeout, log)
        if chosen:
            break
    if chosen is None:
        log(f"[{role}] nothing was stable: falling back to G=1, R=1")
        chosen = {"G": 1, "R": 1, "score": 0.0, "unit": "citizens/s" if role == "s1" else "requests/s"}
    log(f"[{role}] chosen: G={chosen['G']} R={chosen['R']} ({chosen['score']} {chosen['unit']})")
    return {
        "device": device,
        "G": chosen["G"],
        "R": chosen["R"],
        "score": chosen["score"],
        "unit": chosen["unit"],
        "grid": rows,
    }


def run(
    roles=("s1", "s2"),
    layout="auto",
    effort="full",
    mode="real",
    timeout=240.0,
    out=CALIBRATION_FILE,
    hw=None,
    log=print,
) -> int:
    from .up import detect_hardware, plan

    hw = hw or detect_hardware()
    name, backends = plan(hw, tuple(roles), 8100, layout)
    log(f"hardware: {hw.get('backend')} {hw.get('name', '')} ({hw.get('cores')} cores); layout: {name}")
    res = {}
    for b in backends:
        for role in b.roles:
            env = dict(b.env)
            if b.threads:
                env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = str(b.threads)
            kind = "cpu" if b.device == "cpu" else "gpu"
            res[role] = calibrate_role(
                role, b.device, kind, effort, env, mode, timeout * (3 if kind == "cpu" else 1), log
            )
    doc = {
        "version": 1,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "hardware": {k: hw.get(k) for k in ("backend", "name", "cores", "integrated")},
        "layout": name,
        "effort": effort,
        "roles": res,
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    log(f"saved {out}")
    for role, v in res.items():
        what = f"G={v['G']}, " if role == "s1" else ""
        log(f"  {role}: {what}R={v['R']} on {v['device']}")
    return 0


def load(path: Path = CALIBRATION_FILE) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def matches(doc: dict, hw: dict) -> bool:
    h = doc.get("hardware", {})
    return h.get("backend") == hw.get("backend") and h.get("name") == hw.get("name")
