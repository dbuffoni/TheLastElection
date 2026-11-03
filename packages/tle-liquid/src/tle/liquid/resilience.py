"""Surviving a crashing model process (a GPU hang kills it with SIGABRT).

Measured on a Radeon 780M: 14 hangs in an 8 hour game; three were isolated, eleven came in 7 minutes, each right after a restart
and one successful request. A gateway that simply re-sends the request in flight to the restarted process can keep killing it
with the same request. So:

* **State per model process** (``BackendState``): ready flag, recent crash times, and a count of how many crashes each request
  was in flight for.
* **Supervisor** (``supervise_one``): when the process dies, wait with an increasing delay (5 s, 15 s, 45 s, 135 s, 180 s for
  crashes within the last 10 minutes; back to 5 s after a quiet period), start it again, wait for it to load, **warm it up with
  known requests**, and only then mark it ready. Traffic never reaches a process that has not passed the warm-up.
* **Queue at the gateway** (``Quarantine``): at most ``concurrency`` requests per model are in flight (a FIFO per model), so a
  crash is attributable. A request that was in flight for two crashes is answered with HTTP 500 ``poisoned_request`` instead of being
  retried for ever; the game falls back for that one call and goes on.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
import urllib.request
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

BACKOFF_BASE = float(os.environ.get("TLE_LIQUID_BACKOFF_BASE", 5.0))
BACKOFF_FACTOR = 3.0
BACKOFF_MAX = 180.0
QUIET_WINDOW = 600.0  # crashes older than this are forgotten
POISON_LIMIT = 2  # crashes a single request may be in flight for
DATA = Path(__file__).resolve().parents[3] / "data"


def backoff_delay(recent_crashes: int, base: float = BACKOFF_BASE) -> float:
    """Delay before restart number ``recent_crashes`` (1 = first crash in the window)."""
    return min(BACKOFF_MAX, base * BACKOFF_FACTOR ** max(0, recent_crashes - 1))


@dataclass
class BackendState:
    role_label: str = ""
    ready: threading.Event = field(default_factory=threading.Event)
    crashes: deque = field(default_factory=deque)  # monotonic times
    total_crashes: int = 0

    def record_crash(self, now: float | None = None) -> int:
        now = time.monotonic() if now is None else now
        self.crashes.append(now)
        self.total_crashes += 1
        while self.crashes and now - self.crashes[0] > QUIET_WINDOW:
            self.crashes.popleft()
        return len(self.crashes)


class Quarantine:
    """Counts the crashes each request (by content hash) was in flight for."""

    def __init__(self, limit: int = POISON_LIMIT):
        self.limit = limit
        self.hits: dict[str, int] = {}
        self.dropped = 0

    @staticmethod
    def key(path: str, body: bytes) -> str:
        return hashlib.sha1(path.encode() + b"\0" + body).hexdigest()

    def crashed_with(self, key: str) -> bool:
        """The request was in flight when a backend died. True = it has done so too often: do not send it again."""
        self.hits[key] = self.hits.get(key, 0) + 1
        if self.hits[key] >= self.limit:
            self.dropped += 1
            return True
        return False

    def done(self, key: str) -> None:
        self.hits.pop(key, None)


def warmup_requests(role: str, limit: int = 3) -> list[tuple[str, dict]]:
    """Known-good (path, body) pairs from the data shipped for calibration; empty when the data is not there."""
    try:
        if role == "s1":
            from .calib_worker import task_of

            rows = [json.loads(x) for x in (DATA / "s1_requests.jsonl").read_text().splitlines()]
            rows = [r for r in rows if task_of(r) in ("option", "activity")][:limit]
            return [
                ("/api/alpha/decisions", {"state": r["state"], "questions": r["questions"]}) for r in rows
            ]
        rows = [json.loads(x) for x in (DATA / "s2_requests.jsonl").read_text().splitlines()]
        r = rows[0]
        body = {"messages": r["messages"], "max_tokens": 8, "temperature": 0.0}
        return [("/v1/chat/completions", body)]
    except (OSError, ValueError, IndexError):
        return []


def warm_up(port: int, role: str, timeout: float = 300.0) -> bool:
    """Send the warm-up requests straight to a model process. False if one fails (the process is probably dead)."""
    for path, body in warmup_requests(role):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"content-type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if r.status != 200:
                    return False
        except Exception:
            return False
    return True


def supervise_one(
    i: int,
    spawn: Callable[[], subprocess.Popen],
    procs: list,
    state: BackendState,
    ready_check: Callable[[], bool],
    stop: threading.Event,
    log: Callable[[str], None] = print,
    base_delay: float = BACKOFF_BASE,
) -> None:
    """Keeps process ``procs[i]`` alive. ``ready_check()`` runs once the process is up: health, then warm-up."""
    while not stop.is_set():
        p = procs[i]
        if p.poll() is None:
            stop.wait(1.0)
            continue
        state.ready.clear()
        n = state.record_crash()
        delay = backoff_delay(n, base_delay)
        log(
            f"[up] {state.role_label} exited with code {p.returncode}; crash #{state.total_crashes} "
            f"({n} in the last {QUIET_WINDOW / 60:.0f} min); restarting in {delay:.0f}s"
        )
        if stop.wait(delay):
            return
        procs[i] = spawn()
        deadline = time.monotonic() + 1800
        while not stop.is_set() and procs[i].poll() is None and time.monotonic() < deadline:
            if ready_check():
                state.ready.set()
                log(f"[up] {state.role_label} is back (warm-up passed)")
                break
            stop.wait(2.0)
