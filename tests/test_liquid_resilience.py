import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest
from tle.liquid.resilience import BackendState, Quarantine, backoff_delay, supervise_one


def test_backoff_grows_and_caps():
    assert [backoff_delay(n) for n in (1, 2, 3, 4, 5, 9)] == [5, 15, 45, 135, 180, 180]


def test_crash_window_forgets_old_crashes():
    st = BackendState()
    assert st.record_crash(0) == 1 and st.record_crash(10) == 2
    assert st.record_crash(5000) == 1 and st.total_crashes == 3


def test_quarantine_drops_after_two_crashes_and_forgets_on_success():
    q = Quarantine()
    k = q.key("/x", b"body")
    assert not q.crashed_with(k) and q.crashed_with(k) and q.dropped == 1
    q.done(k)
    assert not q.crashed_with(k)  # a request that finally succeeded starts from zero


def test_supervisor_restarts_waits_for_warmup_and_marks_ready():
    state = BackendState(role_label="s1")
    spawned = []

    def spawn():
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        spawned.append(p)
        return p

    procs = [subprocess.Popen([sys.executable, "-c", "raise SystemExit(134)"])]
    time.sleep(0.3)  # it is dead before the supervisor looks
    warm_calls = []
    stop = threading.Event()
    t = threading.Thread(
        target=supervise_one,
        args=(
            0,
            spawn,
            procs,
            state,
            lambda: (warm_calls.append(1), len(warm_calls) >= 2)[1],
            stop,
            lambda m: None,
            0.05,
        ),
        daemon=True,
    )
    t.start()
    assert state.ready.wait(20), "the process was never declared ready"
    assert (
        len(spawned) == 1 and len(warm_calls) >= 2 and state.total_crashes == 1
    )  # not ready until the warm-up passed
    stop.set()
    for p in spawned + procs:
        p.kill()


def test_gateway_retries_a_crashed_request_then_quarantines_a_poisoned_one():
    httpx = pytest.importorskip("httpx")
    from starlette.testclient import TestClient
    from tle.liquid.up import make_gateway

    seen = {"flaky": 0}

    def handler(request: httpx.Request):
        body = request.content.decode()
        if "poison" in body:
            raise httpx.ConnectError("process died")
        if "flaky" in body and seen["flaky"] < 1:
            seen["flaky"] += 1
            raise httpx.ConnectError("process died")
        return httpx.Response(200, json={"ok": True})

    st = BackendState(role_label="s1")
    st.ready.set()
    app = make_gateway(
        {"s1": "http://b1"}, None, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        states={"s1": st}, concurrency={"s1": 1},
    )  # fmt: skip
    c = TestClient(app)
    assert (
        c.post("/api/alpha/decisions", json={"state": "flaky"}).status_code == 200
    )  # one crash: sent again, fine
    r = c.post("/api/alpha/decisions", json={"state": "poison"})
    assert r.status_code == 500 and r.json()["error"]["type"] == "poisoned_request"
    assert c.get("/health").json()["backends"]["s1"]["dropped_requests"] == 1


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(url, body, timeout=60):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_up_survives_a_poisoned_request_end_to_end():
    pytest.importorskip("httpx")
    pytest.importorskip("uvicorn")
    port = _free_port()
    env = {**os.environ, "TLE_LIQUID_TEST_POISON": "KILLER", "TLE_LIQUID_BACKOFF_BASE": "0.2"}
    p = subprocess.Popen(
        [sys.executable, "-m", "tle.liquid.cli", "up", "--mode", "mock", "--layout", "cpu", "--roles", "s1", "--port", str(port)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )  # fmt: skip
    try:
        for line in p.stdout:  # waits for READY (hardware probe + mock backend)
            if "link:" in line:
                break
        base = f"http://127.0.0.1:{port}"
        for _ in range(60):  # the gateway starts right after the link is printed
            try:
                urllib.request.urlopen(base + "/health", timeout=3)
                break
            except OSError:
                time.sleep(0.5)
        q = {"a": {"type": "choice", "instructions": "x", "criteria": {"u": "u", "v": "v"}}}
        assert _post(base + "/api/alpha/decisions", {"state": {"i": 1}, "questions": q})[0] == 200
        code, body = _post(base + "/api/alpha/decisions", {"state": {"i": "KILLER"}, "questions": q}, 120)
        assert code == 500 and body["error"]["type"] == "poisoned_request"
        assert (
            _post(base + "/api/alpha/decisions", {"state": {"i": 2}, "questions": q}, 120)[0] == 200
        )  # it came back
        health = json.loads(urllib.request.urlopen(base + "/health", timeout=30).read())["backends"]["s1"]
        assert health["crashes"] >= 2 and health["dropped_requests"] == 1
    finally:
        p.terminate()
        p.wait(20)
