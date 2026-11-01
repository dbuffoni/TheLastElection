"""tle-liquid: batching, HTTP contract, auth, schema filling for every game contract. No weights, no GPU, no network:
the mock runtimes have the interface of the real ones (the random-weight LFM2 path is covered by `tle-liquid smoke --mode tiny`)."""

import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

pytest.importorskip("starlette")
uvicorn = pytest.importorskip("uvicorn")

from tle.liquid import LiquidConfig, LiquidEngine  # noqa: E402
from tle.liquid.batching import MicroBatcher  # noqa: E402
from tle.liquid.mock import fill_schema  # noqa: E402
from tle.liquid.server import create_app  # noqa: E402
from tle.liquid.smoke import CHAT_SCHEMA, QUESTIONS, validate  # noqa: E402


def test_microbatcher_fuses_concurrent_items_and_isolates_errors():
    b = MicroBatcher(
        lambda xs: [ValueError("bad") if x == 3 else x * 2 for x in xs], max_batch=8, max_wait_ms=100
    )
    fs = [b.submit(i) for i in range(6)]
    assert [f.result() for i, f in enumerate(fs) if i != 3] == [0, 2, 4, 8, 10]
    with pytest.raises(ValueError):
        fs[3].result()
    assert b.batches == [6]  # one fused call
    b.close()


def test_microbatcher_respects_max_batch():
    b = MicroBatcher(lambda xs: xs, max_batch=4, max_wait_ms=100)
    fs = [b.submit(i) for i in range(10)]
    assert [f.result() for f in fs] == list(range(10))
    assert max(b.batches) <= 4 and sum(b.batches) == 10
    b.close()


def test_fill_schema_satisfies_every_game_contract():
    from tle.contracts.schema import export_schema
    from tle.contracts.tasks import TASKS
    from tle.testing import catalogs

    vocab = catalogs().vocab()
    for name, spec in TASKS.items():
        sch = export_schema(spec.output_model, vocab)
        assert validate(fill_schema(sch, 1), sch) is None, name


@pytest.fixture()
def server():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    eng = LiquidEngine(LiquidConfig(api_key="k", port=port, max_wait_ms=30), mode="mock")
    srv = uvicorn.Server(uvicorn.Config(create_app(eng), host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}", eng
    srv.should_exit = True
    eng.close()


def _post(url, body, key="k"):
    h = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    with urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), h)) as r:
        return json.loads(r.read())


def test_auth_required_except_health(server):
    base, _ = server
    assert json.loads(urllib.request.urlopen(base + "/health").read())["status"] == "ok"
    for key in ("", "wrong"):
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base + "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}]}, key)
        assert e.value.code == 401


def test_chat_schema_and_decisions_over_http(server):
    base, eng = server
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "a", "strict": True, "schema": CHAT_SCHEMA},
        },
    }
    out = _post(base + "/v1/chat/completions", body)
    assert out["choices"][0]["finish_reason"] == "stop"
    assert validate(json.loads(out["choices"][0]["message"]["content"]), CHAT_SCHEMA) is None
    d = _post(base + "/api/alpha/decisions", {"state": {"a": 1}, "questions": QUESTIONS})
    assert d["answers"]["pillar"]["choice"] in QUESTIONS["pillar"]["criteria"]
    assert eng.info()["roles"]["s1"]["loaded"]


def test_server_with_one_role_refuses_the_other():
    eng = LiquidEngine(LiquidConfig(roles=("s2",)), mode="mock")
    with pytest.raises(KeyError):
        eng.decide({"state": {}, "questions": QUESTIONS})
    eng.close()
