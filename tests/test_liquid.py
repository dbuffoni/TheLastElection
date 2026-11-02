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


def test_role_device_from_env(monkeypatch):
    from tle.liquid.config import load_config

    monkeypatch.setenv("TLE_LIQUID_S1_DEVICE", "cpu")
    cfg = load_config(None)
    assert cfg.specs["s1"].device == "cpu" and cfg.specs["s2"].device == ""


def test_engines_file_nested_url_default(monkeypatch):
    from tle.models.factory import load_specs

    monkeypatch.delenv("TLE_LIQUID_S1_URL", raising=False)
    monkeypatch.setenv("TLE_LIQUID_URL", "http://a:1")
    monkeypatch.setenv("TLE_LIQUID_S2_URL", "http://b:2")
    s = load_specs("config/engines.liquid.toml")
    assert s["s1"].base_url.startswith("http://a:1/") and s["s2"].base_url.startswith("http://b:2/")


def test_up_plan_layouts():
    from tle.liquid.up import plan

    igpu = {"backend": "rocm", "name": "AMD Radeon 780M Graphics", "cores": 8, "integrated": True}
    name, b = plan(igpu, ("s1", "s2"), 8100)
    assert name == "split-gpu" and [(x.roles, x.device) for x in b] == [(("s1",), "cpu"), (("s2",), "rocm")]
    assert b[0].threads == 6 and b[1].env.get("HSA_OVERRIDE_GFX_VERSION") == "11.0.0"
    assert [x.port for x in b] == [8101, 8102]
    nv = {"backend": "cuda", "name": "NVIDIA RTX 4090", "cores": 16, "integrated": False}
    name, b = plan(nv, ("s1", "s2"), 8100)
    assert name == "together" and len(b) == 1 and b[0].roles == ("s1", "s2") and b[0].device == "cuda"
    cpu = {"backend": "cpu", "name": "", "cores": 8}
    name, b = plan(cpu, ("s1", "s2"), 8100)
    assert name == "cpu" and [x.threads for x in b] == [4, 4]
    name, b = plan(cpu, ("s2",), 8100, max_batch=3)
    assert b[0].threads == 8 and b[0].max_batch == 3
    import pytest

    with pytest.raises(RuntimeError):
        plan(cpu, ("s1",), 8100, layout="together")


def test_up_gateway_routes_and_key():
    pytest.importorskip("httpx")
    import httpx
    from starlette.testclient import TestClient
    from tle.liquid.up import exports, make_gateway, make_link

    hits = []

    def fake(request: httpx.Request):
        hits.append(str(request.url))
        return httpx.Response(200, json={"status": "ok"})

    app = make_gateway(
        {"s1": "http://b1", "s2": "http://b2"},
        "KEY",
        client=httpx.AsyncClient(transport=httpx.MockTransport(fake)),
    )
    c = TestClient(app)
    assert c.post("/api/alpha/decisions", json={}).status_code == 401
    assert c.post("/k/KEY/api/alpha/decisions", json={}).status_code == 200
    assert c.post("/k/KEY/v1/chat/completions", json={}).status_code == 200
    assert c.post("/v1/chat/completions", json={}, headers={"Authorization": "Bearer KEY"}).status_code == 200
    assert hits == [
        "http://b1/api/alpha/decisions",
        "http://b2/v1/chat/completions",
        "http://b2/v1/chat/completions",
    ]
    only_s2 = TestClient(make_gateway({"s2": "http://b2"}, None))
    assert only_s2.post("/api/alpha/decisions", json={}).status_code == 404
    assert make_link("h", 1, "K") == "http://h:1/k/K" and make_link("h", 1, None) == "http://h:1"
    assert exports("L", ("s1", "s2")) == ["export TLE_LIQUID_URL=L"]
    assert exports("L", ("s2",)) == ["export TLE_LIQUID_S2_URL=L"]
