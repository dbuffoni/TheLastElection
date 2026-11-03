"""Integrated smoke test: starts the server in this process, fires concurrent requests at both endpoints and checks the
answers, reports whether requests were really fused into batches, and (``--game``) plays a few game turns through the
server with the game's own engine settings. Modes: ``real`` (downloaded weights), ``tiny`` (random-weight LFM2 on the real S2
code path + mock S1) and ``mock`` (no model at all). Output ends with a PASS/FAIL table; exit code 0 only if all pass."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from .config import LiquidConfig
from .engine import LiquidEngine

CHAT_SCHEMA = {
    "type": "object",
    "properties": {
        "mood": {"type": "string", "enum": ["calm", "angry", "hopeful"]},
        "line": {"type": "string", "maxLength": 40},
        "support": {"type": "integer", "minimum": 0, "maximum": 9},
        "tags": {"type": "array", "maxItems": 2, "items": {"type": "string", "maxLength": 8}},
    },
    "required": ["mood", "line", "support", "tags"],
    "additionalProperties": False,
}
QUESTIONS = {
    "pillar": {
        "type": "choice",
        "instructions": "Which pillar does the decree attack?",
        "criteria": {"none": "none", "press": "free press", "courts": "independent courts"},
    },
    "tone": {
        "type": "choice",
        "instructions": "How forceful is it?",
        "criteria": {"low": "polite", "high": "forceful"},
    },
}


def validate(v, s, defs=None, lengths: bool = True) -> str | None:
    """Minimal JSON-schema check for the strict subset (returns an error string or None). ``lengths=False`` skips maxLength:
    the random tiny model emits invalid UTF-8 byte soup whose decoded length differs from the grammar's count."""
    defs = defs if defs is not None else s.get("$defs", {})
    while "$ref" in s:
        s = defs[s["$ref"].rsplit("/", 1)[1]]
    if "enum" in s:
        return None if v in s["enum"] else f"{v!r} not in {s['enum']}"
    if "anyOf" in s:
        errs = [validate(v, o, defs, lengths) for o in s["anyOf"]]
        return None if None in errs else f"no anyOf branch matches: {errs}"
    t = s.get("type")
    if t == "object":
        if not isinstance(v, dict):
            return "not an object"
        for k in s.get("required", []):
            if k not in v:
                return f"missing {k}"
        extra = set(v) - set(s.get("properties", {}))
        if extra and s.get("additionalProperties") is False:
            return f"extra keys {extra}"
        for k, c in s.get("properties", {}).items():
            if k in v and (e := validate(v[k], c, defs, lengths)):
                return f"{k}: {e}"
    elif t == "array":
        if not isinstance(v, list) or len(v) > s.get("maxItems", 10**9) or len(v) < s.get("minItems", 0):
            return "bad array"
        for x in v:
            if e := validate(x, s.get("items", {}), defs, lengths):
                return e
    elif t == "string" and not isinstance(v, str):
        return "not a string"
    elif t == "string" and lengths and len(v) > s.get("maxLength", 10**9):
        return "string too long"
    elif t == "integer" and (
        not isinstance(v, int)
        or isinstance(v, bool)
        or not s.get("minimum", -(10**18)) <= v <= s.get("maximum", 10**18)
    ):
        return "bad integer"
    elif t == "null" and v is not None:
        return "not null"
    return None


def _post(url: str, body: dict, key: str = "") -> dict:
    h = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    with urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), h), timeout=600) as r:
        return json.loads(r.read())


def run(cfg: LiquidConfig, mode: str = "tiny", game: bool = False, n_requests: int = 32) -> int:
    import uvicorn

    from .server import create_app

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cfg = replace(cfg, host="127.0.0.1", port=port, api_key="")
    t0 = time.perf_counter()
    engine = LiquidEngine(cfg, mode=mode)
    print(f"engine ready in {time.perf_counter() - t0:.1f}s: {json.dumps(engine.info(), indent=None)}")
    srv = uvicorn.Server(uvicorn.Config(create_app(engine), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    base = f"http://127.0.0.1:{port}"
    results: list[tuple[str, bool, str]] = []

    def check(name, fn):
        t = time.perf_counter()
        try:
            note = fn() or ""
            results.append((name, True, f"{note} [{time.perf_counter() - t:.2f}s]"))
        except Exception as e:
            results.append((name, False, f"{type(e).__name__}: {e}"))

    pool = ThreadPoolExecutor(max_workers=max(8, n_requests))
    roles = cfg.roles

    def t_health():
        with urllib.request.urlopen(base + "/health") as r:
            assert json.loads(r.read())["status"] == "ok"
        with urllib.request.urlopen(base + "/v1/models") as r:
            ids = [m["id"] for m in json.loads(r.read())["data"]]
        return f"models {ids}"

    def decide(i):
        q = json.loads(json.dumps(QUESTIONS))
        return _post(
            base + "/api/alpha/decisions",
            {
                "model": "liquid-s1",
                "state": {"decree": f"Close the paper number {i}", "i": i},
                "questions": q,
            },
        )

    def t_decide():
        before = len(engine.batchers["s1"].batches)
        outs = list(pool.map(decide, range(n_requests)))
        for o in outs:
            for name, q in QUESTIONS.items():
                a = o["answers"][name]
                assert a["choice"] in q["criteria"], a
                assert abs(sum(a["probabilities"].values()) - 1) < 1e-3
        sizes = engine.batchers["s1"].batches[before:]
        assert max(sizes) > 1, f"requests were not fused: batch sizes {sizes}"
        return f"{n_requests} concurrent -> batches {sizes}"

    def chat(i, schema=CHAT_SCHEMA):
        body = {
            "model": "liquid-s2",
            "temperature": 0.7,
            "seed": 7,
            "max_tokens": 160,
            "messages": [
                {"role": "system", "content": "You are the narrator."},
                {"role": "user", "content": f"Write item {i}."},
            ],
        }
        if schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "strict": True, "schema": schema},
            }
        return _post(base + "/v1/chat/completions", body)

    def t_chat_schema():
        before = len(engine.batchers["s2"].batches)
        t = time.perf_counter()
        outs = list(pool.map(chat, range(n_requests)))
        dt = time.perf_counter() - t
        toks = 0
        for o in outs:
            c = o["choices"][0]
            assert c["finish_reason"] == "stop", f"finish_reason {c['finish_reason']}"
            err = validate(json.loads(c["message"]["content"]), CHAT_SCHEMA, lengths=mode != "tiny")
            assert err is None, err
            toks += o["usage"]["completion_tokens"]
        sizes = engine.batchers["s2"].batches[before:]
        assert max(sizes) > 1, f"requests were not fused: batch sizes {sizes}"
        return (
            f"{n_requests} concurrent, all valid against the schema, {toks / dt:.0f} tok/s, batches {sizes}"
        )

    def t_chat_free():
        o = chat(0, None)
        assert isinstance(o["choices"][0]["message"]["content"], str)
        return "free text ok"

    def t_errors():
        for url, body in ((base + "/v1/chat/completions", {}), (base + "/api/alpha/decisions", {"state": 1})):
            try:
                _post(url, body)
                raise AssertionError("expected HTTP 400")
            except urllib.error.HTTPError as e:
                assert e.code == 400, e.code
        return "bad bodies -> 400"

    def t_game():
        py = "from tle.cli.main import app; app()"
        env = {**os.environ, "TLE_LIQUID_URL": base}
        with tempfile.TemporaryDirectory() as d:
            cmd = [
                sys.executable,
                "-c",
                py,
                "run",
                "--engines",
                "config/engines.liquid.toml",
                "--n",
                "20",
                "--max-turns",
                "3",
                "--no-telemetry",
                "--runs-dir",
                d,
                "--strategy",
                "random",
            ]
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        assert r.returncode == 0, (r.stdout + r.stderr)[-1500:]
        return "3 game turns through the server: " + r.stdout.strip().splitlines()[-1][:120]

    check("health + models", t_health)
    check("bad requests", t_errors)
    if "s1" in roles:
        check("S1 decisions, concurrent (batched)", t_decide)
    if "s2" in roles:
        check("S2 chat, JSON-schema constrained, concurrent (batched)", t_chat_schema)
        check("S2 chat, free text", t_chat_free)
    if game:
        check("game turns via config/engines.liquid.toml", t_game)
    srv.should_exit = True
    engine.close()
    width = max(len(n) for n, _, _ in results)
    print()
    for n, ok, note in results:
        print(f"{'PASS' if ok else 'FAIL'}  {n:<{width}}  {note}")
    return 0 if all(ok for _, ok, _ in results) else 1
