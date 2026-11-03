"""HTTP front of the engine (Starlette). Three families of endpoints:

* ``POST /v1/chat/completions``     OpenAI chat completions with ``response_format = json_schema`` (S2, any OpenAI client)
* ``POST /api/alpha/decisions``     the decision-model API (state + typed questions -> answers); also at ``/v1/decisions``.
                                    This is what the game's ``kind = "jev"`` engine speaks, so S1 needs no adapter.
* ``GET  /health``, ``GET /v1/models``

Requests are not served one by one: every handler hands its request to the role's micro-batcher, so many concurrent clients
become few GPU calls. Set ``api_key`` to require ``Authorization: Bearer <key>`` (do this when binding 0.0.0.0).
"""

from __future__ import annotations

import asyncio
import hmac
import os
import time
import uuid

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .engine import LiquidEngine


def _err(status: int, msg: str, kind: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse({"error": {"message": msg, "type": kind}}, status_code=status)


def create_app(engine: LiquidEngine) -> Starlette:
    key = engine.cfg.api_key
    needle = os.environ.get("TLE_LIQUID_TEST_POISON", "")

    def unauthorized(request: Request) -> JSONResponse | None:
        if not key:
            return None
        got = request.headers.get("authorization", "")
        if not hmac.compare_digest(got.encode(), f"Bearer {key}".encode()):
            return _err(401, "missing or wrong API key", "authentication_error")
        return None

    async def health(request: Request):
        return JSONResponse({"status": "ok", **engine.info()})

    async def models(request: Request):
        if (r := unauthorized(request)) is not None:
            return r
        data = [
            {
                "id": engine.spec(r_).alias,
                "object": "model",
                "owned_by": "liquid",
                "root": engine.spec(r_).repo,
            }
            for r_ in engine.cfg.roles
        ]
        return JSONResponse({"object": "list", "data": data})

    async def chat(request: Request):
        if (r := unauthorized(request)) is not None:
            return r
        try:
            body = await request.json()
            messages = body["messages"]
            assert isinstance(messages, list) and messages
        except Exception:
            return _err(400, "body must be JSON with a non-empty 'messages' list")
        if "s2" not in engine.cfg.roles:
            return _err(404, f"this instance does not serve the chat model (roles: {list(engine.cfg.roles)})")
        rf = body.get("response_format") or {}
        schema = rf.get("json_schema", {}).get("schema") if rf.get("type") == "json_schema" else None
        n = int(body.get("n", 1))
        req = {
            "messages": [
                {
                    "role": m["role"],
                    "content": m["content"]
                    if isinstance(m["content"], str)
                    else "".join(p.get("text", "") for p in m["content"]),
                }
                for m in messages
            ],
            "schema": schema,
            "temperature": float(body.get("temperature", 0.0) or 0.0),
            "max_tokens": body.get("max_completion_tokens") or body.get("max_tokens"),
            "seed": body.get("seed"),
        }
        try:
            outs = await asyncio.gather(
                *[asyncio.wrap_future(engine.chat(dict(req))) for _ in range(max(1, n))]
            )
        except Exception as e:
            return _err(500, f"{type(e).__name__}: {e}", "server_error")
        return JSONResponse(
            {
                "id": "chatcmpl-" + uuid.uuid4().hex[:24],
                "object": "chat.completion",
                "created": int(time.time()),
                "model": engine.spec("s2").alias,
                "choices": [
                    {
                        "index": i,
                        "message": {"role": "assistant", "content": o["text"]},
                        "finish_reason": o["finish"],
                    }
                    for i, o in enumerate(outs)
                ],
                "usage": {
                    "prompt_tokens": outs[0]["in"],
                    "completion_tokens": sum(o["out"] for o in outs),
                    "total_tokens": outs[0]["in"] + sum(o["out"] for o in outs),
                },
            }
        )

    async def decisions(request: Request):
        if (r := unauthorized(request)) is not None:
            return r
        if (
            needle and needle.encode() in await request.body()
        ):  # tests: a request that kills the process (like a GPU hang)
            os._exit(134)
        try:
            body = await request.json()
            state, questions = body["state"], body["questions"]
            assert isinstance(questions, dict) and questions
        except Exception:
            return _err(400, "body must be JSON with 'state' and a non-empty 'questions' object")
        if "s1" not in engine.cfg.roles:
            return _err(
                404, f"this instance does not serve the decision model (roles: {list(engine.cfg.roles)})"
            )
        try:
            out = await asyncio.wrap_future(engine.decide({"state": state, "questions": questions}))
        except Exception as e:
            return _err(500, f"{type(e).__name__}: {e}", "server_error")
        return JSONResponse(out)

    return Starlette(
        routes=[
            Route("/health", health),
            Route("/v1/models", models),
            Route("/v1/chat/completions", chat, methods=["POST"]),
            Route("/api/alpha/decisions", decisions, methods=["POST"]),
            Route("/v1/decisions", decisions, methods=["POST"]),
        ]
    )


def serve(engine: LiquidEngine) -> None:
    import uvicorn

    uvicorn.run(create_app(engine), host=engine.cfg.host, port=engine.cfg.port, log_level="info")
