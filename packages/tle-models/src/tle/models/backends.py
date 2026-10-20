"""Model backends (spec 3.6 / Technical Implementation 3.6). A backend turns (system, user, JSON schema)
into text; it knows nothing about the game. ``openai_compat`` covers vLLM, llama.cpp server, Ollama,
LM Studio and hosted APIs; its HTTP transport is injectable so tests never open a socket. A ``local``
in-process backend (transformers + outlines) is deliberately not implemented in the cloud slice: see
docs/local-setup.md for how to add it on the machine that has the GPU."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Completion:
    text: str
    in_tokens: int = 0
    out_tokens: int = 0
    latency_s: float = 0.0


class Backend(Protocol):
    name: str
    model: str

    def complete(
        self,
        system: str,
        user: str,
        schema: dict[str, Any] | None,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Completion: ...


@dataclass
class MockBackend:
    """Deterministic scripted backend for tests and dry runs. ``script`` maps a task name (found in the
    user prompt header ``TASK: <name>``) to a response (str) or a callable(user_prompt) -> str.
    Unscripted tasks raise, so a test cannot silently pass through an unexpected call."""

    script: dict[str, Any] = field(default_factory=dict)
    name: str = "mock"
    model: str = "mock-1"
    calls: list[tuple[str, str]] = field(default_factory=list)

    def complete(self, system, user, schema, temperature=0.0, seed=None) -> Completion:
        task = user.split("TASK:", 1)[1].split("\n", 1)[0].strip() if "TASK:" in user else ""
        self.calls.append((task, user))
        if task not in self.script:
            raise KeyError(f"MockBackend: no scripted response for task '{task}'")
        r = self.script[task]
        text = r(user) if callable(r) else r
        if isinstance(r, list):  # a queue of responses
            text = r.pop(0)
        return Completion(text=text, in_tokens=len(user) // 4, out_tokens=len(text) // 4)


Transport = Callable[[str, dict[str, str], bytes, float], bytes]


_POOL = ThreadPoolExecutor(max_workers=128, thread_name_prefix="tle-http")


def _urllib_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
    """POST with a hard deadline: a socket timeout alone does not fire when a gateway keeps the connection alive
    with keepalive bytes while the model never answers, so the request runs in a worker and is abandoned at the deadline."""

    def post() -> bytes:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (user-configured endpoint)
            return resp.read()

    try:
        return _POOL.submit(post).result(timeout=timeout)
    except FutureTimeout as e:
        raise TimeoutError(f"no answer within {timeout:g}s") from e


@dataclass
class OpenAICompatBackend:
    """Chat-completions client with strict JSON-schema structured output (``response_format``)."""

    base_url: str = "http://localhost:8000/v1"
    model: str = "local-model"
    api_key: str = "not-needed"
    timeout: float = 60.0
    name: str = "openai_compat"
    structured: bool = True
    max_tokens: int = (
        2048  # bounds a runaway generation (some providers loop under schema-constrained decoding)
    )
    max_concurrency: int = 8  # parallel per-citizen requests
    retries_429: int = 4  # rate-limited requests wait 1, 2, 4... seconds and try again
    transport: Transport = _urllib_transport

    def complete(self, system, user, schema, temperature=0.0, seed=None) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if seed is not None:
            payload["seed"] = seed
        if schema is not None and self.structured:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "strict": True, "schema": schema},
            }
        t0 = time.perf_counter()
        for k in range(self.retries_429 + 1):
            try:
                raw = self.transport(
                    self.base_url.rstrip("/") + "/chat/completions",
                    {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
                    json.dumps(payload).encode(),
                    self.timeout,
                )
                break
            except urllib.error.HTTPError as e:
                if e.code != 429 or k == self.retries_429:
                    raise
                time.sleep(2.0**k)
        data = json.loads(raw)
        usage = data.get("usage", {})
        return Completion(
            text=data["choices"][0]["message"]["content"],
            in_tokens=usage.get("prompt_tokens", 0),
            out_tokens=usage.get("completion_tokens", 0),
            latency_s=time.perf_counter() - t0,
        )


def make_backend(kind: str, **kw) -> Backend:
    if kind == "mock":
        return MockBackend(**kw)
    if kind == "openai_compat":
        return OpenAICompatBackend(**kw)
    if kind == "jev":
        from .jev import JevBackend

        return JevBackend(**kw)
    if kind == "local":
        raise NotImplementedError(
            "the in-process 'local' backend is added on the GPU machine; "
            "run a local server (vLLM/llama.cpp/Ollama) and use 'openai_compat'"
        )
    raise ValueError(f"unknown backend '{kind}'")
