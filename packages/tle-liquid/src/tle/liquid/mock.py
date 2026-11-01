"""Runtimes without weights: deterministic stand-ins with the exact interface of the real ones. They exist so the server,
the batcher, the HTTP contract and the game integration can be tested on any machine (CI, a laptop with no GPU, a sandbox
with no access to huggingface.co). They say nothing about model quality."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _h(*parts: Any) -> int:
    return int.from_bytes(
        hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).digest()[:8], "big"
    )


class MockDecisionRuntime:
    kind = "decision"

    def __init__(self, *a, **k):
        self.calls: list[int] = []

    def run_batch(self, reqs: list[dict]) -> list[dict]:
        self.calls.append(len(reqs))
        out = []
        for r in reqs:
            answers = {}
            for name, q in r["questions"].items():
                crit = list(q.get("criteria", {}))
                if q.get("type") == "choice" and crit:
                    w = [1.0 + (_h(r["state"], name, c) % 1000) / 1000.0 for c in crit]
                    tot = sum(w)
                    probs = {c: x / tot for c, x in zip(crit, w)}
                    best = max(probs, key=probs.get)
                    answers[name] = {"choice": best, "confidence": probs[best], "probabilities": probs}
                else:
                    answers[name] = {"noul": 0.5}
            out.append(
                {
                    "answers": answers,
                    "usage": {"input_tokens": len(json.dumps(r["state"])) // 4, "output_tokens": 0},
                }
            )
        return out


_WORDS = "the mayor town council market river bridge school farmers workers quiet loud angry hopeful tired proud".split()


def fill_schema(schema: dict, seed: int, defs: dict | None = None, key: str = "") -> Any:
    """A deterministic value that satisfies a (strict-subset) JSON schema."""
    defs = defs if defs is not None else schema.get("$defs", {})
    while "$ref" in schema:
        schema = defs[schema["$ref"].rsplit("/", 1)[1]]
    if "enum" in schema:
        return schema["enum"][_h(seed, key) % len(schema["enum"])]
    if "const" in schema:
        return schema["const"]
    if "anyOf" in schema:
        opts = [o for o in schema["anyOf"] if o.get("type") != "null"] or schema["anyOf"]
        return fill_schema(opts[0], seed, defs, key)
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), t[0])
    if t == "object":
        return {k: fill_schema(v, seed, defs, k) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        n = max(schema.get("minItems", 0), min(schema.get("maxItems", 3), 2))
        return [fill_schema(schema.get("items", {}), seed + j, defs, key) for j in range(n)]
    if t == "boolean":
        return bool(_h(seed, key) & 1)
    if t == "integer":
        lo, hi = int(schema.get("minimum", 0)), int(schema.get("maximum", 10))
        return lo + _h(seed, key) % (hi - lo + 1)
    if t == "number":
        lo, hi = float(schema.get("minimum", 0.0)), float(schema.get("maximum", 1.0))
        return lo + (hi - lo) * ((_h(seed, key) % 1000) / 1000.0)
    n = 6 + _h(seed, key) % 6
    text = " ".join(_WORDS[_h(seed, key, j) % len(_WORDS)] for j in range(n))
    return text[: schema.get("maxLength", len(text))]


class MockChatRuntime:
    kind = "chat"
    constrained = True

    def __init__(self, *a, **k):
        self.calls: list[int] = []

    def run_batch(self, reqs: list[dict]) -> list[dict]:
        self.calls.append(len(reqs))
        out = []
        for r in reqs:
            seed = _h(r["messages"], r.get("seed"))
            text = (
                json.dumps(fill_schema(r["schema"], seed))
                if r.get("schema")
                else "The town waits, and nobody quite knows for what."
            )
            out.append(
                {
                    "text": text,
                    "finish": "stop",
                    "in": sum(len(m["content"]) for m in r["messages"]) // 4,
                    "out": len(text) // 4,
                }
            )
        return out
