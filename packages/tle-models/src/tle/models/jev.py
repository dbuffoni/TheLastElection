"""Jev backend (TypeSafe's System One decision model, served by OpenRouter or a compatible server).

Jev does not chat: it receives a ``state`` and typed ``questions`` and returns ``choice`` answers with probabilities. The
game's S1 tasks already return closed vocabularies, so the contract's JSON schema is turned into questions here:
every enum field becomes a ``choice`` question; nested objects are flattened (``predicate.trade``); optional objects
(``anyOf`` with null) become null; free strings take their neutral default. The one dynamic task, ``choose_reaction``
(S2 wrote the closed options once for the event, Jev picks one per citizen), becomes a single question whose criteria are
the options. Anything else (lists, free-form fields) raises ``JevUnsupported``: the RealEngine counts it as a failed
attempt and falls back to the simulated rule, so the game never stops. S2 stays on an LLM: Jev cannot write text."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from .backends import Completion, Transport, _urllib_transport

DEFAULT_URL = "https://openrouter.ai/api/alpha/decisions"
NEUTRAL_STRING = {"threat": "none"}  # free-string fields that take a default (everything else: "any")


class JevUnsupported(Exception):
    permanent = True


def _resolve(node: dict, defs: dict) -> dict:
    while "$ref" in node:
        node = defs[node["$ref"].rsplit("/", 1)[1]]
    return node


def _flatten(node: dict, defs: dict, path: str, questions: dict, template: dict, rules: dict) -> Any:
    """Walk the schema; fill ``questions`` (path -> choice question) and return the JSON skeleton of the answer."""
    node = _resolve(node, defs)
    if "anyOf" in node:
        if any(x.get("type") == "null" for x in node["anyOf"]):
            return None  # optional structure: not requested
        raise JevUnsupported(f"{path}: union")
    if "enum" in node:
        questions[path] = {
            "type": "choice",
            "instructions": f"Choose the value of '{path}'.",
            "criteria": {v: v for v in node["enum"]},
        }
        return {"__q__": path}
    if node.get("type") == "object":
        return {
            k: _flatten(c, defs, f"{path}.{k}" if path else k, questions, template, rules)
            for k, c in node["properties"].items()
        }
    if node.get("type") == "string":
        return {"__s__": NEUTRAL_STRING.get(path.rsplit(".", 1)[-1], "any")}
    raise JevUnsupported(f"{path}: {node.get('type')}")


def _fill(tpl: Any, answers: dict) -> Any:
    if isinstance(tpl, dict):
        if "__q__" in tpl:
            return answers[tpl["__q__"]]["choice"]
        if "__s__" in tpl:
            return tpl["__s__"]
        return {k: _fill(v, answers) for k, v in tpl.items()}
    return tpl


@dataclass
class JevBackend:
    base_url: str = DEFAULT_URL
    model: str = "typesafe/jev-1.14"
    api_key: str = "not-needed"
    timeout: float = 60.0
    name: str = "jev"
    max_concurrency: int = 16
    transport: Transport = _urllib_transport

    # -- request building ------------------------------------------------------------------------
    def build(self, user: str, schema: dict | None) -> tuple[dict, Any]:
        """Returns (request body, template used to rebuild the contract JSON from the answers)."""
        task = user.split("TASK:", 1)[1].split("\n", 1)[0].strip()
        desc = user.split("\n", 2)[1].strip()
        state = json.loads(user.split("INPUT:\n", 1)[1])
        if task in ("choose_reaction", "choose_influencer_response") and isinstance(
            state.get("options"), list
        ):
            opts = state.pop("options")
            q = {
                "option": {
                    "type": "choice",
                    "instructions": f"{desc} Pick the option this citizen would choose.",
                    "criteria": {str(o["id"]): o["text"] for o in opts},
                }
            }
            return {"model": self.model, "state": state, "questions": q}, {"option": {"__q__": "option"}}
        if schema is None:
            raise JevUnsupported("no schema")
        questions: dict[str, dict] = {}
        tpl = _flatten(schema, schema.get("$defs", {}), "", questions, {}, {})
        if not questions:
            raise JevUnsupported("nothing to ask")
        for q in questions.values():
            q["instructions"] += f" Context: {desc}"
        return {"model": self.model, "state": state, "questions": questions}, tpl

    def complete(self, system, user, schema, temperature=0.0, seed=None) -> Completion:
        body, tpl = self.build(user, schema)
        t0 = time.perf_counter()
        raw = self.transport(
            self.base_url,
            {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            json.dumps(body).encode(),
            self.timeout,
        )
        data = json.loads(raw)
        answers = data["answers"]
        usage = data.get("usage", {})
        return Completion(
            text=json.dumps(_fill(tpl, answers)),
            in_tokens=usage.get("input_tokens", 0),
            out_tokens=usage.get("output_tokens", 0),
            latency_s=time.perf_counter() - t0,
        )
