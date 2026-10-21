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
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .backends import Completion, Transport, _urllib_transport
from .prompts import describe_citizen

DEFAULT_URL = "https://openrouter.ai/api/alpha/decisions"
NEUTRAL_STRING = {"threat": "none"}  # free-string fields that take a default (everything else: "any")


# What a question asks, by the last name of the contract field (prompts are data: tune them here).
FIELD_HELP = {
    "pillar": "Which pillar of democracy does the decree restrict or attack? 'none' if it restricts none.",
    "domain": "Which area of public life does the decree belong to?",
    "aggressiveness": "How forceful is the decree? from 'very low' (a polite request) to 'maximum' (an outright seizure of power).",
    "tec_effect": "Effect of the decree on technology and modernization ('no effect' if it says nothing).",
    "cul_effect": "Effect of the decree on culture and tradition ('no effect' if it says nothing).",
    "econ_effect": "Effect of the decree on the economy ('no effect' if it says nothing).",
    "activity": "Which daily activity would this citizen choose today?",
    "option": "Which reaction would this citizen choose?",
}


OPTION_RULE = (
    "Pick the option that THIS citizen would choose given who they are (ideology, temperament, character, opinion of the "
    "mayor, views on the pillars of democracy, fear). Do not pick an option because it is listed first or because it "
    "supports the actor: citizens differ, and opponents of the actor pick opposing options."
)


def _shuffled(items: list, seed: int) -> list:
    """Options in an order that depends on the request (deterministic): a decision model must not learn that the first
    option is the usual answer."""
    order = np.random.default_rng(seed & 0xFFFFFFFF).permutation(len(items))
    return [items[i] for i in order]


def _in_words(state: dict) -> dict:
    """The context of a citizen as words (see ``describe_citizen``); other state is left as it is."""
    c = state.get("context")
    if isinstance(c, dict) and "temperament" in c:
        state = {**state, "citizen": describe_citizen(c)}
        state.pop("context")
    return state


class JevUnsupported(Exception):
    permanent = True


def _resolve(node: dict, defs: dict) -> dict:
    while "$ref" in node:
        node = defs[node["$ref"].rsplit("/", 1)[1]]
    return node


def _flatten(node: dict, defs: dict, path: str, questions: dict, hints: dict, prefix: str = "") -> Any:
    """Walk the schema; fill ``questions`` (path -> choice question) and return the JSON skeleton of the answer."""
    node = _resolve(node, defs)
    if "anyOf" in node:
        if any(x.get("type") == "null" for x in node["anyOf"]):
            return None  # optional structure: not requested
        raise JevUnsupported(f"{path}: union")
    if "enum" in node:
        field = path.rsplit(".", 1)[-1]
        h = hints.get(field, {})
        key = prefix + path
        questions[key] = {
            "type": "choice",
            "instructions": FIELD_HELP.get(field, f"Choose the value of '{path.replace('.', ' / ')}'."),
            "criteria": {v: h.get(v, v) for v in node["enum"]},
        }
        return {"__q__": key}
    if node.get("type") == "object":
        return {
            k: _flatten(c, defs, f"{path}.{k}" if path else k, questions, hints, prefix)
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
    hints: dict = field(default_factory=dict)  # field name -> {value: description} from the catalogs
    transport: Transport = _urllib_transport

    # -- request building ------------------------------------------------------------------------
    def build(self, user: str, schema: dict | None, seed: int = 0) -> tuple[dict, Any]:
        """Returns (request body, template used to rebuild the contract JSON from the answers)."""
        task = user.split("TASK:", 1)[1].split("\n", 1)[0].strip()
        desc = user.split("\n", 2)[1].strip()
        if "BATCH of" in user:
            return self._build_batch(task, desc, user, schema, seed)
        state = _in_words(json.loads(user.split("INPUT:\n", 1)[1]))
        questions, tpl = self._questions(task, desc, dict(state), schema, "", seed)
        return {"model": self.model, "state": state, "questions": questions}, tpl

    def _questions(
        self, task: str, desc: str, state: dict, schema: dict | None, prefix: str, seed: int = 0
    ) -> tuple[dict, Any]:
        """Questions for one item whose own part of the state is ``state``; ``prefix`` keeps keys apart in a batch."""
        if task in ("choose_reaction", "choose_influencer_response") and isinstance(
            state.get("options"), list
        ):
            opts = _shuffled(state.pop("options"), seed)
            key = prefix + "option"
            q = {
                key: {
                    "type": "choice",
                    "instructions": f"{desc} {OPTION_RULE}",
                    "criteria": {str(o["id"]): o["text"] for o in opts},
                }
            }
            return q, {"option": {"__q__": key}}
        if schema is None:
            raise JevUnsupported("no schema")
        questions: dict[str, dict] = {}
        tpl = _flatten(schema, schema.get("$defs", {}), "", questions, self.hints, prefix)
        if not questions:
            raise JevUnsupported("nothing to ask")
        for q in questions.values():
            q["instructions"] += f" Context: {desc}"
        return questions, tpl

    def _build_batch(
        self, task: str, desc: str, user: str, schema: dict | None, seed: int = 0
    ) -> tuple[dict, Any]:
        """k independent items in one request: the state holds the shared part and the items, every question names its item."""
        shared = json.loads(user.split("SHARED (applies to every item):\n", 1)[1].split("\n\nITEMS", 1)[0])
        items = json.loads(
            user.split("(index = position, starting at 0):\n", 1)[1].split("\n\nAnswer with", 1)[0]
        )
        items = [_in_words(it) for it in items]
        item_schema = dict(schema["properties"]["items"]["items"]) if schema else None
        if item_schema is not None and "$defs" in schema:
            item_schema["$defs"] = schema["$defs"]
        questions: dict[str, dict] = {}
        tpls = []
        for j in range(len(items)):
            view = dict(shared)
            if (
                task == "choose_reaction" and "options" in shared
            ):  # the options are shared: every item picks among them
                qs, tpl = self._questions(
                    task, desc, {"options": shared["options"]}, item_schema, f"i{j}:", seed + 7919 * j
                )
            else:
                qs, tpl = self._questions(task, desc, dict(view), item_schema, f"i{j}:", seed + 7919 * j)
            for q in qs.values():
                q["instructions"] += f" Answer for the item at index {j} of state.items."
            questions.update(qs)
            tpls.append(tpl)
        state = {"shared": shared, "items": items}
        return {"model": self.model, "state": state, "questions": questions}, {"items": tpls}

    def complete(self, system, user, schema, temperature=0.0, seed=None) -> Completion:
        body, tpl = self.build(user, schema, int(seed or 0))
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
