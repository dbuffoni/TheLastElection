"""RealEngine: the Engine protocol over a model backend. Validates every answer against the contract and
the closed vocabularies, retries up to ``n_retry`` times, then falls back (13.3), so the pipeline never
sees a malformed answer. Metrics and warnings are recorded exactly like the simulated engine."""

from __future__ import annotations

import json
import threading
from typing import Any

import numpy as np
from pydantic import BaseModel, ValidationError
from tle.contracts.schema import check_vocab, export_schema
from tle.contracts.tasks import TASKS
from tle.core.rng import GameRng
from tle.engines.protocol import ActorCtx, ExecutionRecord, LabelContext, OptionFeatures

from .backends import Backend
from .prompts import render


class RealEngine:
    def __init__(
        self,
        backend: Backend,
        vocab: dict[str, list[str]],
        fallback=None,
        n_retry: int = 2,
        temperature: float = 0.0,
        name: str = "real",
    ):
        self.backend, self.vocab, self.fallback = backend, vocab, fallback
        self.n_retry, self.temperature, self.name = n_retry, temperature, name
        self._exec: dict[str, ExecutionRecord] = {}
        self._warn: list[dict[str, Any]] = []
        self._schemas: dict[str, dict] = {}
        self._lock = threading.Lock()  # per-citizen calls may run in parallel

    def _rec(self, task: str) -> ExecutionRecord:
        return self._exec.setdefault(
            task, ExecutionRecord(task=task, backend=self.backend.name, model=self.backend.model)
        )

    def schema(self, task: str) -> dict:
        if task not in self._schemas:
            self._schemas[task] = export_schema(TASKS[task].output_model, self.vocab)
        return self._schemas[task]

    @property
    def concurrency(self) -> int:
        return int(getattr(self.backend, "max_concurrency", 1))

    def call(self, task: str, inp: BaseModel, rng: GameRng) -> BaseModel | None:
        if task == "classify_decree" and getattr(inp, "_structure", None) is not None:
            return (
                inp._structure
            )  # a simulated player already structured the decree (15.5): nothing to ask the model
        spec = TASKS[task]
        system, user = render(task, inp)
        with self._lock:
            rec = self._rec(task)
            rec.calls += 1
        last = ""
        for attempt in range(self.n_retry + 1):
            with self._lock:
                rec.attempts += 1
            try:
                c = self.backend.complete(
                    system, user, self.schema(task), self.temperature, seed=int(rng.integers(0, 2**31 - 1))
                )
            except Exception as e:  # transport errors count as failed attempts
                last = f"backend error: {e}"
                if getattr(
                    e, "permanent", False
                ):  # e.g. a task the backend cannot express: retrying cannot help
                    break
                continue
            with self._lock:
                rec.in_tokens += c.in_tokens
                rec.out_tokens += c.out_tokens
                rec.latency_s += c.latency_s
            last = c.text
            try:
                out = spec.output_model.model_validate(json.loads(c.text))
            except (ValidationError, json.JSONDecodeError):
                continue
            if check_vocab(out, self.vocab):
                continue
            return out
        with self._lock:
            rec.fallbacks += 1
            self._warn.append({"task": task, "input": user[-200:], "response": last[:300]})
        if self.fallback is not None:
            return self.fallback.call(task, inp, rng)
        return None

    # Batched choices go one citizen at a time through the generic call (a vLLM server batches them itself).
    def choose_reaction_many(
        self, ctx: LabelContext, actor: ActorCtx, news, options, features: OptionFeatures, rng: GameRng
    ) -> np.ndarray:
        from tle.contracts import models as m

        opts = [m.OptionIn(id=str(k), text=o.text) for k, o in enumerate(options)]

        def one(i):
            res = self.call(
                "choose_reaction",
                m.ChooseReactionIn(context=ctx.row(i), news=news, options=opts),
                rng.child("c", i),
            )
            return (
                int(res.option)
                if res is not None and res.option.isdigit() and int(res.option) < len(opts)
                else len(opts) // 2
            )

        return np.array(self._map(one, len(ctx)), dtype=int)

    def choose_activity_many(self, ctx: LabelContext, rng: GameRng) -> np.ndarray:
        from tle.contracts import models as m

        acts = self.vocab["activities"]

        def one(i):
            res = self.call("choose_activity", m.ChooseActivityIn(context=ctx.row(i)), rng.child("a", i))
            return acts.index(res.activity) if res is not None else 0

        return np.array(self._map(one, len(ctx)), dtype=int)

    def _map(self, fn, n: int) -> list[int]:
        """Per-citizen calls; in parallel when the backend allows it (results keep their order)."""
        workers = self.concurrency
        if workers <= 1 or n <= 1:
            return [fn(i) for i in range(n)]
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(fn, range(n)))

    def drain_execution(self) -> list[ExecutionRecord]:
        out = list(self._exec.values())
        self._exec = {}
        if self.fallback is not None:
            out += self.fallback.drain_execution()
        return out

    def drain_warnings(self) -> list[dict[str, Any]]:
        out, self._warn = self._warn, []
        if self.fallback is not None:
            out += self.fallback.drain_warnings()
        return out
