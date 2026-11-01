"""LiquidEngine: loads the runtimes of the configured roles and puts a micro-batcher in front of each. Transport-free:
the HTTP server, the smoke test and any Python caller use the same two methods, ``chat`` and ``decide``."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future
from dataclasses import replace

from .batching import MicroBatcher
from .config import LiquidConfig, ModelSpec, fetch

log = logging.getLogger("tle.liquid")


class LiquidEngine:
    def __init__(self, cfg: LiquidConfig, mode: str = "real", download: bool = True):
        """``mode``: real (the configured weights) | mock (no weights at all) | tiny (a random-weight LFM2 built locally runs
        the real S2 code path on any machine; S1 stays a mock because d1 ships its own remote code)."""
        self.cfg, self.mode, self.download = cfg, mode, download
        self.mock = mode == "mock"
        self._tiny_dir = None
        self.runtimes: dict[str, object] = {}
        self.batchers: dict[str, MicroBatcher] = {}
        self.device = None
        self.load_seconds: dict[str, float] = {}
        self._lock = threading.Lock()
        if mode != "mock":
            from .devices import detect

            self.device = detect(cfg.device)
            log.info("device: %s", self.device.describe())
        if not cfg.lazy_load:
            for role in cfg.roles:
                self._ensure(role)

    # ---- loading ----------------------------------------------------------------------------------------------
    def spec(self, role: str) -> ModelSpec:
        s = self.cfg.specs[role]
        return replace(s, kind="mock") if self.mock or (self.mode == "tiny" and role == "s1") else s

    def _ensure(self, role: str):
        with self._lock:
            if role in self.runtimes:
                return self.runtimes[role]
            if role not in self.cfg.roles:
                raise KeyError(
                    f"role '{role}' is not served by this instance (roles: {list(self.cfg.roles)})"
                )
            spec, t0 = self.spec(role), time.perf_counter()
            if spec.kind == "mock":
                from .mock import MockChatRuntime, MockDecisionRuntime

                rt = MockDecisionRuntime() if role == "s1" else MockChatRuntime()
            else:
                if self.device.backend != "cpu" and spec.approx_gb > self.device.free_gb:
                    log.warning(
                        "%s needs about %.1f GB and the GPU has %.1f GB free",
                        spec.repo,
                        spec.approx_gb,
                        self.device.free_gb,
                    )
                if self.mode == "tiny":
                    import tempfile

                    from .tiny import build_tiny_lfm2

                    self._tiny_dir = tempfile.mkdtemp(prefix="tle-liquid-tiny-")
                    path = build_tiny_lfm2(self._tiny_dir)
                else:
                    path = fetch(spec) if self.download else spec.repo
                from .runtimes import ChatRuntime, DecisionRuntime

                rt = (
                    DecisionRuntime(spec, path, self.device, self.cfg.dtype)
                    if spec.kind == "decision"
                    else ChatRuntime(spec, path, self.device, self.cfg.dtype, self.cfg.max_new_tokens)
                )
            self.load_seconds[role] = time.perf_counter() - t0
            self.runtimes[role] = rt
            self.batchers[role] = MicroBatcher(
                rt.run_batch, self.cfg.max_batch, self.cfg.max_wait_ms, name=f"tle-liquid-{role}"
            )
            log.info("loaded %s (%s) in %.1fs", spec.repo, role, self.load_seconds[role])
            return rt

    # ---- the two operations -----------------------------------------------------------------------------------
    def chat(self, req: dict) -> Future:
        """{"messages", "schema"?, "temperature"?, "max_tokens"?, "seed"?} -> Future[{"text", "finish", "in", "out"}]"""
        self._ensure("s2")
        return self.batchers["s2"].submit(req)

    def decide(self, req: dict) -> Future:
        """{"state", "questions"} -> Future[{"answers", "usage"}]"""
        self._ensure("s1")
        return self.batchers["s1"].submit(req)

    def chat_many(self, reqs: list[dict]) -> list[dict]:
        fs = [self.chat(r) for r in reqs]
        return [f.result() for f in fs]

    def decide_many(self, reqs: list[dict]) -> list[dict]:
        fs = [self.decide(r) for r in reqs]
        return [f.result() for f in fs]

    def info(self) -> dict:
        return {
            "mode": self.mode,
            "device": "mock" if self.mock else self.device.describe(),
            "roles": {
                r: {
                    "model": self.spec(r).repo,
                    "alias": self.spec(r).alias,
                    "kind": self.spec(r).kind,
                    "loaded": r in self.runtimes,
                }
                for r in self.cfg.roles
            },
            "batches": {
                r: {"n": len(b.batches), "max": max(b.batches, default=0)} for r, b in self.batchers.items()
            },
            "max_batch": self.cfg.max_batch,
        }

    def close(self) -> None:
        for b in self.batchers.values():
            b.close()
