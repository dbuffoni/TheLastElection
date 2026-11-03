"""Micro-batching: concurrent requests are fused into one call of the model. A worker thread takes the first waiting
request, waits up to ``max_wait_ms`` for more (up to ``max_batch``) and runs them together. The game sends many
independent requests (one per citizen, or ``batch_size`` items each), so this is what turns them into GPU batches."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any


class MicroBatcher:
    def __init__(
        self,
        fn: Callable[[list[Any]], list[Any]],
        max_batch: int = 16,
        max_wait_ms: int = 15,
        name: str = "batcher",
    ):
        """``fn`` maps a list of items to a list of results of the same length; a result that is an ``Exception``
        instance fails only its own request."""
        self.fn, self.max_batch, self.max_wait = fn, max(1, max_batch), max_wait_ms / 1000.0
        self._q: queue.Queue = queue.Queue()
        self.batches: list[int] = []  # size of every executed batch (observability / tests)
        self._stop = False
        self._t = threading.Thread(target=self._run, name=name, daemon=True)
        self._t.start()

    def submit(self, item: Any) -> Future:
        f: Future = Future()
        self._q.put((item, f))
        return f

    def close(self) -> None:
        self._stop = True
        self._q.put(None)

    def _run(self) -> None:
        while not self._stop:
            first = self._q.get()
            if first is None:
                return
            group = [first]
            deadline = time.monotonic() + self.max_wait
            while len(group) < self.max_batch:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                try:
                    nxt = self._q.get(timeout=left)
                except queue.Empty:
                    break
                if nxt is None:
                    self._stop = True
                    break
                group.append(nxt)
            self.batches.append(len(group))
            try:
                results = self.fn([g[0] for g in group])
                if len(results) != len(group):
                    raise RuntimeError(
                        f"batch function returned {len(results)} results for {len(group)} items"
                    )
            except (
                Exception
            ) as e:  # the whole batch failed: every request gets the error, the worker lives on
                for _, f in group:
                    f.set_exception(e)
                continue
            for (_, f), r in zip(group, results):
                f.set_exception(r) if isinstance(r, Exception) else f.set_result(r)
