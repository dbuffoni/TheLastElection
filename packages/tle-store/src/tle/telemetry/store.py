"""Run store (spec: Technical Implementation, telemetry). One directory per run:

    manifest.json   run id, seed, scenario, git/params digests, versions, start/end
    params.json     the resolved flat parameters (with overrides)
    status.json     progress, outcome, heartbeat (rewritten each flush, readable while the run is open)
    metrics.arrow   one row per turn: day, capital, every indicator, pillar levels, decree outcome
    events.arrow    one row per published news item
    milestones.arrow  pillar advances/falls, threat phases, outcome
    execution.arrow  per-task engine accounting (calls, attempts, fallbacks, tokens)
    warnings.jsonl  malformed answers and fallbacks

Tables are written as Arrow IPC *streams* while open (crash safe: readable up to the last flushed batch) and
compacted to Arrow IPC *files* at close. Telemetry is a pure observer: it never touches the game RNG, so a
run with telemetry off is bit-identical (tested)."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc

TABLES = ("metrics", "events", "milestones", "execution")


class _Table:
    def __init__(self, path: Path):
        self.path = path
        self.rows: list[dict[str, Any]] = []
        self.sink = None
        self.writer = None
        self.schema: pa.Schema | None = None

    def add(self, row: dict[str, Any]) -> None:
        self.rows.append(row)

    def flush(self) -> None:
        if not self.rows:
            return
        batch = pa.Table.from_pylist(self.rows)
        if self.writer is None:
            self.schema = batch.schema
            self.sink = pa.OSFile(str(self.path.with_suffix(".arrows")), "wb")
            self.writer = ipc.new_stream(self.sink, self.schema)
        else:  # keep the first schema; unseen columns are dropped, missing ones become null
            cols = {
                n: batch[n] if n in batch.column_names else pa.nulls(len(batch), self.schema.field(n).type)
                for n in self.schema.names
            }
            batch = pa.table(cols, schema=self.schema)
        self.writer.write_table(batch)
        self.sink.flush()
        self.rows = []

    def close(self) -> None:
        self.flush()
        if self.writer is None:
            return
        self.writer.close()
        self.sink.close()
        stream = self.path.with_suffix(".arrows")
        table = ipc.open_stream(stream).read_all()
        with ipc.new_file(str(self.path.with_suffix(".arrow")), table.schema) as w:
            w.write_table(table)
        stream.unlink()


class RunStore:
    def __init__(
        self, root: Path, run_id: str, manifest: dict[str, Any], params: dict[str, Any], flush_every: int = 25
    ):
        self.dir = Path(root) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self.t0 = time.time()
        self.tables = {t: _Table(self.dir / t) for t in TABLES}
        self.warnings_path = self.dir / "warnings.jsonl"
        self.manifest = {
            "run_id": run_id,
            "started": self.t0,
            "python": platform.python_version(),
            **manifest,
        }
        (self.dir / "manifest.json").write_text(json.dumps(self.manifest, indent=2, default=str))
        (self.dir / "params.json").write_text(json.dumps(params, indent=2, default=str))
        self.status = {"state": "running", "turn": 0, "day": 0.0, "outcome": ""}
        self._write_status()
        self.closed = False

    def _write_status(self) -> None:
        self.status["heartbeat"] = time.time()
        (self.dir / "status.json").write_text(json.dumps(self.status))

    def log_turn(
        self,
        row: dict,
        events: list[dict],
        milestones: list[dict],
        execution: list[dict],
        warnings: list[dict],
    ) -> None:
        self.tables["metrics"].add(row)
        for e in events:
            self.tables["events"].add(e)
        for ms in milestones:
            self.tables["milestones"].add(ms)
        for x in execution:
            self.tables["execution"].add(x)
        if warnings:
            with self.warnings_path.open("a") as f:
                for wn in warnings:
                    f.write(json.dumps(wn) + "\n")
        self.status.update(turn=row["turn"], day=row["day"], outcome=row.get("outcome", ""))
        if row["turn"] % self.flush_every == 0:
            self.flush()

    def flush(self) -> None:
        for t in self.tables.values():
            t.flush()
        self._write_status()

    def close(self, outcome: str = "", extra: dict | None = None) -> None:
        if self.closed:
            return
        for t in self.tables.values():
            t.close()
        self.status.update(
            state="finished",
            outcome=outcome or self.status["outcome"],
            seconds=time.time() - self.t0,
            **(extra or {}),
        )
        self._write_status()
        self.closed = True


class TelemetryObserver:
    """Game observer feeding a ``RunStore``; pass to ``play(..., observers=[obs])``."""

    def __init__(self, store: RunStore, run_meta: dict | None = None):
        self.store = store
        self.prev_levels: dict[str, int] = {}
        self.meta = run_meta or {}

    def __call__(self, game, pkg) -> None:
        w = game.w
        row = {
            "turn": pkg.turn,
            "day": pkg.day,
            "dt": pkg.dt,
            "capital": pkg.capital,
            "outcome": pkg.outcome,
            **{k: float(v) for k, v in pkg.indicators.items()},
        }
        if pkg.decree:
            row.update(
                decree_pillar=pkg.decree["pillar"],
                decree_aggressiveness=pkg.decree["aggressiveness"],
                decree_result=pkg.decree.get("result", ""),
                decree_cost=pkg.decree["cost"],
            )
        elif pkg.rejected:
            row.update(decree_result="rejected")
        events = [
            {
                "turn": pkg.turn,
                "day": pkg.day,
                "event_id": n.event_id,
                "type": n.event_type,
                "title": n.title,
                "intensity": n.intensity,
                "actor": n.actor_kind,
                "involved": n.involved,
                "tags": ",".join(n.tags),
            }
            for n in pkg.news
        ]
        ms = []
        for pid, lvl in pkg.pillars.items():
            if lvl != self.prev_levels.get(pid, 0):
                ms.append(
                    {
                        "turn": pkg.turn,
                        "day": pkg.day,
                        "kind": "pillar_level",
                        "what": pid,
                        "value": float(lvl),
                    }
                )
            self.prev_levels[pid] = lvl
        for t in pkg.threats:
            ms.append(
                {
                    "turn": pkg.turn,
                    "day": pkg.day,
                    "kind": "threat",
                    "what": t["event"],
                    "value": float(t["phase"]),
                }
            )
        if pkg.outcome:
            ms.append(
                {"turn": pkg.turn, "day": pkg.day, "kind": "outcome", "what": pkg.outcome, "value": 1.0}
            )
        ex = [
            {
                "turn": pkg.turn,
                "task": r.task,
                "calls": r.calls,
                "attempts": r.attempts,
                "fallbacks": r.fallbacks,
                "in_tokens": r.in_tokens,
                "out_tokens": r.out_tokens,
                "latency_s": r.latency_s,
                "backend": r.backend,
            }
            for r in game.engines.drain_execution()
        ]
        self.store.log_turn(row, events, ms, ex, pkg.warnings)
        if pkg.outcome:
            self.store.close(pkg.outcome, {"digest": w.digest()})
