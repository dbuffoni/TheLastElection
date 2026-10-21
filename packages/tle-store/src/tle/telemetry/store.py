"""Run store (spec: Technical Implementation, telemetry). One directory per run:

    manifest.json   run id, seed, scenario, git/params digests, versions, start/end
    params.json     the resolved flat parameters (with overrides)
    status.json     progress, outcome, heartbeat (rewritten each flush, readable while the run is open)
    metrics.arrow   one row per turn: day, capital, every indicator, pillar levels, decree outcome
    events.arrow    one row per published news item
    milestones.arrow  pillar advances/falls, threat phases, outcome
    journal.arrow   the diary of the city: decrees, messages, news, pillar moves, threats, mood changes (readable text)
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

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from .journal import journal_entries

TABLES = ("metrics", "events", "milestones", "execution", "citizens", "bonds", "journal")
SCHEMA_VERSION = "1"
TYPE_HINTS = {
    "decree_pillar": pa.string(),
    "decree_aggressiveness": pa.string(),
    "decree_result": pa.string(),
    "decree_cost": pa.float64(),
}


class _Table:
    def __init__(self, path: Path):
        self.path = path
        self.rows: list[dict[str, Any]] = []
        self.sink = None
        self.writer = None
        self.schema: pa.Schema | None = None
        self.dropped: set[str] = set()

    def add(self, row: dict[str, Any]) -> None:
        self.rows.append(row)

    def add_table(self, table: pa.Table) -> None:
        """Write a ready Arrow table as one record batch (panels, snapshots)."""
        self.flush()
        self._write(table)

    def _write(self, batch: pa.Table) -> None:
        if self.writer is None:
            self.schema = batch.schema.with_metadata({"tle.schema_version": SCHEMA_VERSION})
            self.sink = pa.OSFile(str(self.path.with_suffix(".arrows")), "wb")
            self.writer = ipc.new_stream(self.sink, self.schema)
            batch = batch.replace_schema_metadata(self.schema.metadata)
        else:  # the first schema is kept; a new column cannot join an open stream, so it is reported, not hidden
            self.dropped |= set(batch.column_names) - set(self.schema.names)
            cols = {
                n: batch[n] if n in batch.column_names else pa.nulls(len(batch), self.schema.field(n).type)
                for n in self.schema.names
            }
            batch = pa.table(cols, schema=self.schema)
        self.writer.write_table(batch)
        self.sink.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        batch = pa.Table.from_pylist(self.rows)
        self.rows = []
        for name, typ in TYPE_HINTS.items():  # an all-None first batch must not freeze a column as null-typed
            if name in batch.column_names and pa.types.is_null(batch.schema.field(name).type):
                batch = batch.set_column(batch.column_names.index(name), name, batch[name].cast(typ))
        self._write(batch)

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
        self.checks: dict[str, float] = {}  # invariant name -> first day of violation
        (self.dir / "state").mkdir(exist_ok=True)
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
        journal: list[dict] | None = None,
    ) -> None:
        self.tables["metrics"].add(row)
        for j in journal or ():
            self.tables["journal"].add(j)
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

    def save_state(self, name: str, data: bytes) -> None:
        (self.dir / "state" / f"{name}.npz").write_bytes(data)

    def note_violations(self, names: list[str], day: float) -> None:
        for n in names:
            self.checks.setdefault(n, day)

    def flush(self) -> None:
        for t in self.tables.values():
            t.flush()
        (self.dir / "checks.json").write_text(
            json.dumps({"first_violation_day": self.checks, "ok": not self.checks})
        )
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
    """Game observer feeding a ``RunStore``; pass to ``play(..., observers=[obs])`` (and to ``Game(observers=...)`` to
    also record the warm-up, with negative days and ``phase = "warmup"``).

    Per turn: indicators and every population metric of the registry (``metrics.arrow``), events, milestones, execution
    accounting, invariant checks. Every ``panel_every`` days (plus day 0 and the last day): the citizen panel and the
    graph snapshot. It only reads the world: it never draws from the game RNG."""

    def __init__(self, store: RunStore, run_meta: dict | None = None, panel_every: float | None = None):
        from tle.systems import popstats

        self.store = store
        self.popstats = popstats
        self.prev_levels: dict[str, int] = {}
        self.prev_labels: dict[str, str] = {}
        self.meta = run_meta or {}
        self.panel_every = panel_every
        self.next_panel = 0.0
        self.ref = None
        self.started = False
        self.last_panel_day = -1.0

    # -- lifecycle ---------------------------------------------------------------------------------
    def on_start(self, game) -> None:
        """Called by ``play`` with the world at day 0 (after the warm-up): reference for mobility, first panel."""
        if self.started:
            return
        self.started = True
        if self.panel_every is None:
            self.panel_every = float(game.p.technical.panel_every)
        self.ref = self.popstats.PopRef.capture(game.w)
        self.store.save_state("day0", game.w.to_bytes())
        self._panel(game.w, game.p)

    def _panel(self, w, p) -> None:
        self.store.tables["citizens"].add_table(citizen_panel(w, p))
        self.store.tables["bonds"].add_table(bond_snapshot(w))
        self.last_panel_day = w.day
        self.next_panel = (w.day // self.panel_every + 1) * self.panel_every

    # -- per turn ----------------------------------------------------------------------------------
    def __call__(self, game, pkg) -> None:
        if not self.started and pkg.phase != "warmup":
            self.on_start(game)
        w = game.w
        warm = pkg.phase == "warmup"
        pop = self.popstats.compute(w, game.p, None if warm else self.ref)
        row = {
            "turn": pkg.turn,
            "day": pkg.day,
            "dt": pkg.dt,
            "phase": pkg.phase,
            "capital": pkg.capital,
            "outcome": pkg.outcome,
            "digest": w.digest()[:16],
            **{k: float(v) for k, v in pkg.indicators.items()},
            **{k: float(v) for k, v in pop.items()},
            "decree_pillar": None,
            "decree_aggressiveness": None,
            "decree_result": None,
            "decree_cost": None,
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
                "phase": pkg.phase,
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
        journal = []
        if not warm:  # read before prev_levels is updated below
            journal = journal_entries(pkg, self.prev_labels, self.prev_levels)
            self.prev_labels = dict(pkg.labels)
        ms = []
        if not warm:
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
        self.store.note_violations(self.popstats.check_invariants(w), pkg.day)
        self.store.log_turn(row, events, ms, ex, pkg.warnings, journal)
        if not warm and self.started and (w.day >= self.next_panel or pkg.outcome):
            if pkg.outcome and w.day == self.last_panel_day:
                pass
            else:
                self._panel(w, game.p)
        if pkg.outcome:
            self.store.save_state("final", w.to_bytes())
            self.store.close(pkg.outcome, {"digest": w.digest()})


def citizen_panel(w, p) -> pa.Table:
    """One row per living citizen at the current day (spec 5.1 ``citizens.arrow``)."""
    from tle.core.prims import express

    al = np.flatnonzero(w.alive[: w.n_used])
    E = express(w.O[al], w.fear[al])
    cols: dict[str, Any] = {
        "day": np.full(len(al), w.day),
        "id": al.astype(np.int32),
        "arrival_day": w.arrival_day[al],
        "age": w.age[al],
        "cls": w.cls[al].astype(np.int8),
        "trade": w.trade[al].astype(np.int16),
        "circle": w.circle[al].astype(np.int16),
        "wealth": w.r[al],
        "fear": w.fear[al],
        "councillor": w.councillor[al],
        "couple": w.couple[al],
        "faction": np.where(w.fac_member[al].any(axis=1), w.fac_m[al].argmax(axis=1), -1).astype(np.int16),
        "influencer": w.inf_on[al],
        "coalition": w.coalition[al],
    }
    for j in range(4):
        cols[f"T{j + 1}"] = w.T[al, j]
    for j in range(3):
        cols[f"C{j + 1}"] = w.C[al, j]
        cols[f"I{j + 1}"] = w.I[al, j]
        cols[f"B{j + 1}"] = w.B[al, j]
    for j, nm in enumerate(("mayor", "p1", "p2", "p3", "p4", "p5")):
        cols[f"op_{nm}"] = w.O[al, j]
        cols[f"ex_{nm}"] = E[:, j]
    return pa.table(cols)


def bond_snapshot(w) -> pa.Table:
    n = w.bn
    return pa.table(
        {
            "day": np.full(n, w.day),
            "source": w.bsrc[:n],
            "target": w.bdst[:n],
            "type": w.btype[:n],
            "intensity": w.bw[:n],
            "stable": w.bstable[:n],
        }
    )
