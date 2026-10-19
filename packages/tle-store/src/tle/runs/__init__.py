"""Read library for run directories (Polars / DuckDB). Works on open runs (Arrow streams) and closed ones."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.ipc as ipc

TABLES = ("metrics", "events", "milestones", "execution")


def runs_dir(root: str | Path | None = None) -> Path:
    return Path(root) if root else Path("runs")


def list_runs(root: str | Path | None = None) -> list[dict]:
    out = []
    for d in sorted(runs_dir(root).glob("*/status.json")):
        st = json.loads(d.read_text())
        man = json.loads((d.parent / "manifest.json").read_text())
        out.append({"run_id": d.parent.name, **{k: man.get(k) for k in ("scenario", "seed", "label")}, **st})
    return out


def table(run: str | Path, name: str, root: str | Path | None = None) -> pl.DataFrame:
    d = Path(run) if Path(run).exists() else runs_dir(root) / str(run)
    f, s = d / f"{name}.arrow", d / f"{name}.arrows"
    if f.exists():
        return pl.from_arrow(ipc.open_file(f).read_all())
    if s.exists():
        reader = ipc.open_stream(s)
        batches = []
        try:
            for b in reader:
                batches.append(b)
        except pa.ArrowInvalid:  # a half-written trailing batch of a crashed run
            pass
        return pl.from_arrow(pa.Table.from_batches(batches)) if batches else pl.DataFrame()
    return pl.DataFrame()


def manifest(run: str | Path, root: str | Path | None = None) -> dict:
    d = Path(run) if Path(run).exists() else runs_dir(root) / str(run)
    return json.loads((d / "manifest.json").read_text())


def params(run: str | Path, root: str | Path | None = None) -> dict:
    d = Path(run) if Path(run).exists() else runs_dir(root) / str(run)
    return json.loads((d / "params.json").read_text())


def sql(query: str, root: str | Path | None = None):
    """DuckDB over all closed runs: tables are exposed as ``metrics``, ``events``, ``milestones`` with a ``run_id`` column."""
    import duckdb

    con = duckdb.connect()
    for t in TABLES:
        frames = []
        for d in runs_dir(root).iterdir() if runs_dir(root).exists() else []:
            df = table(d, t)
            if len(df):
                frames.append(df.with_columns(pl.lit(d.name).alias("run_id")))
        if frames:
            con.register(t, pl.concat(frames, how="diagonal").to_arrow())
    return con.execute(query).pl()
