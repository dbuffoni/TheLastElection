"""Read library for run directories (Polars / DuckDB). Works on open runs (Arrow streams) and closed ones."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.ipc as ipc

TABLES = ("metrics", "events", "milestones", "execution", "journal")


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


# ---------------------------------------------------------------------------------------------------------
# Object API (spec 9.1): store.open(run) -> Run with metrics / panel / bonds / events / milestones / checks
class Run:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.id = self.path.name

    # -- documents
    @property
    def manifest(self) -> dict:
        return json.loads((self.path / "manifest.json").read_text())

    @property
    def params(self) -> dict:
        return json.loads((self.path / "params.json").read_text())

    @property
    def status(self) -> dict:
        return json.loads((self.path / "status.json").read_text())

    def checks(self) -> dict:
        f = self.path / "checks.json"
        return json.loads(f.read_text()) if f.exists() else {"ok": True, "first_violation_day": {}}

    # -- tables
    def metrics(self, cols: list[str] | None = None, phase: str | None = "game") -> Metrics:
        df = table(self.path, "metrics")
        if phase and "phase" in df.columns:
            df = df.filter(pl.col("phase") == phase)
        keep = [
            "turn",
            "day",
            "dt",
            *[c for c in (cols or df.columns) if c in df.columns and c not in ("turn", "day", "dt")],
        ]
        return Metrics(df.select(list(dict.fromkeys(keep))))

    def warmup(self, cols: list[str] | None = None) -> Metrics:
        return self.metrics(cols, phase="warmup")

    def panel_days(self) -> list[float]:
        df = table(self.path, "citizens")
        return sorted(df["day"].unique().to_list()) if len(df) else []

    def panel(self, day: float | None = None) -> pl.DataFrame:
        """Citizen panel at the snapshot closest to ``day`` (default: the last one)."""
        df = table(self.path, "citizens")
        if not len(df):
            return df
        days = df["day"].unique().sort().to_list()
        d = days[-1] if day is None else min(days, key=lambda x: abs(x - day))
        return df.filter(pl.col("day") == d)

    def bonds(self, day: float | None = None) -> pl.DataFrame:
        df = table(self.path, "bonds")
        if not len(df):
            return df
        days = df["day"].unique().sort().to_list()
        d = days[-1] if day is None else min(days, key=lambda x: abs(x - day))
        return df.filter(pl.col("day") == d)

    def events(self, phase: str | None = "game", **filters) -> pl.DataFrame:
        df = table(self.path, "events")
        if phase and "phase" in df.columns:
            df = df.filter(pl.col("phase") == phase)
        for k, v in filters.items():
            df = df.filter(pl.col(k) == v)
        return df

    def journal(self, min_importance: int = 0, kinds: list[str] | None = None) -> pl.DataFrame:
        """The diary of the city (one row per entry), optionally filtered by importance and kind."""
        df = table(self.path, "journal")
        if not len(df):
            return df
        df = df.filter(pl.col("importance") >= min_importance)
        if kinds:
            df = df.filter(pl.col("kind").is_in(kinds))
        return df

    def milestones(self) -> pl.DataFrame:
        return table(self.path, "milestones")

    def execution(self) -> pl.DataFrame:
        return table(self.path, "execution")

    def state(self, name: str = "final"):
        """The serialized world saved at day 0 or at the end (fork or inspect it with ``World.from_bytes``)."""
        from tle.world.state import World

        return World.from_bytes((self.path / "state" / f"{name}.npz").read_bytes())


class Metrics:
    """Per-turn metric table with the daily view of spec 5.2 (forward-filled to one row per day)."""

    def __init__(self, df: pl.DataFrame):
        self.df = df

    def daily(self) -> pl.DataFrame:
        if not len(self.df):
            return self.df
        d0, d1 = int(self.df["day"].min()), int(self.df["day"].max())
        days = pl.DataFrame({"day": pl.Series(range(d0, d1 + 1), dtype=pl.Float64)})
        return days.join_asof(
            self.df.sort("day").with_columns(pl.col("day").cast(pl.Float64)), on="day", strategy="backward"
        )

    def __getattr__(self, item):
        return getattr(self.df, item)

    def __len__(self) -> int:
        return len(self.df)


def open_run(run: str | Path, root: str | Path | None = None) -> Run:
    p = Path(run) if Path(run).exists() else runs_dir(root) / str(run)
    if not p.exists():
        raise FileNotFoundError(f"no run '{run}' under {runs_dir(root)}")
    return Run(p)
