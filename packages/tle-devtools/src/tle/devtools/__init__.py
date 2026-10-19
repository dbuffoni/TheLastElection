"""Project health checks (also run by the Claude Code hooks and CI)."""

from __future__ import annotations

import ast
from pathlib import Path

from tle.registry.loader import repo_root

# Deterministic-core packages must not read global randomness or the clock (spec: no global random).
PURE_DIRS = ("core", "systems", "loop", "world")
BANNED_CALLS = {("random", None), ("np", "random"), ("numpy", "random")}
BANNED_IMPORTS = {"random"}
BANNED_ATTR_CALLS = {"time.time", "time.perf_counter", "datetime.now", "datetime.utcnow", "os.urandom"}


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def boundary_violations(root: Path | None = None) -> list[str]:
    """AST scan of the deterministic core: global RNG, wall clock, or unseeded sources."""
    root = root or repo_root()
    base = root / "packages" / "tle-engine" / "src" / "tle"
    out = []
    for d in PURE_DIRS:
        for f in sorted((base / d).rglob("*.py")):
            if f.name == "rng.py":  # the one sanctioned place
                continue
            tree = ast.parse(f.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import) and any(a.name in BANNED_IMPORTS for a in node.names):
                    out.append(f"{f.relative_to(root)}:{node.lineno}: imports global 'random'")
                elif isinstance(node, ast.Call):
                    name = _dotted(node.func)
                    if name in BANNED_ATTR_CALLS or name.startswith(
                        ("np.random.", "numpy.random.", "random.")
                    ):
                        out.append(f"{f.relative_to(root)}:{node.lineno}: forbidden call {name}()")
    return out


def metrics_lint_errors() -> list[str]:
    """Registry <-> telemetry consistency (spec 6): run a 3-turn game with telemetry and compare the written columns."""
    import tempfile

    from tle.registry.metrics import load_metrics
    from tle.systems import popstats

    specs = load_metrics()
    errs = []
    for mid, s in specs.items():
        t = s.target
        if t is None:
            continue
        if t.kind == "band" and (t.lo is None or t.hi is None or t.lo >= t.hi):
            errs.append(f"{mid}: band target needs lo < hi")
        if t.kind in ("min", "max", "drift_max") and t.value is None:
            errs.append(f"{mid}: {t.kind} target needs a value")
    from tle.catalogs.loader import load_catalogs
    from tle.loop.game import Game, play
    from tle.params import default_params, to_flat
    from tle.runs import open_run
    from tle.telemetry import RunStore, TelemetryObserver

    with tempfile.TemporaryDirectory() as d:
        p = default_params().with_overrides({"T_warm": 14})
        store = RunStore(d, "lint", {"scenario": "industrial_secular", "seed": 1}, to_flat(p))
        ob = TelemetryObserver(store)
        g = Game(load_catalogs(), p, seed=1, n=60, observers=[ob])
        play(g, None, 3, [ob])
        store.close()
        cols = set(open_run(Path(d) / "lint").metrics(phase=None).df.columns)
        computed = set(popstats.compute(g.w, p, popstats.PopRef.capture(g.w)))
    for mid in computed - set(specs):
        errs.append(f"{mid}: computed by popstats but not registered in config/metrics")
    for mid, s in specs.items():
        if s.source == "turn" and mid not in cols:
            errs.append(f"{mid}: registered with source=turn but never written to metrics.arrow")
        if s.target and mid not in computed and s.source == "turn":
            errs.append(f"{mid}: has a target but is not computed")
    return errs


def run_checks(console) -> bool:
    ok = True

    def report(name: str, errs: list[str]) -> None:
        nonlocal ok
        if errs:
            ok = False
            console.print(f"[red]FAIL[/] {name}")
            for e in errs[:20]:
                console.print(f"   {e}")
        else:
            console.print(f"[green]ok[/]   {name}")

    from tle.registry import codegen

    report("generated parameters are current", [] if codegen.is_current() else ["run `tle params codegen`"])
    from tle.catalogs.loader import load_catalogs

    cat = None
    try:
        cat = load_catalogs()
        report("catalogs load and validate", [])
    except Exception as e:  # noqa: BLE001
        report("catalogs load and validate", [str(e)])
    if cat:
        from tle.contracts.schema import check_strict_subset, export_all

        errs = []
        for name, sch in export_all(cat.vocab()).items():
            errs += [f"{name}: {e}" for e in check_strict_subset(sch)]
        report("contract schemas are in the strict subset", errs)
    report("no global RNG / wall clock in the deterministic core", boundary_violations())
    report("metric registry matches telemetry", metrics_lint_errors())
    return ok
