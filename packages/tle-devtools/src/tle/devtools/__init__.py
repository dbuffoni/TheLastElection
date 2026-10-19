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
    return ok
