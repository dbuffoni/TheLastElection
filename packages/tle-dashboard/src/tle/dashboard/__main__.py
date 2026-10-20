"""``python -m tle.dashboard [--runs-dir runs] [--port 5006] [--live]`` (launched by ``tle dash``)."""

from __future__ import annotations

import argparse

import panel as pn

from .app import build_app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--port", type=int, default=5006)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()
    pn.serve(
        lambda: build_app(a.runs_dir, a.live), port=a.port, show=not a.no_show, title="The Last Election"
    )


if __name__ == "__main__":
    main()
