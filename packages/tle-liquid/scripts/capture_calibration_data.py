"""Regenerates packages/tle-liquid/data/*.jsonl: real S1/S2 requests of the game, used by `tle-liquid calibrate`.
Starts a mock server (no weights), plays a short game against it, keeps an even sample of each kind of request.
    uv run python packages/tle-liquid/scripts/capture_calibration_data.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "packages/tle-liquid/data"
KEEP_S1 = {"option": 40, "activity": 40, "classify": 20, "extract": 10}
KEEP_S2 = 8
PORT = 8199

SERVER = """
import json, sys
from tle.liquid import mock
A, B = sys.argv[1], sys.argv[2]
O1, O2 = mock.MockDecisionRuntime.run_batch, mock.MockChatRuntime.run_batch
def w(path, reqs):
    with open(path, "a") as f:
        for r in reqs: f.write(json.dumps(r) + "\\n")
mock.MockDecisionRuntime.run_batch = lambda self, reqs: (w(A, reqs), O1(self, reqs))[1]
mock.MockChatRuntime.run_batch = lambda self, reqs: (w(B, reqs), O2(self, reqs))[1]
from tle.liquid.cli import main
sys.argv = ["x", "serve", "--mode", "mock", "--port", sys.argv[3]]
main()
"""


def kind(r: dict) -> str:
    names = {k.split(":", 1)[-1] for k in r["questions"]}
    if "option" in names:
        return "option"
    if "activity" in names:
        return "activity"
    return "classify" if len(names) == 11 else "extract"


def spread(rows: list, n: int) -> list:
    if len(rows) <= n:
        return rows
    return [rows[int(i * len(rows) / n)] for i in range(n)]


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "s1.jsonl", Path(d) / "s2.jsonl"
        srv = subprocess.Popen([sys.executable, "-I", "-c", SERVER, str(a), str(b), str(PORT)])
        try:
            for _ in range(60):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                    break
                except Exception:
                    time.sleep(1)
            env = {"TLE_LIQUID_URL": f"http://127.0.0.1:{PORT}"}
            import os

            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from tle.cli.main import app; app()",
                    "run",
                    "--engines",
                    "config/engines.liquid.toml",
                    "--n",
                    "200",
                    "--max-turns",
                    "8",
                    "--no-telemetry",
                    "--runs-dir",
                    d,
                    "--strategy",
                    "random",
                ],
                env={**os.environ, **env},
                cwd=ROOT,
                check=True,
                capture_output=True,
            )
        finally:
            srv.terminate()
        s1 = [json.loads(x) for x in a.read_text().splitlines()]
        s2 = [json.loads(x) for x in b.read_text().splitlines()]
    OUT.mkdir(parents=True, exist_ok=True)
    kept = []
    for k, n in KEEP_S1.items():
        kept += spread([r for r in s1 if kind(r) == k], n)
    (OUT / "s1_requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
    (OUT / "s2_requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in spread(s2, KEEP_S2)))
    print(f"s1: {len(kept)} requests, s2: {min(len(s2), KEEP_S2)} requests -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
