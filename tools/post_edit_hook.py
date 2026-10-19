#!/usr/bin/env python3
"""PostToolUse hook: after editing the parameter registry, regenerate the typed Params so code and TOML never drift."""

import json
import subprocess
import sys

try:
    payload = json.load(sys.stdin)
    path = (payload.get("tool_input") or {}).get("file_path", "")
except Exception:
    sys.exit(0)
if "/config/params/" in path and path.endswith(".toml"):
    r = subprocess.run(["uv", "run", "tle", "params", "codegen"], capture_output=True, text=True)
    print("regenerated tle.params._generated" if r.returncode == 0 else f"codegen failed:\n{r.stderr[-500:]}")
