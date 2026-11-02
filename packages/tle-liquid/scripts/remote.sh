#!/usr/bin/env bash
# Run this on the OTHER machine (a clone of this repo): installs what is missing, detects the hardware, starts the Liquid
# models and prints one link for the machine that plays.   packages/tle-liquid/scripts/remote.sh [--roles s1 s2] [--port 8100]
# Everything is passed to `tle-liquid up --share`. See packages/tle-liquid/REMOTE.md.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PATH="$HOME/.local/bin:$PATH"
if ! uv run --no-sync python -c "import torch, transformers, starlette, httpx" >/dev/null 2>&1; then
  packages/tle-liquid/scripts/install.sh auto --fetch
fi
exec uv run tle-liquid up --share "$@"
