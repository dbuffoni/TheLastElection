#!/usr/bin/env bash
# The three steps in one command, for a machine that will serve the models (a clone of this repo):
#   1. configuration: install.sh picks the torch wheel for the hardware and downloads the weights (skipped when ready)
#   2. calibration:   `tle-liquid calibrate` finds the best G and R for this machine and saves them (skipped when saved;
#                     --recalibrate forces it, --quick uses a small grid)
#   3. activation:    `tle-liquid up --share` starts the models and prints ONE link for the machine that plays
#   packages/tle-liquid/scripts/remote.sh [--recalibrate] [--quick] [--roles s1 s2] [--port 8100]
# See packages/tle-liquid/REMOTE.md.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PATH="$HOME/.local/bin:$PATH"
recal=0; quick=()
args=()
for a in "$@"; do
  case "$a" in
    --recalibrate) recal=1 ;;
    --quick) quick=(--quick) ;;
    *) args+=("$a") ;;
  esac
done
if ! uv run --no-sync python -c "import torch, transformers, starlette, httpx" >/dev/null 2>&1; then
  packages/tle-liquid/scripts/install.sh auto --fetch
fi
if [[ $recal -eq 1 || ! -f .tle-liquid/calibration.json ]]; then
  uv run tle-liquid calibrate "${quick[@]}"
fi
exec uv run tle-liquid up --share "${args[@]}"
