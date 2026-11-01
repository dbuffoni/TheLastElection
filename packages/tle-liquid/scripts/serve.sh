#!/usr/bin/env bash
# Starts the Liquid engine. Local use:   serve.sh
# Remote desktop (other machines reach it):   TLE_LIQUID_API_KEY=<secret> serve.sh --host 0.0.0.0
# Any `tle-liquid serve` flag works (--device rocm, --roles s2, --lazy, --mode mock ...).
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec uv run tle-liquid serve "$@"
