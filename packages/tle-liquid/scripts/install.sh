#!/usr/bin/env bash
# Installs the Liquid engine on this machine: the right PyTorch wheel for the GPU (NVIDIA CUDA, AMD ROCm or CPU),
# the serving libraries, and (optionally) downloads the models.
#   packages/tle-liquid/scripts/install.sh [auto|cuda|rocm|cpu] [--fetch]
# Override the wheel index of uv with TORCH_BACKEND (e.g. cu128, rocm6.4, cpu) if auto-detection picks the wrong one.
set -euo pipefail
cd "$(dirname "$0")/../../.."

kind="${1:-auto}"
fetch=0
[[ "${1:-}" == "--fetch" || "${2:-}" == "--fetch" ]] && fetch=1
[[ "$kind" == "--fetch" ]] && kind=auto

if [[ "$kind" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then kind=cuda
  elif [[ -e /dev/kfd ]] || command -v rocminfo >/dev/null 2>&1; then kind=rocm
  else kind=cpu; fi
  echo "detected: $kind"
fi
case "$kind" in
  cuda) backend="${TORCH_BACKEND:-cu128}" ;;
  rocm) backend="${TORCH_BACKEND:-rocm6.4}" ;;
  cpu)  backend="${TORCH_BACKEND:-cpu}" ;;
  *) echo "usage: $0 [auto|cuda|rocm|cpu] [--fetch]"; exit 2 ;;
esac

command -v uv >/dev/null 2>&1 || { echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
uv sync --inexact                                          # the game itself
uv pip install --torch-backend "$backend" torch            # GPU-matched wheel
uv pip install -e "packages/tle-liquid[serve]"             # transformers, xgrammar, starlette, uvicorn, huggingface_hub
uv run tle-liquid doctor || true
if [[ $fetch -eq 1 ]]; then uv run tle-liquid fetch; fi
echo "done. Start the engine with: packages/tle-liquid/scripts/serve.sh"
