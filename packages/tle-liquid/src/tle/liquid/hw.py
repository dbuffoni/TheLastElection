"""What this machine has, as JSON. Run in a subprocess by ``tle-liquid up`` so that the launcher itself never holds a GPU
context (and so that ``HSA_OVERRIDE_GFX_VERSION`` can be set before torch starts). ``python -m tle.liquid.hw``."""

from __future__ import annotations

import json
import os

# AMD APUs whose gfx target PyTorch's ROCm wheels do not ship kernels for: the nearest supported target is used instead.
# (Only applied when the user has not set the variable.) "Graphics" in the name = integrated GPU.
HSA_OVERRIDES = {"780M": "11.0.0", "760M": "11.0.0", "740M": "11.0.0"}


def hsa_override_for(name: str) -> str | None:
    for tag, ver in HSA_OVERRIDES.items():
        if tag in name:
            return ver
    return None


def integrated(backend: str, name: str) -> bool:
    """AMD integrated GPUs are named '... Graphics'; discrete cards and NVIDIA are treated as dedicated."""
    return backend == "rocm" and name.rstrip().endswith("Graphics")


def probe() -> dict:
    info = {"backend": "cpu", "name": "", "total_gb": 0.0, "cores": os.cpu_count() or 1, "integrated": False}
    try:
        import torch

        if torch.cuda.is_available():
            hip = getattr(torch.version, "hip", None)
            info["backend"] = "rocm" if hip else "cuda"
            info["name"] = torch.cuda.get_device_name(0)
            info["total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
            info["integrated"] = integrated(info["backend"], info["name"])
    except Exception as e:  # no torch, broken driver...: say so and fall back to the CPU
        info["error"] = f"{type(e).__name__}: {e}"
    return info


if __name__ == "__main__":
    print(json.dumps(probe()))
