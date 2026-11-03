"""Device detection: NVIDIA (CUDA), AMD (ROCm/HIP) and CPU all appear as a torch device. PyTorch's ROCm build answers to
the ``cuda`` device name, so the vendor is told apart by ``torch.version.hip``."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    backend: str  # "cuda" | "rocm" | "cpu"
    torch_device: str  # "cuda:0" | "cpu"
    name: str
    total_gb: float
    free_gb: float
    bf16: bool

    def describe(self) -> str:
        if self.backend == "cpu":
            return f"cpu ({self.name})"
        return (
            f"{self.backend} {self.torch_device}: {self.name}, {self.free_gb:.1f}/{self.total_gb:.1f} GB free"
        )


def detect(prefer: str = "auto", index: int = 0) -> DeviceInfo:
    """``prefer``: auto | cuda | rocm | cpu. ``cuda`` and ``rocm`` both need a GPU build of torch; asking for one that is
    absent raises instead of silently running a language model on the CPU."""
    import platform

    import torch

    gpu = torch.cuda.is_available()
    hip = getattr(torch.version, "hip", None)
    if prefer == "cpu" or (prefer == "auto" and not gpu):
        return DeviceInfo("cpu", "cpu", platform.processor() or platform.machine(), 0.0, 0.0, False)
    if not gpu:
        built = (
            f"ROCm {hip}"
            if hip
            else ("CUDA " + str(torch.version.cuda) if torch.version.cuda else "CPU-only")
        )
        raise RuntimeError(
            f"device '{prefer}' requested but torch ({torch.__version__}, {built} build) sees no GPU. "
            "Run `packages/tle-liquid/scripts/install.sh` to install the wheel that matches your GPU, or use --device cpu."
        )
    backend = "rocm" if hip else "cuda"
    if prefer in ("cuda", "rocm") and prefer != backend:
        raise RuntimeError(f"device '{prefer}' requested but this torch is a {backend} build")
    free, total = torch.cuda.mem_get_info(index)
    return DeviceInfo(
        backend,
        f"cuda:{index}",
        torch.cuda.get_device_name(index),
        total / 2**30,
        free / 2**30,
        bool(torch.cuda.is_bf16_supported()),
    )


def pick_dtype(dev: DeviceInfo, requested: str = "auto"):
    import torch

    if requested != "auto":
        return {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }[requested]
    if dev.backend == "cpu":
        return torch.float32  # bf16 on CPU is slow without AMX/AVX512-bf16; the models are small
    return torch.bfloat16 if dev.bf16 else torch.float16
