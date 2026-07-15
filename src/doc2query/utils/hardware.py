"""CPU/GPU capability reporting with no accelerator requirement."""

import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch


def collect_hardware_report() -> dict[str, Any]:
    """Return a JSON-serializable snapshot of relevant runtime capabilities."""
    cuda_available = bool(torch.cuda.is_available())
    devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_vram_bytes": properties.total_memory,
                }
            )
    bf16_supported = bool(cuda_available and torch.cuda.is_bf16_supported())
    return {
        "status": "ok",
        "cpu_only_supported": True,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
            if cuda_available
            else None
        ),
        "bf16_supported": bf16_supported,
        "gpu_count": len(devices),
        "gpus": devices,
        "message": (
            "CUDA GPU detected; verify memory before training."
            if cuda_available
            else "No CUDA GPU detected; CPU mode is supported."
        ),
    }


def write_hardware_report(path: Path, report: dict[str, Any] | None = None) -> Path:
    """Atomically write a hardware report as UTF-8 JSON."""
    payload = collect_hardware_report() if report is None else report
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path
