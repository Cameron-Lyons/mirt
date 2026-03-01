"""Runtime backend selection utilities for :mod:`mirt`."""

from __future__ import annotations

from typing import Any, Literal

from mirt._gpu_backend import (
    GPU_AVAILABLE,
    get_gpu_device_name,
    get_gpu_memory_info,
    is_torch_available,
)
from mirt._rust_backend import RUST_AVAILABLE

_CURRENT_BACKEND: Literal["auto", "gpu", "rust", "numpy"] = "auto"


def set_backend(backend: Literal["auto", "gpu", "rust", "numpy"]) -> None:
    """Set the computational backend for IRT operations."""
    global _CURRENT_BACKEND

    if backend not in ("auto", "gpu", "rust", "numpy"):
        raise ValueError(
            f"Invalid backend '{backend}'. Must be one of: 'auto', 'gpu', 'rust', 'numpy'"
        )

    if backend == "gpu" and not GPU_AVAILABLE:
        raise ValueError(
            "GPU backend requested but not available. "
            "Install PyTorch with CUDA support: pip install torch"
        )

    if backend == "rust" and not RUST_AVAILABLE:
        raise ValueError(
            "Rust backend requested but not available. "
            "Ensure the package was installed with Rust extension."
        )

    _CURRENT_BACKEND = backend


def get_backend() -> Literal["auto", "gpu", "rust", "numpy"]:
    """Get the currently configured backend."""
    return _CURRENT_BACKEND


def get_backend_info() -> dict[str, Any]:
    """Get information about available computational backends."""
    effective = _CURRENT_BACKEND
    if effective == "auto":
        if GPU_AVAILABLE:
            effective = "gpu"
        elif RUST_AVAILABLE:
            effective = "rust"
        else:
            effective = "numpy"

    return {
        "current_backend": _CURRENT_BACKEND,
        "effective_backend": effective,
        "gpu_available": GPU_AVAILABLE,
        "gpu_device": get_gpu_device_name(),
        "gpu_memory": get_gpu_memory_info(),
        "rust_available": RUST_AVAILABLE,
        "torch_available": is_torch_available(),
    }
