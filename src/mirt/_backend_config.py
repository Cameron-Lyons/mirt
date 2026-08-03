"""Runtime backend selection utilities for :mod:`mirt`."""

from __future__ import annotations

from typing import Any, Literal

from mirt._backend_state import (
    BackendName,
    get_backend_preference,
    set_backend_preference,
)
from mirt._gpu_backend import (
    get_gpu_device_name,
    get_gpu_memory_info,
    is_gpu_available,
    is_torch_available,
)
from mirt._rust_backend import RUST_AVAILABLE

BackendChoice = Literal["auto", "gpu", "rust", "numpy"]


def set_backend(backend: BackendChoice) -> None:
    """Set the computational backend for IRT operations.

    Parameters
    ----------
    backend : {"auto", "gpu", "rust", "numpy"}
        Preferred backend. ``"numpy"`` disables Rust acceleration globally.
        ``"auto"`` prefers GPU when available, otherwise Rust, otherwise NumPy.
    """
    if backend not in ("auto", "gpu", "rust", "numpy"):
        raise ValueError(
            f"Invalid backend '{backend}'. Must be one of: 'auto', 'gpu', 'rust', 'numpy'"
        )

    if backend == "gpu" and not is_gpu_available():
        raise ValueError(
            "GPU backend requested but not available. "
            "Install PyTorch with CUDA support: pip install torch"
        )

    if backend == "rust" and not RUST_AVAILABLE:
        raise ValueError(
            "Rust backend requested but not available. "
            "Ensure the package was installed with Rust extension."
        )

    set_backend_preference(backend)


def get_backend() -> BackendChoice:
    """Get the currently configured backend."""
    return get_backend_preference()


def should_use_rust(use_rust: bool = True) -> bool:
    """Whether a hot path should use Rust acceleration.

    Honors both the per-call ``use_rust`` flag and the global backend
    preference from :func:`set_backend`.

    Parameters
    ----------
    use_rust : bool, default=True
        Per-call preference. ``False`` always disables Rust for that call.

    Returns
    -------
    bool
        ``True`` when Rust should be used for the call.
    """
    if not use_rust:
        return False
    if get_backend_preference() == "numpy":
        return False
    return bool(RUST_AVAILABLE)


def get_backend_info() -> dict[str, Any]:
    """Get information about available computational backends."""
    current: BackendName = get_backend_preference()
    effective: BackendChoice = current
    gpu_available = is_gpu_available()
    if effective == "auto":
        if gpu_available:
            effective = "gpu"
        elif RUST_AVAILABLE:
            effective = "rust"
        else:
            effective = "numpy"

    return {
        "current_backend": current,
        "effective_backend": effective,
        "gpu_available": gpu_available,
        "gpu_device": get_gpu_device_name(),
        "gpu_memory": get_gpu_memory_info(),
        "rust_available": RUST_AVAILABLE,
        "torch_available": is_torch_available(),
    }
