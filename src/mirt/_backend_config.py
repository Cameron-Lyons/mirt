"""Runtime backend selection utilities for :mod:`mirt`."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

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
from mirt.backends.rust._helpers import RUST_AVAILABLE
from mirt.exceptions import MirtValidationError

BackendChoice = Literal["auto", "gpu", "rust", "numpy"]


def set_backend(backend: BackendChoice) -> None:
    """Set the computational backend for IRT operations.

    Parameters
    ----------
    backend : {"auto", "gpu", "rust", "numpy"}
        Preferred backend. ``"auto"`` prefers the GPU when available,
        otherwise Rust, otherwise NumPy; ``"gpu"`` requires a GPU and
        otherwise behaves like ``"auto"``. ``"rust"`` disables GPU dispatch
        and ``"numpy"`` disables both GPU dispatch and Rust acceleration.

    Notes
    -----
    An estimator's explicit ``use_gpu=True`` or ``use_gpu=False`` (for
    example :class:`~mirt.estimation.em.EMEstimator` and ``fit_mirt``)
    takes precedence over this preference for GPU dispatch, and
    ``use_gpu="auto"`` follows it. ``use_rust=False`` disables Rust for that
    call under any preference.
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


def validate_use_gpu(use_gpu: Any) -> bool | Literal["auto"]:
    """Return a per-call ``use_gpu`` flag as a bool or ``"auto"``.

    Raises
    ------
    MirtValidationError
        If ``use_gpu`` is neither a boolean nor ``"auto"``.
    """
    if isinstance(use_gpu, (bool, np.bool_)):
        return bool(use_gpu)
    if isinstance(use_gpu, str) and use_gpu == "auto":
        return "auto"
    raise MirtValidationError(
        "use_gpu must be a boolean or 'auto'",
        parameter="use_gpu",
        value=use_gpu,
        expected="True, False, or 'auto'",
    )


def gpu_dispatch_allowed(use_gpu: bool | Literal["auto"] = "auto") -> bool:
    """Whether the per-call flag and global preference permit GPU dispatch.

    An explicit per-call ``use_gpu`` takes precedence over the global
    preference from :func:`set_backend`, which ``"auto"`` follows: the
    ``"auto"`` and ``"gpu"`` preferences permit the GPU, ``"rust"`` and
    ``"numpy"`` do not. GPU availability is checked separately, see
    :func:`should_use_gpu`.

    Parameters
    ----------
    use_gpu : bool or "auto", default="auto"
        Per-call preference. ``False`` never uses the GPU and ``True`` uses
        it whenever one is available.

    Returns
    -------
    bool
        ``True`` when the call may use an available GPU.
    """
    if isinstance(use_gpu, str):
        return get_backend_preference() not in ("rust", "numpy")
    return bool(use_gpu)


def should_use_gpu(use_gpu: bool | Literal["auto"] = "auto") -> bool:
    """Whether a hot path should dispatch to the GPU.

    Parameters
    ----------
    use_gpu : bool or "auto", default="auto"
        Per-call preference, resolved as in :func:`gpu_dispatch_allowed`.

    Returns
    -------
    bool
        ``True`` when GPU dispatch is permitted and a GPU is available.
    """
    return gpu_dispatch_allowed(use_gpu) and is_gpu_available()


def get_backend_info() -> dict[str, Any]:
    """Get information about available computational backends."""
    current: BackendName = get_backend_preference()
    effective: BackendChoice = current
    gpu_available = is_gpu_available()
    if effective in ("auto", "gpu"):
        if should_use_gpu():
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
