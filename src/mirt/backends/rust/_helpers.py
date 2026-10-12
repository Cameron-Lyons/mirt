"""Shared helpers and Rust availability for the mirt Rust backend.

Fallback contract
-----------------
Each wrapper module declares ``FALLBACK_MODE`` as one of:

* ``"numpy"`` — pure NumPy implementation runs when Rust is unavailable or
  disabled via ``mirt.set_backend("numpy")``.
* ``"optional"`` — returns ``None`` when Rust cannot be used; callers must
  provide their own Python path.
* ``"required"`` — raises :class:`RuntimeError` when Rust cannot be used
  (accelerated-only entry points; higher-level public APIs still have Python
  estimators).
* ``"mixed"`` — module contains more than one of the modes above; see the
  module docstring for per-function notes.

Wrappers validate shapes, indices and controls before calling the extension,
because a Rust panic surfaces as ``pyo3_runtime.PanicException``, a
:class:`BaseException` that ``except Exception`` does not catch.
:func:`_call_native` re-raises any remaining panic as :class:`RuntimeError`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NoReturn, TypeVar

import numpy as np
from numpy.typing import NDArray

try:
    from mirt import mirt_rs

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    mirt_rs = None


def is_rust_available() -> bool:
    """Check if the Rust extension module is importable."""
    return RUST_AVAILABLE


def rust_enabled() -> bool:
    """Whether Rust wrappers should dispatch to the extension.

    Returns ``False`` when the extension is missing or the global backend
    preference is ``"numpy"`` (see :func:`mirt.set_backend`).
    """
    if not RUST_AVAILABLE:
        return False
    from mirt._backend_state import get_backend_preference

    return get_backend_preference() != "numpy"


def rust_required(name: str) -> NoReturn:
    """Raise a consistent error for accelerated-only entry points."""
    raise RuntimeError(
        f"Rust backend required for {name}. "
        "Build the extension (`maturin develop --release`) or use the "
        "corresponding public Python estimator API instead."
    )


def _ensure_f64(arr: NDArray[np.floating] | None) -> NDArray[np.float64] | None:
    """Return a contiguous float64 array, avoiding copies when possible."""
    if arr is None:
        return None
    return np.ascontiguousarray(arr, dtype=np.float64)


def _ensure_i32(arr: NDArray[np.integer] | None) -> NDArray[np.int32] | None:
    """Return a contiguous int32 array, avoiding copies when possible."""
    if arr is None:
        return None
    return np.ascontiguousarray(arr, dtype=np.int32)


_INT32_MIN = int(np.iinfo(np.int32).min)
_INT32_MAX = int(np.iinfo(np.int32).max)
_T = TypeVar("_T")


def _category_codes(responses: NDArray[Any]) -> NDArray[np.int32]:
    """Return contiguous int32 category codes with every missing code as -1.

    Negative and ``NaN`` entries are missing. An int32 cast would wrap
    observed codes above ``2**31 - 1`` into valid categories (and turn ``NaN``
    into an arbitrary integer), so such codes raise instead.

    Raises
    ------
    IndexError
        If an observed code does not fit in int32.
    ValueError
        If an observed floating-point code is not an integer.
    """
    values = np.asarray(responses)
    if values.dtype == np.int32:
        return np.ascontiguousarray(values)
    if values.dtype.kind in "iu" and values.size:
        # Codes that fit in int32 keep their sign, hence their missingness.
        if _INT32_MIN <= values.min() and values.max() <= _INT32_MAX:
            return np.ascontiguousarray(values, dtype=np.int32)
    observed = values >= 0
    if values.dtype.kind == "f" and np.any(observed & (values != np.trunc(values))):
        raise ValueError("observed response codes must be integers")
    if np.any(observed & (values > _INT32_MAX)):
        raise IndexError("response category is outside the item category range")
    return np.where(observed, values, -1).astype(np.int32)


def _is_native_panic(error: BaseException) -> bool:
    """Return whether ``error`` is the exception of a panicking Rust kernel.

    pyo3 raises ``pyo3_runtime.PanicException``, which derives from
    :class:`BaseException` and is not importable, so it is matched by name.
    """
    error_type = type(error)
    return (
        error_type.__name__ == "PanicException"
        and error_type.__module__ == "pyo3_runtime"
    )


def _call_native(kernel: Callable[..., _T], *args: Any) -> _T:
    """Call a Rust kernel, re-raising a Rust panic as :class:`RuntimeError`.

    Callers with a NumPy fallback catch ``RuntimeError`` (or ``Exception``),
    which a panic would otherwise bypass.
    """
    try:
        return kernel(*args)
    except BaseException as error:
        if not _is_native_panic(error):
            raise
        name = getattr(kernel, "__name__", "kernel")
        raise RuntimeError(f"native {name} failed: {error}") from error


_MAX_VECTOR_CHUNK_ENTRIES = 1_000_000


def _entry_chunk_size(n_rows: int, entries_per_row: int) -> int:
    """Select a row chunk size with bounded temporary memory."""
    return max(
        1,
        min(n_rows, _MAX_VECTOR_CHUNK_ENTRIES // max(1, entries_per_row)),
    )
