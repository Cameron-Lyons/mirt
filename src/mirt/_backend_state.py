"""Leaf module holding the global backend preference.

Kept separate from :mod:`mirt._backend_config` and the Rust wrappers so that
backend modules can honor ``set_backend("numpy")`` without circular imports.
"""

from __future__ import annotations

from typing import Literal

BackendName = Literal["auto", "gpu", "rust", "numpy"]

_CURRENT_BACKEND: BackendName = "auto"


def get_backend_preference() -> BackendName:
    """Return the configured backend preference."""
    return _CURRENT_BACKEND


def set_backend_preference(backend: BackendName) -> None:
    """Set the configured backend preference (no availability checks)."""
    global _CURRENT_BACKEND
    _CURRENT_BACKEND = backend
