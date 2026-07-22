"""Compatibility shim — prefer mirt.backends.rust."""

from mirt.backends import rust as _rust
from mirt.backends.rust._helpers import _MAX_VECTOR_CHUNK_ENTRIES

_RUST_EXPORTS: list[str] = list(getattr(_rust, "__all__"))
globals().update({name: getattr(_rust, name) for name in _RUST_EXPORTS})

__all__ = [*_RUST_EXPORTS, "_MAX_VECTOR_CHUNK_ENTRIES"]
