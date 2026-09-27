"""Compatibility shim — prefer mirt.backends.rust."""

from typing import TYPE_CHECKING, Any

from mirt.backends import rust as _rust

if TYPE_CHECKING:
    from mirt.backends.rust._helpers import _MAX_VECTOR_CHUNK_ENTRIES

__all__ = [*_rust.__all__, "_MAX_VECTOR_CHUNK_ENTRIES"]


def __getattr__(name: str) -> Any:
    if name == "_MAX_VECTOR_CHUNK_ENTRIES":
        from mirt.backends.rust._helpers import _MAX_VECTOR_CHUNK_ENTRIES

        value = _MAX_VECTOR_CHUNK_ENTRIES
    elif name in _rust.__all__:
        value = getattr(_rust, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
