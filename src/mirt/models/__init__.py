"""Public model API with on-demand imports.

Importing this namespace keeps numerical dependencies and individual model
modules deferred until one of their public symbols is accessed.
"""

from __future__ import annotations

import importlib
from typing import Any

from mirt._api_registry import MODELS_EXPORTS as _MODELS_EXPORTS

# The registry is pure data, so importing it keeps this namespace numpy-free.
_LAZY_IMPORTS = dict(_MODELS_EXPORTS)

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name: str) -> Any:
    """Resolve and cache a public model symbol on first access."""
    try:
        module_name = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return loaded attributes together with deferred public symbols."""
    return sorted(set(globals()) | set(__all__))
