"""Public result containers with deferred implementation imports."""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_IMPORTS = {
    "AbilityPosteriorResult": (
        "mirt.results.ability_posterior",
        "AbilityPosteriorResult",
    ),
    "FitResult": ("mirt.results.fit_result", "FitResult"),
    "ScoreResult": ("mirt.results.score_result", "ScoreResult"),
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_name, symbol_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, symbol_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'mirt.results' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
