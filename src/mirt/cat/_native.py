"""Authored CAT methods required by native simulation shortcuts."""

from __future__ import annotations

from inspect import getattr_static
from types import FunctionType
from typing import Any, TypeVar

_Class = TypeVar("_Class")
_DEFAULT_METHODS: dict[type, tuple[tuple[str, type, Any], ...]] = {}
_MISSING = object()


def register_native_defaults(cls: type[_Class]) -> type[_Class]:
    """Capture original methods when a supported class is defined.

    Capture inherited definitions too, so replacing a base method before the
    engine is imported cannot silently change the native eligibility baseline.
    """
    definitions: dict[str, list[tuple[type, Any]]] = {}
    for owner in cls.__mro__:
        if owner is object:
            continue
        for name, value in vars(owner).items():
            if isinstance(value, (FunctionType, classmethod, staticmethod, property)):
                definitions.setdefault(name, []).append((owner, value))

    methods = []
    for name, originals in definitions.items():
        methods.append((name, cls, originals[0][1]))
        methods.extend((name, owner, value) for owner, value in originals[1:])
    _DEFAULT_METHODS[cls] = tuple(methods)
    return cls


def uses_native_defaults(value: object, cls: type) -> bool:
    """Require an exact supported type and its unmodified authored methods."""
    if type(value) is not cls:
        return False
    methods = _DEFAULT_METHODS.get(cls)
    if methods is None:
        return False
    namespace = vars(value)
    return all(
        name not in namespace and getattr_static(owner, name, _MISSING) is original
        for name, owner, original in methods
    )
