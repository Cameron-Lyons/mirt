"""Name lookup shared by the CAT factories and engine options.

Every factory and every string-valued engine option resolves names the same
way: case and surrounding whitespace are ignored, and ``-`` and ``_`` are
interchangeable. ``"sympson_hetter"`` therefore selects ``"sympson-hetter"``
and ``"Max-Items"`` selects ``"max_items"``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

_Value = TypeVar("_Value")


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def resolve_name(
    name: object,
    table: Mapping[str, _Value],
    *,
    kind: str,
) -> tuple[str, _Value]:
    """Return the canonical key and value that ``name`` selects in ``table``.

    Raises
    ------
    TypeError
        If ``name`` is not a string.
    ValueError
        If ``name`` matches no key of ``table``.
    """
    if not isinstance(name, str):
        raise TypeError(f"{kind} must be a string")
    normalized = _normalize(name)
    for key, value in table.items():
        if _normalize(key) == normalized:
            return key, value
    valid = ", ".join(table)
    raise ValueError(f"Unknown {kind} '{name}'. Valid options: {valid}")


def canonical_name(
    name: object, names: Mapping[str, object] | tuple[str, ...], *, kind: str
) -> str:
    """Return the canonical spelling of ``name`` among ``names``."""
    table = names if isinstance(names, Mapping) else dict.fromkeys(names)
    return resolve_name(name, table, kind=kind)[0]
