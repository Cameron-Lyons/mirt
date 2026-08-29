"""Regression coverage for explicit iteration and exception contracts."""

from __future__ import annotations

import builtins
from types import SimpleNamespace
from typing import Any

import pytest

from mirt.cat._engine_common import build_administered_response_matrix
from mirt.diagnostics.drf import plot_drf


def test_response_matrix_rejects_desynchronized_session_history() -> None:
    engine = SimpleNamespace(
        model=SimpleNamespace(n_items=3),
        _items_administered=[0, 1],
        _responses=[1],
    )

    with pytest.raises(ValueError, match="zip.*shorter"):
        build_administered_response_matrix(engine)


def test_plot_drf_preserves_optional_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fail_matplotlib_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "matplotlib.pyplot":
            raise ImportError("simulated missing plotting dependency")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_matplotlib_import)

    with pytest.raises(ImportError, match="matplotlib required") as error:
        plot_drf({})

    assert isinstance(error.value.__cause__, ImportError)
    assert "simulated missing" in str(error.value.__cause__)
