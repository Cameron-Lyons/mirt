"""Regression tests for lazy top-level API loading."""

from __future__ import annotations

from types import ModuleType

import mirt


def test_lazy_equating_symbol_resolution() -> None:
    """Heavy equating symbols should resolve lazily via __getattr__."""
    assert "link" not in mirt.__dict__
    link_fn = mirt.link
    assert callable(link_fn)
    assert "link" in mirt.__dict__


def test_lazy_module_namespace_resolution() -> None:
    """Top-level module namespaces should be exposed lazily."""
    equating_module = mirt.equating
    assert isinstance(equating_module, ModuleType)
    assert mirt.__dict__["equating"] is equating_module


def test_lazy_dataset_constant_materialization() -> None:
    """Dataset constants should be built on first access, not import."""
    from mirt.utils import datasets

    assert "LSAT6" not in datasets.__dict__
    dataset = datasets.LSAT6
    assert "data" in dataset
    assert "LSAT6" in datasets.__dict__
