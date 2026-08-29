"""Regression tests for lazy loading in :mod:`mirt.diagnostics`."""

from __future__ import annotations

import importlib


def _reload_diagnostics_module():
    diagnostics_module = importlib.import_module("mirt.diagnostics")
    return importlib.reload(diagnostics_module)


def test_lazy_diagnostics_symbol_resolution() -> None:
    diagnostics = _reload_diagnostics_module()

    assert "compute_dif" not in diagnostics.__dict__
    assert "adjust_p_values" not in diagnostics.__dict__

    compute_dif = diagnostics.compute_dif
    adjust_p_values = diagnostics.adjust_p_values
    assert callable(compute_dif)
    assert callable(adjust_p_values)
    assert diagnostics.__dict__["compute_dif"] is compute_dif
    assert diagnostics.__dict__["adjust_p_values"] is adjust_p_values


def test_lazy_diagnostics_bayesian_symbol_resolution() -> None:
    diagnostics = _reload_diagnostics_module()

    assert "psis_loo" not in diagnostics.__dict__

    psis_loo = diagnostics.psis_loo
    assert callable(psis_loo)
    assert diagnostics.__dict__["psis_loo"] is psis_loo
