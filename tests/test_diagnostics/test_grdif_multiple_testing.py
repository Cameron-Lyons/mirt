"""Multiple-testing contracts for GRDIF workflows."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import mirt.diagnostics.dif as dif_module
from mirt.diagnostics.dif import compute_grdif, compute_pairwise_rdif
from mirt.diagnostics.multiple_testing import adjust_p_values


def _data() -> tuple[np.ndarray, np.ndarray]:
    responses = np.tile([0, 1, 0, 1], (8, 1)).astype(np.int64)
    groups = np.repeat(["reference", "focal"], 4)
    return responses, groups


def _stub_fit_and_scores(
    monkeypatch: pytest.MonkeyPatch,
    n_rows: int,
) -> Any:
    model = object()
    monkeypatch.setattr(
        "mirt.fit_mirt",
        lambda *args, **kwargs: SimpleNamespace(model=model),
    )
    monkeypatch.setattr(
        dif_module,
        "_score_grdif_responses",
        lambda *args, **kwargs: np.zeros((n_rows, 1)),
    )
    return model


@pytest.mark.parametrize("method", ["bonferroni", "holm", "fdr_bh"])
def test_grdif_adjusts_each_statistic_family_across_items(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    data, groups = _data()
    _stub_fit_and_scores(monkeypatch, len(data))
    raw = (
        np.array([0.001, 0.02, 0.04, 0.20]),
        np.array([0.002, 0.03, 0.06, 0.30]),
        np.array([0.003, 0.04, 0.08, 0.40]),
    )
    zeros = np.zeros(data.shape[1])
    monkeypatch.setattr(
        dif_module,
        "_compute_grdif_statistics",
        lambda *args, **kwargs: (zeros, zeros, zeros, *raw),
    )

    result = compute_grdif(data, groups, p_adjust=method)

    for suffix, p_values in zip(("r", "s", "rs"), raw, strict=True):
        expected = adjust_p_values(p_values, method)
        assert_allclose(result[f"p_value_{suffix}_adjusted"], expected)
        assert_array_equal(result[f"flagged_{suffix}"], expected < 0.05)
    assert result["p_adjustment"] == method


def test_grdif_default_preserves_raw_flagging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, groups = _data()
    _stub_fit_and_scores(monkeypatch, len(data))
    raw = np.array([0.001, 0.02, 0.08, 0.40])
    zeros = np.zeros(data.shape[1])
    monkeypatch.setattr(
        dif_module,
        "_compute_grdif_statistics",
        lambda *args, **kwargs: (zeros, zeros, zeros, raw, raw, raw),
    )

    result = compute_grdif(data, groups)

    assert_allclose(result["p_value_rs_adjusted"], raw)
    assert_array_equal(result["flagged_rs"], raw < 0.05)
    assert result["p_adjustment"] == "none"


def test_grdif_purification_uses_adjusted_p_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, groups = _data()
    _stub_fit_and_scores(monkeypatch, len(data))
    raw = np.array([0.02, 0.50, 0.50, 0.50])
    zeros = np.zeros(data.shape[1])
    monkeypatch.setattr(
        dif_module,
        "_compute_grdif_statistics",
        lambda *args, **kwargs: (zeros, zeros, zeros, raw, raw, raw),
    )

    result = compute_grdif(
        data,
        groups,
        purify=True,
        p_adjust="bonferroni",
    )

    assert result["purification_history"][0]["n_flagged"] == 0
    assert_array_equal(result["anchor_items"], np.ones(4, dtype=np.bool_))
    assert result["purification_complete"] is True


def _stub_pairwise_statistics(
    monkeypatch: pytest.MonkeyPatch,
    n_rows: int,
) -> None:
    _stub_fit_and_scores(monkeypatch, n_rows)
    monkeypatch.setattr(
        dif_module,
        "_expected_response_matrix",
        lambda *args, **kwargs: np.zeros((n_rows, 4)),
    )
    r = np.array([6.63, 5.02, 3.84, 2.71])
    s = np.array([7.88, 5.99, 4.22, 2.71])
    rs = np.array([9.21, 7.38, 5.99, 4.61])
    zeros = np.zeros(4)
    monkeypatch.setattr(
        dif_module,
        "_compute_grdif_statistics",
        lambda *args, **kwargs: (r, s, rs, zeros, zeros, zeros),
    )


@pytest.mark.parametrize("method", ["bonferroni", "holm", "fdr_bh"])
def test_pairwise_rdif_adjusts_each_full_pair_item_family(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    data = np.tile([0, 1, 0, 1], (9, 1)).astype(np.int64)
    groups = np.repeat(["a", "b", "c"], 3)
    _stub_pairwise_statistics(monkeypatch, len(data))

    result = compute_pairwise_rdif(data, groups, p_adjust=method)

    for suffix in ("r", "s", "rs"):
        raw = result[f"p_values_{suffix}"]
        expected = adjust_p_values(raw, method)
        assert_allclose(result[f"p_values_{suffix}_adjusted"], expected)
        assert_array_equal(result[f"flagged_{suffix}"], expected < 0.05)
    assert result["p_adjustment"] == method


def test_pairwise_bonferroni_controls_pairs_and_items_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = np.tile([0, 1, 0, 1], (9, 1)).astype(np.int64)
    groups = np.repeat(["a", "b", "c"], 3)
    _stub_pairwise_statistics(monkeypatch, len(data))

    result = compute_pairwise_rdif(data, groups, p_adjust="bonferroni")

    raw = result["p_values_r"]
    assert_allclose(
        result["p_values_r_adjusted"],
        np.clip(raw * raw.size, 0.0, 1.0),
    )


@pytest.mark.parametrize("function", [compute_grdif, compute_pairwise_rdif])
def test_grdif_workflows_reject_unknown_adjustment_before_fitting(
    monkeypatch: pytest.MonkeyPatch,
    function,
) -> None:
    def unexpected_fit(*args: Any, **kwargs: Any) -> None:
        pytest.fail("fit should not run for invalid inputs")

    monkeypatch.setattr("mirt.fit_mirt", unexpected_fit)
    data, groups = _data()
    with pytest.raises(ValueError, match="p_adjust"):
        function(data, groups, p_adjust="sidak")
