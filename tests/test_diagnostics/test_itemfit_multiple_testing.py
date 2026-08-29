"""Multiple-testing contracts for S-X2 item-fit statistics."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import mirt.diagnostics as diagnostics
from mirt import itemfit
from mirt.diagnostics.itemfit import compute_itemfit, compute_s_x2
from mirt.diagnostics.multiple_testing import adjust_p_values


def test_compute_s_x2_is_available_from_diagnostics_namespace() -> None:
    assert diagnostics.compute_s_x2 is compute_s_x2


@pytest.mark.parametrize("method", ["bonferroni", "holm", "fdr_bh"])
def test_compute_s_x2_exposes_requested_adjustment(
    fitted_2pl_model,
    dichotomous_responses,
    method,
) -> None:
    model = fitted_2pl_model.model
    responses = dichotomous_responses["responses"]

    result = compute_s_x2(model, responses, n_groups=5, p_adjust=method)

    assert_allclose(
        result["p_value_adjusted"],
        adjust_p_values(result["p_value"], method),
    )
    assert np.all(result["p_value_adjusted"] >= result["p_value"])


def test_compute_itemfit_applies_adjustment_to_s_x2_only(
    fitted_2pl_model,
    dichotomous_responses,
) -> None:
    model = fitted_2pl_model.model
    responses = dichotomous_responses["responses"]

    result = compute_itemfit(
        model,
        responses,
        statistics=["infit", "outfit", "S_X2"],
        n_groups=5,
        p_adjust="holm",
    )

    assert {"infit", "outfit", "S_X2", "df", "p_value"}.issubset(result)
    assert_allclose(
        result["p_value_adjusted"],
        adjust_p_values(result["p_value"], "holm"),
    )


def test_default_result_shape_remains_unchanged(
    fitted_2pl_model,
    dichotomous_responses,
) -> None:
    model = fitted_2pl_model.model
    responses = dichotomous_responses["responses"]

    direct = compute_s_x2(model, responses, n_groups=5)
    combined = compute_itemfit(
        model,
        responses,
        statistics=["S_X2"],
        n_groups=5,
    )

    assert set(direct) == {"S_X2", "df", "p_value"}
    assert set(combined) == {"S_X2", "df", "p_value"}


def test_top_level_itemfit_forwards_adjustment(
    fitted_2pl_model,
    dichotomous_responses,
) -> None:
    responses = dichotomous_responses["responses"]

    result = itemfit(
        fitted_2pl_model,
        responses,
        statistics=["S_X2"],
        n_groups=5,
        p_adjust="fdr_bh",
    )

    assert "p_value_adjusted" in result.columns
    assert_allclose(
        np.asarray(result["p_value_adjusted"]),
        adjust_p_values(np.asarray(result["p_value"]), "fdr_bh"),
    )


@pytest.mark.parametrize("function", [compute_itemfit, compute_s_x2])
def test_itemfit_rejects_unknown_adjustment(
    function,
    fitted_2pl_model,
    dichotomous_responses,
) -> None:
    with pytest.raises(ValueError, match="p_adjust"):
        function(
            fitted_2pl_model.model,
            dichotomous_responses["responses"],
            p_adjust="sidak",
        )
