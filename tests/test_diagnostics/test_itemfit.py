"""Tests for item fit statistics."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.diagnostics.itemfit import compute_itemfit, compute_s_x2


class TestComputeItemfit:
    """Tests for compute_itemfit function."""

    def test_basic_itemfit(self, fitted_2pl_model, dichotomous_responses):
        """Test basic item fit computation."""
        model = fitted_2pl_model.model
        result = compute_itemfit(model, responses=dichotomous_responses["responses"])

        assert "infit" in result
        assert "outfit" in result
        assert len(result["infit"]) == dichotomous_responses["n_items"]
        assert len(result["outfit"]) == dichotomous_responses["n_items"]

    def test_itemfit_values_positive(self, fitted_2pl_model, dichotomous_responses):
        """Test that fit statistics are positive."""
        model = fitted_2pl_model.model
        result = compute_itemfit(model, responses=dichotomous_responses["responses"])

        assert np.all(result["infit"] > 0)
        assert np.all(result["outfit"] > 0)

    def test_itemfit_values_reasonable_range(
        self, fitted_2pl_model, dichotomous_responses
    ):
        """Test that fit statistics are in reasonable range."""
        model = fitted_2pl_model.model
        result = compute_itemfit(model, responses=dichotomous_responses["responses"])

        assert np.all(result["infit"] < 3.0)
        assert np.all(result["outfit"] < 3.0)

    def test_itemfit_with_theta(self, fitted_2pl_model, dichotomous_responses):
        """Test item fit with provided theta."""
        from mirt.scoring import fscores

        model = fitted_2pl_model.model
        scores = fscores(model, dichotomous_responses["responses"], method="EAP")

        result = compute_itemfit(
            model,
            responses=dichotomous_responses["responses"],
            theta=scores.theta,
        )

        assert "infit" in result
        assert "outfit" in result

    def test_itemfit_statistics_subset(self, fitted_2pl_model, dichotomous_responses):
        """Test computing only subset of statistics."""
        model = fitted_2pl_model.model
        result = compute_itemfit(
            model,
            responses=dichotomous_responses["responses"],
            statistics=["infit"],
        )

        assert "infit" in result
        assert "outfit" not in result

    def test_itemfit_no_responses_raises_error(self, fitted_2pl_model):
        """Test that missing responses raises error."""
        model = fitted_2pl_model.model
        with pytest.raises(ValueError, match="responses required"):
            compute_itemfit(model, responses=None)

    def test_itemfit_default_statistics(self, fitted_2pl_model, dichotomous_responses):
        """Test default statistics are infit and outfit."""
        model = fitted_2pl_model.model
        result = compute_itemfit(model, responses=dichotomous_responses["responses"])

        assert "infit" in result
        assert "outfit" in result


class TestComputeSX2:
    """Tests for compute_s_x2 function."""

    def test_basic_s_x2(self, fitted_2pl_model, dichotomous_responses):
        """Test basic S-X2 computation."""
        model = fitted_2pl_model.model
        result = compute_s_x2(model, dichotomous_responses["responses"])

        assert "S_X2" in result
        assert "df" in result
        assert "p_value" in result
        assert len(result["S_X2"]) == dichotomous_responses["n_items"]

    def test_s_x2_values_positive(self, fitted_2pl_model, dichotomous_responses):
        """Test that S-X2 values are non-negative."""
        model = fitted_2pl_model.model
        result = compute_s_x2(model, dichotomous_responses["responses"])

        assert np.all(result["S_X2"] >= 0)

    def test_s_x2_df_positive(self, fitted_2pl_model, dichotomous_responses):
        """Test that degrees of freedom are positive."""
        model = fitted_2pl_model.model
        result = compute_s_x2(model, dichotomous_responses["responses"])

        assert np.all(result["df"] >= 1)

    def test_s_x2_p_values_in_range(self, fitted_2pl_model, dichotomous_responses):
        """Test that p-values are in [0, 1]."""
        model = fitted_2pl_model.model
        result = compute_s_x2(model, dichotomous_responses["responses"])

        assert np.all(result["p_value"] >= 0)
        assert np.all(result["p_value"] <= 1)

    def test_s_x2_with_theta(self, fitted_2pl_model, dichotomous_responses):
        """Test S-X2 with provided theta."""
        from mirt.scoring import fscores

        model = fitted_2pl_model.model
        scores = fscores(model, dichotomous_responses["responses"], method="EAP")

        result = compute_s_x2(
            model,
            dichotomous_responses["responses"],
            theta=scores.theta,
        )

        assert "S_X2" in result

    def test_s_x2_custom_n_groups(self, fitted_2pl_model, dichotomous_responses):
        """Test S-X2 with custom number of groups."""
        model = fitted_2pl_model.model
        result = compute_s_x2(
            model,
            dichotomous_responses["responses"],
            n_groups=5,
        )

        assert "S_X2" in result


class TestItemfitWithPolytomousModel:
    """Tests for item fit with polytomous models."""

    def test_itemfit_polytomous(self, polytomous_responses):
        """Test item fit with polytomous model."""
        from mirt import fit_mirt

        result = fit_mirt(
            polytomous_responses["responses"],
            model="GRM",
            max_iter=15,
            n_quadpts=11,
        )

        fit_result = compute_itemfit(
            result.model, responses=polytomous_responses["responses"]
        )

        assert "infit" in fit_result
        assert "outfit" in fit_result
        assert len(fit_result["infit"]) == polytomous_responses["n_items"]


class TestItemfitEdgeCases:
    """Tests for edge cases in item fit computation."""

    def test_itemfit_perfect_fit(self, fitted_2pl_model):
        """Test item fit when responses match model expectations perfectly."""
        model = fitted_2pl_model.model
        n_persons = 30
        responses = np.zeros((n_persons, model.n_items), dtype=int)

        from mirt.scoring import fscores

        theta = fscores(model, responses, method="EAP").theta

        result = compute_itemfit(model, responses=responses, theta=theta)

        assert np.all(np.isfinite(result["infit"]))
        assert np.all(np.isfinite(result["outfit"]))

    def test_itemfit_consistency(self, fitted_2pl_model, dichotomous_responses):
        """Test that item fit is consistent across calls."""
        model = fitted_2pl_model.model
        result1 = compute_itemfit(model, responses=dichotomous_responses["responses"])
        result2 = compute_itemfit(model, responses=dichotomous_responses["responses"])

        assert_allclose(result1["infit"], result2["infit"])
        assert_allclose(result1["outfit"], result2["outfit"])
