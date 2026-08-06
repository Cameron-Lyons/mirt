"""Tests for model fit statistics (M2, RMSEA, CFI, TLI)."""

import numpy as np
import pytest

from mirt import compute_fit_indices, compute_m2
from mirt.diagnostics.modelfit import (
    _compute_expected_margins,
    _compute_rmsea,
    _compute_rmsea_ci,
)
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.models import GradedResponseModel, TwoParameterLogistic


class TestM2:
    """Tests for M2 limited-information statistic."""

    def test_compute_m2(self, fitted_2pl_model, dichotomous_responses):
        """Test M2 computation."""
        responses = dichotomous_responses["responses"]

        m2_result = compute_m2(fitted_2pl_model.model, responses)

        assert "M2" in m2_result
        assert "df" in m2_result
        assert "p_value" in m2_result

        assert m2_result["M2"] >= 0
        assert m2_result["df"] > 0
        assert 0 <= m2_result["p_value"] <= 1

    def test_m2_with_fit_result(self, fitted_2pl_model, dichotomous_responses):
        """Test M2 computation with FitResult object."""
        responses = dichotomous_responses["responses"]

        m2_result = compute_m2(fitted_2pl_model.model, responses)
        assert "M2" in m2_result

    def test_supplied_theta_changes_expected_moments(self):
        model = TwoParameterLogistic(n_items=3)
        responses = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1]], dtype=int)

        centered = compute_m2(model, responses, theta=np.zeros(4))
        high_ability = compute_m2(model, responses, theta=np.full(4, 2.0))

        assert centered["M2"] == pytest.approx(0.25)
        assert high_ability["M2"] > centered["M2"]

    def test_polytomous_collapsed_score_moments(self):
        model = GradedResponseModel(n_items=3, n_categories=4)
        responses = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 0], [3, 0, 1]], dtype=int)

        result = compute_m2(model, responses, theta=np.linspace(-1.5, 1.5, 4))
        quadrature_result = compute_m2(model, responses, n_quadpts=9)

        assert np.isfinite(result["M2"])
        assert result["M2"] >= 0.0
        assert 0.0 <= result["p_value"] <= 1.0
        assert np.isfinite(quadrature_result["M2"])

    @pytest.mark.parametrize(
        ("responses", "message"),
        [
            (np.zeros((1, 3)), "at least 2 persons"),
            (np.zeros((3, 2)), "expected 3"),
            (np.full((3, 3), np.nan), "no observed"),
            (np.full((3, 3), 0.5), "integer category"),
            (np.full((3, 3), 2), "between 0 and 1"),
        ],
    )
    def test_input_validation(self, responses, message):
        with pytest.raises(ValueError, match=message):
            compute_m2(TwoParameterLogistic(n_items=3), responses)

    def test_theta_and_quadrature_validation(self):
        model = TwoParameterLogistic(n_items=3)
        responses = np.zeros((4, 3), dtype=int)

        with pytest.raises(ValueError, match="theta must have shape"):
            compute_m2(model, responses, theta=np.zeros(3))
        with pytest.raises(ValueError, match="n_quadpts"):
            compute_m2(model, responses, n_quadpts=1)

    def test_vectorized_quadrature_matches_itemwise_calculation(self):
        model = TwoParameterLogistic(n_items=4)
        model.set_parameters(
            discrimination=np.array([0.7, 1.0, 1.3, 1.8]),
            difficulty=np.array([-1.0, -0.25, 0.5, 1.25]),
        )
        quadrature = GaussHermiteQuadrature(n_points=15, n_dimensions=1)

        expected_uni, expected_bi = _compute_expected_margins(model, 15)
        item_probabilities = np.column_stack(
            [model.probability(quadrature.nodes, idx) for idx in range(model.n_items)]
        )
        manual_uni = quadrature.weights @ item_probabilities
        manual_bi = (item_probabilities * quadrature.weights[:, None]).T @ (
            item_probabilities
        )

        np.testing.assert_allclose(expected_uni, manual_uni)
        np.testing.assert_allclose(expected_bi, manual_bi)


class TestFitIndices:
    """Tests for RMSEA, CFI, TLI, SRMSR."""

    def test_compute_fit_indices(self, fitted_2pl_model, dichotomous_responses):
        """Test fit indices computation."""
        responses = dichotomous_responses["responses"]

        fit_stats = compute_fit_indices(fitted_2pl_model.model, responses)

        assert "RMSEA" in fit_stats
        assert "CFI" in fit_stats
        assert "TLI" in fit_stats
        assert "SRMSR" in fit_stats

    def test_rmsea_range(self, fitted_2pl_model, dichotomous_responses):
        """Test that RMSEA is in valid range."""
        responses = dichotomous_responses["responses"]

        fit_stats = compute_fit_indices(fitted_2pl_model.model, responses)

        assert fit_stats["RMSEA"] >= 0

    def test_cfi_tli_range(self, fitted_2pl_model, dichotomous_responses):
        """Test that CFI/TLI are in valid range."""
        responses = dichotomous_responses["responses"]

        fit_stats = compute_fit_indices(fitted_2pl_model.model, responses)

        assert fit_stats["CFI"] >= 0
        assert fit_stats["TLI"] >= -0.5

    def test_rmsea_ci(self, fitted_2pl_model, dichotomous_responses):
        """Test RMSEA confidence intervals."""
        responses = dichotomous_responses["responses"]

        fit_stats = compute_fit_indices(fitted_2pl_model.model, responses)

        if "RMSEA_CI_lower" in fit_stats:
            assert fit_stats["RMSEA_CI_lower"] <= fit_stats["RMSEA"]
            assert fit_stats["RMSEA_CI_upper"] >= fit_stats["RMSEA"]

    def test_srmsr_range(self, fitted_2pl_model, dichotomous_responses):
        """Test that SRMSR is in valid range."""
        responses = dichotomous_responses["responses"]

        fit_stats = compute_fit_indices(fitted_2pl_model.model, responses)

        assert fit_stats["SRMSR"] >= 0

    def test_missing_codes_are_equivalent(self):
        model = TwoParameterLogistic(n_items=3)
        responses = np.array(
            [[-1, 1, 0], [1, 0, 1], [1, -1, 0], [0, 0, 1]], dtype=float
        )
        theta = np.linspace(-1.0, 1.0, 4)
        nan_responses = responses.copy()
        nan_responses[nan_responses < 0] = np.nan

        coded = compute_fit_indices(model, responses, theta=theta)
        missing = compute_fit_indices(model, nan_responses, theta=theta)

        np.testing.assert_allclose(
            list(coded.values()),
            list(missing.values()),
            equal_nan=True,
        )

    def test_probability_matrix_is_evaluated_once(self, monkeypatch):
        model = TwoParameterLogistic(n_items=4)
        responses = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]])
        original_probability = model.probability
        calls: list[int | None] = []

        def counted_probability(theta, item_idx=None):
            calls.append(item_idx)
            return original_probability(theta, item_idx)

        monkeypatch.setattr(model, "probability", counted_probability)

        compute_fit_indices(model, responses, theta=np.linspace(-1.0, 1.0, 4))

        assert calls == [None]

    def test_rmsea_interval_is_ordered_and_contains_estimate(self):
        estimate = _compute_rmsea(10.2, 1, 4)
        lower, upper = _compute_rmsea_ci(10.2, 1, 4)

        assert lower <= estimate <= upper
