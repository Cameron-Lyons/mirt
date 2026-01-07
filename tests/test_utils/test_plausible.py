"""Tests for plausible value generation."""

import numpy as np
import pytest

from mirt import (
    combine_plausible_values,
    fit_mirt,
    generate_plausible_values,
    plausible_value_regression,
    plausible_value_statistics,
)


class TestGeneratePlausibleValues:
    """Tests for plausible value generation."""

    def test_generate_pv_posterior(self, dichotomous_responses):
        """Test posterior sampling method."""
        responses = dichotomous_responses["responses"]
        result = fit_mirt(responses, model="2PL", max_iter=50)

        pvs = generate_plausible_values(
            result,
            responses,
            n_plausible=5,
            method="posterior",
            seed=42,
        )

        n_persons = responses.shape[0]
        assert pvs.shape == (n_persons, 1, 5)

    def test_generate_pv_mcmc(self, dichotomous_responses):
        """Test MCMC sampling method."""
        responses = dichotomous_responses["responses"][:50]  # Smaller for speed
        result = fit_mirt(responses, model="2PL", max_iter=50)

        pvs = generate_plausible_values(
            result,
            responses,
            n_plausible=3,
            method="mcmc",
            seed=42,
        )

        n_persons = responses.shape[0]
        assert pvs.shape == (n_persons, 1, 3)

    def test_pv_variability(self, dichotomous_responses):
        """Test that PVs show appropriate variability."""
        responses = dichotomous_responses["responses"]
        result = fit_mirt(responses, model="2PL", max_iter=50)

        pvs = generate_plausible_values(
            result,
            responses,
            n_plausible=5,
            method="posterior",
            seed=42,
        )

        pv_variance = np.var(pvs, axis=2)
        assert np.mean(pv_variance) > 0

    def test_pv_correlation_with_ability(self, dichotomous_responses):
        """Test that PVs correlate with true ability."""
        responses = dichotomous_responses["responses"]
        true_theta = dichotomous_responses["theta"]
        result = fit_mirt(responses, model="2PL", max_iter=50)

        pvs = generate_plausible_values(
            result,
            responses,
            n_plausible=5,
            seed=42,
        )

        avg_pv = pvs[:, 0, :].mean(axis=1)
        correlation = np.corrcoef(avg_pv, true_theta)[0, 1]

        assert correlation > 0.5


class TestCombinePlausibleValues:
    """Tests for Rubin's combining rules."""

    def test_combine_estimates(self):
        """Test combining point estimates."""
        estimates = [1.0, 1.1, 0.9, 1.05, 0.95]

        result = combine_plausible_values(estimates)

        assert "estimate" in result
        assert result["estimate"] == pytest.approx(1.0, abs=0.05)

    def test_combine_with_variances(self):
        """Test combining with within-imputation variances."""
        estimates = [1.0, 1.1, 0.9, 1.05, 0.95]
        variances = [0.01, 0.01, 0.01, 0.01, 0.01]

        result = combine_plausible_values(estimates, variances)

        assert "variance" in result
        assert "se" in result
        assert "within_var" in result
        assert "between_var" in result

    def test_rubin_variance_formula(self):
        """Test Rubin's variance formula."""
        estimates = [1.0, 1.2, 0.8]
        variances = [0.1, 0.1, 0.1]

        result = combine_plausible_values(estimates, variances)

        m = 3
        within = 0.1
        between = np.var(estimates, ddof=1)
        expected_total = within + (1 + 1 / m) * between

        assert result["variance"] == pytest.approx(expected_total, rel=0.01)


class TestPlausibleValueRegression:
    """Tests for regression using plausible values."""

    def test_pv_regression(self, dichotomous_responses):
        """Test regression with PVs as predictor."""
        responses = dichotomous_responses["responses"]
        result = fit_mirt(responses, model="2PL", max_iter=50)

        pvs = generate_plausible_values(
            result,
            responses,
            n_plausible=5,
            seed=42,
        )

        true_theta = dichotomous_responses["theta"]
        y = true_theta + np.random.randn(len(true_theta)) * 0.5

        reg_result = plausible_value_regression(pvs, y)

        assert "coefficients" in reg_result
        assert "se" in reg_result
        assert "pvalues" in reg_result

    def test_pv_regression_significance(self, dichotomous_responses):
        """Test that regression detects true relationship."""
        responses = dichotomous_responses["responses"]
        result = fit_mirt(responses, model="2PL", max_iter=50)

        pvs = generate_plausible_values(
            result,
            responses,
            n_plausible=5,
            seed=42,
        )

        true_theta = dichotomous_responses["theta"]
        y = 2 * true_theta + np.random.randn(len(true_theta)) * 0.1

        reg_result = plausible_value_regression(pvs, y)

        assert reg_result["coefficients"][1] > 0


class TestPlausibleValueStatistics:
    """Tests for computing statistics with PVs."""

    def test_pv_mean(self, dichotomous_responses):
        """Test population mean estimation."""
        responses = dichotomous_responses["responses"]
        result = fit_mirt(responses, model="2PL", max_iter=50)

        pvs = generate_plausible_values(
            result,
            responses,
            n_plausible=5,
            seed=42,
        )

        stats = plausible_value_statistics(pvs, statistic="mean")

        assert "estimate" in stats
        assert "se" in stats
        assert abs(stats["estimate"]) < 1.0

    def test_pv_variance(self, dichotomous_responses):
        """Test population variance estimation."""
        responses = dichotomous_responses["responses"]
        result = fit_mirt(responses, model="2PL", max_iter=50)

        pvs = generate_plausible_values(
            result,
            responses,
            n_plausible=5,
            seed=42,
        )

        stats = plausible_value_statistics(pvs, statistic="variance")

        assert "estimate" in stats
        assert 0.5 < stats["estimate"] < 2.0

    def test_pv_percentile(self, dichotomous_responses):
        """Test percentile estimation."""
        responses = dichotomous_responses["responses"]
        result = fit_mirt(responses, model="2PL", max_iter=50)

        pvs = generate_plausible_values(
            result,
            responses,
            n_plausible=5,
            seed=42,
        )

        stats = plausible_value_statistics(pvs, statistic="percentile_50")

        assert "estimate" in stats
        assert abs(stats["estimate"]) < 1.0
