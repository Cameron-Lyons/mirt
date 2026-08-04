"""Tests for plausible value generation."""

import numpy as np
import pytest

from mirt import (
    combine_plausible_values,
    generate_plausible_values,
    plausible_value_regression,
    plausible_value_statistics,
)


class TestGeneratePlausibleValues:
    """Tests for plausible value generation."""

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"n_plausible": 0}, "n_plausible"),
            ({"n_plausible": 1.5}, "n_plausible"),
            ({"n_quadpts": 0}, "n_quadpts"),
            ({"method": "mcmc", "n_iter": 0}, "n_iter"),
        ],
    )
    def test_generation_parameters_must_be_positive_integers(
        self,
        fitted_2pl_model,
        dichotomous_responses,
        kwargs,
        message,
    ):
        """Reject invalid sample, quadrature, and iteration counts."""
        with pytest.raises(ValueError, match=message):
            generate_plausible_values(
                fitted_2pl_model,
                dichotomous_responses["responses"],
                **kwargs,
            )

    def test_requires_fitted_model(self):
        """Plausible values require estimated item parameters."""
        from mirt.models.dichotomous import TwoParameterLogistic

        model = TwoParameterLogistic(n_items=2)

        with pytest.raises(ValueError, match="fitted"):
            generate_plausible_values(model, np.array([[0, 1]]))

    def test_validates_response_shape_and_codes(self, fitted_2pl_model):
        """Reject malformed and non-dichotomous response matrices."""
        with pytest.raises(ValueError, match="2D"):
            generate_plausible_values(fitted_2pl_model, np.array([0, 1]))
        with pytest.raises(ValueError, match="coded as 0 or 1"):
            generate_plausible_values(
                fitted_2pl_model,
                np.full((1, fitted_2pl_model.model.n_items), 2),
            )

    def test_validates_polytomous_response_codes(self):
        """Reject category codes outside an item's configured range."""
        from mirt.models.polytomous import GradedResponseModel

        model = GradedResponseModel(n_items=2, n_categories=[3, 4])
        model._is_fitted = True

        with pytest.raises(ValueError, match="item 0 must be below 3"):
            generate_plausible_values(model, np.array([[3, 0]]))

    def test_generate_pv_posterior(self, fitted_2pl_model, dichotomous_responses):
        """Test posterior sampling method."""
        responses = dichotomous_responses["responses"]

        pvs = generate_plausible_values(
            fitted_2pl_model,
            responses,
            n_plausible=3,
            method="posterior",
            seed=42,
        )

        n_persons = responses.shape[0]
        assert pvs.shape == (n_persons, 1, 3)

    def test_posterior_uses_single_batch_likelihood(
        self,
        fitted_2pl_model,
        dichotomous_responses,
        monkeypatch,
    ):
        """Posterior sampling evaluates every respondent in one batch."""
        model = fitted_2pl_model.model
        original = model.log_likelihood_batch
        call_count = 0

        def counted_batch(responses, theta):
            nonlocal call_count
            call_count += 1
            return original(responses, theta)

        monkeypatch.setattr(model, "log_likelihood_batch", counted_batch)

        generate_plausible_values(
            model,
            dichotomous_responses["responses"],
            n_plausible=3,
            method="posterior",
            seed=42,
        )

        assert call_count == 1

    def test_generate_pv_mcmc(self, fitted_2pl_model_small):
        """Test MCMC sampling method."""
        responses = fitted_2pl_model_small["responses"]
        result = fitted_2pl_model_small["result"]

        pvs = generate_plausible_values(
            result,
            responses,
            n_plausible=3,
            method="mcmc",
            seed=42,
        )

        n_persons = responses.shape[0]
        assert pvs.shape == (n_persons, 1, 3)
        assert not np.array_equal(pvs[:, :, 0], pvs[:, :, -1])

    def test_mcmc_reuses_current_log_density(
        self,
        fitted_2pl_model_small,
        monkeypatch,
    ):
        """MCMC recomputes only proposed states after initialization."""
        model = fitted_2pl_model_small["result"].model
        responses = fitted_2pl_model_small["responses"][:2]
        original = model.log_likelihood
        call_count = 0

        def counted_likelihood(responses, theta):
            nonlocal call_count
            call_count += 1
            return original(responses, theta)

        monkeypatch.setattr(model, "log_likelihood", counted_likelihood)

        generate_plausible_values(
            model,
            responses,
            n_plausible=3,
            method="mcmc",
            n_iter=4,
            seed=42,
        )

        assert call_count == responses.shape[0] * (1 + 3 * 4)

    def test_pv_variability(self, fitted_2pl_model, dichotomous_responses):
        """Test that PVs show appropriate variability."""
        responses = dichotomous_responses["responses"]

        pvs = generate_plausible_values(
            fitted_2pl_model,
            responses,
            n_plausible=3,
            method="posterior",
            seed=42,
        )

        pv_variance = np.var(pvs, axis=2)
        assert np.mean(pv_variance) > 0

    def test_pv_correlation_with_ability(self, fitted_2pl_model, dichotomous_responses):
        """Test that PVs correlate with true ability."""
        responses = dichotomous_responses["responses"]
        true_theta = dichotomous_responses["theta"]

        pvs = generate_plausible_values(
            fitted_2pl_model,
            responses,
            n_plausible=3,
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

    def test_pv_regression(self, fitted_2pl_model, dichotomous_responses):
        """Test regression with PVs as predictor."""
        responses = dichotomous_responses["responses"]

        pvs = generate_plausible_values(
            fitted_2pl_model,
            responses,
            n_plausible=3,
            seed=42,
        )

        true_theta = dichotomous_responses["theta"]
        rng = np.random.default_rng(42)
        y = true_theta + rng.standard_normal(len(true_theta)) * 0.5

        reg_result = plausible_value_regression(pvs, y)

        assert "coefficients" in reg_result
        assert "se" in reg_result
        assert "pvalues" in reg_result

    def test_pv_regression_significance(self, fitted_2pl_model, dichotomous_responses):
        """Test that regression detects true relationship."""
        responses = dichotomous_responses["responses"]

        pvs = generate_plausible_values(
            fitted_2pl_model,
            responses,
            n_plausible=3,
            seed=42,
        )

        true_theta = dichotomous_responses["theta"]
        rng = np.random.default_rng(42)
        y = 2 * true_theta + rng.standard_normal(len(true_theta)) * 0.1

        reg_result = plausible_value_regression(pvs, y)

        assert reg_result["coefficients"][1] > 0


class TestPlausibleValueStatistics:
    """Tests for computing statistics with PVs."""

    def test_pv_mean(self, fitted_2pl_model, dichotomous_responses):
        """Test population mean estimation."""
        responses = dichotomous_responses["responses"]

        pvs = generate_plausible_values(
            fitted_2pl_model,
            responses,
            n_plausible=3,
            seed=42,
        )

        stats = plausible_value_statistics(pvs, statistic="mean")

        assert "estimate" in stats
        assert "se" in stats
        assert abs(stats["estimate"]) < 1.0

    def test_pv_variance(self, fitted_2pl_model, dichotomous_responses):
        """Test population variance estimation."""
        responses = dichotomous_responses["responses"]

        pvs = generate_plausible_values(
            fitted_2pl_model,
            responses,
            n_plausible=3,
            seed=42,
        )

        stats = plausible_value_statistics(pvs, statistic="variance")

        assert "estimate" in stats
        assert 0.5 < stats["estimate"] < 2.0

    def test_pv_percentile(self, fitted_2pl_model, dichotomous_responses):
        """Test percentile estimation."""
        responses = dichotomous_responses["responses"]

        pvs = generate_plausible_values(
            fitted_2pl_model,
            responses,
            n_plausible=3,
            seed=42,
        )

        stats = plausible_value_statistics(pvs, statistic="percentile_50")

        assert "estimate" in stats
        assert abs(stats["estimate"]) < 1.0
