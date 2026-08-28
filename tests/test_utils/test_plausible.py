"""Tests for plausible value generation."""

import numpy as np
import pytest

from mirt import (
    combine_plausible_values,
    generate_plausible_values,
    plausible_value_regression,
    plausible_value_statistics,
)
from mirt.utils.plausible import _inverse_cdf_rows


class TestGeneratePlausibleValues:
    """Tests for plausible value generation."""

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"n_plausible": 0}, "n_plausible"),
            ({"n_plausible": 1.5}, "n_plausible"),
            ({"n_quadpts": 0}, "n_quadpts"),
            ({"method": "mcmc", "n_iter": 0}, "n_iter"),
            ({"method": "mcmc", "burn_in": -1}, "burn_in"),
            ({"method": "mcmc", "proposal_scale": 0}, "proposal_scale"),
            ({"method": "mcmc", "proposal_scale": np.inf}, "proposal_scale"),
            ({"method": "posterior", "chunk_size": 0}, "chunk_size"),
            ({"method": "mcmc", "chunk_size": 0}, "chunk_size"),
        ],
    )
    def test_generation_parameters_are_validated(
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

    def test_posterior_batches_people_without_changing_seeded_draws(
        self,
        fitted_2pl_model,
        dichotomous_responses,
        monkeypatch,
    ):
        """Posterior row batching is reproducible across chunk sizes."""
        model = fitted_2pl_model.model
        responses = dichotomous_responses["responses"][:17]
        expected = generate_plausible_values(
            model,
            responses,
            n_plausible=4,
            method="posterior",
            n_quadpts=15,
            chunk_size=len(responses),
            seed=42,
        )

        original = model.log_likelihood_batch
        batch_sizes = []

        def counted_batch(responses, theta):
            batch_sizes.append(len(responses))
            return original(responses, theta)

        monkeypatch.setattr(model, "log_likelihood_batch", counted_batch)
        actual = generate_plausible_values(
            model,
            responses,
            n_plausible=4,
            method="posterior",
            n_quadpts=15,
            chunk_size=6,
            seed=42,
        )

        np.testing.assert_array_equal(actual, expected)
        assert batch_sizes == [6, 6, 5]

    def test_posterior_rejects_malformed_batch_likelihood(
        self,
        fitted_2pl_model,
        dichotomous_responses,
        monkeypatch,
    ):
        """Posterior generation checks the model's batch likelihood contract."""
        model = fitted_2pl_model.model
        monkeypatch.setattr(
            model,
            "log_likelihood_batch",
            lambda responses, theta: np.zeros(len(responses)),
        )

        with pytest.raises(ValueError, match="quadrature node"):
            generate_plausible_values(
                model,
                dichotomous_responses["responses"][:3],
                n_plausible=2,
                method="posterior",
                seed=42,
            )

    def test_inverse_cdf_search_matches_reference_scan(self):
        """Flattened row search agrees with direct categorical scanning."""
        cumulative = np.array(
            [
                [0.1, 0.4, 1.0],
                [0.0, 0.25, 1.0],
                [0.7, 0.9, 1.0],
            ]
        )
        uniforms = np.array(
            [
                [0.0, 0.1, 0.4, 0.999],
                [0.0, 0.2, 0.25, 0.8],
                [0.0, 0.7, 0.95, 0.999],
            ]
        )
        expected = np.sum(
            uniforms[:, :, None] > cumulative[:, None, :],
            axis=2,
        )

        actual = _inverse_cdf_rows(cumulative, uniforms)

        np.testing.assert_array_equal(actual, expected)

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

    def test_mcmc_batches_people_and_reuses_current_log_density(
        self,
        fitted_2pl_model_small,
        monkeypatch,
    ):
        """MCMC uses one paired likelihood call per iteration and batch."""
        model = fitted_2pl_model_small["result"].model
        responses = fitted_2pl_model_small["responses"][:5]
        original = model.log_likelihood
        batch_sizes = []

        def counted_likelihood(responses, theta):
            batch_sizes.append(len(responses))
            return original(responses, theta)

        monkeypatch.setattr(model, "log_likelihood", counted_likelihood)

        generate_plausible_values(
            model,
            responses,
            n_plausible=3,
            method="mcmc",
            n_iter=4,
            burn_in=2,
            chunk_size=2,
            seed=42,
        )

        n_batches = 3
        assert len(batch_sizes) == n_batches * (1 + 2 + 3 * 4)
        assert batch_sizes == [2, 2, 1] * (1 + 2 + 3 * 4)

    def test_mcmc_controls_are_reproducible(
        self,
        fitted_2pl_model_small,
    ):
        """Burn-in and proposal tuning retain seeded reproducibility."""
        model = fitted_2pl_model_small["result"].model
        responses = fitted_2pl_model_small["responses"][:8]
        kwargs = {
            "n_plausible": 3,
            "method": "mcmc",
            "n_iter": 5,
            "burn_in": 4,
            "proposal_scale": 0.35,
            "chunk_size": 3,
            "seed": 42,
        }

        first = generate_plausible_values(model, responses, **kwargs)
        second = generate_plausible_values(model, responses, **kwargs)
        unchunked = generate_plausible_values(
            model,
            responses,
            **{**kwargs, "chunk_size": len(responses)},
        )

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, unchunked)
        assert np.isfinite(first).all()
        assert not np.array_equal(first[:, :, 0], first[:, :, -1])

    def test_mcmc_supports_paired_polytomous_likelihoods(self):
        """Batched sampling supports fitted ordinal item models."""
        from mirt.models.polytomous import GradedResponseModel

        model = GradedResponseModel(n_items=3, n_categories=[3, 4, 3])
        model._is_fitted = True
        responses = np.array(
            [[0, 1, 2], [2, 3, 1], [1, -1, 0], [2, 0, 2]],
            dtype=int,
        )

        pvs = generate_plausible_values(
            model,
            responses,
            n_plausible=2,
            method="mcmc",
            n_iter=3,
            burn_in=2,
            seed=7,
        )

        assert pvs.shape == (4, 1, 2)
        assert np.isfinite(pvs).all()

    def test_mcmc_supports_multidimensional_likelihoods(self):
        """Batched sampling advances every latent dimension."""
        from mirt.models.dichotomous import TwoParameterLogistic

        model = TwoParameterLogistic(n_items=3, n_factors=2)
        model._is_fitted = True
        responses = np.array([[0, 1, 0], [1, 1, 0], [1, -1, 1], [0, 0, 1]])

        pvs = generate_plausible_values(
            model,
            responses,
            n_plausible=3,
            method="mcmc",
            n_iter=4,
            burn_in=3,
            seed=11,
        )

        assert pvs.shape == (4, 2, 3)
        assert np.isfinite(pvs).all()
        assert np.any(pvs[:, 0, :] != 0.0)
        assert np.any(pvs[:, 1, :] != 0.0)

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
