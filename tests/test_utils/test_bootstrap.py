"""Tests for bootstrap standard errors and confidence intervals."""

import numpy as np

from mirt import bootstrap_ci, bootstrap_se, parametric_bootstrap


class TestBootstrapSE:
    """Tests for bootstrap standard errors."""

    def test_bootstrap_se(self, fitted_2pl_model, dichotomous_responses):
        """Test bootstrap SE computation."""
        responses = dichotomous_responses["responses"]

        se = bootstrap_se(
            fitted_2pl_model,
            responses,
            n_bootstrap=3,
            seed=42,
        )

        assert "discrimination" in se or "discrimination_se" in se.keys()
        assert "difficulty" in se or "difficulty_se" in se.keys()

    def test_bootstrap_se_positive(self, fitted_2pl_model, dichotomous_responses):
        """Test that bootstrap SEs are positive."""
        responses = dichotomous_responses["responses"]

        se = bootstrap_se(fitted_2pl_model, responses, n_bootstrap=3, seed=42)

        for key, values in se.items():
            if isinstance(values, np.ndarray):
                assert np.all(values >= 0)


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_bootstrap_ci_percentile(self, fitted_2pl_model, dichotomous_responses):
        """Test percentile bootstrap CI."""
        responses = dichotomous_responses["responses"]

        ci = bootstrap_ci(
            fitted_2pl_model,
            responses,
            n_bootstrap=3,
            method="percentile",
            alpha=0.05,
            seed=42,
        )

        assert "discrimination" in ci or "difficulty" in ci
        for key, value in ci.items():
            if isinstance(value, tuple):
                assert len(value) == 2

    def test_bootstrap_ci_basic(self, fitted_2pl_model, dichotomous_responses):
        """Test basic bootstrap CI."""
        responses = dichotomous_responses["responses"]

        ci = bootstrap_ci(
            fitted_2pl_model,
            responses,
            n_bootstrap=3,
            method="basic",
            alpha=0.05,
            seed=42,
        )

        assert ci is not None

    def test_bootstrap_ci_bca(self, fitted_2pl_model, dichotomous_responses):
        """Test BCa bootstrap CI."""
        responses = dichotomous_responses["responses"]

        ci = bootstrap_ci(
            fitted_2pl_model,
            responses,
            n_bootstrap=3,
            method="BCa",
            alpha=0.05,
            seed=42,
        )

        assert ci is not None


class TestParametricBootstrap:
    """Tests for parametric bootstrap."""

    def test_parametric_bootstrap(self, fitted_2pl_model):
        """Test parametric bootstrap."""
        bootstrap_results = parametric_bootstrap(
            fitted_2pl_model,
            n_bootstrap=3,
            seed=42,
        )

        assert isinstance(bootstrap_results, dict)
        assert "discrimination" in bootstrap_results
        assert "difficulty" in bootstrap_results

    def test_parametric_bootstrap_variance(self, fitted_2pl_model):
        """Test parametric bootstrap variance estimation."""
        bootstrap_results = parametric_bootstrap(
            fitted_2pl_model,
            n_bootstrap=3,
            seed=42,
        )

        disc_estimates = bootstrap_results["discrimination"]

        variances = np.var(disc_estimates, axis=0)
        assert np.all(variances >= 0)
