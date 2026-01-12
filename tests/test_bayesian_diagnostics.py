"""Tests for Bayesian model diagnostics (PSIS-LOO, WAIC, PPC)."""

import numpy as np
import pytest

from mirt.diagnostics.bayesian import (
    PSISResult,
    WAICResult,
    compare_models,
    dic,
    psis_loo,
    waic,
)


class TestPSISLOO:
    """Tests for PSIS-LOO cross-validation."""

    def test_basic_computation(self):
        """Test basic PSIS-LOO computation."""
        rng = np.random.default_rng(42)

        n_samples = 1000
        n_obs = 50
        log_lik = rng.normal(-1.0, 0.5, (n_samples, n_obs))

        result = psis_loo(log_lik)

        assert isinstance(result, PSISResult)
        assert result.pointwise.shape == (n_obs,)
        assert result.pareto_k.shape == (n_obs,)
        assert result.looic == pytest.approx(-2 * result.elpd_loo)

    def test_psis_loo_dimensions(self):
        """Test PSIS-LOO handles different dimensions."""
        rng = np.random.default_rng(42)

        log_lik_1d = rng.normal(-1.0, 0.5, 100)
        result = psis_loo(log_lik_1d)
        assert result.pointwise.shape == (1,)

        log_lik_2d = rng.normal(-1.0, 0.5, (100, 20))
        result = psis_loo(log_lik_2d)
        assert result.pointwise.shape == (20,)

    def test_pareto_k_diagnostics(self):
        """Test Pareto k diagnostic values."""
        rng = np.random.default_rng(42)

        n_samples = 500
        n_obs = 30
        log_lik = rng.normal(-1.0, 0.3, (n_samples, n_obs))

        result = psis_loo(log_lik)

        assert np.all(result.pareto_k >= 0)
        assert result.n_high_k >= 0

    def test_summary(self):
        """Test summary generation."""
        rng = np.random.default_rng(42)
        log_lik = rng.normal(-1.0, 0.5, (500, 20))

        result = psis_loo(log_lik)
        summary = result.summary()

        assert "PSIS-LOO" in summary
        assert "elpd_loo" in summary
        assert "Pareto k" in summary


class TestWAIC:
    """Tests for WAIC computation."""

    def test_basic_computation(self):
        """Test basic WAIC computation."""
        rng = np.random.default_rng(42)

        n_samples = 1000
        n_obs = 50
        log_lik = rng.normal(-1.0, 0.5, (n_samples, n_obs))

        result = waic(log_lik)

        assert isinstance(result, WAICResult)
        assert result.pointwise.shape == (n_obs,)
        assert result.waic == pytest.approx(-2 * result.elpd_waic)
        assert result.p_waic >= 0

    def test_waic_dimensions(self):
        """Test WAIC handles different dimensions."""
        rng = np.random.default_rng(42)

        log_lik_1d = rng.normal(-1.0, 0.5, 100)
        result = waic(log_lik_1d)
        assert result.pointwise.shape == (1,)

    def test_p_waic_interpretation(self):
        """Test that p_waic is reasonable."""
        rng = np.random.default_rng(42)

        log_lik_low_var = rng.normal(-1.0, 0.1, (1000, 20))
        result_low = waic(log_lik_low_var)

        log_lik_high_var = rng.normal(-1.0, 1.0, (1000, 20))
        result_high = waic(log_lik_high_var)

        assert result_high.p_waic > result_low.p_waic

    def test_summary(self):
        """Test summary generation."""
        rng = np.random.default_rng(42)
        log_lik = rng.normal(-1.0, 0.5, (500, 20))

        result = waic(log_lik)
        summary = result.summary()

        assert "WAIC" in summary
        assert "elpd_waic" in summary
        assert "p_waic" in summary


class TestDIC:
    """Tests for DIC computation."""

    def test_basic_computation(self):
        """Test basic DIC computation."""
        rng = np.random.default_rng(42)

        log_lik_at_mean = -50.0
        log_lik_samples = rng.normal(-50.0, 5.0, 1000)

        dic_val, p_dic = dic(log_lik_at_mean, log_lik_samples)

        assert dic_val > 0
        assert p_dic >= 0

    def test_dic_relationship(self):
        """Test DIC = D(theta_bar) + 2*p_D relationship."""
        log_lik_at_mean = -100.0
        log_lik_samples = np.array([-95, -100, -105, -98, -102])

        dic_val, p_dic = dic(log_lik_at_mean, log_lik_samples)

        deviance_at_mean = -2 * log_lik_at_mean
        expected_dic = deviance_at_mean + 2 * p_dic

        assert dic_val == pytest.approx(expected_dic)


class TestModelComparison:
    """Tests for model comparison."""

    def test_compare_psis_results(self):
        """Test comparing PSIS-LOO results."""
        rng = np.random.default_rng(42)

        log_lik1 = rng.normal(-1.0, 0.5, (500, 30))
        log_lik2 = rng.normal(-1.5, 0.5, (500, 30))

        result1 = psis_loo(log_lik1)
        result2 = psis_loo(log_lik2)

        comparison = compare_models(result1, result2, names=["Model A", "Model B"])

        assert "Model Comparison" in comparison
        assert "Model A" in comparison
        assert "Model B" in comparison
        assert "LOOIC" in comparison

    def test_compare_waic_results(self):
        """Test comparing WAIC results."""
        rng = np.random.default_rng(42)

        log_lik1 = rng.normal(-1.0, 0.5, (500, 30))
        log_lik2 = rng.normal(-1.5, 0.5, (500, 30))

        result1 = waic(log_lik1)
        result2 = waic(log_lik2)

        comparison = compare_models(result1, result2, names=["Model A", "Model B"])

        assert "Model Comparison" in comparison
        assert "WAIC" in comparison


class TestIntegration:
    """Integration tests for Bayesian diagnostics."""

    def test_loo_waic_agreement(self):
        """Test that LOO and WAIC give similar results."""
        rng = np.random.default_rng(42)

        n_samples = 2000
        n_obs = 100
        log_lik = rng.normal(-1.0, 0.3, (n_samples, n_obs))

        loo_result = psis_loo(log_lik)
        waic_result = waic(log_lik)

        relative_diff = abs(loo_result.elpd_loo - waic_result.elpd_waic) / abs(
            loo_result.elpd_loo
        )
        assert relative_diff < 0.2

    def test_better_model_detection(self):
        """Test that better models have higher elpd."""
        rng = np.random.default_rng(42)

        n_samples = 1000
        n_obs = 50

        log_lik_good = rng.normal(-0.5, 0.2, (n_samples, n_obs))

        log_lik_bad = rng.normal(-2.0, 0.2, (n_samples, n_obs))

        loo_good = psis_loo(log_lik_good)
        loo_bad = psis_loo(log_lik_bad)

        assert loo_good.elpd_loo > loo_bad.elpd_loo
        assert loo_good.looic < loo_bad.looic


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
