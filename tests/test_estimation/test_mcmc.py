"""Tests for MCMC estimation methods."""

import numpy as np
import pytest

from mirt import GibbsSampler, MCMCResult, MHRMEstimator, TwoParameterLogistic
from mirt.results.fit_result import FitResult


class TestMHRMEstimator:
    """Tests for MHRM estimator."""

    def test_init(self):
        """Test MHRM estimator initialization."""
        estimator = MHRMEstimator(n_cycles=50, burnin=20)
        assert estimator.n_cycles == 50
        assert estimator.burnin == 20

    def test_init_all_parameters(self):
        """Test MHRM estimator initialization with all parameters."""
        estimator = MHRMEstimator(
            n_cycles=100,
            burnin=30,
            n_chains=2,
            proposal_sd=0.3,
            gain_sequence="adaptive",
            verbose=False,
            use_rust=True,
            seed=42,
        )
        assert estimator.n_cycles == 100
        assert estimator.burnin == 30
        assert estimator.n_chains == 2
        assert estimator.proposal_sd == 0.3
        assert estimator.gain_sequence == "adaptive"
        assert estimator.seed == 42

    def test_fit(self, dichotomous_responses):
        """Test MHRM fitting."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        estimator = MHRMEstimator(
            n_cycles=30,
            burnin=10,
        )

        result = estimator.fit(model, dichotomous_responses["responses"])

        assert result.model._is_fitted
        assert "discrimination" in result.model._parameters
        assert "difficulty" in result.model._parameters

    def test_fit_returns_fit_result(self, dichotomous_responses):
        """Test that MHRM fit returns FitResult."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        estimator = MHRMEstimator(n_cycles=25, burnin=10, seed=42)

        result = estimator.fit(model, dichotomous_responses["responses"])

        assert isinstance(result, FitResult)
        assert result.converged is True
        assert result.n_iterations == 25

    def test_convergence_tracking(self, dichotomous_responses):
        """Test that convergence is tracked."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        estimator = MHRMEstimator(n_cycles=20, burnin=5)

        result = estimator.fit(model, dichotomous_responses["responses"])

        assert hasattr(result, "log_likelihood")

    def test_parameter_bounds(self, dichotomous_responses):
        """Test that estimated parameters are within reasonable bounds."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        estimator = MHRMEstimator(n_cycles=40, burnin=15, seed=42)

        result = estimator.fit(model, dichotomous_responses["responses"])

        disc = result.model.parameters["discrimination"]
        diff = result.model.parameters["difficulty"]

        assert np.all(disc > 0), "Discrimination should be positive"
        assert np.all(disc < 10), "Discrimination should be reasonable"
        assert np.all(np.abs(diff) < 10), "Difficulty should be reasonable"

    def test_seed_reproducibility(self, dichotomous_responses):
        """Test that same seed produces same results."""
        model1 = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        model2 = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])

        estimator1 = MHRMEstimator(n_cycles=30, burnin=10, seed=123)
        estimator2 = MHRMEstimator(n_cycles=30, burnin=10, seed=123)

        result1 = estimator1.fit(model1, dichotomous_responses["responses"])
        result2 = estimator2.fit(model2, dichotomous_responses["responses"])

        np.testing.assert_allclose(
            result1.model.parameters["discrimination"],
            result2.model.parameters["discrimination"],
            rtol=1e-5,
        )

    def test_gain_sequence_standard(self, dichotomous_responses):
        """Test standard gain sequence."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        estimator = MHRMEstimator(
            n_cycles=25, burnin=10, gain_sequence="standard", seed=42
        )

        result = estimator.fit(model, dichotomous_responses["responses"])
        assert result.model._is_fitted

    def test_gain_sequence_adaptive(self, dichotomous_responses):
        """Test adaptive gain sequence."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        estimator = MHRMEstimator(
            n_cycles=25, burnin=10, gain_sequence="adaptive", seed=42
        )

        result = estimator.fit(model, dichotomous_responses["responses"])
        assert result.model._is_fitted

    def test_standard_errors_computed(self, dichotomous_responses):
        """Test that standard errors are computed."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        estimator = MHRMEstimator(n_cycles=40, burnin=15, seed=42)

        result = estimator.fit(model, dichotomous_responses["responses"])

        assert "discrimination" in result.standard_errors
        assert "difficulty" in result.standard_errors
        assert (
            len(result.standard_errors["discrimination"])
            == dichotomous_responses["n_items"]
        )

    def test_aic_bic_computed(self, dichotomous_responses):
        """Test that AIC and BIC are computed."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        estimator = MHRMEstimator(n_cycles=30, burnin=10, seed=42)

        result = estimator.fit(model, dichotomous_responses["responses"])

        assert result.aic is not None
        assert result.bic is not None
        assert result.aic > 0
        assert result.bic > 0
        assert result.bic > result.aic  # BIC penalizes more for n > e^2

    def test_with_missing_data(self, responses_with_missing):
        """Test MHRM with missing data."""
        model = TwoParameterLogistic(n_items=responses_with_missing["n_items"])
        estimator = MHRMEstimator(n_cycles=30, burnin=10, seed=42)

        result = estimator.fit(model, responses_with_missing["responses"])

        assert result.model._is_fitted
        assert not np.any(np.isnan(result.model.parameters["discrimination"]))
        assert not np.any(np.isnan(result.model.parameters["difficulty"]))


class TestGibbsSampler:
    """Tests for Gibbs sampler."""

    def test_init(self):
        """Test Gibbs sampler initialization."""
        sampler = GibbsSampler(n_iter=100, burnin=20, thin=2)
        assert sampler.n_iter == 100
        assert sampler.burnin == 20
        assert sampler.thin == 2

    def test_init_all_parameters(self):
        """Test Gibbs sampler initialization with all parameters."""
        sampler = GibbsSampler(
            n_iter=200,
            burnin=50,
            thin=2,
            n_chains=2,
            priors={"discrimination": {"mean": 1.0, "sd": 0.5}},
            verbose=False,
            use_rust=True,
            seed=42,
        )
        assert sampler.n_iter == 200
        assert sampler.burnin == 50
        assert sampler.thin == 2
        assert sampler.n_chains == 2
        assert sampler.seed == 42
        assert "discrimination" in sampler.priors

    def test_fit(self, dichotomous_responses):
        """Test Gibbs sampler fitting."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        sampler = GibbsSampler(
            n_iter=50,
            burnin=15,
            thin=1,
        )

        result = sampler.fit(model, dichotomous_responses["responses"])

        assert isinstance(result, MCMCResult)
        assert result.model._is_fitted

    def test_mcmc_result_has_chains(self, dichotomous_responses):
        """Test MCMCResult has chains."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        sampler = GibbsSampler(n_iter=40, burnin=15, thin=1)

        result = sampler.fit(model, dichotomous_responses["responses"])

        assert hasattr(result, "chains")
        assert "discrimination" in result.chains
        assert "difficulty" in result.chains

    def test_chain_dimensions(self, dichotomous_responses):
        """Test that chain dimensions are correct."""
        n_iter = 50
        burnin = 20
        thin = 1
        n_items = dichotomous_responses["n_items"]

        model = TwoParameterLogistic(n_items=n_items)
        sampler = GibbsSampler(n_iter=n_iter, burnin=burnin, thin=thin, seed=42)

        result = sampler.fit(model, dichotomous_responses["responses"])

        expected_samples = (n_iter - burnin) // thin
        assert result.chains["discrimination"].shape[0] == expected_samples
        assert result.chains["discrimination"].shape[1] == n_items
        assert result.chains["difficulty"].shape[0] == expected_samples
        assert result.chains["difficulty"].shape[1] == n_items

    def test_seed_reproducibility(self, dichotomous_responses):
        """Test that same seed produces same results."""
        model1 = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        model2 = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])

        sampler1 = GibbsSampler(n_iter=40, burnin=15, thin=1, seed=456)
        sampler2 = GibbsSampler(n_iter=40, burnin=15, thin=1, seed=456)

        result1 = sampler1.fit(model1, dichotomous_responses["responses"])
        result2 = sampler2.fit(model2, dichotomous_responses["responses"])

        np.testing.assert_allclose(
            result1.chains["discrimination"],
            result2.chains["discrimination"],
            rtol=1e-5,
        )

    def test_posterior_mean_as_estimate(self, dichotomous_responses):
        """Test that model parameters are posterior means."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        sampler = GibbsSampler(n_iter=50, burnin=20, thin=1, seed=42)

        result = sampler.fit(model, dichotomous_responses["responses"])

        posterior_mean_disc = np.mean(result.chains["discrimination"], axis=0)
        np.testing.assert_allclose(
            result.model.parameters["discrimination"],
            posterior_mean_disc,
            rtol=1e-5,
        )

    def test_thinning_reduces_samples(self, dichotomous_responses):
        """Test that thinning reduces the number of samples."""
        model1 = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        model2 = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])

        sampler_thin1 = GibbsSampler(
            n_iter=60, burnin=20, thin=1, seed=42, use_rust=False
        )
        sampler_thin2 = GibbsSampler(
            n_iter=60, burnin=20, thin=2, seed=42, use_rust=False
        )

        result1 = sampler_thin1.fit(model1, dichotomous_responses["responses"])
        result2 = sampler_thin2.fit(model2, dichotomous_responses["responses"])

        assert result1.chains["discrimination"].shape[0] == 40  # (60-20)/1
        assert result2.chains["discrimination"].shape[0] == 20  # (60-20)/2

    def test_with_missing_data(self, responses_with_missing):
        """Test Gibbs sampler with missing data."""
        model = TwoParameterLogistic(n_items=responses_with_missing["n_items"])
        sampler = GibbsSampler(n_iter=40, burnin=15, thin=1, seed=42)

        result = sampler.fit(model, responses_with_missing["responses"])

        assert result.model._is_fitted
        assert not np.any(np.isnan(result.model.parameters["discrimination"]))

    def test_theta_chain_stored(self, dichotomous_responses):
        """Test that theta chain is stored."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        sampler = GibbsSampler(n_iter=40, burnin=15, thin=1, seed=42)

        result = sampler.fit(model, dichotomous_responses["responses"])

        assert "theta" in result.chains
        n_persons = dichotomous_responses["n_persons"]
        expected_samples = 40 - 15
        assert result.chains["theta"].shape[0] == expected_samples
        assert result.chains["theta"].shape[1] == n_persons

    def test_log_likelihood_chain_stored(self, dichotomous_responses):
        """Test that log-likelihood chain is stored."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        sampler = GibbsSampler(n_iter=40, burnin=15, thin=1, seed=42)

        result = sampler.fit(model, dichotomous_responses["responses"])

        assert "log_likelihood" in result.chains
        assert len(result.chains["log_likelihood"]) == 40 - 15


class TestMCMCResult:
    """Tests for MCMCResult class."""

    def test_summary(self, dichotomous_responses):
        """Test posterior summary."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        sampler = GibbsSampler(n_iter=50, burnin=20, thin=1)

        result = sampler.fit(model, dichotomous_responses["responses"])
        summary = result.summary()

        assert isinstance(summary, str)
        assert "MCMC" in summary or "Iteration" in summary

    def test_summary_contains_settings(self, dichotomous_responses):
        """Test that summary contains MCMC settings."""
        n_iter = 60
        burnin = 25
        thin = 1

        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        sampler = GibbsSampler(n_iter=n_iter, burnin=burnin, thin=thin, seed=42)

        result = sampler.fit(model, dichotomous_responses["responses"])
        summary = result.summary()

        assert str(n_iter) in summary
        assert str(burnin) in summary
        assert str(thin) in summary

    def test_summary_contains_log_likelihood(self, dichotomous_responses):
        """Test that summary contains log-likelihood."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        sampler = GibbsSampler(n_iter=50, burnin=20, thin=1, seed=42)

        result = sampler.fit(model, dichotomous_responses["responses"])
        summary = result.summary()

        assert "Log-likelihood" in summary or "LL" in summary

    def test_convergence_diagnostics(self, dichotomous_responses):
        """Test convergence diagnostics."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        sampler = GibbsSampler(n_iter=50, burnin=20, thin=1)

        result = sampler.fit(model, dichotomous_responses["responses"])

        assert hasattr(result, "rhat")
        assert hasattr(result, "ess")
        assert "discrimination" in result.rhat
        assert "difficulty" in result.rhat

    def test_rhat_values_reasonable(self, dichotomous_responses):
        """Test that R-hat values are reasonable (close to 1 for converged chains)."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        sampler = GibbsSampler(n_iter=80, burnin=30, thin=1, seed=42)

        result = sampler.fit(model, dichotomous_responses["responses"])

        for name, rhat_val in result.rhat.items():
            if not np.isnan(rhat_val):
                assert rhat_val > 0, f"R-hat for {name} should be positive"
                assert rhat_val < 2.0, f"R-hat for {name} should be reasonable"

    def test_ess_values_positive(self, dichotomous_responses):
        """Test that ESS values are positive."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        sampler = GibbsSampler(n_iter=60, burnin=25, thin=1, seed=42)

        result = sampler.fit(model, dichotomous_responses["responses"])

        for name, ess_val in result.ess.items():
            assert ess_val > 0, f"ESS for {name} should be positive"

    @pytest.mark.slow
    def test_dic_waic(self, dichotomous_responses):
        """Test DIC and WAIC computation (slow)."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        sampler = GibbsSampler(n_iter=100, burnin=30, thin=1)

        result = sampler.fit(model, dichotomous_responses["responses"])

        assert hasattr(result, "dic")
        assert hasattr(result, "waic")
        assert result.dic > 0
        assert result.waic > 0

    def test_dic_waic_basic(self, dichotomous_responses):
        """Test basic DIC and WAIC computation (non-slow version)."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        sampler = GibbsSampler(n_iter=50, burnin=20, thin=1, seed=42)

        result = sampler.fit(model, dichotomous_responses["responses"])

        assert result.dic > 0
        assert result.waic > 0
        assert abs(result.waic - result.dic) < result.dic

    def test_mcmc_result_attributes(self, dichotomous_responses):
        """Test that MCMCResult has all expected attributes."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        sampler = GibbsSampler(n_iter=50, burnin=20, thin=1, seed=42)

        result = sampler.fit(model, dichotomous_responses["responses"])

        assert hasattr(result, "model")
        assert hasattr(result, "chains")
        assert hasattr(result, "log_likelihood")
        assert hasattr(result, "dic")
        assert hasattr(result, "waic")
        assert hasattr(result, "rhat")
        assert hasattr(result, "ess")
        assert hasattr(result, "n_iterations")
        assert hasattr(result, "burnin")
        assert hasattr(result, "thin")

    def test_mcmc_result_settings_preserved(self, dichotomous_responses):
        """Test that MCMC settings are preserved in result."""
        n_iter = 55
        burnin = 22
        thin = 1

        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        sampler = GibbsSampler(n_iter=n_iter, burnin=burnin, thin=thin, seed=42)

        result = sampler.fit(model, dichotomous_responses["responses"])

        assert result.n_iterations == n_iter
        assert result.burnin == burnin
        assert result.thin == thin


class TestMCMCEdgeCases:
    """Tests for MCMC edge cases."""

    def test_small_dataset(self):
        """Test MCMC with small dataset."""
        rng = np.random.default_rng(42)
        n_persons, n_items = 20, 4

        theta = rng.standard_normal(n_persons)
        disc = np.ones(n_items)
        diff = rng.normal(0, 1, n_items)
        probs = 1 / (1 + np.exp(-disc * (theta[:, None] - diff)))
        responses = (rng.random((n_persons, n_items)) < probs).astype(int)

        model = TwoParameterLogistic(n_items=n_items)
        sampler = GibbsSampler(n_iter=30, burnin=10, thin=1, seed=42)

        result = sampler.fit(model, responses)

        assert result.model._is_fitted
        assert len(result.chains["discrimination"].shape) == 2

    def test_high_thinning(self, dichotomous_responses):
        """Test MCMC with high thinning rate."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        sampler = GibbsSampler(n_iter=100, burnin=20, thin=10, seed=42, use_rust=False)

        result = sampler.fit(model, dichotomous_responses["responses"])

        expected_samples = (100 - 20) // 10
        assert result.chains["discrimination"].shape[0] == expected_samples

    def test_minimal_burnin(self, dichotomous_responses):
        """Test MCMC with minimal burnin."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        sampler = GibbsSampler(n_iter=50, burnin=5, thin=1, seed=42)

        result = sampler.fit(model, dichotomous_responses["responses"])

        assert result.model._is_fitted
        assert result.chains["discrimination"].shape[0] == 45


class TestMCMCParameterRecovery:
    """Tests for MCMC parameter recovery."""

    def test_parameter_recovery_direction(self, dichotomous_responses):
        """Test that estimated parameters have correct sign/direction."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        sampler = GibbsSampler(n_iter=80, burnin=30, thin=1, seed=42)

        result = sampler.fit(model, dichotomous_responses["responses"])

        assert np.all(result.model.parameters["discrimination"] > 0)
        assert np.all(np.isfinite(result.model.parameters["difficulty"]))

    def test_posterior_uncertainty(self, dichotomous_responses):
        """Test that posterior chains show uncertainty."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        sampler = GibbsSampler(n_iter=60, burnin=25, thin=1, seed=42)

        result = sampler.fit(model, dichotomous_responses["responses"])

        disc_std = np.std(result.chains["discrimination"], axis=0)
        assert np.all(disc_std > 0), "Posterior should show variation"

        diff_std = np.std(result.chains["difficulty"], axis=0)
        assert np.all(diff_std > 0), "Posterior should show variation"
