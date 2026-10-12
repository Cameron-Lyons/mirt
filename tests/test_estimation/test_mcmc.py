"""Tests for MCMC estimation methods."""

import numpy as np
import pytest

from mirt import GibbsSampler, MCMCResult, MHRMEstimator, TwoParameterLogistic
from mirt.estimation.priors import LogNormalPrior
from mirt.exceptions import MirtEstimationError
from mirt.results.fit_result import FitResult


def _manual_posterior_waic(
    model, responses: np.ndarray, chains: dict[str, np.ndarray]
) -> float:
    """Evaluate the pointwise WAIC definition with paired posterior draws."""
    evaluation_model = model.copy()
    pointwise = []
    for sample_index, theta in enumerate(chains["theta"]):
        for name in model.parameters:
            evaluation_model._parameters[name] = chains[name][sample_index]
        pointwise.append(evaluation_model.log_likelihood(responses, theta))

    log_likelihood = np.stack(pointwise)
    maximum = np.max(log_likelihood, axis=0)
    lppd = np.sum(
        maximum + np.log(np.mean(np.exp(log_likelihood - maximum[None, :]), axis=0))
    )
    effective_parameters = np.sum(np.var(log_likelihood, axis=0))
    return float(-2.0 * (lppd - effective_parameters))


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

    @pytest.mark.slow
    def test_fit_longer_chain(self, dichotomous_responses):
        """Longer MHRM chain for the weekly slow suite."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        estimator = MHRMEstimator(n_cycles=200, burnin=50, seed=7)
        result = estimator.fit(model, dichotomous_responses["responses"])
        assert np.isfinite(result.log_likelihood)

    def test_fit_returns_fit_result(self, dichotomous_responses):
        """Test that MHRM fit returns FitResult."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        estimator = MHRMEstimator(n_cycles=25, burnin=10, seed=42)

        # Fifteen decreasing-gain cycles are too few for the stopping rule.
        with pytest.warns(RuntimeWarning, match="increase n_cycles"):
            result = estimator.fit(model, dichotomous_responses["responses"])

        assert isinstance(result, FitResult)
        assert result.converged is False
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
        assert result.bic > result.aic

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
            priors={"discrimination": LogNormalPrior(0.0, 0.5)},
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

        assert result1.chains["discrimination"].shape[0] == 40
        assert result2.chains["discrimination"].shape[0] == 20

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

    @staticmethod
    def _synthetic_result() -> MCMCResult:
        """Build a deterministic result for posterior-statistic tests."""
        model = TwoParameterLogistic(n_items=2)
        chains = {
            "discrimination": np.array(
                [[0.8, 1.2], [1.0, 1.4], [1.2, 1.6], [1.4, 1.8]]
            ),
            "difficulty": np.array([[-1.0, 0.0], [-0.5, 0.5], [0.0, 1.0], [0.5, 1.5]]),
            "theta": np.arange(24, dtype=float).reshape(4, 3, 2),
            "log_likelihood": np.array([-12.0, -11.0, -10.0, -9.0]),
        }
        return MCMCResult(
            model=model,
            chains=chains,
            log_likelihood=-10.5,
            dic=25.0,
            waic=24.0,
            rhat={"discrimination": 1.01, "difficulty": 1.02},
            ess={"discrimination": 75.0, "difficulty": 64.5},
            n_iterations=100,
            burnin=20,
            thin=2,
        )

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

    def test_summary_contains_effective_sample_sizes(self):
        """Test that text summaries expose both convergence diagnostics."""
        summary = self._synthetic_result().summary()

        assert "Effective sample size:" in summary
        assert "discrimination: 75.0" in summary
        assert "difficulty: 64.5" in summary

    def test_posterior_summary_preserves_parameter_shapes(self):
        """Posterior summaries reduce only the leading draw dimension."""
        result = self._synthetic_result()

        summary = result.posterior_summary(
            credible_level=0.8,
            parameters=["discrimination", "theta"],
        )

        assert set(summary) == {"discrimination", "theta"}
        for name in summary:
            chain = result.chains[name]
            expected = np.quantile(chain, [0.1, 0.5, 0.9], axis=0)
            np.testing.assert_allclose(summary[name]["mean"], np.mean(chain, axis=0))
            np.testing.assert_allclose(summary[name]["std"], np.std(chain, axis=0))
            np.testing.assert_allclose(summary[name]["ci_lower"], expected[0])
            np.testing.assert_allclose(summary[name]["median"], expected[1])
            np.testing.assert_allclose(summary[name]["ci_upper"], expected[2])
        assert np.shape(summary["theta"]["mean"]) == (3, 2)

    def test_posterior_summary_defaults_to_every_stored_chain(self):
        """The default summary includes scalar and multidimensional chains."""
        result = self._synthetic_result()

        summary = result.posterior_summary()

        assert set(summary) == set(result.chains)
        assert summary["log_likelihood"]["mean"] == pytest.approx(-10.5)
        assert np.shape(summary["log_likelihood"]["ci_lower"]) == ()

    def test_credible_intervals_accept_single_chain_name(self):
        """A single chain can be selected without wrapping its name."""
        result = self._synthetic_result()

        intervals = result.credible_intervals(
            credible_level=0.5,
            parameters="difficulty",
        )

        assert set(intervals) == {"difficulty"}
        expected = np.quantile(result.chains["difficulty"], [0.25, 0.75], axis=0)
        np.testing.assert_allclose(intervals["difficulty"][0], expected[0])
        np.testing.assert_allclose(intervals["difficulty"][1], expected[1])

    @pytest.mark.parametrize(
        "credible_level",
        [0.0, 1.0, -0.1, np.nan, np.inf, True, "0.95"],
    )
    def test_posterior_summary_rejects_invalid_credible_levels(self, credible_level):
        """Credible levels must be finite probabilities inside the unit interval."""
        result = self._synthetic_result()

        with pytest.raises(ValueError, match="credible_level"):
            result.posterior_summary(credible_level=credible_level)

    def test_posterior_summary_rejects_unknown_or_invalid_chains(self):
        """Selection errors and malformed draws fail with clear messages."""
        result = self._synthetic_result()
        with pytest.raises(ValueError, match="unknown posterior chain: guessing"):
            result.posterior_summary(parameters="guessing")
        with pytest.raises(ValueError, match="parameters must be a chain name"):
            result.posterior_summary(parameters=42)

        result.chains["difficulty"] = np.full((4, 2), np.nan)
        with pytest.raises(ValueError, match="must contain only finite values"):
            result.posterior_summary(parameters="difficulty")

    def test_posterior_summary_rejects_misaligned_draw_counts(self):
        """Joint summaries require chains based on the same posterior draws."""
        result = self._synthetic_result()
        result.chains["difficulty"] = result.chains["difficulty"][:-1]

        with pytest.raises(ValueError, match="equal draw counts"):
            result.posterior_summary(parameters=["discrimination", "difficulty"])

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
        """R-hat values are finite and near one for a longer chain."""
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        sampler = GibbsSampler(n_iter=2000, burnin=400, thin=1, seed=42, use_rust=False)

        result = sampler.fit(model, dichotomous_responses["responses"])

        for name, rhat_val in result.rhat.items():
            assert np.isfinite(rhat_val), f"R-hat for {name} should be finite"
            assert 0.9 < rhat_val < 1.2, f"R-hat for {name} should be reasonable"

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


class TestWAICComputation:
    """Tests for stable posterior-draw WAIC evaluation."""

    def test_fit_reports_waic_from_paired_posterior_draws(self, dichotomous_responses):
        responses = dichotomous_responses["responses"]
        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        sampler = GibbsSampler(
            n_iter=30,
            burnin=10,
            thin=2,
            use_rust=False,
            seed=824,
        )

        result = sampler.fit(model, responses)
        expected = _manual_posterior_waic(result.model, responses, result.chains)

        assert result.waic == pytest.approx(expected, rel=1e-13, abs=1e-13)

    def test_matches_paired_definition_with_one_likelihood_call_per_draw(
        self, monkeypatch
    ):
        rng = np.random.default_rng(20260827)
        n_samples, n_persons, n_items = 7, 11, 5
        responses = rng.integers(0, 2, size=(n_persons, n_items))
        model = TwoParameterLogistic(n_items=n_items)
        model._initialize_parameters()
        model._parameters["discrimination"] = np.linspace(0.8, 1.4, n_items)
        model._parameters["difficulty"] = np.linspace(-0.7, 0.6, n_items)
        original_parameters = {
            name: values.copy() for name, values in model.parameters.items()
        }
        chains = {
            "theta": rng.normal(size=(n_samples, n_persons, 1)),
            "discrimination": rng.uniform(0.6, 1.8, size=(n_samples, n_items)),
            "difficulty": rng.normal(0.0, 0.9, size=(n_samples, n_items)),
            "log_likelihood": np.zeros(n_samples),
        }
        expected = _manual_posterior_waic(model, responses, chains)

        original_log_likelihood = TwoParameterLogistic.log_likelihood
        call_count = 0

        def counted_log_likelihood(self, response_values, theta_values):
            nonlocal call_count
            call_count += 1
            return original_log_likelihood(self, response_values, theta_values)

        monkeypatch.setattr(
            TwoParameterLogistic,
            "log_likelihood",
            counted_log_likelihood,
        )
        sampler = GibbsSampler(n_iter=10, burnin=2, use_rust=False)
        actual = sampler._compute_waic(chains, model, responses)

        assert actual == pytest.approx(expected, rel=1e-13, abs=1e-13)
        assert call_count == n_samples
        for name, values in original_parameters.items():
            np.testing.assert_array_equal(model.parameters[name], values)

    def test_remains_finite_when_direct_exponentiation_underflows(self):
        n_samples, n_persons, n_items = 3, 2, 1200
        responses = np.ones((n_persons, n_items), dtype=int)
        model = TwoParameterLogistic(n_items=n_items)
        model._initialize_parameters()
        model._parameters["discrimination"] = np.ones(n_items)
        model._parameters["difficulty"] = np.zeros(n_items)
        theta = np.zeros((n_samples, n_persons, 1))
        chains = {
            "theta": theta,
            "discrimination": np.ones((n_samples, n_items)),
            "difficulty": np.zeros((n_samples, n_items)),
            "log_likelihood": np.zeros(n_samples),
        }
        pointwise = model.log_likelihood(responses, theta[0])
        assert np.all(np.exp(pointwise) == 0.0)

        sampler = GibbsSampler(n_iter=10, burnin=2, use_rust=False)
        actual = sampler._compute_waic(chains, model, responses)

        assert np.isfinite(actual)
        assert actual == pytest.approx(float(-2.0 * np.sum(pointwise)))

    def test_rejects_misaligned_parameter_chains(self):
        model = TwoParameterLogistic(n_items=2)
        model._initialize_parameters()
        responses = np.zeros((3, 2), dtype=int)
        chains = {
            "theta": np.zeros((4, 3, 1)),
            "discrimination": np.ones((4, 2)),
            "difficulty": np.zeros((3, 2)),
            "log_likelihood": np.zeros(4),
        }

        sampler = GibbsSampler(n_iter=10, burnin=2, use_rust=False)
        with pytest.raises(
            MirtEstimationError,
            match="parameter chain has an unexpected shape",
        ):
            sampler._compute_waic(chains, model, responses)


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


class TestGibbsChains:
    """Several Gibbs chains, serially or in spawned worker processes."""

    @staticmethod
    def _responses() -> np.ndarray:
        rng = np.random.default_rng(0)
        return (rng.random((40, 3)) < 0.6).astype(int)

    def test_parallel_chains_match_serial_chains(self, monkeypatch):
        import mirt.utils._parallel as parallel

        pools = []
        original = parallel._process_pool

        def record(max_workers, *args, **kwargs):
            pools.append(max_workers)
            return original(max_workers, *args, **kwargs)

        monkeypatch.setattr(parallel, "_process_pool", record)
        results = {}
        for parallel_chains in (False, True):
            sampler = GibbsSampler(
                n_iter=12,
                burnin=4,
                n_chains=2,
                seed=5,
                use_rust=False,
                parallel_chains=parallel_chains,
            )
            results[parallel_chains] = sampler.fit(
                TwoParameterLogistic(3), self._responses()
            )

        # Workers come from the spawn pool that carries the backend choice.
        assert pools == [2]
        reference = GibbsSampler(n_iter=12, burnin=4, use_rust=False)
        chains = [
            reference._run_single_chain(
                TwoParameterLogistic(3), self._responses(), 40, s, True
            )
            for s in (5, 1005)
        ]
        for name, chain in results[False].chains.items():
            np.testing.assert_array_equal(results[True].chains[name], chain)
            np.testing.assert_array_equal(
                chain, np.concatenate([single[name] for single in chains])
            )

    def test_2pl_chains_do_not_depend_on_use_rust(self):
        responses = self._responses()
        # A native 2PL kernel used to replace these chains with draws from a
        # sampler that clipped its proposals to the box.
        results = [
            GibbsSampler(n_iter=12, burnin=4, n_chains=2, seed=5, use_rust=flag).fit(
                TwoParameterLogistic(3), responses
            )
            for flag in (True, False)
        ]
        for name, chain in results[1].chains.items():
            np.testing.assert_array_equal(results[0].chains[name], chain)

    @pytest.mark.parametrize("n_chains", [0, 1.5, True])
    def test_invalid_chain_counts_are_rejected(self, n_chains):
        from mirt.exceptions import MirtValidationError

        with pytest.raises(MirtValidationError, match="n_chains"):
            GibbsSampler(n_chains=n_chains)

    def test_parallel_chains_reject_unpicklable_models(self):
        from mirt.exceptions import MirtValidationError

        class LocalTwoPL(TwoParameterLogistic):
            pass

        sampler = GibbsSampler(
            n_iter=4, burnin=1, n_chains=2, use_rust=False, parallel_chains=True
        )
        with pytest.raises(MirtValidationError, match="parallel_chains=False"):
            sampler.fit(LocalTwoPL(3), self._responses())


def _reference_rank_rhat(draws: np.ndarray) -> np.ndarray:
    """Rank-normalized split R-hat of each column, one column at a time."""
    from scipy.special import ndtri
    from scipy.stats import rankdata

    n_chains, n_draws, n_values = draws.shape
    half = n_draws // 2
    values = np.empty(n_values)
    for column in range(n_values):
        split = np.concatenate(
            [draws[:, :half, column], draws[:, n_draws - half :, column]]
        )

        def rhat(x):
            ranks = rankdata(x.ravel()).reshape(x.shape)
            z = ndtri((ranks - 0.375) / (x.size + 0.25))
            within = np.mean(np.var(z, axis=1, ddof=1))
            between = half * np.var(np.mean(z, axis=1), ddof=1)
            return np.sqrt((between / within + half - 1) / half)

        values[column] = max(rhat(split), rhat(np.abs(split - np.median(split))))
    return values


def _stacked_result(draws: np.ndarray, model=None) -> MCMCResult:
    """Wrap ``(n_chains, n_draws, n_items)`` discrimination draws in a result."""
    from mirt.estimation.mcmc import _diagnose_chains

    n_chains, n_draws, n_items = draws.shape
    if model is None:
        model = TwoParameterLogistic(n_items=n_items)
    chains = {"discrimination": draws.reshape(n_chains * n_draws, n_items)}
    _, rhat, ess, ess_tail = _diagnose_chains(chains, model, n_chains)
    return MCMCResult(
        model=model,
        chains=chains,
        log_likelihood=0.0,
        dic=0.0,
        waic=0.0,
        rhat=rhat,
        ess=ess,
        n_iterations=n_draws,
        burnin=0,
        thin=1,
        n_chains=n_chains,
        ess_tail=ess_tail,
    )


class TestConvergenceDiagnostics:
    """Rank-normalized split R-hat and bulk/tail ESS (Vehtari et al., 2021)."""

    def test_rhat_is_assessed_per_coordinate_within_chains(self):
        # Two coordinates disagree between chains in opposite directions, so
        # their item average looks converged. It was what R-hat assessed, on
        # the stacked draws cut in two halves.
        rng = np.random.default_rng(3)
        draws = rng.normal(0.0, 0.1, (2, 400, 2))
        draws[0] += [1.0, -1.0]
        draws[1] += [-1.0, 1.0]
        result = _stacked_result(draws)

        per_coordinate = result.convergence_diagnostics()["discrimination"]["rhat"]
        # Rank normalization bounds R-hat; separated chains of 200 split
        # draws give about 1.8.
        assert np.all(per_coordinate > 1.5)
        assert result.rhat["discrimination"] == pytest.approx(per_coordinate.max())
        assert result.converged is False

    def test_rhat_matches_a_columnwise_reference(self):
        rng = np.random.default_rng(4)
        draws = np.cumsum(rng.normal(size=(3, 301, 4)), axis=1) * 0.05
        draws += rng.normal(size=draws.shape)
        result = _stacked_result(draws)

        np.testing.assert_allclose(
            result.convergence_diagnostics()["discrimination"]["rhat"],
            _reference_rank_rhat(draws),
            rtol=1e-12,
        )

    def test_independent_chains_pass_with_full_effective_size(self):
        rng = np.random.default_rng(5)
        result = _stacked_result(rng.normal(size=(4, 1000, 3)))

        diagnostics = result.convergence_diagnostics()["discrimination"]
        assert np.all(diagnostics["rhat"] < 1.01)
        np.testing.assert_allclose(diagnostics["ess_bulk"], 4000, rtol=0.2)
        np.testing.assert_allclose(diagnostics["ess_tail"], 4000, rtol=0.25)
        assert result.converged is True
        assert "Converged: yes" in result.summary()

    def test_bulk_ess_matches_the_autoregressive_value(self):
        rng = np.random.default_rng(6)
        rho, n_chains, n_draws = 0.8, 4, 4000
        draws = np.zeros((n_chains, n_draws, 1))
        noise = rng.normal(size=draws.shape)
        for t in range(1, n_draws):
            draws[:, t] = rho * draws[:, t - 1] + noise[:, t]
        result = _stacked_result(draws)

        expected = n_chains * n_draws * (1 - rho) / (1 + rho)
        assert result.ess["discrimination"] == pytest.approx(expected, rel=0.15)
        # 100 effective draws per chain are required.
        short = _stacked_result(draws[:, :400])
        assert short.ess["discrimination"] < 400
        assert short.converged is False

    def test_fixed_coordinates_are_skipped_and_stuck_ones_fail(self):
        rng = np.random.default_rng(7)
        draws = rng.normal(size=(2, 500, 3))
        draws[:, :, 0] = 1.5
        model = TwoParameterLogistic(n_items=3)
        model.set_free_parameter_masks(
            {"discrimination": np.array([False, True, True])}
        )
        result = _stacked_result(draws, model)

        rhat = result.convergence_diagnostics()["discrimination"]["rhat"]
        assert np.isnan(rhat[0]) and np.all(np.isfinite(rhat[1:]))
        assert result.converged is True

        stuck = _stacked_result(draws)
        assert np.isnan(stuck.rhat["discrimination"])
        assert stuck.converged is False

    def test_sampler_reports_the_chains_it_ran(self):
        responses = TestGibbsChains._responses()
        result = GibbsSampler(
            n_iter=60, burnin=10, n_chains=3, seed=2, use_rust=False
        ).fit(TwoParameterLogistic(3), responses)

        assert result.n_chains == 3
        draws = result.chains["difficulty"].reshape(3, 50, 3)
        np.testing.assert_allclose(
            result.convergence_diagnostics()["difficulty"]["rhat"],
            _reference_rank_rhat(draws),
            rtol=1e-12,
        )
        assert result.rhat["difficulty"] == pytest.approx(
            np.max(_reference_rank_rhat(draws))
        )


def _two_pl_data(n_persons=600, n_items=10, seed=11):
    rng = np.random.default_rng(seed)
    discrimination = rng.uniform(0.8, 2.0, n_items)
    difficulty = np.linspace(-1.5, 1.5, n_items)
    theta = rng.standard_normal(n_persons)
    probability = 1 / (1 + np.exp(-discrimination * (theta[:, None] - difficulty)))
    return (rng.random(probability.shape) < probability).astype(int)


def _per_item_acceptance(chain: np.ndarray) -> np.ndarray:
    return np.mean(np.any(np.diff(chain, axis=0) != 0, axis=2), axis=0)


class TestNumPySampler:
    """Itemwise Metropolis-within-Gibbs updates of the NumPy sampler."""

    def test_items_are_accepted_individually(self):
        # One joint accept/reject per parameter array accepted ~2% of moves.
        responses = _two_pl_data()
        result = GibbsSampler(n_iter=500, burnin=200, seed=1, use_rust=False).fit(
            TwoParameterLogistic(10), responses
        )
        moved = np.diff(result.chains["discrimination"], axis=0) != 0
        acceptance = moved.mean(axis=0)
        assert np.all((acceptance > 0.15) & (acceptance < 0.7))
        # Items move independently of one another.
        assert np.mean(moved.all(axis=1)) < 0.05

    def test_guessing_and_upper_draws_stay_inside_their_boxes(self):
        from mirt.models.dichotomous import (
            FourParameterLogistic,
            ThreeParameterLogistic,
        )

        responses = _two_pl_data(n_persons=400, n_items=6, seed=5)
        for model in (ThreeParameterLogistic(6), FourParameterLogistic(6)):
            result = GibbsSampler(n_iter=300, burnin=100, seed=2).fit(model, responses)
            guessing = result.chains["guessing"]
            assert np.all((guessing > 0.0) & (guessing < 0.5))
            if "upper" in result.chains:
                upper = result.chains["upper"]
                assert np.all((upper > 0.5) & (upper < 1.0))
            # Negative guessing used to leave a state set_parameters rejects.
            result.model.set_parameters(**result.model.parameters)

    def test_graded_thresholds_stay_ordered_and_track_em(self):
        import mirt
        from mirt.models.polytomous import GradedResponseModel

        n_items = 10
        responses = mirt.simdata(
            "GRM",
            n_persons=600,
            n_items=n_items,
            n_categories=4,
            discrimination=np.linspace(1.0, 1.8, n_items),
            thresholds=np.array([[-1.5, 0.0, 1.2], [-1.0, 0.2, 1.5]] * 5),
            seed=4,
        )
        em = mirt.fit_mirt(responses, "GRM").model.parameters
        # Joint proposals for all thresholds used to miss EM by over 0.6.
        result = GibbsSampler(n_iter=1500, burnin=500, seed=3).fit(
            GradedResponseModel(n_items, 4), responses
        )

        assert np.all(np.diff(result.chains["thresholds"], axis=2) > 0)
        for name in ("discrimination", "thresholds"):
            difference = result.model.parameters[name] - em[name]
            assert np.max(np.abs(difference)) < 0.15

    def test_masked_models_keep_fixed_coordinates(self):
        responses = _two_pl_data(n_persons=300, n_items=4)
        model = TwoParameterLogistic(4)
        model.set_parameters(discrimination=np.array([2.0, 1.0, 1.0, 1.0]))
        model.set_free_parameter_masks(
            {"discrimination": np.array([False, True, True, True])}
        )
        result = GibbsSampler(n_iter=200, burnin=50, seed=1).fit(model, responses)

        assert np.all(result.chains["discrimination"][:, 0] == 2.0)
        assert np.isnan(result.convergence_diagnostics()["discrimination"]["rhat"][0])
        assert result.model.parameters["discrimination"][0] == 2.0

    @pytest.mark.parametrize("estimator", ["gibbs", "mhrm"])
    def test_a_common_estimated_slope_is_refused(self, estimator):
        from mirt.exceptions import MirtModelError
        from mirt.models.dichotomous import OneParameterLogistic

        responses = _two_pl_data(n_persons=200, n_items=4)
        make = {
            "gibbs": lambda: GibbsSampler(n_iter=20, burnin=5, seed=1),
            "mhrm": lambda: MHRMEstimator(n_cycles=20, burnin=5, seed=1),
        }[estimator]
        with pytest.raises(
            MirtModelError, match="cannot estimate discrimination.*equal across"
        ):
            make().fit(OneParameterLogistic(4, estimate_discrimination=True), responses)
        # The unit-slope Rasch model keeps its slopes at one.
        with __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            fitted = make().fit(OneParameterLogistic(4), responses)
        np.testing.assert_array_equal(
            fitted.model.parameters["discrimination"], np.ones(4)
        )


class TestMarginalLogLikelihood:
    """Reported log-likelihoods are marginal, as EM's are."""

    @staticmethod
    def _grid_log_likelihood(model, responses):
        from scipy.special import logsumexp

        from mirt.estimation.quadrature import build_quadrature, default_quadpts

        quadrature = build_quadrature(default_quadpts(model.n_factors), model.n_factors)
        joint = model.log_likelihood_batch(responses, quadrature.nodes)
        return float(logsumexp(joint + np.log(quadrature.weights), axis=1).sum())

    def test_gibbs_log_likelihood_is_marginal_at_posterior_means(self):
        import mirt

        responses = _two_pl_data(n_persons=500, n_items=8)
        em = mirt.fit_mirt(responses, "2PL", compute_standard_errors=False)
        result = GibbsSampler(n_iter=600, burnin=200, seed=2).fit(
            TwoParameterLogistic(8), responses
        )

        assert result.log_likelihood == pytest.approx(
            self._grid_log_likelihood(result.model, responses), rel=1e-10
        )
        # The posterior mean of the conditional log-likelihood was reported,
        # hundreds of units above the marginal likelihood.
        assert abs(result.log_likelihood - em.log_likelihood) < 3.0
        assert np.mean(result.chains["log_likelihood"]) > result.log_likelihood + 100

    def test_importance_sampling_matches_quadrature_for_four_factors(self):
        from scipy.special import logsumexp

        from mirt.estimation.mcmc import _marginal_log_likelihood, _theta_moments
        from mirt.estimation.quadrature import GaussHermiteQuadrature

        rng = np.random.default_rng(0)
        n_items, n_factors = 12, 4
        model = TwoParameterLogistic(n_items=n_items, n_factors=n_factors)
        slopes = np.zeros((n_items, n_factors))
        for item in range(n_items):
            slopes[item, item % n_factors] = rng.uniform(0.8, 1.6)
            slopes[item, (item + 1) % n_factors] = 0.4
        model.set_parameters(
            discrimination=slopes, difficulty=rng.normal(0.0, 0.8, n_items)
        )
        theta = rng.standard_normal((150, n_factors))
        responses = (rng.random((150, n_items)) < model.probability(theta)).astype(int)
        grid = GaussHermiteQuadrature(11, n_factors)
        exact = logsumexp(
            model.log_likelihood_batch(responses, grid.nodes) + np.log(grid.weights),
            axis=1,
        ).sum()
        draws = theta + 0.6 * rng.standard_normal((400, *theta.shape))
        estimate = _marginal_log_likelihood(
            model,
            responses,
            posterior=_theta_moments(draws),
            rng=np.random.default_rng(1),
        )
        assert estimate == pytest.approx(exact, abs=0.5)
