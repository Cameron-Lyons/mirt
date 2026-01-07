"""Tests for MCMC estimation methods."""

from mirt import GibbsSampler, MCMCResult, MHRMEstimator, TwoParameterLogistic


class TestMHRMEstimator:
    """Tests for MHRM estimator."""

    def test_init(self):
        """Test MHRM estimator initialization."""
        estimator = MHRMEstimator(n_cycles=100, burnin=50)
        assert estimator.n_cycles == 100
        assert estimator.burnin == 50

    def test_fit(self, dichotomous_responses):
        """Test MHRM fitting."""
        model = TwoParameterLogistic(n_items=10)
        estimator = MHRMEstimator(
            n_cycles=50,
            burnin=20,
        )

        result = estimator.fit(model, dichotomous_responses["responses"])

        assert result.model._is_fitted
        assert "discrimination" in result.model._parameters
        assert "difficulty" in result.model._parameters

    def test_convergence_tracking(self, dichotomous_responses):
        """Test that convergence is tracked."""
        model = TwoParameterLogistic(n_items=10)
        estimator = MHRMEstimator(n_cycles=30, burnin=10)

        result = estimator.fit(model, dichotomous_responses["responses"])

        assert hasattr(result, "log_likelihood")


class TestGibbsSampler:
    """Tests for Gibbs sampler."""

    def test_init(self):
        """Test Gibbs sampler initialization."""
        sampler = GibbsSampler(n_iter=1000, burnin=200, thin=2)
        assert sampler.n_iter == 1000
        assert sampler.burnin == 200
        assert sampler.thin == 2

    def test_fit(self, dichotomous_responses):
        """Test Gibbs sampler fitting."""
        model = TwoParameterLogistic(n_items=10)
        sampler = GibbsSampler(
            n_iter=100,
            burnin=30,
            thin=1,
        )

        result = sampler.fit(model, dichotomous_responses["responses"])

        assert isinstance(result, MCMCResult)
        assert result.model._is_fitted

    def test_mcmc_result_has_chains(self, dichotomous_responses):
        """Test MCMCResult has chains."""
        model = TwoParameterLogistic(n_items=10)
        sampler = GibbsSampler(n_iter=50, burnin=20, thin=1)

        result = sampler.fit(model, dichotomous_responses["responses"])

        assert hasattr(result, "chains")
        assert "discrimination" in result.chains
        assert "difficulty" in result.chains


class TestMCMCResult:
    """Tests for MCMCResult class."""

    def test_summary(self, dichotomous_responses):
        """Test posterior summary."""
        model = TwoParameterLogistic(n_items=10)
        sampler = GibbsSampler(n_iter=100, burnin=30, thin=1)

        result = sampler.fit(model, dichotomous_responses["responses"])
        summary = result.summary()

        assert isinstance(summary, str)
        assert "MCMC" in summary or "Iteration" in summary

    def test_convergence_diagnostics(self, dichotomous_responses):
        """Test convergence diagnostics."""
        model = TwoParameterLogistic(n_items=10)
        sampler = GibbsSampler(n_iter=100, burnin=30, thin=1)

        result = sampler.fit(model, dichotomous_responses["responses"])

        assert hasattr(result, "rhat")
        assert hasattr(result, "ess")
        assert "discrimination" in result.rhat
        assert "difficulty" in result.rhat

    def test_dic_waic(self, dichotomous_responses):
        """Test DIC and WAIC computation."""
        model = TwoParameterLogistic(n_items=10)
        sampler = GibbsSampler(n_iter=100, burnin=30, thin=1)

        result = sampler.fit(model, dichotomous_responses["responses"])

        assert hasattr(result, "dic")
        assert hasattr(result, "waic")
        assert result.dic > 0
        assert result.waic > 0
