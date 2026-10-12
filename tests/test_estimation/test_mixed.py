"""Tests for Mixed Effects IRT."""

import numpy as np
import pytest
from scipy.special import expit, logsumexp

from mirt import LLTM, MixedEffectsFitResult, MixedEffectsIRT
from mirt.estimation.mixed import _LatentRegressionEM
from mirt.models import OneParameterLogistic, TwoParameterLogistic


def _latent_regression_data(n_persons=3000, n_items=10, slopes=None, seed=4):
    """Simulate ``theta = 0.8 z + e`` with ``Var(e) = 0.36`` (total variance 1)."""
    rng = np.random.default_rng(seed)
    covariate = rng.normal(0, 1, n_persons)
    theta = 0.8 * covariate + rng.normal(0, 0.6, n_persons)
    discrimination = rng.uniform(0.8, 1.8, n_items) if slopes is None else slopes
    difficulty = rng.normal(0, 1, n_items)
    probability = expit(discrimination * (theta[:, None] - difficulty))
    responses = (rng.random((n_persons, n_items)) < probability).astype(int)
    return responses, covariate


def _direct_marginal_log_likelihood(result, responses, covariates):
    """Integrate each respondent's latent-regression likelihood on a wide grid."""
    nodes = np.linspace(-10.0, 10.0, 2001)
    parameters = result.model.parameters
    logits = parameters["discrimination"] * (nodes[:, None] - parameters["difficulty"])
    log_likelihood = (
        responses @ -np.logaddexp(0, -logits).T
        + (1 - responses) @ -np.logaddexp(0, logits).T
    )
    mean = result.person_intercept + covariates @ result.person_effects
    variance = result.residual_variance
    log_prior = -0.5 * (nodes[None, :] - mean[:, None]) ** 2 / variance - 0.5 * np.log(
        2 * np.pi * variance
    )
    spacing = nodes[1] - nodes[0]
    return float(
        np.sum(logsumexp(log_likelihood + log_prior + np.log(spacing), axis=1))
    )


class TestMixedEffectsIRT:
    """Tests for MixedEffectsIRT model."""

    def test_init(self):
        """Test MixedEffectsIRT initialization."""
        model = MixedEffectsIRT(base_model="2PL")
        assert model is not None

    def test_with_person_covariates(self, dichotomous_responses):
        """Test fitting with person-level covariates."""
        n_persons = dichotomous_responses["n_persons"]
        responses = dichotomous_responses["responses"]

        person_covariates = np.random.randn(n_persons, 1)

        model = MixedEffectsIRT(
            base_model="2PL",
            person_covariates=person_covariates,
        )
        result = model.fit(responses, max_iter=20)

        assert isinstance(result, MixedEffectsFitResult)
        assert result.model is not None

    def test_with_item_covariates(self, dichotomous_responses):
        """Test fitting with item-level covariates."""
        n_items = dichotomous_responses["n_items"]
        responses = dichotomous_responses["responses"]

        item_covariates = np.random.randn(n_items, 1)

        model = MixedEffectsIRT(
            base_model="2PL",
            item_covariates=item_covariates,
        )
        result = model.fit(responses, max_iter=20)

        assert result.model is not None


@pytest.fixture(scope="module")
def rasch_data():
    """Rasch responses with known difficulties."""
    rng = np.random.default_rng(0)
    difficulty = rng.normal(0, 1, 12)
    theta = rng.normal(0, 1, 1000)
    probability = expit(theta[:, None] - difficulty[None, :])
    responses = (rng.random(probability.shape) < probability).astype(int)
    return responses, difficulty


class TestLatentRegression:
    """Marginal maximum-likelihood latent regression."""

    def test_matches_r_mirt_latent_regression(self):
        """Estimate the regression inside the marginal likelihood.

        R mirt 1.48, ``mirt(resp, 1, "2PL", covdata=, formula=~z,
        quadpts=201, technical=list(theta_lim=c(-10, 10)))`` on these data
        gives beta 1.373815 with unit residual variance, which is 0.812426
        and residual variance 0.349713 on the unit-total-variance scale, and
        logLik -15637.84. Regressing EAP scores gave an attenuated 0.587.
        """
        responses, covariate = _latent_regression_data()
        estimator = MixedEffectsIRT(
            "2PL",
            person_covariates=covariate[:, None],
            person_covariate_names=["z"],
        )

        result = estimator.fit(responses)

        assert result.converged
        assert result.person_effects[0] == pytest.approx(0.812426, abs=2e-3)
        assert result.residual_variance == pytest.approx(0.349713, abs=2e-3)
        assert result.person_effects[0] / np.sqrt(
            result.residual_variance
        ) == pytest.approx(1.373815, abs=5e-3)
        assert result.log_likelihood == pytest.approx(-15637.84, abs=0.02)
        assert result.log_likelihood == pytest.approx(
            _direct_marginal_log_likelihood(result, responses, covariate[:, None]),
            abs=0.01,
        )
        # 2 parameters per item and one regression slope; the intercept and
        # residual variance are fixed by the unit-variance scale.
        assert result.aic == pytest.approx(-2 * result.log_likelihood + 2 * 21)
        assert 0.005 < result.person_effect_se[0] < 0.02
        predicted = estimator.predict_theta(covariate[:, None])
        assert predicted.mean() == pytest.approx(0.0, abs=1e-6)
        assert predicted.var() + result.residual_variance == pytest.approx(1.0)
        assert result.theta.mean() == pytest.approx(0.0, abs=1e-3)
        assert "z: 0.81" in result.summary()

    def test_common_slope_one_parameter_model(self):
        """The 1PL rescales its common slope with the ability distribution."""
        responses, covariate = _latent_regression_data(
            n_persons=1500, n_items=8, slopes=np.full(8, 1.3), seed=6
        )

        result = MixedEffectsIRT("1PL", person_covariates=covariate).fit(responses)

        discrimination = result.model.parameters["discrimination"]
        assert result.converged
        assert np.ptp(discrimination) < 1e-12
        assert discrimination[0] == pytest.approx(1.3, abs=0.15)
        assert result.person_effects[0] == pytest.approx(0.8, abs=0.05)
        assert result.residual_variance == pytest.approx(0.36, abs=0.06)
        assert result.aic == pytest.approx(-2 * result.log_likelihood + 2 * 10)
        assert result.log_likelihood == pytest.approx(
            _direct_marginal_log_likelihood(result, responses, covariate[:, None]),
            abs=0.01,
        )

    @pytest.mark.parametrize(
        ("base_model", "start"),
        [
            ("2PL", TwoParameterLogistic(4)),
            ("3PL", None),
            ("1PL", OneParameterLogistic(4, estimate_discrimination=True)),
            ("1PL", OneParameterLogistic(4)),
        ],
    )
    def test_information_uses_the_exact_marginal_score(self, base_model, start):
        """The score used for standard errors is the likelihood gradient."""
        from mirt.models import ThreeParameterLogistic

        responses, covariate = _latent_regression_data(n_persons=200, n_items=4, seed=8)
        covariates = np.column_stack([covariate, np.sin(covariate)])
        model = ThreeParameterLogistic(4) if start is None else start.copy()
        regression = _LatentRegressionEM(
            base_model, responses, covariates, n_quadpts=41
        )
        fit = regression.fit(model, max_iter=3, tol=1e-12, verbose=False)
        assert np.isfinite(fit.log_likelihood)
        items = regression._item_values(fit.model)
        variance = 0.6 if regression.slopes == "fixed" else 0.7
        effects = np.array([0.3, -0.2])
        vector = regression._pack(items, effects, variance)

        def log_likelihood(values):
            unpacked_items, unpacked_effects, residual_variance = regression._unpack(
                values
            )
            _, log_marginal = regression._posterior(
                unpacked_items,
                regression.centered @ unpacked_effects,
                residual_variance,
            )
            return float(np.sum(log_marginal))

        numerical = np.empty_like(vector)
        for index in range(vector.size):
            step = np.zeros_like(vector)
            step[index] = 1e-6
            numerical[index] = (
                log_likelihood(vector + step) - log_likelihood(vector - step)
            ) / 2e-6
        np.testing.assert_allclose(
            regression._score(vector), numerical, rtol=1e-4, atol=1e-4
        )

    def test_rejects_unsupported_base_model_for_person_covariates(self):
        estimator = MixedEffectsIRT("GRM", person_covariates=np.zeros(4))
        with pytest.raises(ValueError, match="1PL, 2PL or 3PL"):
            estimator.fit(np.zeros((4, 3), dtype=int))


class TestLLTM:
    """Tests for Linear Logistic Test Model."""

    def test_identity_design_reproduces_the_rasch_likelihood(self, rasch_data):
        """With Q = I the LLTM is the Rasch model, so the LR statistic is 0."""
        responses, _ = rasch_data

        result = LLTM(np.eye(12)).fit(responses)

        assert result["log_likelihood_lltm"] == pytest.approx(
            result["log_likelihood_rasch"], abs=1e-6
        )
        assert result["chi_sq"] == pytest.approx(0.0, abs=1e-5)
        assert result["df"] == 0
        assert np.isnan(result["p_value"])
        np.testing.assert_allclose(
            result["difficulty_lltm"], result["difficulty_rasch"], atol=1e-4
        )

    def test_likelihood_ratio_test_compares_marginal_likelihoods(self, rasch_data):
        """A true LLTM is retained and a constant design is rejected."""
        responses, difficulty = rasch_data
        true_design = np.column_stack([np.ones(12), difficulty])

        nested = LLTM(true_design).fit(responses)
        constant = LLTM(np.ones((12, 1))).fit(responses)

        assert nested["df"] == 10
        assert 0.0 <= nested["chi_sq"] < 25.0
        assert nested["p_value"] > 0.01
        assert nested["log_likelihood_lltm"] <= nested["log_likelihood_rasch"]
        assert constant["df"] == 11
        assert constant["p_value"] < 1e-10
        # Both models also estimate the common 1PL slope.
        assert nested["aic_lltm"] == pytest.approx(
            -2 * nested["log_likelihood_lltm"] + 2 * 3
        )
        assert nested["aic_rasch"] == pytest.approx(
            -2 * nested["log_likelihood_rasch"] + 2 * 13
        )

    def test_init(self):
        """Test LLTM initialization."""
        q_matrix = np.random.randint(0, 2, (10, 3))
        model = LLTM(q_matrix=q_matrix)

        assert model is not None

    def test_difficulty_reconstruction(self):
        """Test that difficulties are reconstructed from Q-matrix."""
        q_matrix = np.array(
            [
                [1, 0],
                [0, 1],
                [1, 1],
            ]
        )

        model = LLTM(q_matrix=q_matrix)
        np.testing.assert_array_equal(model._q_matrix, q_matrix)

    def test_fit(self, dichotomous_responses):
        """Test LLTM fitting."""
        n_items = dichotomous_responses["n_items"]
        responses = dichotomous_responses["responses"]

        q_matrix = np.random.randint(0, 2, (n_items, 3)).astype(float)
        for i in range(n_items):
            if q_matrix[i].sum() == 0:
                q_matrix[i, 0] = 1

        model = LLTM(q_matrix=q_matrix)
        result = model.fit(responses)

        assert result is not None
        assert model._eta is not None


class TestMixedEffectsFitResult:
    """Tests for MixedEffectsFitResult."""

    def test_result_attributes(self, dichotomous_responses):
        """Test result object attributes."""
        model = MixedEffectsIRT(base_model="2PL")
        responses = dichotomous_responses["responses"]

        result = model.fit(responses, max_iter=10)

        assert hasattr(result, "model")
        assert hasattr(result, "log_likelihood")

    def test_variance_components(self, dichotomous_responses):
        """Test variance component extraction."""
        n_persons = dichotomous_responses["n_persons"]
        responses = dichotomous_responses["responses"]
        person_covariates = np.random.randn(n_persons, 1)

        model = MixedEffectsIRT(
            base_model="2PL",
            person_covariates=person_covariates,
        )
        result = model.fit(responses, max_iter=10)

        assert result is not None
