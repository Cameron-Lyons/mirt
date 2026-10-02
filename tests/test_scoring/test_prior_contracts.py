"""Normal scoring-prior contracts and independent numerical references."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit

from mirt import get_backend, set_backend
from mirt.backends.rust._helpers import RUST_AVAILABLE
from mirt.models import TwoParameterLogistic
from mirt.scoring._common import resolve_prior_distribution
from mirt.scoring.eap import EAPScorer
from mirt.scoring.eapsum import EAPSumScorer, sum_score_to_theta
from mirt.scoring.map import MAPScorer


def _model(n_factors: int = 1) -> TwoParameterLogistic:
    model = TwoParameterLogistic(n_items=4, n_factors=n_factors)
    if n_factors == 1:
        model.set_parameters(
            discrimination=np.array([0.7, 0.9, 1.1, 1.3]),
            difficulty=np.array([-0.8, -0.1, 0.5, 1.2]),
        )
    model._is_fitted = True
    return model


@pytest.mark.parametrize(
    "covariance",
    [
        np.array([[np.nextafter(0.0, 1.0)]]),
        np.nextafter(0.0, 1.0) * np.array([[2.0, 1.0], [1.0, 2.0]]),
        np.diag([1e308, np.nextafter(0.0, 1.0)]),
    ],
)
def test_prior_symmetrization_preserves_positive_subnormal_covariance(covariance):
    original = covariance.copy()
    mean, resolved = resolve_prior_distribution(
        n_factors=len(covariance), prior_mean=None, prior_cov=covariance
    )
    np.testing.assert_array_equal(resolved, original)
    np.testing.assert_array_equal(covariance, original)
    np.testing.assert_array_equal(mean, np.zeros(len(covariance)))
    assert np.all(np.diag(resolved) > 0.0)


@pytest.mark.parametrize("method", ["EAP", "posterior", "MAP", "EAPsum"])
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prior_mean": 0.0}, "prior_mean must have shape"),
        ({"prior_mean": [0.0, 1.0]}, "prior_mean must have shape"),
        ({"prior_mean": [np.nan]}, "prior_mean.*finite"),
        ({"prior_mean": [np.inf]}, "prior_mean.*finite"),
        ({"prior_cov": [1.0]}, "prior_cov must have shape"),
        ({"prior_cov": np.eye(2)}, "prior_cov must have shape"),
        ({"prior_cov": [[np.nan]]}, "prior_cov.*finite"),
        ({"prior_cov": [[np.inf]]}, "prior_cov.*finite"),
        ({"prior_cov": [[0.0]]}, "positive definite"),
        ({"prior_cov": [[-1.0]]}, "positive definite"),
    ],
)
def test_bayesian_scoring_rejects_invalid_priors_before_likelihood(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    kwargs: dict[str, object],
    message: str,
) -> None:
    model = _model()

    def unexpected(*args, **kwargs):
        pytest.fail("an invalid prior must fail before evaluating likelihoods")

    monkeypatch.setattr(model, "log_likelihood", unexpected)
    monkeypatch.setattr(model, "log_likelihood_batch", unexpected)
    monkeypatch.setattr(model, "probability", unexpected)
    scorer = (
        MAPScorer(**kwargs)
        if method == "MAP"
        else EAPSumScorer(**kwargs)
        if method == "EAPsum"
        else EAPScorer(**kwargs)
    )
    score = scorer.posterior if method == "posterior" else scorer.score
    with pytest.raises(ValueError, match=message):
        score(model, np.array([[0, 1, 0, 1]]))


@pytest.mark.parametrize("method", ["EAP", "posterior", "MAP"])
@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        ([[1.0, 0.8], [0.1, 1.0]], "symmetric"),
        ([[1.0, 2.0], [2.0, 1.0]], "positive definite"),
        ([[1.0, 1.0], [1.0, 1.0]], "positive definite"),
        ([[1.0, 0.0], [0.0, np.nan]], "finite"),
    ],
)
def test_multidimensional_scoring_rejects_invalid_covariance(
    method: str, covariance: list[list[float]], message: str
) -> None:
    scorer = (
        MAPScorer(prior_cov=covariance)
        if method == "MAP"
        else EAPScorer(n_quadpts=5, prior_cov=covariance)
    )
    score = scorer.posterior if method == "posterior" else scorer.score
    with pytest.raises(ValueError, match=message):
        score(_model(2), np.full((1, 4), -1))


def test_correlated_prior_is_recovered_without_observed_responses() -> None:
    mean = np.array([0.75, -1.25])
    covariance = np.array([[1.8, 0.6], [0.6, 0.9]])
    responses = np.array([[-1, -9, -1, -2]])
    model = _model(2)
    scorer = EAPScorer(n_quadpts=7, prior_mean=mean, prior_cov=covariance)

    posterior = scorer.posterior(model, responses)
    scores = scorer.score(model, responses)
    centered = posterior.points - mean
    recovered_covariance = centered.T @ (centered * posterior.weights[0, :, None])

    assert_allclose(posterior.mean, mean[None, :], atol=1e-14)
    assert_allclose(recovered_covariance, covariance, atol=1e-14)
    assert_allclose(scores.theta, mean[None, :], atol=1e-14)
    assert_allclose(scores.standard_error**2, covariance.diagonal()[None, :])
    assert_allclose(posterior.log_marginal_likelihood, 0.0, atol=1e-14)


@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_custom_prior_scores_match_independent_logistic_integral(backend: str) -> None:
    if backend == "rust" and not RUST_AVAILABLE:
        pytest.skip("native backend is unavailable")
    previous = get_backend()
    set_backend(backend)
    try:
        model = _model()
        responses = np.array([[1, 0, -9, 1]])
        mean, variance = 0.8, 1.3
        observed = responses[0] >= 0
        slopes = model.discrimination[observed]
        difficulties = model.difficulty[observed]
        outcomes = responses[0, observed]

        def joint(theta: float) -> float:
            probability = expit(slopes * (theta - difficulties))
            likelihood = np.prod(np.where(outcomes == 1, probability, 1 - probability))
            prior = np.exp(-0.5 * (theta - mean) ** 2 / variance)
            return float(likelihood * prior / np.sqrt(2 * np.pi * variance))

        marginal = quad(joint, -np.inf, np.inf, epsabs=1e-12)[0]
        expected_mean = (
            quad(lambda x: x * joint(x), -np.inf, np.inf, epsabs=1e-12)[0] / marginal
        )
        expected_variance = (
            quad(
                lambda x: (x - expected_mean) ** 2 * joint(x),
                -np.inf,
                np.inf,
                epsabs=1e-12,
            )[0]
            / marginal
        )
        prior = {"prior_mean": np.array([mean]), "prior_cov": np.array([[variance]])}
        posterior = EAPScorer(n_quadpts=61, **prior).posterior(model, responses)
        scores = EAPScorer(n_quadpts=61, **prior).score(model, responses)
        assert_allclose(posterior.mean, expected_mean, atol=2e-10)
        assert_allclose(scores.theta, expected_mean, atol=2e-10)
        assert_allclose(scores.standard_error**2, expected_variance, atol=2e-10)
        assert_allclose(posterior.log_marginal_likelihood, np.log(marginal), atol=2e-10)

        def posterior_derivative(theta: float) -> float:
            probability = expit(slopes * (theta - difficulties))
            return float(slopes @ (outcomes - probability) - (theta - mean) / variance)

        expected_mode = brentq(posterior_derivative, -6.0, 6.0)
        probability = expit(slopes * (expected_mode - difficulties))
        curvature = np.sum(slopes**2 * probability * (1 - probability)) + 1 / variance
        modes = MAPScorer(**prior).score(model, responses)
        assert_allclose(modes.theta, expected_mode, atol=2e-6)
        assert_allclose(modes.standard_error, 1 / np.sqrt(curvature), rtol=2e-5)
    finally:
        set_backend(previous)


@pytest.mark.parametrize("n_quadpts", [True, 4, 5.5, "7"])
def test_eapsum_rejects_invalid_grid_size(n_quadpts: object) -> None:
    with pytest.raises(ValueError, match="at least 5"):
        EAPSumScorer(n_quadpts=n_quadpts)


def test_eapsum_owns_prior_configuration() -> None:
    mean, covariance = np.array([0.5]), np.array([[1.5]])
    scorer = EAPSumScorer(n_quadpts=21, prior_mean=mean, prior_cov=covariance)
    expected = scorer.score(_model(), np.array([[0, 1, 0, 1]]))
    mean[0] = -4.0
    covariance[0, 0] = 100.0
    actual = scorer.score(_model(), np.array([[0, 1, 0, 1]]))
    assert_allclose(actual.theta, expected.theta, rtol=0.0, atol=0.0)
    assert_allclose(actual.standard_error, expected.standard_error, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("method", ["lookup", "sum_score_to_theta"])
@pytest.mark.parametrize("invalid_model", ["unfitted", "multidimensional"])
def test_sum_score_lookup_preserves_model_scoring_contracts(
    method: str, invalid_model: str
) -> None:
    model = _model(2 if invalid_model == "multidimensional" else 1)
    if invalid_model == "unfitted":
        model._is_fitted = False
    with pytest.raises(
        ValueError, match="unidimensional" if model.n_factors == 2 else "fitted"
    ):
        if method == "lookup":
            EAPSumScorer().get_lookup_table(model)
        else:
            sum_score_to_theta(model, [0, 1, 2])
