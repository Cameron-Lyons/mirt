"""Regression tests for sparse Bayesian inference and reporting."""

from __future__ import annotations

from math import lgamma
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.constants import PROB_EPSILON, REGULARIZATION_EPSILON
from mirt.estimation.sparse_bayesian import (
    SparseBayesianEstimator,
    SparseBayesianResult,
    SpikeSlabLassoPrior,
)
from mirt.exceptions import MirtValidationError
from mirt.models.dichotomous import (
    OneParameterLogistic,
    TwoParameterLogistic,
)


def _slow_e_step(
    estimator: SparseBayesianEstimator,
    responses: np.ndarray,
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the shifted-prior update one person and item at a time."""
    mu = estimator._mu.copy()
    sigma = estimator._sigma.copy()
    xi = estimator._xi.copy()
    valid = responses >= 0
    prior_natural_mean = prior_precision @ prior_mean

    for _ in range(estimator.n_inner_iter):
        lam = estimator._lambda_jj(xi)
        for person in range(responses.shape[0]):
            observed = valid[person]
            loadings = estimator._loadings[observed]
            item_lam = lam[person, observed]
            precision = prior_precision + np.einsum(
                "j,jf,jg->fg",
                2.0 * item_lam,
                loadings,
                loadings,
            )
            sigma[person] = np.linalg.inv(precision)
            coefficients = (
                responses[person, observed]
                - 0.5
                - 2.0 * item_lam * estimator._intercepts[observed]
            )
            natural_mean = coefficients @ loadings + prior_natural_mean
            mu[person] = sigma[person] @ natural_mean

        for person, item in np.argwhere(valid):
            loading = estimator._loadings[item]
            eta_mean = loading @ mu[person] + estimator._intercepts[item]
            eta_variance = loading @ sigma[person] @ loading
            xi[person, item] = np.sqrt(
                max(eta_variance + eta_mean**2, np.finfo(float).eps)
            )

    return mu, sigma, xi


def _slow_m_step(
    estimator: SparseBayesianEstimator,
    responses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the sparse M-step item by item for a reference result."""
    loadings = estimator._loadings.copy()
    intercepts = estimator._intercepts.copy()
    gamma = estimator._gamma.copy()
    valid = responses >= 0
    lam = estimator._lambda_jj(estimator._xi)
    second_moments = estimator._sigma + np.einsum(
        "if,ig->ifg",
        estimator._mu,
        estimator._mu,
    )

    for item in range(responses.shape[1]):
        observed = valid[:, item]
        curvature = np.einsum(
            "i,ifg->fg",
            2.0 * lam[observed, item],
            second_moments[observed],
        )
        curvature += np.eye(estimator.k_max) * REGULARIZATION_EPSILON
        coefficients = (
            responses[observed, item]
            - 0.5
            - 2.0 * lam[observed, item] * intercepts[item]
        )
        score = coefficients @ estimator._mu[observed]
        unpenalized = np.linalg.solve(curvature, score)
        gamma[item] = estimator._ssl_prior.compute_posterior_inclusion(unpenalized)
        penalty = estimator._ssl_prior.compute_effective_penalty(gamma[item])
        threshold = penalty / (np.diag(curvature) + PROB_EPSILON)
        loadings[item] = estimator._ssl_prior.soft_threshold(
            unpenalized,
            threshold,
        )

    for item in range(responses.shape[1]):
        observed = valid[:, item]
        if not np.any(observed):
            continue
        linear = estimator._mu[observed] @ loadings[item]
        numerator = np.sum(
            responses[observed, item] - 0.5 - 2.0 * lam[observed, item] * linear
        )
        denominator = 2.0 * np.sum(lam[observed, item])
        intercepts[item] = np.clip(numerator / denominator, -10.0, 10.0)

    return loadings, intercepts, gamma


def _slow_elbo(
    estimator: SparseBayesianEstimator,
    responses: np.ndarray,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
) -> float:
    """Evaluate the ELBO without batching for a reference result."""
    valid = responses >= 0
    lam = estimator._lambda_jj(estimator._xi)
    value = 0.0

    for person, item in np.argwhere(valid):
        loading = estimator._loadings[item]
        intercept = estimator._intercepts[item]
        eta_mean = loading @ estimator._mu[person] + intercept
        eta_variance = loading @ estimator._sigma[person] @ loading
        eta_second = eta_variance + eta_mean**2
        xi = estimator._xi[person, item]
        value += (
            -np.logaddexp(0.0, -xi)
            + (responses[person, item] - 0.5) * eta_mean
            - 0.5 * xi
            - lam[person, item] * (eta_second - xi**2)
        )

    prior_precision = np.linalg.inv(prior_cov)
    log_det_prior = np.linalg.slogdet(prior_cov)[1]
    for mu, sigma in zip(estimator._mu, estimator._sigma, strict=True):
        diff = mu - prior_mean
        value -= 0.5 * (
            diff @ prior_precision @ diff
            + np.trace(prior_precision @ sigma)
            + log_det_prior
            - np.linalg.slogdet(sigma)[1]
            - estimator.k_max
        )

    if not estimator._fixed_loadings:
        value += np.sum(estimator._ssl_prior.log_pdf(estimator._loadings))
    return float(value)


def test_inclusion_probability_uses_normalized_laplace_densities() -> None:
    prior = SpikeSlabLassoPrior(lambda_0=0.04, lambda_1=1.0, theta=0.5)
    values = np.array([0.0, 0.2, 1.5, 100.0])
    spike = (
        (1.0 - prior.theta)
        * np.exp(-np.abs(values) / prior.lambda_0)
        / (2.0 * prior.lambda_0)
    )
    slab = (
        prior.theta * np.exp(-np.abs(values) / prior.lambda_1) / (2.0 * prior.lambda_1)
    )
    expected = slab / (spike + slab)

    actual = prior.compute_posterior_inclusion(values)

    assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert actual[0] == pytest.approx(1.0 / 26.0)


def test_effective_penalty_returns_mixture_rate() -> None:
    prior = SpikeSlabLassoPrior(lambda_0=0.1, lambda_1=2.0)
    gamma = np.array([0.0, 0.25, 1.0])
    expected = (1.0 - gamma) / 0.1 + gamma / 2.0

    assert_allclose(prior.compute_effective_penalty(gamma), expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lambda_0": np.nan}, "lambda_0"),
        ({"lambda_1": np.inf}, "lambda_1"),
        ({"theta": True}, "theta"),
        ({"adaptive": 1}, "adaptive"),
    ],
)
def test_prior_rejects_invalid_controls(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(MirtValidationError, match=message):
        SpikeSlabLassoPrior(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"k_max": 1.5}, "k_max"),
        ({"k_max": True}, "k_max"),
        ({"n_inner_iter": 0}, "n_inner_iter"),
        ({"n_inner_iter": 1.5}, "n_inner_iter"),
        ({"sparsity_threshold": np.nan}, "sparsity_threshold"),
        ({"sparsity_threshold": 1.1}, "sparsity_threshold"),
        ({"adaptive_theta": 1}, "adaptive_theta"),
    ],
)
def test_estimator_rejects_invalid_controls(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(MirtValidationError, match=message):
        SparseBayesianEstimator(**kwargs)


@pytest.mark.parametrize(
    ("prior_mean", "prior_cov", "message"),
    [
        (np.zeros(1), np.eye(2), "prior_mean shape"),
        (np.array([0.0, np.inf]), np.eye(2), "prior_mean.*finite"),
        (np.zeros(2), np.eye(1), "prior_cov shape"),
        (np.zeros(2), np.array([[1.0, np.inf], [np.inf, 1.0]]), "prior_cov.*finite"),
        (np.zeros(2), np.array([[1.0, 0.2], [0.1, 1.0]]), "symmetric"),
        (np.zeros(2), np.array([[1.0, 2.0], [2.0, 1.0]]), "positive definite"),
    ],
)
def test_fit_validates_prior(
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    message: str,
) -> None:
    model = TwoParameterLogistic(n_items=2, n_factors=2)
    responses = np.array([[0, 1], [1, 0]], dtype=np.int64)

    with pytest.raises(MirtValidationError, match=message):
        SparseBayesianEstimator(k_max=2, max_iter=1).fit(
            model,
            responses,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
        )


def test_fit_rejects_model_dimension_mismatch() -> None:
    model = TwoParameterLogistic(n_items=2, n_factors=2)
    responses = np.array([[0, 1], [1, 0]], dtype=np.int64)

    with pytest.raises(MirtValidationError, match="n_factors.*k_max"):
        SparseBayesianEstimator(k_max=3).fit(model, responses)


def test_vectorized_e_step_matches_shifted_prior_reference() -> None:
    responses = np.array(
        [[1, 0, 1], [0, -1, 1], [-1, -1, -1], [1, 1, -1]],
        dtype=np.int64,
    )
    prior_mean = np.array([0.45, -0.3])
    prior_cov = np.array([[1.4, 0.25], [0.25, 0.9]])
    prior_precision = np.linalg.inv(prior_cov)
    estimator = SparseBayesianEstimator(
        k_max=2,
        n_inner_iter=3,
        adaptive_theta=False,
    )
    estimator._loadings = np.array([[0.8, 0.3], [1.2, -0.4], [0.5, 1.1]])
    estimator._intercepts = np.array([-0.2, 0.35, -0.1])
    estimator._gamma = np.full((3, 2), 0.5)
    estimator._mu = np.broadcast_to(prior_mean, (4, 2)).copy()
    estimator._sigma = np.broadcast_to(prior_cov, (4, 2, 2)).copy()
    estimator._xi = np.ones(responses.shape)
    expected_mu, expected_sigma, expected_xi = _slow_e_step(
        estimator,
        responses,
        prior_mean,
        prior_precision,
    )

    estimator._e_step(responses, prior_mean, prior_precision)

    assert_allclose(estimator._mu, expected_mu, rtol=1e-12, atol=1e-12)
    assert_allclose(estimator._sigma, expected_sigma, rtol=1e-12, atol=1e-12)
    assert_allclose(estimator._xi, expected_xi, rtol=1e-12, atol=1e-12)
    assert_allclose(estimator._mu[2], prior_mean, atol=1e-14)
    assert_allclose(estimator._sigma[2], prior_cov, atol=1e-13)


def test_vectorized_m_step_matches_itemwise_reference() -> None:
    rng = np.random.default_rng(20260828)
    responses = rng.integers(0, 2, size=(24, 5), dtype=np.int64)
    responses[rng.random(responses.shape) < 0.2] = -1
    responses[:, 4] = -1
    estimator = SparseBayesianEstimator(
        k_max=3,
        adaptive_theta=False,
    )
    estimator._loadings = rng.normal(0.0, 0.5, size=(5, 3))
    estimator._intercepts = rng.normal(0.0, 0.4, size=5)
    estimator._gamma = np.full((5, 3), 0.5)
    estimator._mu = rng.normal(0.0, 0.6, size=(24, 3))
    matrices = rng.normal(size=(24, 3, 3))
    estimator._sigma = (
        np.einsum("ifg,ihg->ifh", matrices, matrices) / 3.0 + np.eye(3) * 0.2
    )
    estimator._xi = rng.uniform(0.4, 2.0, size=responses.shape)
    expected_loadings, expected_intercepts, expected_gamma = _slow_m_step(
        estimator,
        responses,
    )

    estimator._m_step_ssl(responses)

    assert_allclose(estimator._loadings, expected_loadings, rtol=1e-11, atol=1e-11)
    assert_allclose(estimator._intercepts, expected_intercepts, rtol=1e-12, atol=1e-12)
    assert_allclose(estimator._gamma, expected_gamma, rtol=1e-12, atol=1e-12)
    assert_allclose(estimator._loadings[4], 0.0, atol=0.0)


def test_vectorized_elbo_matches_scalar_reference() -> None:
    responses = np.array(
        [[1, 0, -1], [0, 1, 1], [-1, -1, -1], [1, -1, 0]],
        dtype=np.int64,
    )
    prior_mean = np.array([0.35, -0.2])
    prior_cov = np.array([[1.3, 0.2], [0.2, 0.85]])
    estimator = SparseBayesianEstimator(k_max=2, adaptive_theta=False)
    estimator._loadings = np.array([[0.7, 0.4], [1.1, -0.3], [0.6, 0.9]])
    estimator._intercepts = np.array([-0.25, 0.1, 0.3])
    estimator._gamma = np.full((3, 2), 0.5)
    estimator._mu = np.array([[0.2, -0.1], [-0.4, 0.3], [0.35, -0.2], [0.7, 0.1]])
    estimator._sigma = np.array(
        [
            [[0.7, 0.1], [0.1, 0.6]],
            [[0.8, -0.05], [-0.05, 0.5]],
            prior_cov,
            [[0.55, 0.08], [0.08, 0.75]],
        ]
    )
    estimator._xi = np.array(
        [[1.1, 0.8, 1.0], [0.7, 1.3, 0.9], [1.0, 1.0, 1.0], [1.4, 1.0, 0.6]]
    )

    expected = _slow_elbo(estimator, responses, prior_mean, prior_cov)
    actual = estimator._compute_elbo(responses, prior_mean, prior_cov)

    assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_fit_handles_all_missing_rows_and_items_with_shifted_prior() -> None:
    responses = np.array(
        [[-1, -1, -1], [1, 0, -1], [0, 1, -1], [1, 1, -1]],
        dtype=np.int64,
    )
    prior_mean = np.array([0.6, -0.4])
    prior_cov = np.array([[1.3, 0.2], [0.2, 0.9]])
    estimator = SparseBayesianEstimator(
        k_max=2,
        max_iter=2,
        n_inner_iter=2,
        adaptive_theta=False,
    )

    result = estimator.fit(
        TwoParameterLogistic(n_items=3, n_factors=2),
        responses,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
    )

    assert_allclose(estimator._mu[0], prior_mean, atol=1e-14)
    assert_allclose(estimator._sigma[0], prior_cov, atol=1e-13)
    assert_allclose(result.loadings[2], 0.0, atol=0.0)
    assert result.intercepts[2] == pytest.approx(0.0)
    assert result.model.difficulty[2] == pytest.approx(0.0)
    assert estimator.variational_means.shape == (4, 2)
    assert estimator.variational_covariances.shape == (4, 2, 2)

    means = estimator.variational_means
    means[0] = 100.0
    assert_allclose(estimator.variational_means[0], prior_mean, atol=1e-14)


def test_one_factor_fit_preserves_model_parameter_shape() -> None:
    responses = np.array(
        [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]],
        dtype=np.int64,
    )
    estimator = SparseBayesianEstimator(
        k_max=1,
        max_iter=2,
        adaptive_theta=False,
    )

    result = estimator.fit(OneParameterLogistic(n_items=3), responses)

    assert result.model.discrimination.shape == (3,)
    assert_allclose(result.model.discrimination, 1.0, atol=0.0)
    assert_allclose(result.loadings, 1.0, atol=0.0)
    assert_allclose(result.inclusion_probabilities, 1.0, atol=0.0)


def test_difficulty_conversion_preserves_signed_intercept() -> None:
    estimator = SparseBayesianEstimator(k_max=2)
    estimator._loadings = np.array([[-1.0, -0.5], [1.0, -1.0]])
    estimator._intercepts = np.array([0.6, 0.5])

    difficulty = estimator._compute_difficulty_from_intercept()

    assert difficulty[0] == pytest.approx(0.4)
    assert difficulty[1] == pytest.approx(0.0)
    assert -np.sum(estimator._loadings[0]) * difficulty[0] == pytest.approx(0.6)


def test_fit_reports_data_likelihood_and_combinatorial_ebic() -> None:
    responses = np.array(
        [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
        dtype=np.int64,
    )
    estimator = SparseBayesianEstimator(
        k_max=2,
        max_iter=2,
        adaptive_theta=False,
    )
    result = estimator.fit(
        TwoParameterLogistic(n_items=3, n_factors=2),
        responses,
    )
    n_selected = int(np.count_nonzero(result.sparsity_pattern))
    n_candidates = result.n_items * estimator.k_max
    n_params = result.n_items + n_selected
    expected_bic = -2.0 * result.log_likelihood + n_params * np.log(len(responses))
    expected_log_models = (
        lgamma(n_candidates + 1)
        - lgamma(n_selected + 1)
        - lgamma(n_candidates - n_selected + 1)
    )

    assert isinstance(result, SparseBayesianResult)
    assert result.log_likelihood == pytest.approx(
        estimator._compute_log_likelihood(responses, result.sparse_loadings)
    )
    assert result.log_likelihood != pytest.approx(result.elbo)
    assert result.bic == pytest.approx(expected_bic)
    assert result.ebic == pytest.approx(expected_bic + expected_log_models)
