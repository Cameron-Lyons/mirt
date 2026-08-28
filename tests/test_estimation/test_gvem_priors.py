"""Regression tests for GVEM priors and vectorized NumPy kernels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

import mirt.estimation.gvem as gvem_module
from mirt.estimation.gvem import GVEMEstimator
from mirt.exceptions import MirtValidationError
from mirt.models.dichotomous import TwoParameterLogistic


def _slow_e_step(
    estimator: GVEMEstimator,
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
        lam = estimator._lambda(xi)
        for person in range(responses.shape[0]):
            observed = valid[person]
            slopes = estimator._slopes[observed]
            item_lam = lam[person, observed]
            precision = prior_precision + np.einsum(
                "j,jf,jg->fg",
                2.0 * item_lam,
                slopes,
                slopes,
            )
            sigma[person] = np.linalg.inv(precision)
            coeffs = (
                responses[person, observed]
                - 0.5
                - 2.0 * item_lam * estimator._intercepts[observed]
            )
            natural_mean = coeffs @ slopes + prior_natural_mean
            mu[person] = sigma[person] @ natural_mean

        for person, item in np.argwhere(valid):
            slope = estimator._slopes[item]
            eta_mean = slope @ mu[person] + estimator._intercepts[item]
            eta_variance = slope @ sigma[person] @ slope
            xi[person, item] = np.sqrt(max(eta_variance + eta_mean**2, 0.0))

    return mu, sigma, xi


def _slow_elbo(
    estimator: GVEMEstimator,
    responses: np.ndarray,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
) -> float:
    """Evaluate the ELBO without batching for a reference result."""
    valid = responses >= 0
    lam = estimator._lambda(estimator._xi)
    value = 0.0

    for person, item in np.argwhere(valid):
        slope = estimator._slopes[item]
        intercept = estimator._intercepts[item]
        eta_mean = slope @ estimator._mu[person] + intercept
        eta_variance = slope @ estimator._sigma[person] @ slope
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
        kl = 0.5 * (
            diff @ prior_precision @ diff
            + np.trace(prior_precision @ sigma)
            + log_det_prior
            - np.linalg.slogdet(sigma)[1]
            - prior_mean.size
        )
        value -= kl

    return float(value)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_inner_iter": 1.5}, "n_inner_iter"),
        ({"n_inner_iter": True}, "n_inner_iter"),
        ({"se_step_size": 0.0}, "se_step_size"),
        ({"se_step_size": np.inf}, "se_step_size"),
        ({"use_gpu": "sometimes"}, "use_gpu"),
    ],
)
def test_constructor_rejects_invalid_controls(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(MirtValidationError, match=message):
        GVEMEstimator(**kwargs)


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
        GVEMEstimator(max_iter=1).fit(
            model,
            responses,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
        )


def test_vectorized_e_step_matches_shifted_prior_reference() -> None:
    responses = np.array(
        [[1, 0, 1], [0, -1, 1], [-1, -1, -1], [1, 1, -1]],
        dtype=np.int64,
    )
    prior_mean = np.array([0.45, -0.3])
    prior_cov = np.array([[1.4, 0.25], [0.25, 0.9]])
    prior_precision = np.linalg.inv(prior_cov)
    model = TwoParameterLogistic(n_items=3, n_factors=2)
    model._initialize_parameters()
    estimator = GVEMEstimator(n_inner_iter=3, use_gpu=False)
    estimator._slopes = np.array([[0.8, 0.3], [1.2, -0.4], [0.5, 1.1]])
    estimator._intercepts = np.array([-0.2, 0.35, -0.1])
    estimator._initialize_variational_params(
        responses.shape[0],
        model.n_factors,
        model.n_items,
        prior_mean,
        prior_cov,
    )
    expected_mu, expected_sigma, expected_xi = _slow_e_step(
        estimator,
        responses,
        prior_mean,
        prior_precision,
    )

    estimator._e_step_python(responses, prior_mean, prior_precision)

    assert_allclose(estimator._mu, expected_mu, rtol=1e-12, atol=1e-12)
    assert_allclose(estimator._sigma, expected_sigma, rtol=1e-12, atol=1e-12)
    assert_allclose(estimator._xi, expected_xi, rtol=1e-12, atol=1e-12)
    assert_allclose(estimator._mu[2], prior_mean, rtol=0.0, atol=1e-14)
    assert_allclose(estimator._sigma[2], prior_cov, rtol=1e-12, atol=1e-12)


def test_native_e_step_recenters_shifted_prior(monkeypatch: pytest.MonkeyPatch) -> None:
    prior_mean = np.array([0.4, -0.25])
    prior_cov = np.array([[1.2, 0.1], [0.1, 0.8]])
    model = TwoParameterLogistic(n_items=2, n_factors=2)
    model._initialize_parameters()
    estimator = GVEMEstimator(n_inner_iter=1, use_gpu=False)
    estimator._slopes = np.array([[0.8, 0.3], [1.1, -0.2]])
    estimator._intercepts = np.array([-0.15, 0.25])
    estimator._initialize_variational_params(1, 2, 2, prior_mean, prior_cov)
    captured: dict[str, np.ndarray] = {}

    def fake_native(
        responses: np.ndarray,
        slopes: np.ndarray,
        intercepts: np.ndarray,
        prior_precision: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        xi: np.ndarray,
        n_inner_iter: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del responses, slopes, prior_precision, n_inner_iter
        captured["intercepts"] = intercepts.copy()
        captured["mu"] = mu.copy()
        return mu, sigma, xi

    monkeypatch.setattr(gvem_module, "_rust_gvem_e_step", fake_native)
    estimator._e_step(
        model,
        np.array([[-1, -1]], dtype=np.int64),
        prior_mean,
        prior_cov,
    )

    assert_allclose(captured["mu"], np.zeros((1, 2)), atol=0.0)
    assert_allclose(
        captured["intercepts"],
        estimator._intercepts + estimator._slopes @ prior_mean,
    )
    assert_allclose(estimator._mu[0], prior_mean, atol=0.0)


def test_fit_keeps_all_missing_posterior_at_shifted_prior() -> None:
    responses = np.array(
        [[-1, -1], [1, 0], [0, 1], [1, 1]],
        dtype=np.int64,
    )
    prior_mean = np.array([0.6, -0.4])
    prior_cov = np.array([[1.3, 0.2], [0.2, 0.9]])
    estimator = GVEMEstimator(max_iter=2, n_inner_iter=2, use_gpu=False)

    estimator.fit(
        TwoParameterLogistic(n_items=2, n_factors=2),
        responses,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
    )

    assert_allclose(estimator.variational_means[0], prior_mean, atol=1e-14)
    assert_allclose(estimator.variational_covariances[0], prior_cov, atol=1e-14)


def test_vectorized_elbo_matches_scalar_reference() -> None:
    responses = np.array(
        [[1, 0, -1], [0, 1, 1], [-1, -1, -1], [1, -1, 0]],
        dtype=np.int64,
    )
    prior_mean = np.array([0.35, -0.2])
    prior_cov = np.array([[1.3, 0.2], [0.2, 0.85]])
    model = TwoParameterLogistic(n_items=3, n_factors=2)
    estimator = GVEMEstimator(use_gpu=False)
    estimator._slopes = np.array([[0.7, 0.4], [1.1, -0.3], [0.6, 0.9]])
    estimator._intercepts = np.array([-0.25, 0.1, 0.3])
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
    actual = estimator._compute_elbo_python(
        model,
        responses,
        prior_mean,
        prior_cov,
    )

    assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_standard_errors_reuse_center_and_custom_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = TwoParameterLogistic(n_items=2)
    model._initialize_parameters()
    estimator = GVEMEstimator(se_step_size=1e-3, use_gpu=False)
    estimator._convert_to_slope_intercept(model)
    prior_mean = np.array([0.6])
    prior_cov = np.array([[1.7]])
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def quadratic_elbo(
        fitted_model: TwoParameterLogistic,
        responses: np.ndarray,
        mean: np.ndarray,
        cov: np.ndarray,
    ) -> float:
        del responses
        calls.append((mean.copy(), cov.copy()))
        return -sum(
            float(np.sum(values**2)) for values in fitted_model.parameters.values()
        )

    monkeypatch.setattr(estimator, "_compute_elbo", quadratic_elbo)
    standard_errors = estimator._compute_standard_errors(
        model,
        np.ones((2, 2), dtype=np.int64),
        prior_mean,
        prior_cov,
    )

    assert len(calls) == 2 * model.n_parameters + 1
    assert all(np.array_equal(mean, prior_mean) for mean, _ in calls)
    assert all(np.array_equal(cov, prior_cov) for _, cov in calls)
    assert_allclose(standard_errors["discrimination"], np.sqrt(0.5), rtol=1e-5)
    assert_allclose(standard_errors["difficulty"], np.sqrt(0.5), rtol=1e-5)
