"""Regression coverage for regularized multidimensional estimation."""

from __future__ import annotations

import numpy as np
import pytest

import mirt.estimation.regularized as regularized_module
from mirt.estimation.latent_density import GaussianDensity
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.estimation.regularized import (
    PenaltySpec,
    RegularizedMIRTEstimator,
    _expected_item_counts,
)
from mirt.exceptions import MirtDataError
from mirt.utils.numeric import logsumexp


def _slow_e_step(
    estimator: RegularizedMIRTEstimator,
    responses: np.ndarray,
    loadings: np.ndarray,
    intercepts: np.ndarray,
    density: GaussianDensity,
) -> tuple[np.ndarray, np.ndarray]:
    quadrature = estimator._quadrature
    assert quadrature is not None
    quad_points = quadrature.nodes
    quad_weights = quadrature.weights
    log_likelihoods = np.zeros((responses.shape[0], len(quad_weights)))

    for q, theta_q in enumerate(quad_points):
        probabilities = regularized_module.sigmoid(theta_q @ loadings.T + intercepts)
        probabilities = np.clip(
            probabilities,
            regularized_module.PROB_EPSILON,
            1 - regularized_module.PROB_EPSILON,
        )
        for item_idx in range(responses.shape[1]):
            observed = responses[:, item_idx] >= 0
            item_responses = responses[observed, item_idx]
            log_likelihoods[observed, q] += item_responses * np.log(
                probabilities[item_idx]
            ) + (1 - item_responses) * np.log1p(-probabilities[item_idx])

    log_prior_mass = density.log_quadrature_mass(quad_points, quad_weights)
    log_joint = log_likelihoods + log_prior_mass[None, :]
    log_marginal = logsumexp(log_joint, axis=1, keepdims=True)
    return np.exp(log_joint - log_marginal), np.exp(log_marginal.ravel())


def test_vectorized_e_step_matches_reference_with_forced_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(314)
    estimator = RegularizedMIRTEstimator(n_factors=2, n_quadpts=7)
    estimator._quadrature = GaussHermiteQuadrature(n_points=7, n_dimensions=2)
    density = GaussianDensity(
        mean=np.array([0.25, -0.4]),
        cov=np.array([[1.2, 0.25], [0.25, 0.8]]),
        n_dimensions=2,
    )
    responses = rng.integers(0, 2, size=(18, 9), dtype=np.int32)
    responses[rng.random(responses.shape) < 0.2] = -1
    responses[0] = -1
    responses[:, 3] = -1
    loadings = rng.normal(0.0, 0.8, size=(9, 2))
    intercepts = rng.normal(0.0, 1.0, size=9)

    expected_posterior, expected_marginal = _slow_e_step(
        estimator,
        responses,
        loadings,
        intercepts,
        density,
    )
    monkeypatch.setattr(regularized_module, "_MAX_ESTEP_TEMP_ENTRIES", 20)

    actual_posterior, actual_marginal = estimator._e_step(
        responses,
        loadings,
        intercepts,
        density,
    )

    np.testing.assert_allclose(
        actual_posterior,
        expected_posterior,
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        actual_marginal,
        expected_marginal,
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(actual_posterior.sum(axis=1), 1.0, atol=1e-14)


def test_vectorized_expected_counts_match_observed_item_reference() -> None:
    rng = np.random.default_rng(2718)
    responses = rng.integers(0, 2, size=(23, 8), dtype=np.int32)
    responses[rng.random(responses.shape) < 0.25] = -1
    responses[:, 5] = -1
    posterior = rng.random((23, 17))
    posterior /= posterior.sum(axis=1, keepdims=True)

    actual_correct, actual_observed = _expected_item_counts(responses, posterior)
    expected_correct = np.zeros_like(actual_correct)
    expected_observed = np.zeros_like(actual_observed)
    for item_idx in range(responses.shape[1]):
        observed = responses[:, item_idx] >= 0
        expected_correct[item_idx] = np.sum(
            responses[observed, item_idx, None] * posterior[observed],
            axis=0,
        )
        expected_observed[item_idx] = posterior[observed].sum(axis=0)

    np.testing.assert_allclose(actual_correct, expected_correct, atol=1e-14)
    np.testing.assert_allclose(actual_observed, expected_observed, atol=1e-14)


def test_vectorized_lambda_max_matches_coordinatewise_reference() -> None:
    rng = np.random.default_rng(1618)
    responses = rng.integers(0, 2, size=(31, 7), dtype=np.int32)
    responses[rng.random(responses.shape) < 0.2] = -1
    estimator = RegularizedMIRTEstimator(n_factors=2, n_quadpts=5)

    actual = estimator._compute_lambda_max(responses)

    quadrature = estimator._quadrature
    assert quadrature is not None
    density = GaussianDensity(
        mean=np.zeros(2),
        cov=np.eye(2),
        n_dimensions=2,
    )
    posterior, _ = estimator._e_step(
        responses,
        np.zeros((responses.shape[1], 2)),
        np.zeros(responses.shape[1]),
        density,
    )
    max_gradient = 0.0
    for item_idx in range(responses.shape[1]):
        observed = responses[:, item_idx] >= 0
        expected_correct = np.sum(
            responses[observed, item_idx, None] * posterior[observed],
            axis=0,
        )
        expected_observed = posterior[observed].sum(axis=0)
        residual = expected_correct - 0.5 * expected_observed
        for factor_idx in range(estimator.n_factors):
            gradient = abs(np.sum(residual * quadrature.nodes[:, factor_idx]))
            max_gradient = max(max_gradient, gradient)

    assert actual == pytest.approx(max_gradient * 1.1, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("penalty_type", ["unknown", "", 3])
def test_penalty_spec_rejects_unknown_types(penalty_type: object) -> None:
    with pytest.raises(ValueError, match="penalty type"):
        PenaltySpec(penalty_type, 0.1)  # type: ignore[arg-type]


@pytest.mark.parametrize("lambda_value", [-0.1, np.nan, np.inf, True, "0.1"])
def test_penalty_spec_rejects_invalid_strengths(lambda_value: object) -> None:
    with pytest.raises(ValueError, match="lambda_val"):
        PenaltySpec("lasso", lambda_value)  # type: ignore[arg-type]


@pytest.mark.parametrize("alpha", [-0.1, 1.1, np.nan, np.inf, True, "0.5"])
def test_penalty_spec_rejects_invalid_mixing(alpha: object) -> None:
    with pytest.raises(ValueError, match="alpha"):
        PenaltySpec("elastic_net", 0.1, alpha)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_factors": 1}, "n_factors"),
        ({"n_factors": 2.5}, "n_factors"),
        ({"n_quadpts": 0}, "n_quadpts"),
        ({"cd_max_iter": 0}, "cd_max_iter"),
        ({"cd_tol": 0.0}, "cd_tol"),
        ({"cd_tol": np.inf}, "cd_tol"),
    ],
)
def test_estimator_rejects_invalid_solver_configuration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RegularizedMIRTEstimator(**kwargs)  # type: ignore[arg-type]


def test_estimator_requires_boolean_adaptive_flag() -> None:
    with pytest.raises(TypeError, match="adaptive"):
        RegularizedMIRTEstimator(adaptive=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("responses", "message"),
    [
        (np.array([[0.5, 1.0], [0.0, 1.0]]), "integer response codes"),
        (np.array([[2, 1], [0, 1]]), "binary responses"),
        (np.array([[np.nan, 1.0], [0.0, 1.0]]), "finite response codes"),
    ],
)
def test_fit_rejects_invalid_binary_responses(
    responses: np.ndarray,
    message: str,
) -> None:
    estimator = RegularizedMIRTEstimator(
        n_factors=2,
        n_quadpts=3,
        max_iter=1,
        cd_max_iter=1,
    )

    with pytest.raises(MirtDataError, match=message):
        estimator.fit(responses)


@pytest.mark.parametrize("lambda_value", [-1.0, np.nan, np.inf])
def test_fit_rejects_invalid_lambda_override(lambda_value: float) -> None:
    estimator = RegularizedMIRTEstimator(max_iter=1, cd_max_iter=1)
    responses = np.array([[0, 1], [1, 0]], dtype=np.int32)

    with pytest.raises(ValueError, match="lambda_val"):
        estimator.fit(responses, lambda_val=lambda_value)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lambda_values": []}, "lambda_values"),
        ({"lambda_values": [-0.1]}, "lambda_values"),
        ({"lambda_values": [np.nan]}, "lambda_values"),
        ({"n_lambda": 0}, "n_lambda"),
        ({"lambda_min_ratio": 0.0}, "lambda_min_ratio"),
        ({"lambda_min_ratio": 1.1}, "lambda_min_ratio"),
    ],
)
def test_fit_path_rejects_invalid_grid_configuration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    estimator = RegularizedMIRTEstimator(max_iter=1, cd_max_iter=1)
    responses = np.array([[0, 1], [1, 0]], dtype=np.int32)

    with pytest.raises(ValueError, match=message):
        estimator.fit_path(responses, **kwargs)  # type: ignore[arg-type]


def test_fit_path_handles_zero_lambda_max_without_logarithmic_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator = RegularizedMIRTEstimator(max_iter=1, cd_max_iter=1)
    responses = np.full((4, 3), -1, dtype=np.int32)
    fitted_lambdas: list[float] = []

    def record_fit(
        _responses: np.ndarray,
        lambda_val: float | None = None,
    ) -> object:
        assert lambda_val is not None
        fitted_lambdas.append(lambda_val)
        return object()

    monkeypatch.setattr(estimator, "fit", record_fit)

    results = estimator.fit_path(responses, n_lambda=3)

    assert len(results) == 3
    assert fitted_lambdas == [0.0, 0.0, 0.0]
