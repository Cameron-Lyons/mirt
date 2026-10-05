"""Tests for factor-correlation estimation with FactorCovarianceDensity."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize
from scipy.special import logsumexp

from mirt import EMEstimator, MultidimensionalModel
from mirt.estimation.latent_density import (
    FactorCovarianceDensity,
    GaussianDensity,
    _constrained_gaussian_covariance,
)
from mirt.estimation.priors import LogNormalPrior
from mirt.estimation.quadrature import GaussHermiteQuadrature


def _random_second_moment(n_dimensions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(n_dimensions, n_dimensions + 3))
    return factor @ factor.T / (n_dimensions + 3)


def _deviance(cov: np.ndarray, moment: np.ndarray) -> float:
    sign, log_det = np.linalg.slogdet(cov)
    if sign <= 0:
        return np.inf
    return float(log_det + np.trace(np.linalg.solve(cov, moment)))


def _reference_optimum(
    moment: np.ndarray, start: np.ndarray, free: np.ndarray
) -> np.ndarray:
    """Direct numerical maximum over the free upper-triangle entries."""
    rows, cols = np.nonzero(np.triu(free))

    def build(values: np.ndarray) -> np.ndarray:
        cov = start.copy()
        cov[rows, cols] = values
        cov[cols, rows] = values
        return cov

    result = minimize(
        lambda values: _deviance(build(values), moment),
        start[rows, cols],
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 40000},
    )
    return build(result.x)


@pytest.mark.parametrize(
    "free",
    [
        # Chain F1-F2-F3 with free variances: no closed form.
        np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=bool),
        # Unit-variance correlation matrix with every pair free.
        np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool),
        # One fixed variance, one zero correlation.
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=bool),
    ],
)
def test_constrained_covariance_reaches_the_numerical_optimum(free) -> None:
    moment = _random_second_moment(3, seed=4)
    start = np.eye(3)

    estimate = _constrained_gaussian_covariance(moment, start, free)
    reference = _reference_optimum(moment, start, free)

    assert_allclose(estimate[~free], start[~free])
    assert _deviance(estimate, moment) <= _deviance(reference, moment) + 1e-9
    assert_allclose(estimate, reference, atol=1e-4)


def test_constrained_covariance_uses_block_closed_form() -> None:
    moment = _random_second_moment(4, seed=1)
    free = np.zeros((4, 4), dtype=bool)
    free[:2, :2] = True
    free[2:, 2:] = True

    estimate = _constrained_gaussian_covariance(moment, np.eye(4), free)

    assert_allclose(estimate, np.where(free, moment, 0.0))
    reference = _reference_optimum(moment, np.eye(4), free)
    assert _deviance(estimate, moment) <= _deviance(reference, moment) + 1e-12


def test_density_validates_its_free_mask() -> None:
    with pytest.raises(ValueError, match="Boolean"):
        FactorCovarianceDensity(2, free=np.ones((2, 2)))
    with pytest.raises(ValueError, match="shape"):
        FactorCovarianceDensity(2, free=np.ones((3, 3), dtype=bool))
    with pytest.raises(ValueError, match="symmetric"):
        FactorCovarianceDensity(2, free=np.array([[False, True], [False, False]]))


def test_density_counts_free_entries_and_reports_correlation() -> None:
    free = np.array([[True, True, False], [True, False, False], [False] * 3])
    cov = np.array([[4.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    density = FactorCovarianceDensity(3, free=free, cov=cov)

    assert density.n_parameters == 2
    assert FactorCovarianceDensity(3).n_parameters == 3
    assert density.estimate_cov
    assert_allclose(density.correlation[0, 1], 0.5)
    assert_allclose(np.diag(density.correlation), 1.0)
    assert_allclose(density.free, free)


def test_update_estimates_correlation_with_fixed_unit_variances() -> None:
    quadrature = GaussHermiteQuadrature(n_points=21, n_dimensions=2)
    target = np.array([[1.0, 0.3], [0.3, 1.0]])
    weights = np.exp(
        GaussianDensity(cov=target).log_quadrature_mass(
            quadrature.nodes, quadrature.weights
        )
    )
    density = FactorCovarianceDensity(2)

    density.update(quadrature.nodes, weights)

    assert_allclose(np.diag(density.cov), 1.0)
    assert_allclose(density.cov[0, 1], 0.3, atol=1e-8)


def test_update_estimates_a_free_variance() -> None:
    quadrature = GaussHermiteQuadrature(n_points=15, n_dimensions=1)
    weights = np.exp(
        GaussianDensity(cov=np.array([[2.0]])).log_quadrature_mass(
            quadrature.nodes, quadrature.weights
        )
    )
    density = FactorCovarianceDensity(1, free=np.array([[True]]))

    density.update(quadrature.nodes, weights)

    assert_allclose(density.cov, [[2.0]], rtol=1e-4)
    assert density.n_parameters == 1


def _simulate_cfa(seed: int, n_persons: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    loadings = np.zeros((8, 2))
    loadings[:4, 0] = rng.uniform(1.0, 2.0, 4)
    loadings[4:, 1] = rng.uniform(1.0, 2.0, 4)
    theta = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_persons)
    logits = theta @ loadings.T + rng.normal(size=8)
    responses = (rng.random(logits.shape) < 1 / (1 + np.exp(-logits))).astype(int)
    return responses, loadings > 0


def _marginal_log_likelihood(model, responses, cov, n_points) -> float:
    quadrature = GaussHermiteQuadrature(n_points=n_points, n_dimensions=2)
    log_mass = GaussianDensity(cov=cov).log_quadrature_mass(
        quadrature.nodes, quadrature.weights
    )
    probabilities = np.clip(model.probability(quadrature.nodes), 1e-12, 1 - 1e-12)
    log_likelihood = (
        responses @ np.log(probabilities).T
        + (1 - responses) @ np.log(1 - probabilities).T
    )
    return float(np.sum(logsumexp(log_likelihood + log_mass, axis=1)))


def _fit(responses, pattern, density, tol, priors):
    model = MultidimensionalModel(
        8, 2, model_type="confirmatory", loading_pattern=pattern.astype(float)
    )
    estimator = EMEstimator(
        n_quadpts=11,
        tol=tol,
        max_iter=300,
        latent_density=density,
        compute_standard_errors=False,
        item_priors=priors,
    )
    return estimator, estimator.fit(model, responses)


@pytest.mark.parametrize("priors", [None, {"slopes": LogNormalPrior(0.0, 0.2)}])
def test_correlation_em_ascends_and_converges(priors) -> None:
    # Regression: rescaling the slopes after each M-step (parameter-expanded
    # EM) cycled without converging under quadrature, and with slope priors
    # it moved away from the posterior mode.
    responses, pattern = _simulate_cfa(seed=1, n_persons=600)
    density = FactorCovarianceDensity(2)

    estimator, result = _fit(responses, pattern, density, tol=1e-8, priors=priors)

    assert result.converged
    assert np.all(np.diff(estimator._convergence_history) >= -1e-9)
    assert result.n_parameters == 8 + 8 + 1
    rho = density.cov[0, 1]

    def log_likelihood(value: float) -> float:
        cov = np.array([[1.0, value], [value, 1.0]])
        return _marginal_log_likelihood(result.model, responses, cov, 11)

    best = log_likelihood(rho)
    assert best == pytest.approx(result.log_likelihood, abs=1e-6)
    assert best > log_likelihood(rho - 0.005)
    assert best > log_likelihood(rho + 0.005)
