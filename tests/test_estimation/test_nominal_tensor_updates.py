"""Regression coverage for category-by-factor item parameters."""

import numpy as np
import pytest
from scipy.special import logsumexp

from mirt.estimation.em import EMEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.estimation.se_methods import compute_se
from mirt.estimation.weighted import WeightedEMEstimator
from mirt.exceptions import MirtValidationError
from mirt.models.polytomous import NominalResponseModel


def _expected_curvature(model, points, responses, posterior, weights):
    slopes = np.zeros_like(model.slopes)
    intercepts = np.zeros_like(model.intercepts)
    for item, k in enumerate(model.n_categories):
        valid = responses[:, item] >= 0
        total = (posterior[valid] * weights[valid, None]).sum(axis=0)
        probability = model.probability(points, item)
        assert np.all((probability > 1e-10) & (probability < 1 - 1e-10))
        information = total[:, None] * probability * (1 - probability)
        intercepts[item, 1:k] = 1 / np.sqrt(information[:, 1:].sum(axis=0))
        slopes[item, 1:k] = 1 / np.sqrt(information[:, 1:].T @ points**2)
    return {"slopes": slopes, "intercepts": intercepts}


@pytest.mark.parametrize("factors", [2, 3])
def test_tensor_item_parameters_roundtrip_without_aliasing(factors):
    model = NominalResponseModel(2, n_categories=[2, 4], n_factors=factors)
    original = model.slopes.copy()
    values = np.arange(4 * factors, dtype=float).reshape(4, factors) / 10
    expected = values.copy()
    model.set_item_parameter(0, "slopes", values)
    values.fill(99.0)
    actual = model.get_item_parameters(0)
    np.testing.assert_array_equal(actual["slopes"], expected)
    np.testing.assert_array_equal(model.slopes[1], original[1])
    actual["slopes"].fill(-99.0)
    np.testing.assert_array_equal(model.slopes[0], expected)
    snapshot = model.slopes.copy()
    with pytest.raises(MirtValidationError, match="Invalid per-item value"):
        model.set_item_parameter(0, "slopes", np.ones((5, factors)))
    np.testing.assert_array_equal(model.slopes, snapshot)


@pytest.mark.parametrize("weighted", [False, True])
def test_tensor_standard_errors_match_independent_softmax_curvature(weighted):
    model = NominalResponseModel(2, n_categories=[2, 4], n_factors=2)
    estimator = (
        WeightedEMEstimator(n_quadpts=5) if weighted else EMEstimator(n_quadpts=5)
    )
    estimator._quadrature = GaussHermiteQuadrature(5, 2)
    points = estimator._quadrature.nodes
    rng = np.random.default_rng(619)
    responses = np.column_stack([rng.integers(-1, k, 31) for k in model.n_categories])
    posterior = rng.uniform(0.1, 1.0, (31, len(points)))
    posterior /= posterior.sum(axis=1, keepdims=True)
    weights = rng.uniform(0.2, 2.0, 31) if weighted else np.ones(31)
    weights[::7] = 0.0 if weighted else 1.0
    original = model.parameters
    if weighted:
        actual = estimator._compute_weighted_standard_errors(
            model, responses, posterior, weights
        )
    else:
        actual = estimator._compute_standard_errors(model, responses, posterior)
    expected = _expected_curvature(model, points, responses, posterior, weights)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=3e-4, atol=1e-6)
    for name, values in original.items():
        np.testing.assert_array_equal(model.parameters[name], values)


@pytest.mark.parametrize("method", ["central", "forward", "richardson"])
@pytest.mark.parametrize("jobs", [1, 2])
def test_standalone_numerical_methods_preserve_tensor_coordinates(method, jobs):
    model = NominalResponseModel(2, n_categories=[2, 4], n_factors=2)
    quadrature = GaussHermiteQuadrature(5, 2)
    rng = np.random.default_rng(831)
    responses = np.column_stack([rng.integers(-1, k, 40) for k in model.n_categories])
    posterior = rng.uniform(0.1, 1.0, (40, len(quadrature.nodes)))
    posterior /= posterior.sum(axis=1, keepdims=True)
    original = model.parameters
    actual = compute_se(
        model,
        responses,
        quadrature,
        posterior,
        method=method,
        step_size=1e-4,
        n_jobs=jobs,
    )
    expected = _expected_curvature(
        model, quadrature.nodes, responses, posterior, np.ones(40)
    )
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=3e-4, atol=1e-6)
        np.testing.assert_array_equal(model.parameters[name], original[name])


def test_failed_standalone_tensor_curvature_restores_parameters():
    model = NominalResponseModel(1, n_categories=3, n_factors=2)
    original = model.parameters
    probability = model.probability

    def fail(*args):
        if not np.array_equal(model.slopes, original["slopes"]):
            raise RuntimeError("failed tensor trial")
        return probability(*args)

    model.probability = fail
    quadrature = GaussHermiteQuadrature(5, 2)
    responses = np.array([[0], [1], [2]])
    posterior = np.full((3, 25), 1 / 25)
    with pytest.raises(RuntimeError, match="failed tensor trial"):
        compute_se(model, responses, quadrature, posterior)
    for name, values in original.items():
        np.testing.assert_array_equal(model.parameters[name], values)


@pytest.mark.parametrize(("weighted", "jobs"), [(False, 1), (False, 2), (True, 1)])
def test_multidimensional_nominal_fit_reports_final_parameters_and_curvature(
    weighted, jobs
):
    model = NominalResponseModel(3, n_categories=[2, 3, 4], n_factors=2)
    rng = np.random.default_rng(754)
    responses = np.column_stack([rng.integers(0, k, 80) for k in model.n_categories])
    responses[rng.random(responses.shape) < 0.1] = -1
    weights = rng.uniform(0.2, 2.0, 80)
    estimator = (
        WeightedEMEstimator(n_quadpts=5, max_iter=3, tol=1e-12)
        if weighted
        else EMEstimator(n_quadpts=5, max_iter=3, tol=1e-12, n_jobs=jobs, use_gpu=False)
    )
    result = (
        estimator.fit(model, responses, weights=weights)
        if weighted
        else estimator.fit(model, responses)
    )
    assert model.is_fitted
    assert result.n_iterations == 3
    assert np.all(np.isfinite(estimator.convergence_history))
    probabilities = model.probability(rng.normal(size=(11, 2)))
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0)
    quadrature = estimator._quadrature
    marginals = logsumexp(
        model.log_likelihood_batch(responses, quadrature.nodes)
        + np.log(quadrature.weights),
        axis=1,
    )
    expected = float(estimator._weights @ marginals) if weighted else marginals.sum()
    assert result.log_likelihood == pytest.approx(expected, abs=1e-8)
    for name, values in model.parameters.items():
        se = result.standard_errors[name]
        assert se.shape == values.shape
        np.testing.assert_array_equal(se[~model.free_parameter_masks[name]], 0.0)
        np.testing.assert_array_equal(values[:, 0], 0.0)
    for item, k in enumerate(model.n_categories):
        np.testing.assert_array_equal(model.slopes[item, k:], 0.0)
        np.testing.assert_array_equal(model.intercepts[item, k:], 0.0)
