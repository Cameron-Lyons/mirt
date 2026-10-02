"""Independent raw response-curve oracles for identified spline/polynomial storage."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.interpolate import BSpline
from scipy.special import comb, expit

from mirt.constants import PROB_EPSILON
from mirt.estimation.bl import BLEstimator
from mirt.models.nonparametric import (
    MonotonicPolynomialModel,
    MonotonicSplineModel,
)


def _raw_increments(logs):
    weights = np.exp(
        np.maximum(
            logs - np.max(logs, axis=1, keepdims=True),
            np.log(np.nextafter(0.0, 1.0)),
        )
    )
    return weights / weights.sum(axis=1, keepdims=True)


def _raw_polynomial(theta, logs, location, scale, lower, upper):
    """Evaluate the originally supplied Bernstein formula and exact derivative."""
    degree = logs.shape[1] - 1
    coefficients = np.cumsum(_raw_increments(logs), axis=1)
    t = expit(scale[None, :] * (theta[:, None] - location[None, :]))
    probability = np.zeros_like(t)
    derivative = np.zeros_like(t)
    for k in range(degree + 1):
        probability += (
            comb(degree, k) * t**k * (1 - t) ** (degree - k) * coefficients[:, k]
        )
    for k in range(degree):
        derivative += (
            degree
            * comb(degree - 1, k)
            * t**k
            * (1 - t) ** (degree - 1 - k)
            * (coefficients[:, k + 1] - coefficients[:, k])
        )
    probability = lower + (upper - lower) * probability
    derivative *= (upper - lower) * scale * t * (1 - t)
    information = derivative**2 / (probability * (1 - probability) + PROB_EPSILON)
    return probability, information


@pytest.mark.parametrize("degree", [1, 5, 13, 40])
def test_polynomial_identification_preserves_supplied_raw_formula(degree):
    rng = np.random.default_rng(190 + degree)
    logs = rng.normal(0, 2, size=(3, degree + 1)) + np.array([40, -50, 2])[:, None]
    location = np.array([-0.7, 0.2, 1.3])
    scale = np.array([0.4, 1.2, 2.8])
    lower = np.array([0.0, 0.12, 0.35])
    upper = np.array([1.0, 0.92, 0.83])
    theta = np.linspace(-7, 7, 31)
    expected_p, expected_i = _raw_polynomial(theta, logs, location, scale, lower, upper)
    original_logs = logs.copy()
    model = MonotonicPolynomialModel(3, degree).set_parameters(
        log_coefficients=logs,
        location=location,
        scale=scale,
        lower=lower,
        upper=upper,
    )

    assert_allclose(model.probability(theta), expected_p, rtol=2e-11, atol=2e-12)
    assert_allclose(model.information(theta), expected_i, rtol=4e-10, atol=2e-12)
    assert_array_equal(model.parameters["log_coefficients"][:, 0], 0.0)
    assert_array_equal(model.lower, 0.0)
    assert_array_equal(logs, original_logs)
    assert model.n_parameters == 3 * (degree + 3)
    for item in range(3):
        assert_allclose(model.probability(theta, item), expected_p[:, item], atol=2e-12)
        assert_allclose(model.information(theta, item), expected_i[:, item], atol=2e-12)


def test_polynomial_equivalent_lower_and_increment_gauges_identify_same_model():
    logs = np.array([[0.4, -0.6, 1.1, -0.8], [-0.2, 0.8, 0.5, -1.0]])
    lower = np.array([0.1, 0.3])
    upper = np.array([0.9, 0.8])
    increments = _raw_increments(logs)
    identified_increments = increments * (1 - lower / upper)[:, None]
    identified_increments[:, 0] += lower / upper
    raw = MonotonicPolynomialModel(2, 3).set_parameters(
        log_coefficients=logs, lower=lower, upper=upper
    )
    equivalent = MonotonicPolynomialModel(2, 3).set_parameters(
        log_coefficients=np.log(identified_increments) + np.array([9, -7])[:, None],
        upper=upper,
    )

    for name, values in raw.parameters.items():
        assert_allclose(values, equivalent.parameters[name], atol=3e-15)
    assert_allclose(
        raw.probability(np.linspace(-5, 5, 21)),
        equivalent.probability(np.linspace(-5, 5, 21)),
    )


def test_spline_identification_preserves_independent_integrated_bspline_formula():
    n_knots, degree = 3, 2
    n_basis = n_knots + degree + 1
    interior = np.linspace(-3, 3, n_knots + 2)[1:-1]
    knots = np.r_[np.full(degree + 1, -4), interior, np.full(degree + 1, 4)]
    raw_basis = BSpline(knots, np.eye(n_basis), degree)
    integral = raw_basis.antiderivative()
    norm = integral(4) - integral(-4)
    theta = np.linspace(-5, 5, 31)
    basis = (integral(np.clip(theta, -4, 4)) - integral(-4)) / norm
    basis_derivative = raw_basis(np.clip(theta, -4, 4)) / norm
    basis_derivative[np.abs(theta) > 4] = 0
    logs = np.array([[10, 9, 7, 8, 9, 6], [-7, -6, -3, -5, -4, -2]], dtype=float)
    lower = np.array([0.1, 0.25])
    upper = np.array([0.9, 0.85])
    increments = _raw_increments(logs)
    expected_p = lower + (upper - lower) * (basis @ increments.T)
    derivative = (upper - lower) * (basis_derivative @ increments.T)
    expected_i = derivative**2 / (expected_p * (1 - expected_p) + PROB_EPSILON)

    model = MonotonicSplineModel(2, n_knots, degree).set_parameters(
        log_weights=logs, lower=lower, upper=upper
    )
    shifted = model.copy().set_parameters(
        log_weights=logs + np.array([30, -20])[:, None]
    )

    assert_allclose(model.probability(theta), expected_p, atol=1e-15)
    assert_allclose(model.information(theta), expected_i, atol=1e-15)
    assert_allclose(model.parameters["log_weights"], logs - logs[:, :1], atol=1e-15)
    assert_allclose(
        shifted.parameters["log_weights"], model.parameters["log_weights"], atol=1e-15
    )
    assert model.n_parameters == 2 * (n_basis + 1)


@pytest.mark.parametrize("family", ["spline", "polynomial"])
def test_identified_curve_masks_copies_and_valid_estimator_unpacking(family):
    if family == "spline":
        model = MonotonicSplineModel(2, n_knots=2, degree=2)
        name = "log_weights"
        structural_count = 12
    else:
        model = MonotonicPolynomialModel(2, degree=3)
        name = "log_coefficients"
        structural_count = 12
    intrinsic = model.free_parameter_masks
    assert not np.any(intrinsic[name][:, 0])
    assert np.all(intrinsic[name][:, 1:])
    assert model.n_parameters == structural_count
    restricted = intrinsic[name].copy()
    restricted[0, 1] = False
    model.set_free_parameter_masks({name: restricted})
    restricted[1, 1] = False
    copy = model.copy()
    assert model.n_parameters == copy.n_parameters == structural_count - 1
    copy.set_free_parameter_masks(None)
    assert copy.n_parameters == structural_count
    assert model.n_parameters == structural_count - 1
    with pytest.raises(ValueError, match="cannot free"):
        model.set_free_parameter_masks({name: np.ones_like(intrinsic[name])})

    estimator = BLEstimator(n_quadpts=5, max_iter=1)
    vector, _, layout = estimator._flatten_parameters(model)
    assert vector.size == model.n_parameters
    vector[layout[name]["start_idx"] : layout[name]["end_idx"]] += 0.2
    estimator._unflatten_parameters(model, vector, layout)
    assert_array_equal(model.parameters[name][:, 0], 0)
    assert model.parameters[name][0, 1] == 0
    assert np.all(np.isfinite(model.probability(np.array([-2, 0, 2]))))


def test_extreme_polynomial_bounds_and_logs_preserve_raw_curves():
    logs = np.array([[1000, -1000, 500, 0], [-800, 900, 0, 1]], dtype=float)
    lower = np.array([0.2, np.nextafter(0.9, 0)])
    upper = np.array([0.8, 0.9])
    theta = np.array([-1000, -2, 0, 2, 1000], dtype=float)
    expected_p, expected_i = _raw_polynomial(
        theta, logs, np.zeros(2), np.ones(2), lower, upper
    )
    model = MonotonicPolynomialModel(2, 3).set_parameters(
        log_coefficients=logs, lower=lower, upper=upper
    )
    assert np.all(np.isfinite(model.parameters["log_coefficients"]))
    assert_allclose(model.probability(theta), expected_p, atol=1e-14)
    assert_allclose(model.information(theta), expected_i, atol=1e-14)
