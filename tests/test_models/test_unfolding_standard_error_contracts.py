"""Derived threshold uncertainty follows the exact GGUM reflection map."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mirt.estimation.bl import BLEstimator
from mirt.estimation.em import EMEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.estimation.se_methods import compute_se
from mirt.estimation.standard_errors import (
    _flatten_parameters,
    _unflatten_se,
    compute_crossprod_se,
)
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.unfolding import GeneralizedGradedUnfolding


def test_ggum_standard_error_expansion_matches_full_linear_covariance():
    model = GeneralizedGradedUnfolding(3, [3, 5, 4])
    parameters = model.parameters
    coordinates = [
        (name, index)
        for name, mask in model.free_parameter_masks.items()
        for index in np.flatnonzero(mask)
    ]
    rng = np.random.default_rng(77791)
    factor = rng.normal(size=(15, 15))
    independent_covariance = factor @ factor.T
    # Build the linear full-storage map independently: threshold mirrors have
    # coefficient -1 on the reversed first half; center and padding are zero.
    expansion = np.zeros((sum(values.size for values in parameters.values()), 15))
    offset = 0
    for name, values in parameters.items():
        for flat_index in range(values.size):
            coordinate = (name, flat_index)
            if coordinate in coordinates:
                expansion[offset + flat_index, coordinates.index(coordinate)] = 1
            elif name == "thresholds":
                item, threshold = np.unravel_index(flat_index, values.shape)
                c = model.n_categories[item] - 1
                if c < threshold <= 2 * c:
                    source = np.ravel_multi_index(
                        (item, 2 * c - threshold), values.shape
                    )
                    expansion[
                        offset + flat_index, coordinates.index((name, source))
                    ] = -1
        offset += values.size
    full_covariance = expansion @ independent_covariance @ expansion.T
    expected = np.sqrt(np.diag(full_covariance))
    _, layouts = _flatten_parameters(model)
    standard_errors = _unflatten_se(
        np.sqrt(np.diag(independent_covariance)), layouts, model
    )
    actual = np.concatenate([standard_errors[name].ravel() for name in parameters])
    assert_allclose(actual, expected, atol=1e-14)
    assert np.all(standard_errors["thresholds"][1, 5:] > 0)


def test_ggum_expansion_preserves_unknown_errors_and_caller_arrays():
    model = GeneralizedGradedUnfolding(3, [3, 5, 4])
    errors = np.zeros_like(model.thresholds)
    errors[0, :2] = [np.nan, np.inf]
    errors[1, :4] = [0.0, 0.2, 0.3, 0.4]
    original = errors.copy()
    expanded = model._expand_parameter_standard_errors("thresholds", errors)
    assert_array_equal(errors, original)
    assert not np.shares_memory(expanded, errors)
    assert_allclose(expanded[0], [np.nan, np.inf, 0, np.inf, np.nan, 0, 0, 0, 0])
    assert_allclose(expanded[1], [0, 0.2, 0.3, 0.4, 0, 0.4, 0.3, 0.2, 0])
    assert_array_equal(expanded[2], 0)


@pytest.mark.parametrize("shape", [(), (3,), (2, 3)])
def test_default_standard_error_hook_preserves_shape_and_values(shape):
    errors = np.full(shape, np.nan)
    model = TwoParameterLogistic(3)
    expanded = model._expand_parameter_standard_errors("custom", errors)
    assert expanded.shape == shape
    assert_array_equal(expanded, errors)
    assert not np.shares_memory(expanded, errors)


def _raw_ggum(theta, alpha, location, independent):
    c = len(independent)
    m = 2 * c + 1
    tau = np.r_[0.0, independent, 0.0, -independent[::-1]]
    f = np.exp(
        alpha
        * (
            np.arange(m + 1)[None, :] * (theta[:, None] - location)
            - np.cumsum(tau)[None, :]
        )
    )
    weights = np.column_stack(
        [f[:, category] + f[:, m - category] for category in range(c + 1)]
    )
    return weights / weights.sum(axis=1, keepdims=True)


def _restricted_model_data():
    model = GeneralizedGradedUnfolding(3, [3, 5, 4])
    model.set_parameters(
        discrimination=np.array([0.7, 1.4, 1.05]), location=np.array([-1.1, 0.5, 1.3])
    )
    for item, independent in enumerate(
        [[-1.5, -0.55], [-2.4, -1.3, -0.75, -0.35], [-1.8, -0.85, -0.2]]
    ):
        model.set_independent_thresholds(item, independent)
    restrictions = {
        name: np.zeros_like(mask) for name, mask in model.free_parameter_masks.items()
    }
    restrictions["thresholds"][1, 1] = True
    model.set_free_parameter_masks(restrictions)
    rng = np.random.default_rng(61072)
    probabilities = model.probability(rng.normal(size=320))
    responses = (rng.random((320, 3, 1)) > probabilities.cumsum(axis=2)).sum(axis=2)
    return model, responses


def _posterior(model, responses, grid):
    joint = np.ones((len(responses), len(grid.nodes)))
    for item in range(model.n_items):
        raw = _raw_ggum(
            grid.nodes[:, 0],
            model.discrimination[item],
            model.location[item],
            model.independent_thresholds(item),
        )
        joint *= raw[:, responses[:, item]].T
    joint *= grid.weights
    return joint / joint.sum(axis=1, keepdims=True)


def test_public_crossprod_error_matches_independent_score_and_reflected_covariance():
    model, responses = _restricted_model_data()
    grid = GaussHermiteQuadrature(11)
    nodes = grid.nodes[:, 0]
    step = 1e-5
    log_probabilities = []
    for offset in (step, -step):
        joint = np.ones((len(responses), len(nodes)))
        for item in range(3):
            independent = model.independent_thresholds(item)
            if item == 1:
                independent[1] += offset
            raw = _raw_ggum(
                nodes, model.discrimination[item], model.location[item], independent
            )
            joint *= raw[:, responses[:, item]].T
        log_probabilities.append(np.log(joint @ grid.weights))
    score = (log_probabilities[0] - log_probabilities[1]) / (2 * step)
    independent_variance = 1 / (score @ score)
    expected = np.sqrt(independent_variance)
    original = model.parameters
    posterior = _posterior(model, responses, grid)
    errors = compute_crossprod_se(
        model, responses, posterior, grid, prior_mass=grid.weights
    )
    assert model.n_parameters == 1
    assert_allclose(errors["thresholds"][1, [1, 7]], expected, rtol=2e-8)
    mask = np.zeros(model.thresholds.shape, dtype=bool)
    mask[1, [1, 7]] = True
    assert_array_equal(errors["thresholds"][~mask], 0)
    assert_array_equal(errors["discrimination"], 0)
    assert_array_equal(errors["location"], 0)
    for name, values in original.items():
        assert_array_equal(model.parameters[name], values)


@pytest.mark.parametrize("method", ["EM", "BL", "numerical"])
def test_computed_ggum_errors_propagate_reflection_and_keep_fixed_coordinates_zero(
    method,
):
    model, responses = _restricted_model_data()
    if method == "EM":
        errors = (
            EMEstimator(
                n_quadpts=11, max_iter=3, compute_standard_errors=True, use_rust=False
            )
            .fit(model, responses)
            .standard_errors
        )
    elif method == "BL":
        errors = (
            BLEstimator(n_quadpts=11, max_iter=3).fit(model, responses).standard_errors
        )
    else:
        grid = GaussHermiteQuadrature(11)
        posterior = _posterior(model, responses, grid)
        errors = compute_se(model, responses, grid, posterior, method="numerical")
    assert errors["thresholds"][1, 1] > 0
    assert_allclose(errors["thresholds"][1, 7], errors["thresholds"][1, 1])
    fixed = np.ones(model.thresholds.shape, dtype=bool)
    fixed[1, [1, 7]] = False
    assert_array_equal(errors["thresholds"][fixed], 0)
    assert_array_equal(errors["discrimination"], 0)
    assert_array_equal(errors["location"], 0)
