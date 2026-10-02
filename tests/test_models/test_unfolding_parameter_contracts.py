"""Independent unfolding equations identify estimated model coordinates."""

from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mirt.estimation.bl import BLEstimator
from mirt.estimation.em import EMEstimator
from mirt.estimation.standard_errors import _flatten_parameters, _set_flat_parameters
from mirt.exceptions import MirtValidationError
from mirt.models.unfolding import (
    GeneralizedGradedUnfolding,
    HyperbolicCosineModel,
    IdealPointModel,
)
from mirt.multigroup.estimator import MultigroupEMEstimator


def _ggum_reference(theta, alpha, location, independent):
    """Enumerate subjective categories from the defining GGUM equation."""
    c = len(independent)
    m = 2 * c + 1
    tau = np.zeros(m + 1)
    tau[1 : c + 1] = independent
    for index in range(1, c + 1):
        tau[m - index + 1] = -tau[index]
    f = np.exp(
        alpha
        * (
            np.arange(m + 1)[None, :] * (np.asarray(theta)[:, None] - location)
            - np.cumsum(tau)[None, :]
        )
    )
    weights = np.column_stack([f[:, z] + f[:, m - z] for z in range(c + 1)])
    return weights / weights.sum(axis=1, keepdims=True)


def _random_ggum(counts):
    rng = np.random.default_rng(92016)
    model = GeneralizedGradedUnfolding(len(counts), counts)
    model.set_parameters(
        discrimination=rng.uniform(0.65, 1.6, len(counts)),
        location=rng.uniform(-0.8, 0.8, len(counts)),
    )
    for item, count in enumerate(counts):
        # Deliberately irregular: first-half values need not be equally spaced.
        model.set_independent_thresholds(item, rng.uniform(-2.3, -0.15, count - 1))
    return model


@pytest.mark.parametrize("counts", [[2], [3, 5, 4], [6, 2, 3, 4, 6, 5]])
def test_ggum_counts_only_independent_subjective_thresholds(counts):
    model = GeneralizedGradedUnfolding(len(counts), counts)
    expected = np.zeros(model.thresholds.shape, dtype=bool)
    for item, categories in enumerate(counts):
        expected[item, : categories - 1] = True
    masks = model.free_parameter_masks
    assert_array_equal(masks["thresholds"], expected)
    assert_array_equal(masks["discrimination"], np.ones(len(counts), dtype=bool))
    assert_array_equal(masks["location"], np.ones(len(counts), dtype=bool))
    assert model.n_parameters == 2 * len(counts) + sum(count - 1 for count in counts)


def test_ggum_raw_equation_and_item_group_paths_ignore_derived_storage_noise():
    # Four items of every width exercise the vectorized all-items path.
    counts = [3, 5, 4] * 4
    model = _random_ggum(counts)
    theta = np.linspace(-2.6, 3.1, 19)
    expected = np.zeros((len(theta), len(counts), max(counts)))
    for item, categories in enumerate(counts):
        expected[:, item, :categories] = _ggum_reference(
            theta,
            model.discrimination[item],
            model.location[item],
            model.independent_thresholds(item),
        )
    assert_allclose(model.probability(theta), expected, atol=1e-14)
    for item, categories in enumerate(counts):
        assert_allclose(model.probability(theta, item), expected[:, item, :categories])
    original_information = model.information(theta)
    original_storage = model.parameters["thresholds"]
    rng = np.random.default_rng(427391)
    derived = ~model.free_parameter_masks["thresholds"]
    model._parameters["thresholds"][derived] += rng.normal(0, 3, derived.sum())
    assert_allclose(model.probability(theta), expected, atol=1e-14)
    assert_array_equal(model.information(theta), original_information)
    for item, categories in enumerate(counts):
        assert_allclose(model.probability(theta, item), expected[:, item, :categories])
    canonical = model._canonical_parameter_values("thresholds", model.thresholds)
    assert_array_equal(canonical, original_storage)
    # User-facing full matrices retain strict validation of the canonical form.
    with pytest.raises(MirtValidationError, match="center threshold"):
        model.set_parameters(thresholds=model.thresholds)


def test_ggum_coordinate_derivatives_match_independent_symmetric_equation():
    model = _random_ggum([3, 5, 4])
    theta = np.linspace(-2.6, 3.1, 23)
    step = 1e-6
    for item, categories in enumerate(model.n_categories):
        alpha = model.discrimination[item]
        location = model.location[item]
        independent = model.independent_thresholds(item)
        reference_columns = []
        for coordinate in range(categories + 1):
            upper = [alpha, location, *independent]
            lower = upper.copy()
            upper[coordinate] += step
            lower[coordinate] -= step
            derivative = (
                _ggum_reference(theta, upper[0], upper[1], upper[2:])
                - _ggum_reference(theta, lower[0], lower[1], lower[2:])
            ) / (2 * step)
            reference_columns.append(derivative.ravel())
            worker = deepcopy(model)
            if coordinate < 2:
                name = "discrimination" if coordinate == 0 else "location"
                worker._parameters[name][item] += step
                actual_upper = worker.probability(theta, item)
                worker._parameters[name][item] -= 2 * step
            else:
                # The stored reflected partner is intentionally left unchanged,
                # as in numerical moment/Hessian coordinate perturbations.
                worker._parameters["thresholds"][item, coordinate - 2] += step
                actual_upper = worker.probability(theta, item)
                worker._parameters["thresholds"][item, coordinate - 2] -= 2 * step
            actual = (actual_upper - worker.probability(theta, item)) / (2 * step)
            assert_allclose(actual, derivative, atol=2e-9)
        assert (
            np.linalg.matrix_rank(np.column_stack(reference_columns), tol=1e-8)
            == categories + 1
        )


@pytest.mark.parametrize("column", [2, 3, 5])
def test_ggum_constraints_cannot_be_freed_by_explicit_masks(column):
    model = GeneralizedGradedUnfolding(3, [3, 5, 4])
    mask = model.free_parameter_masks["thresholds"]
    mask[0, column] = True  # Center, reflection, or padding respectively.
    with pytest.raises(MirtValidationError, match="model-family fixed"):
        model.set_free_parameter_masks({"thresholds": mask})
    assert model.n_parameters == 15


def test_ggum_restrictions_and_copies_preserve_independent_counts():
    model = _random_ggum([3, 5, 4])
    masks = model.free_parameter_masks
    masks["discrimination"][0] = False
    masks["thresholds"][1, 2] = False
    before = model.probability(np.array([-1.0, 0.0, 1.0]))
    model.set_free_parameter_masks(masks)
    assert model.n_parameters == 13
    for copied in [model.copy(), deepcopy(model)]:
        assert copied.n_parameters == 13
        assert_array_equal(copied.probability(np.array([-1.0, 0.0, 1.0])), before)
        copied.set_free_parameter_masks(None)
        assert copied.n_parameters == 15
        assert model.n_parameters == 13
    masks["thresholds"].fill(True)
    assert model.n_parameters == 13
    model.set_free_parameter_masks(None)
    assert model.n_parameters == 15


@pytest.mark.parametrize("adapter", ["EM", "BL", "SE", "multigroup"])
def test_ggum_optimizer_updates_reconstruct_dependent_thresholds(adapter):
    model = _random_ggum([3, 5, 4])
    fixed = model.free_parameter_masks["thresholds"]
    fixed[1, 1] = False
    model.set_free_parameter_masks({"thresholds": fixed})
    original = model.parameters
    target = original["thresholds"].copy()
    target[1, 2] += 0.21
    if adapter == "EM":
        estimator = EMEstimator(compute_standard_errors=False)
        parameters, _ = estimator._get_item_params_and_bounds(model, 1)
        # a, location, and thresholds 0, 2, 3 are the five independent slots.
        assert len(parameters) == 5
        parameters[3] += 0.21
        estimator._set_item_params(model, 1, parameters)
    elif adapter == "BL":
        estimator = BLEstimator()
        parameters, _, layout = estimator._flatten_parameters(model)
        indices = layout["thresholds"]["free_indices"]
        position = np.flatnonzero(
            indices == np.ravel_multi_index((1, 2), target.shape)
        )[0]
        parameters[layout["thresholds"]["start_idx"] + position] += 0.21
        estimator._unflatten_parameters(model, parameters, layout)
    elif adapter == "SE":
        parameters, layouts = _flatten_parameters(model)
        index = list(np.flatnonzero(fixed)).index(
            np.ravel_multi_index((1, 2), target.shape)
        )
        parameters[6 + index] += 0.21
        _set_flat_parameters(model, parameters, layouts)
    else:
        MultigroupEMEstimator()._set_param(model, 1, "thresholds", target[1])
    assert model.n_parameters == 14
    assert_array_equal(model.discrimination, original["discrimination"])
    assert_array_equal(model.location, original["location"])
    for item, categories in enumerate(model.n_categories):
        independent = target[item, : categories - 1]
        expected = np.r_[independent, 0.0, -independent[::-1]]
        assert_array_equal(model.thresholds_for_item(item), expected)
        assert_array_equal(model.thresholds[item, 2 * categories - 1 :], 0)
    model.set_parameters(thresholds=model.thresholds)  # Stored form is valid.


def test_ggum_fit_reports_independent_aic_count_and_exact_pattern_likelihood():
    rng = np.random.default_rng(100923)
    model = _random_ggum([3, 5, 4])
    theta = rng.normal(size=160)
    probabilities = model.probability(theta)
    responses = (rng.random((160, 3, 1)) > probabilities.cumsum(axis=2)).sum(axis=2)
    fitted = EMEstimator(
        n_quadpts=11, max_iter=3, compute_standard_errors=False, use_rust=False
    ).fit(model, responses)
    nodes, weights = np.polynomial.hermite.hermgauss(11)
    nodes = nodes * np.sqrt(2)
    weights /= np.sqrt(np.pi)
    joint = np.ones((len(responses), len(nodes)))
    for item in range(3):
        raw = _ggum_reference(
            nodes,
            model.discrimination[item],
            model.location[item],
            model.independent_thresholds(item),
        )
        joint *= raw[:, responses[:, item]].T
    reference_ll = np.log(joint @ weights).sum()
    assert fitted.n_parameters == model.n_parameters == 15
    assert_allclose(fitted.log_likelihood, reference_ll, atol=1e-10)
    assert_allclose(fitted.aic, -2 * reference_ll + 2 * 15)
    assert_allclose(fitted.bic, -2 * reference_ll + 15 * np.log(len(responses)))


def test_hcm_gauge_spans_exact_two_dimensional_curve_family():
    theta = np.linspace(-2.3, 2.8, 21)
    a, location, gamma = 1.35, 0.4, -0.65
    z = a * (theta - location) - gamma
    p = 1 / (1 + np.cosh(z))
    dp_dz = -np.sinh(z) / (1 + np.cosh(z)) ** 2
    jacobian = np.column_stack((dp_dz * (theta - location), -a * dp_dz, -dp_dz))
    assert np.linalg.matrix_rank(jacobian, tol=1e-10) == 2
    assert np.linalg.matrix_rank(jacobian[:, :2], tol=1e-10) == 2
    assert_allclose(jacobian[:, 2], jacobian[:, 1] / a)
    model = HyperbolicCosineModel(1).set_parameters(
        discrimination=np.array([a]),
        location=np.array([location]),
        asymmetry=np.array([gamma]),
    )
    assert_allclose(model.probability(theta, 0), p)
    assert model.n_parameters == 2
    assert_array_equal(model.free_parameter_masks["asymmetry"], [False])
    # A gauge shift changes the stored representation while preserving curves.
    shift = 0.73
    model.set_parameters(
        location=np.array([location - shift / a]), asymmetry=np.array([gamma + shift])
    )
    assert_allclose(model.probability(theta, 0), p)
    assert_allclose(model.peak_location, [location + gamma / a])
    with pytest.raises(MirtValidationError, match="model-family fixed"):
        model.set_free_parameter_masks({"asymmetry": np.array([True])})


@pytest.mark.parametrize("method", ["EM", "BL"])
def test_hcm_fits_preserve_supplied_gauge_and_report_two_parameters_per_item(method):
    rng = np.random.default_rng(153798)
    gauge = np.array([0.4, -0.7, 0.9, 1.2])
    model = HyperbolicCosineModel(4).set_parameters(
        discrimination=np.array([0.8, 1.2, 1.4, 1.0]),
        location=np.array([-1.0, 0.0, 0.5, 1.2]),
        asymmetry=gauge,
    )
    theta = rng.normal(size=250)
    responses = (rng.random((250, 4)) < model.probability(theta)).astype(int)
    estimator = (
        EMEstimator(
            n_quadpts=11, max_iter=3, compute_standard_errors=False, use_rust=False
        )
        if method == "EM"
        else BLEstimator(n_quadpts=11, max_iter=3)
    )
    fitted = estimator.fit(model, responses)
    assert_array_equal(model.asymmetry, gauge)
    assert fitted.n_parameters == model.n_parameters == 8
    assert np.isfinite(fitted.log_likelihood)
    assert_allclose(fitted.aic, -2 * fitted.log_likelihood + 2 * 8)
    copied = model.copy()
    assert copied.n_parameters == 8
    assert_array_equal(copied.asymmetry, gauge)
    assert_array_equal(copied.probability(theta), model.probability(theta))


def test_ideal_point_has_three_independent_active_coordinates():
    theta = np.linspace(-2.2, 2.7, 21)
    a, location, height = 1.2, 0.35, 0.8
    p = height * np.exp(-a * (theta - location) ** 2)
    jacobian = np.column_stack(
        (-p * (theta - location) ** 2, 2 * a * (theta - location) * p, p / height)
    )
    assert np.linalg.matrix_rank(jacobian, tol=1e-10) == 3
    model = IdealPointModel(1).set_parameters(
        discrimination=np.array([a]),
        location=np.array([location]),
        peak_height=np.array([height]),
    )
    assert model.n_parameters == 3
    assert all(np.all(mask) for mask in model.free_parameter_masks.values())
    assert_allclose(model.probability(theta, 0), p)
