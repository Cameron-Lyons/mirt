"""Native model restrictions preserve identification, copies, and diagnostics."""

from copy import deepcopy

import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss
from numpy.testing import assert_array_equal
from scipy.special import expit, logsumexp

from mirt._model_defaults import uses_builtin_model_hooks
from mirt.estimation.bl import BLEstimator
from mirt.estimation.em import EMEstimator
from mirt.exceptions import MirtValidationError
from mirt.models.dichotomous import (
    FourParameterLogistic,
    OneParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.multidimensional import MultidimensionalModel
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    NominalResponseModel,
    PartialCreditModel,
)


@pytest.mark.parametrize(
    "constructor",
    [
        OneParameterLogistic,
        TwoParameterLogistic,
        ThreeParameterLogistic,
        FourParameterLogistic,
        GradedResponseModel,
        GeneralizedPartialCredit,
        PartialCreditModel,
        NominalResponseModel,
        MultidimensionalModel,
    ],
)
def test_restrictions_preserve_native_types_and_structural_masks_after_copy(
    constructor,
):
    kwargs = {"n_items": 3}
    if constructor in {
        GradedResponseModel,
        GeneralizedPartialCredit,
        PartialCreditModel,
        NominalResponseModel,
    }:
        kwargs["n_categories"] = [2, 4, 3]
    if constructor is MultidimensionalModel:
        kwargs["n_factors"] = 2
        kwargs["loading_pattern"] = np.array([[1, 0], [1, 1], [0, 1]])
    model = constructor(**kwargs)
    family_masks = model.free_parameter_masks
    original = model.parameters
    restrictions = {name: mask.copy() for name, mask in family_masks.items()}
    for mask in restrictions.values():
        mask[1] = False

    assert model.set_free_parameter_masks(restrictions) is model
    assert uses_builtin_model_hooks(model, likelihood=True)
    for name, mask in restrictions.items():
        assert_array_equal(model.free_parameter_masks[name], mask)
        assert_array_equal(model.parameters[name], original[name])
    for cloned in [model.copy(), deepcopy(model)]:
        assert type(cloned) is type(model)
        assert uses_builtin_model_hooks(cloned, likelihood=True)
        for name, mask in restrictions.items():
            assert_array_equal(cloned.free_parameter_masks[name], mask)
        cloned.set_free_parameter_masks(None)
        for name, mask in family_masks.items():
            assert_array_equal(cloned.free_parameter_masks[name], mask)
        for name, mask in restrictions.items():
            assert_array_equal(model.free_parameter_masks[name], mask)
    for mask in restrictions.values():
        mask.fill(True)
    exposed = model.free_parameter_masks
    for mask in exposed.values():
        mask.fill(True)
    assert all(not np.any(mask[1]) for mask in model.free_parameter_masks.values())
    model.set_free_parameter_masks(None)
    for name, mask in family_masks.items():
        assert_array_equal(model.free_parameter_masks[name], mask)


@pytest.mark.parametrize(
    "invalid",
    [
        [],
        {"unknown": np.zeros(3, dtype=bool)},
        {"difficulty": np.zeros(3, dtype=int)},
        {"difficulty": np.zeros((3, 1), dtype=bool)},
        {"difficulty": np.zeros(2, dtype=bool)},
        {"discrimination": np.ones(3, dtype=bool)},
    ],
)
def test_mask_validation_is_atomic_and_cannot_free_identification_constraints(invalid):
    model = OneParameterLogistic(3)
    model.set_free_parameter_masks({"difficulty": np.array([False, True, True])})
    original = model.free_parameter_masks

    with pytest.raises(MirtValidationError):
        model.set_free_parameter_masks(invalid)

    for name, mask in original.items():
        assert_array_equal(model.free_parameter_masks[name], mask)


def test_unchanged_family_masks_do_not_disable_prepared_parameter_layouts():
    model = GradedResponseModel(3, [2, 4, 3])
    masks = model.free_parameter_masks
    model.set_free_parameter_masks(masks)
    assert model._free_parameter_restrictions == {}


@pytest.mark.parametrize("constructor", [GradedResponseModel, NominalResponseModel])
def test_polytomous_restrictions_cannot_free_padding_or_reference_categories(
    constructor,
):
    model = constructor(2, [2, 4])
    name = "thresholds" if constructor is GradedResponseModel else "slopes"
    masks = model.free_parameter_masks
    masks[name][0, -1] = True
    with pytest.raises(MirtValidationError, match="model-family fixed"):
        model.set_free_parameter_masks(masks)
    if constructor is NominalResponseModel:
        masks = model.free_parameter_masks
        masks[name][:, 0] = True
        with pytest.raises(MirtValidationError, match="model-family fixed"):
            model.set_free_parameter_masks(masks)


def test_native_em_preserves_restricted_item_and_estimates_remaining_item():
    rng = np.random.default_rng(6479)
    true_a = np.array([0.8, 1.2, 1.6])
    true_b = np.array([-0.7, 0.0, 0.8])
    theta = rng.normal(size=700)
    probabilities = expit(true_a * (theta[:, None] - true_b))
    responses = (rng.random(probabilities.shape) < probabilities).astype(int)
    model = TwoParameterLogistic(3)
    model.set_parameters(discrimination=true_a, difficulty=true_b + [0, 0.5, 0.5])
    restrictions = model.free_parameter_masks
    for mask in restrictions.values():
        mask[0] = False
    model.set_free_parameter_masks(restrictions)
    model._is_fitted = True
    initial = model.parameters

    result = EMEstimator(
        use_rust=False, use_gpu=False, compute_standard_errors=False
    ).fit(model.copy(), responses)

    assert result.converged
    assert result.n_parameters == 4
    assert result.model.discrimination[0] == initial["discrimination"][0]
    assert result.model.difficulty[0] == initial["difficulty"][0]
    assert np.max(np.abs(result.model.difficulty[1:] - initial["difficulty"][1:])) > 0.2


@pytest.mark.parametrize("constructor", [TwoParameterLogistic, ThreeParameterLogistic])
@pytest.mark.parametrize("use_rust", [False, True])
def test_partial_binary_coordinates_fit_and_se_match_independent_curvature(
    constructor, use_rust
):
    rng = np.random.default_rng(5937)
    model = constructor(4)
    values = {
        "discrimination": np.array([0.8, 1.1, 1.4, 1.7]),
        "difficulty": np.array([-1.0, -0.3, 0.4, 1.1]),
    }
    if constructor is ThreeParameterLogistic:
        values["guessing"] = np.full(4, 0.18)
    model.set_parameters(**values)
    theta = rng.normal(size=700)
    probability = model.probability(theta)
    responses = (rng.random(probability.shape) < probability).astype(int)
    model.set_parameters(difficulty=values["difficulty"] + [0.0, 0.45, 0.45, 0.45])
    masks = {
        name: np.zeros_like(value, dtype=bool)
        for name, value in model.parameters.items()
    }
    masks["difficulty"][1:] = True
    model.set_free_parameter_masks(masks)
    model._is_fitted = True
    initial = model.parameters

    result = EMEstimator(n_quadpts=31, use_rust=use_rust, use_gpu=False).fit(
        model.copy(), responses
    )

    assert result.converged
    assert result.n_parameters == 3
    for name, mask in masks.items():
        assert_array_equal(result.model.parameters[name][~mask], initial[name][~mask])
        assert_array_equal(result.standard_errors[name][~mask], 0.0)
    assert np.max(np.abs(result.model.difficulty[1:] - initial["difficulty"][1:])) > 0.2
    nodes, weights = hermgauss(31)
    points = np.sqrt(2.0) * nodes
    a, b = result.model.discrimination, result.model.difficulty
    c = result.model.parameters.get("guessing", np.zeros(4))
    probabilities = c + (1 - c) * expit(a * (points[:, None] - b))
    log_joint = (
        responses @ np.log(probabilities).T
        + (1 - responses) @ np.log1p(-probabilities).T
        + np.log(weights / np.sqrt(np.pi))
    )
    posterior = np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))
    observed = posterior.sum(axis=0)
    correct = responses.T @ posterior
    for item in range(1, 4):

        def objective(location):
            p = c[item] + (1 - c[item]) * expit(a[item] * (points - location))
            return np.sum(
                correct[item] * np.log(p) + (observed - correct[item]) * np.log1p(-p)
            )

        h = 1e-4
        curvature = (
            objective(b[item] + h) - 2 * objective(b[item]) + objective(b[item] - h)
        ) / h**2
        expected_se = np.sqrt(-1.0 / curvature)
        assert result.standard_errors["difficulty"][item] == pytest.approx(
            expected_se, rel=1e-4
        )


def test_partial_grm_thresholds_preserve_fixed_neighbors_and_zero_standard_errors():
    rng = np.random.default_rng(931)
    model = GradedResponseModel(4, [2, 4, 3, 3])
    thresholds = np.array(
        [[0.0, 0.0, 0.0], [-1.2, 0.0, 1.2], [-0.6, 0.6, 0.0], [-0.8, 0.8, 0.0]]
    )
    model.set_parameters(
        discrimination=np.array([0.8, 1.1, 1.4, 1.7]), thresholds=thresholds
    )
    theta = rng.normal(size=700)
    probability = model.probability(theta)
    draw = rng.random(probability.shape[:2])
    responses = np.sum(draw[:, :, None] > np.cumsum(probability, axis=2), axis=2)
    starts = thresholds.copy()
    starts[1, 1] = 0.5
    starts[2:, :2] += 0.4
    model.set_parameters(thresholds=starts)
    masks = model.free_parameter_masks
    masks["discrimination"].fill(False)
    masks["thresholds"][0].fill(False)
    masks["thresholds"][1, [0, 2]] = False
    model.set_free_parameter_masks(masks)
    model._is_fitted = True
    initial = model.parameters

    result = EMEstimator(n_quadpts=31, use_rust=True, use_gpu=False).fit(
        model.copy(), responses
    )

    assert result.converged
    assert result.n_parameters == 5
    for name, mask in masks.items():
        assert_array_equal(result.model.parameters[name][~mask], initial[name][~mask])
        assert_array_equal(result.standard_errors[name][~mask], 0.0)
        assert np.all(np.isfinite(result.standard_errors[name][mask]))
        assert np.all(result.standard_errors[name][mask] > 0)
    assert abs(result.model.thresholds[1, 1] - starts[1, 1]) > 0.2
    assert np.all(np.diff(result.model.thresholds[1]) > 0)


@pytest.mark.parametrize("warm_start", [False, True])
def test_partial_binary_coordinates_fit_with_joint_bl_optimizer(warm_start):
    rng = np.random.default_rng(9034)
    a = np.array([0.8, 1.1, 1.4, 1.7])
    b = np.array([-1.0, -0.3, 0.4, 1.1])
    theta = rng.normal(size=600)
    probability = expit(a * (theta[:, None] - b))
    responses = (rng.random(probability.shape) < probability).astype(int)
    model = TwoParameterLogistic(4)
    model.set_parameters(discrimination=a, difficulty=b + [0.0, 0.5, 0.5, 0.5])
    masks = model.free_parameter_masks
    masks["discrimination"].fill(False)
    masks["difficulty"][0] = False
    model.set_free_parameter_masks(masks)
    model._is_fitted = warm_start
    initial = model.parameters

    result = BLEstimator(n_quadpts=31).fit(model.copy(), responses)

    assert result.converged
    assert result.n_parameters == 3
    for name, mask in masks.items():
        assert_array_equal(result.model.parameters[name][~mask], initial[name][~mask])
        assert_array_equal(result.standard_errors[name][~mask], 0.0)
    assert np.max(np.abs(result.model.difficulty[1:] - initial["difficulty"][1:])) > 0.2
