"""Moment tangents use public parameter hooks and independent derivatives."""

from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import expit

from mirt import compute_m2
from mirt.diagnostics.modelfit import _model_moment_jacobian, _moment_design
from mirt.models.cdm_advanced import GDINA
from mirt.models.dichotomous import ThreeParameterLogistic


class _DomainCanonical3PL(ThreeParameterLogistic):
    def _canonical_parameter_values(self, name, values):
        canonical = super()._canonical_parameter_values(name, values)
        if name == "guessing" and np.any((canonical < 0) | (canonical > 1)):
            raise ValueError("custom canonical hook requires guessing in [0,1]")
        return canonical


def test_canonical_boundary_uses_valid_one_sided_moment_derivative():
    rng = np.random.default_rng(53169)
    model = _DomainCanonical3PL(7).set_parameters(
        discrimination=np.linspace(0.75, 1.45, 7),
        difficulty=np.linspace(-1.3, 1.2, 7),
        guessing=np.zeros(7),
    )
    restrictions = {
        name: np.zeros_like(mask) for name, mask in model.free_parameter_masks.items()
    }
    restrictions["guessing"][0] = True
    model.set_free_parameter_masks(restrictions)
    theta = rng.normal(size=(200, 1))
    probabilities = expit(
        model.discrimination[None, :] * (theta - model.difficulty[None, :])
    )
    responses = (rng.random((200, 7)) < probabilities).astype(int)
    original = model.parameters
    # At c_0=0, dp_0/dc_0=1-logistic(a_0*(theta-b_0)); every other
    # item's derivative is zero. Pair-product derivatives follow the product
    # rule, which is exact here because each feature contains item 0 at most once.
    dp = np.zeros_like(probabilities)
    dp[:, 0] = 1 - probabilities[:, 0]
    expected = np.r_[
        dp.mean(axis=0),
        [
            (dp[:, j] * probabilities[:, k] + probabilities[:, j] * dp[:, k]).mean()
            for j in range(7)
            for k in range(j + 1, 7)
        ],
    ]
    jacobian = _model_moment_jacobian(
        model, responses, theta, 11, _moment_design(responses)
    )
    assert jacobian.shape == (28, 1)
    assert_allclose(jacobian[:, 0], expected, rtol=2e-9, atol=2e-11)
    result = compute_m2(model, responses, theta=theta)
    assert result["df"] == 27
    assert np.isfinite(result["M2"])
    for name, values in original.items():
        assert_array_equal(model.parameters[name], values)


class _CachedGDINA(GDINA):
    def __init__(self, *args, **kwargs):
        self.setter_calls = 0
        super().__init__(*args, **kwargs)

    def set_parameters(self, **params):
        self.setter_calls += 1
        return super().set_parameters(**params)


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("restrict", [False, True])
def test_cached_gdina_fixed_attribute_moment_derivatives_and_isolation(
    missing, restrict
):
    q_matrix = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    model = _CachedGDINA(4, 2, q_matrix)
    coefficients = [
        np.array([0.23, 0.71]),
        np.array([0.18, 0.62]),
        np.array([0.15, 0.31, 0.52, 0.87]),
        np.array([0.38]),
    ]
    delta = model.parameters["delta"]
    for item, values in enumerate(coefficients):
        delta[item, : len(values)] = values
    model.set_parameters(delta=delta)
    if restrict:
        mask = model.free_parameter_masks["delta"]
        mask[2, 2] = False
        model.set_free_parameter_masks({"delta": mask})
    theta = np.repeat(
        np.array([[0, 0], [1, 0], [0, 1], [1, 1]]), [20, 30, 10, 40], axis=0
    )
    # Conditional saturated probabilities are a lookup indexed by the required
    # binary attributes, with the first required attribute as the low bit.
    groups = np.column_stack(
        (
            theta[:, 0],
            theta[:, 1],
            theta[:, 0] + 2 * theta[:, 1],
            np.zeros(len(theta), dtype=int),
        )
    )
    probabilities = np.column_stack(
        [values[groups[:, item]] for item, values in enumerate(coefficients)]
    )
    rng = np.random.default_rng(819743)
    responses = (rng.random(probabilities.shape) < probabilities).astype(float)
    if missing:
        responses[::7, 0] = -1
        responses[3::11, 2] = np.nan
    observed = np.isfinite(responses) & (responses >= 0)
    availability = np.column_stack(
        [
            observed,
            *[
                observed[:, j] & observed[:, k]
                for j in range(4)
                for k in range(j + 1, 4)
            ],
        ]
    )
    coordinates = np.argwhere(model.free_parameter_masks["delta"])
    expected = np.empty((10, len(coordinates)))
    for column, (item, coefficient) in enumerate(coordinates):
        dp = np.zeros_like(probabilities)
        dp[:, item] = groups[:, item] == coefficient
        features = np.column_stack(
            [
                dp,
                *[
                    dp[:, j] * probabilities[:, k] + probabilities[:, j] * dp[:, k]
                    for j in range(4)
                    for k in range(j + 1, 4)
                ],
            ]
        )
        expected[:, column] = (features * availability).sum(axis=0) / availability.sum(
            axis=0
        )
    stored = deepcopy(model.parameters)
    caches = model.delta_parameters
    cache_ids = [id(values) for values in model._delta_params]
    setter_calls = model.setter_calls
    assert_allclose(model.probability(theta), probabilities)
    for _ in range(2):
        actual = _model_moment_jacobian(
            model, responses, theta, 11, _moment_design(responses)
        )
        assert_allclose(actual, expected, rtol=2e-9, atol=2e-11)
        result = compute_m2(model, responses, theta=theta)
        assert result["df"] == 10 - np.linalg.matrix_rank(expected, tol=1e-9)
        assert np.isfinite(result["M2"])
        assert model.setter_calls == setter_calls
        assert [id(values) for values in model._delta_params] == cache_ids
        for name, values in stored.items():
            assert_array_equal(model.parameters[name], values)
        for before, after in zip(caches, model.delta_parameters, strict=True):
            assert_array_equal(after, before)
        assert_allclose(model.probability(theta), probabilities)


def test_public_gdina_setter_refreshes_caches_without_touching_its_original_copy():
    model = _CachedGDINA(2, 1, np.ones((2, 1), dtype=int))
    theta = np.array([[0], [1]])
    before = model.probability(theta)
    copied = deepcopy(model)
    delta = copied.parameters["delta"]
    delta[0, 1] += 0.07
    copied.set_parameters(delta=delta)
    expected = before.copy()
    expected[1, 0] += 0.07
    assert_allclose(copied.probability(theta), expected)
    assert_array_equal(model.probability(theta), before)
    assert not np.shares_memory(model._delta_params[0], copied._delta_params[0])
