"""Equivalence of one-hot categorical batch likelihoods with per-item loops."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

import mirt._categorical as categorical
from mirt._categorical import (
    categorical_log_likelihood_batch,
    category_offsets,
    item_category_table,
)
from mirt.constants import PROB_EPSILON
from mirt.models.base import PolytomousItemModel
from mirt.models.custom import CustomItemModel, create_item_type
from mirt.models.nested import FourPLNestedLogit, TwoPLNestedLogit
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedRatingScaleModel,
    GradedResponseModel,
    NominalResponseModel,
    RatingScaleModel,
)
from mirt.models.sequential import (
    AdjacentCategoryModel,
    SequentialResponseModel,
)
from mirt.models.unfolding import GeneralizedGradedUnfolding


def _old_base_batch(
    model: PolytomousItemModel, responses: NDArray, theta: NDArray[np.float64]
) -> NDArray[np.float64]:
    responses = model._validate_polytomous_responses(responses)
    n_theta = model._ensure_theta_2d(theta).shape[0]
    result = np.zeros((responses.shape[0], n_theta))
    for item_idx in range(model.n_items):
        probabilities = np.clip(
            model.probability(theta, item_idx), PROB_EPSILON, 1 - PROB_EPSILON
        )
        log_probabilities = np.log(probabilities)
        valid = responses[:, item_idx] >= 0
        if np.any(valid):
            codes = responses[valid, item_idx].astype(np.intp)
            result[valid, :] += log_probabilities[:, codes].T
    return result


def _old_nested_batch(
    model: TwoPLNestedLogit, responses: NDArray, theta: NDArray[np.float64]
) -> NDArray[np.float64]:
    values = model._validate_responses(responses)
    theta_values = model._validate_theta(theta)
    result = np.zeros((values.shape[0], theta_values.size))
    for item_idx in range(model.n_items):
        probability, _ = model._item_curves_from_theta(theta_values, item_idx)
        log_probability = np.log(np.clip(probability, PROB_EPSILON, 1.0))
        observed = values[:, item_idx] >= 0
        if np.any(observed):
            result[observed] += log_probability[:, values[observed, item_idx]].T
    return result


def _old_ggum_batch(
    model: GeneralizedGradedUnfolding, responses: NDArray, theta: NDArray[np.float64]
) -> NDArray[np.float64]:
    codes, observed = model._validated_responses(responses)
    log_probabilities = np.log(np.clip(model.probability(theta), PROB_EPSILON, 1.0))
    result = np.zeros((codes.shape[0], log_probabilities.shape[0]))
    for item in range(model.n_items):
        valid = observed[:, item]
        if np.any(valid):
            result[valid] += log_probabilities[:, item, codes[valid, item]].T
    return result


def _old_custom_batch(
    model: CustomItemModel, responses: NDArray, theta: NDArray[np.float64]
) -> NDArray[np.float64]:
    values = model._validated_responses(responses)
    theta_2d = model._validated_theta(theta)
    valid = values >= 0
    probabilities = model.probability(theta_2d)
    result = np.zeros((values.shape[0], theta_2d.shape[0]))
    safe = np.where(valid, values, 0)
    for item in range(model.n_items):
        log_probabilities = np.log(
            np.clip(probabilities[:, item, :], PROB_EPSILON, None)
        )
        contribution = log_probabilities[:, safe[:, item]].T
        result += np.where(valid[:, item, None], contribution, 0.0)
    return result


def _old_sequential_batch(
    model: SequentialResponseModel, responses: NDArray, theta: NDArray[np.float64]
) -> NDArray[np.float64]:
    safe, observed = model._validated_responses(responses)
    probabilities = model.probability(theta)
    log_probabilities = np.log(np.clip(probabilities, PROB_EPSILON, 1.0))
    result = np.zeros((safe.shape[0], probabilities.shape[0]))
    for category in range(model.max_categories):
        mask = (safe == category) & observed
        if np.any(mask):
            result += mask @ log_probabilities[:, :, category].T
    return result


def _perturb(model: PolytomousItemModel, seed: int) -> None:
    rng = np.random.default_rng(seed)
    parameters = model.parameters
    if "discrimination" in parameters:
        values = parameters["discrimination"]
        parameters["discrimination"] = values * rng.uniform(0.6, 1.6, values.shape)
    for name in ("thresholds", "steps", "intercepts", "difficulty"):
        if name in parameters:
            values = parameters[name]
            parameters[name] = values + rng.normal(scale=0.2, size=values.shape)
            if name == "thresholds" and values.ndim == 2:
                parameters[name] = np.sort(parameters[name], axis=1)
    if "slopes" in parameters:
        values = parameters["slopes"]
        parameters["slopes"] = values + rng.normal(scale=0.3, size=values.shape)
    model.set_parameters(**parameters)


def _responses(
    counts: list[int], n_persons: int, seed: int, missing: float = 0.15
) -> NDArray[np.int_]:
    rng = np.random.default_rng(seed)
    responses = np.column_stack(
        [rng.integers(0, count, size=n_persons) for count in counts]
    )
    responses[rng.random(responses.shape) < missing] = -1
    responses[0] = -1
    return responses


BASE_FACTORIES: list[Callable[[], PolytomousItemModel]] = [
    lambda: GradedResponseModel(7, [2, 5, 3, 4, 3, 6, 2]),
    lambda: GradedResponseModel(7, [2, 5, 3, 4, 3, 6, 2], n_factors=2),
    lambda: GeneralizedPartialCredit(7, [4, 2, 5, 3, 3, 2, 4], n_factors=3),
    lambda: NominalResponseModel(9, [3, 4, 2, 5, 3, 4, 2, 3, 5]),
    lambda: NominalResponseModel(9, [3, 4, 2, 5, 3, 4, 2, 3, 5], n_factors=2),
    lambda: RatingScaleModel(6, 4),
    lambda: GradedRatingScaleModel(6, 5),
]


def _theta_grid(n_factors: int, n_points: int) -> NDArray[np.float64]:
    grid = np.linspace(-2.5, 2.5, n_points)
    if n_factors == 1:
        return grid
    return np.column_stack([np.roll(grid, shift) for shift in range(n_factors)])


@pytest.mark.parametrize("factory", BASE_FACTORIES)
@pytest.mark.parametrize("n_points", [1, 13])
def test_base_batch_likelihood_matches_item_loop_exactly(
    factory: Callable[[], PolytomousItemModel], n_points: int
) -> None:
    model = factory()
    _perturb(model, 3)
    theta = _theta_grid(model.n_factors, n_points)
    responses = _responses(model.n_categories, 40, seed=5)

    actual = PolytomousItemModel.log_likelihood_batch(model, responses, theta)
    expected = _old_base_batch(model, responses, theta)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[0], 0.0)

    # Native sigmoid evaluation can differ from NumPy by a few floating-point ulps.
    dispatched = model.log_likelihood_batch(responses, theta)
    np.testing.assert_allclose(dispatched, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(dispatched[0], 0.0)


@pytest.mark.parametrize("factory", BASE_FACTORIES[:4])
def test_base_batch_likelihood_is_block_invariant(
    factory: Callable[[], PolytomousItemModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    model = factory()
    _perturb(model, 7)
    theta = _theta_grid(model.n_factors, 9)
    responses = _responses(model.n_categories, 23, seed=11)
    expected = _old_base_batch(model, responses, theta)

    monkeypatch.setattr(categorical, "_LIKELIHOOD_BLOCK_ELEMENTS", 20)
    actual = PolytomousItemModel.log_likelihood_batch(model, responses, theta)

    np.testing.assert_array_equal(actual, expected)


def test_base_batch_likelihood_accepts_float_and_single_rows() -> None:
    model = GradedResponseModel(5, [3, 2, 4, 3, 5], n_factors=2)
    _perturb(model, 13)
    theta = _theta_grid(2, 6)
    integer_responses = np.array([[2, -1, 3, 0, 4]])
    float_responses = np.array([[2.0, np.nan, 3.0, 0.0, 4.0]])

    actual = PolytomousItemModel.log_likelihood_batch(model, float_responses, theta)
    expected = _old_base_batch(model, integer_responses, theta)

    assert actual.shape == (1, 6)
    np.testing.assert_array_equal(actual, expected)


def test_base_batch_likelihood_accepts_boolean_binary_codes() -> None:
    model = GeneralizedPartialCredit(4, 2)
    _perturb(model, 17)
    theta = np.linspace(-2.0, 2.0, 5)
    responses = np.array([[True, False, True, True], [False, False, True, False]])

    actual = PolytomousItemModel.log_likelihood_batch(model, responses, theta)
    expected = _old_base_batch(model, responses.astype(int), theta)

    np.testing.assert_array_equal(actual, expected)


def test_base_batch_likelihood_supports_tables_wider_than_kernel_blocks() -> None:
    counts = [2, 3, 4, 5, 6] * 30
    model = GradedResponseModel(len(counts), counts, n_factors=2)
    _perturb(model, 19)
    theta = _theta_grid(2, 25)
    responses = _responses(counts, 60, seed=23, missing=0.3)
    assert sum(counts) > 512

    actual = PolytomousItemModel.log_likelihood_batch(model, responses, theta)
    expected = _old_base_batch(model, responses, theta)

    np.testing.assert_array_equal(actual, expected)


def test_base_batch_likelihood_honors_overridden_item_curves() -> None:
    class ShiftedGRM(GradedResponseModel):
        def probability(self, theta, item_idx=None):
            if item_idx is None:
                return super().probability(theta)
            return super().probability(np.asarray(theta) + 0.5, item_idx)

    model = ShiftedGRM(4, [3, 4, 2, 5])
    theta = np.linspace(-1.0, 1.0, 5)
    responses = _responses(model.n_categories, 12, seed=29)

    np.testing.assert_array_equal(
        model.log_likelihood_batch(responses, theta),
        _old_base_batch(model, responses, theta),
    )
    np.testing.assert_array_equal(
        model.log_likelihood_batch(responses, theta)[1:],
        _old_base_batch(GradedResponseModel(4, [3, 4, 2, 5]), responses, theta + 0.5)[
            1:
        ],
    )


@pytest.mark.parametrize("model_class", [TwoPLNestedLogit, FourPLNestedLogit])
def test_nested_batch_likelihood_matches_item_loop_exactly(
    model_class: type[TwoPLNestedLogit], monkeypatch: pytest.MonkeyPatch
) -> None:
    counts = [3, 4, 5, 4, 2, 6]
    model = model_class(6, counts, correct_response=[0, 3, 2, 1, 1, 5])
    model.set_parameters(
        discrimination=np.linspace(0.6, 1.8, 6),
        difficulty=np.linspace(-1.0, 1.0, 6),
        distractor_slopes=np.linspace(-1.0, 1.0, 36).reshape(6, 6),
        distractor_intercepts=np.cos(np.arange(36.0)).reshape(6, 6),
    )
    theta = np.linspace(-3.0, 3.0, 17)
    responses = _responses(counts, 31, seed=31)
    expected = _old_nested_batch(model, responses, theta)

    np.testing.assert_array_equal(
        model.log_likelihood_batch(responses, theta), expected
    )
    monkeypatch.setattr(categorical, "_LIKELIHOOD_BLOCK_ELEMENTS", 1)
    np.testing.assert_array_equal(
        model.log_likelihood_batch(responses, theta), expected
    )


def test_ggum_batch_likelihood_matches_item_loop_exactly() -> None:
    counts = [2, 3, 4, 5, 3, 4]
    model = GeneralizedGradedUnfolding(6, counts)
    model.set_parameters(
        discrimination=np.linspace(0.7, 1.9, 6),
        location=np.linspace(-1.5, 1.5, 6),
    )
    theta = np.linspace(-3.0, 3.0, 21)
    responses = _responses(counts, 33, seed=37).astype(np.float64)
    responses[3, 2] = np.nan

    np.testing.assert_array_equal(
        model.log_likelihood_batch(responses, theta),
        _old_ggum_batch(model, responses, theta),
    )
    np.testing.assert_array_equal(
        model.log_likelihood_batch(responses[:1], theta[:1]), np.zeros((1, 1))
    )


def _ordinal(theta: NDArray[np.float64], shift: float) -> NDArray[np.float64]:
    eta = np.ravel(theta) - shift
    weights = np.column_stack((np.ones_like(eta), np.exp(eta), np.exp(2.0 * eta)))
    return weights / weights.sum(axis=1, keepdims=True)


def test_custom_polytomous_batch_likelihood_matches_item_loop_exactly() -> None:
    spec = create_item_type(
        "Ordinal",
        _ordinal,
        par_names=["shift"],
        par_defaults={"shift": 0.0},
        n_categories=3,
    )
    model = CustomItemModel(5, spec).set_parameters(shift=[-0.5, 0.0, 0.75, 1.5, -1.0])
    theta = np.linspace(-2.0, 2.0, 11)
    responses = _responses([3] * 5, 19, seed=41)

    np.testing.assert_array_equal(
        model.log_likelihood_batch(responses, theta),
        _old_custom_batch(model, responses, theta),
    )


@pytest.mark.parametrize(
    "model_class", [SequentialResponseModel, AdjacentCategoryModel]
)
def test_sequential_batch_likelihood_matches_category_masks(
    model_class: type[SequentialResponseModel],
) -> None:
    counts = [2, 5, 3, 4, 6, 3]
    model = model_class(6, counts)
    model.set_parameters(discrimination=np.linspace(0.6, 1.7, 6))
    theta = np.linspace(-3.0, 3.0, 15)
    responses = _responses(counts, 27, seed=43)

    np.testing.assert_allclose(
        model.log_likelihood_batch(responses, theta),
        _old_sequential_batch(model, responses, theta),
        rtol=0.0,
        atol=1e-12,
    )


def test_categorical_kernel_ignores_unselected_table_entries() -> None:
    log_table = np.array(
        [
            [-0.1, -0.2],
            [np.nan, np.inf],
            [-0.3, -0.4],
            [-0.5, -0.6],
            [-0.7, -0.8],
        ]
    )
    offsets = category_offsets([2, 3])
    codes = np.array([[0, 2], [-1, 0], [0, -1], [-1, -1]])

    result = categorical_log_likelihood_batch(log_table, offsets, codes)

    np.testing.assert_array_equal(
        result,
        [[-0.1 + -0.7, -0.2 + -0.8], [-0.3, -0.4], [-0.1, -0.2], [0.0, 0.0]],
    )


def test_categorical_kernel_uses_explicit_observation_mask() -> None:
    log_table = np.log(np.array([[0.2], [0.8], [0.5], [0.25], [0.25]]))
    codes = np.array([[1, 0], [0, 2]])
    observed = np.array([[True, False], [False, True]])

    result = categorical_log_likelihood_batch(
        log_table, category_offsets([2, 3]), codes, observed
    )

    np.testing.assert_array_equal(result, [[np.log(0.8)], [np.log(0.25)]])
    empty = categorical_log_likelihood_batch(
        log_table, category_offsets([2, 3]), codes[:0]
    )
    assert empty.shape == (0, 1)


def test_item_category_table_stacks_active_categories_item_major() -> None:
    values = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)

    table = item_category_table(values, [2, 4, 3])

    expected = np.concatenate(
        [values[:, 0, :2].T, values[:, 1, :4].T, values[:, 2, :3].T]
    )
    np.testing.assert_array_equal(table, expected)
    assert table.flags.c_contiguous and table.flags.owndata
    np.testing.assert_array_equal(category_offsets([2, 4, 3]), [0, 2, 6])
