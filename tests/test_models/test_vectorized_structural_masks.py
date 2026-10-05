"""Vectorized structural masks and canonical values match the item loops."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from mirt.models.base import PolytomousItemModel
from mirt.models.nested import FourPLNestedLogit, TwoPLNestedLogit
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    NominalResponseModel,
)
from mirt.models.unfolding import GeneralizedGradedUnfolding

COUNTS = [3, 2, 5, 4, 5, 2, 3]


def _old_padding_masks(model: PolytomousItemModel, name: str) -> NDArray[np.bool_]:
    mask = np.zeros(model.parameters[name].shape, dtype=np.bool_)
    for item, count in enumerate(model.n_categories):
        mask[item, : count - 1] = True
    return mask


def _old_padding_canonical(
    model: PolytomousItemModel, values: NDArray[np.float64]
) -> NDArray[np.float64]:
    canonical = values.copy()
    for item, count in enumerate(model.n_categories):
        canonical[item, count - 1 :] = 0.0
    return canonical


def _old_nominal_masks(model: NominalResponseModel) -> dict[str, NDArray[np.bool_]]:
    slopes = np.zeros(model.slopes.shape, dtype=np.bool_)
    intercepts = np.zeros(model.intercepts.shape, dtype=np.bool_)
    for item, count in enumerate(model.n_categories):
        slopes[item, 1:count, ...] = True
        intercepts[item, 1:count] = True
    return {"slopes": slopes, "intercepts": intercepts}


def _old_nominal_canonical(
    model: NominalResponseModel, values: NDArray[np.float64]
) -> NDArray[np.float64]:
    canonical = values.copy()
    for item, count in enumerate(model.n_categories):
        reference = canonical[item, 0].copy()
        canonical[item, :count] -= reference
        canonical[item, count:] = 0.0
    return canonical


def _old_nested_mask(model: TwoPLNestedLogit, name: str) -> NDArray[np.bool_]:
    active = np.zeros(model.parameters[name].shape, dtype=np.bool_)
    for item, (count, correct) in enumerate(
        zip(model.n_categories, model.correct_response, strict=True)
    ):
        reference = 0 if correct != 0 else 1
        active[item, :count] = True
        active[item, [correct, reference]] = False
    return active


def _old_nested_canonical(
    model: TwoPLNestedLogit, values: NDArray[np.float64]
) -> NDArray[np.float64]:
    canonical = values.copy()
    for item, (count, correct) in enumerate(
        zip(model.n_categories, model.correct_response, strict=True)
    ):
        reference = 0 if correct != 0 else 1
        canonical[item, :count] -= canonical[item, reference]
        canonical[item, correct] = 0.0
        canonical[item, count:] = 0.0
    return canonical


def _old_ggum_canonical(
    model: GeneralizedGradedUnfolding, values: NDArray[np.float64]
) -> NDArray[np.float64]:
    canonical = values.copy()
    for item, count in enumerate(model.n_categories):
        independent = canonical[item, : count - 1].copy()
        canonical[item] = 0.0
        canonical[item, : 2 * count - 1] = np.concatenate(
            (independent, [0.0], -independent[::-1])
        )
    return canonical


def _noise(shape: tuple[int, ...], seed: int) -> NDArray[np.float64]:
    return np.random.default_rng(seed).normal(size=shape)


PADDED: list[tuple[Callable[[], PolytomousItemModel], str]] = [
    (lambda: GradedResponseModel(7, COUNTS), "thresholds"),
    (lambda: GradedResponseModel(7, COUNTS, n_factors=3), "thresholds"),
    (lambda: GeneralizedPartialCredit(7, COUNTS), "steps"),
    (lambda: GeneralizedPartialCredit(7, COUNTS, n_factors=2), "steps"),
]


@pytest.mark.parametrize(("factory", "name"), PADDED)
def test_padded_threshold_masks_and_canonical_values_match_item_loops(
    factory: Callable[[], PolytomousItemModel], name: str
) -> None:
    model = factory()
    values = _noise(model.parameters[name].shape, 1)

    np.testing.assert_array_equal(
        model.free_parameter_masks[name], _old_padding_masks(model, name)
    )
    np.testing.assert_array_equal(
        model._canonical_parameter_values(name, values),
        _old_padding_canonical(model, values),
    )


@pytest.mark.parametrize("n_factors", [1, 3])
def test_nominal_masks_and_canonical_values_match_item_loops(n_factors: int) -> None:
    model = NominalResponseModel(7, COUNTS, n_factors=n_factors)
    expected_masks = _old_nominal_masks(model)

    masks = model.free_parameter_masks
    for name in ("slopes", "intercepts"):
        values = _noise(model.parameters[name].shape, 2)
        np.testing.assert_array_equal(masks[name], expected_masks[name])
        np.testing.assert_array_equal(
            model._canonical_parameter_values(name, values),
            _old_nominal_canonical(model, values),
        )
    assert not np.shares_memory(masks["slopes"], masks["intercepts"])


@pytest.mark.parametrize("model_class", [TwoPLNestedLogit, FourPLNestedLogit])
def test_nested_masks_and_canonical_values_match_item_loops(
    model_class: type[TwoPLNestedLogit],
) -> None:
    # Keys at zero use reference one; keys at the last category are covered.
    model = model_class(7, COUNTS, correct_response=[0, 1, 4, 0, 2, 0, 2])
    masks = model.free_parameter_masks

    for name in ("distractor_slopes", "distractor_intercepts"):
        values = _noise(model.parameters[name].shape, 3)
        np.testing.assert_array_equal(masks[name], _old_nested_mask(model, name))
        np.testing.assert_array_equal(
            model._canonical_parameter_values(name, values),
            _old_nested_canonical(model, values),
        )
    assert not np.shares_memory(
        masks["distractor_slopes"], masks["distractor_intercepts"]
    )


def test_ggum_masks_and_canonical_thresholds_match_item_loops() -> None:
    counts = [2, 5, 3, 4, 5, 2]
    model = GeneralizedGradedUnfolding(6, counts)
    values = _noise(model.thresholds.shape, 4)

    np.testing.assert_array_equal(
        model.free_parameter_masks["thresholds"],
        _old_padding_masks(model, "thresholds"),
    )
    np.testing.assert_array_equal(
        model._canonical_parameter_values("thresholds", values),
        _old_ggum_canonical(model, values),
    )
    np.testing.assert_array_equal(
        model._canonical_parameter_values("location", values[:, 0]), values[:, 0]
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: GradedResponseModel(7, COUNTS, n_factors=2),
        lambda: NominalResponseModel(7, COUNTS, n_factors=2),
        lambda: TwoPLNestedLogit(7, COUNTS, correct_response=1),
        lambda: GeneralizedGradedUnfolding(7, COUNTS),
    ],
)
def test_vectorized_masks_are_fresh_and_respect_restrictions(
    factory: Callable[[], PolytomousItemModel],
) -> None:
    model = factory()
    intrinsic = model.free_parameter_masks
    name = next(key for key, mask in intrinsic.items() if mask.ndim >= 2 and mask.any())

    first = model.free_parameter_masks
    first[name][:] = False
    assert np.array_equal(model.free_parameter_masks[name], intrinsic[name])

    restriction = intrinsic[name].copy()
    restriction[np.argwhere(restriction)[0][0]] = False
    model.set_free_parameter_masks({name: restriction})

    np.testing.assert_array_equal(model.free_parameter_masks[name], restriction)
    for other, mask in model.free_parameter_masks.items():
        if other != name:
            np.testing.assert_array_equal(mask, intrinsic[other])
