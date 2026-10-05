"""All-item polytomous information agrees with the per-item definitions."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

import mirt.models.base as model_base
import mirt.models.polytomous as polytomous
from mirt.exceptions import MirtValidationError
from mirt.models.base import PolytomousItemModel
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedRatingScaleModel,
    GradedResponseModel,
    NominalResponseModel,
    PartialCreditModel,
    RatingScaleModel,
)
from mirt.models.sequential import (
    AdjacentCategoryModel,
    ContinuationRatioModel,
    SequentialResponseModel,
)
from mirt.utils import information as information_utils
from mirt.utils.information import iteminfo

MIXED = [2, 5, 3, 4, 6, 3, 2, 5, 4]


def _randomized(model: PolytomousItemModel, seed: int) -> PolytomousItemModel:
    rng = np.random.default_rng(seed)
    parameters = model.parameters
    for name, values in parameters.items():
        if name == "discrimination":
            if isinstance(model, PartialCreditModel):
                continue
            values = values * rng.uniform(0.5, 1.8, values.shape)
            if isinstance(model, GradedRatingScaleModel):
                parameters[name] = values[0]
                continue
        else:
            values = values + rng.normal(scale=0.4, size=values.shape)
        if name == "thresholds":
            values = np.sort(values, axis=-1)
        parameters[name] = values
    if isinstance(model, PartialCreditModel):
        parameters.pop("discrimination")
    model.set_parameters(**parameters)
    return model


FACTORIES: list[Callable[[], PolytomousItemModel]] = [
    lambda: GradedResponseModel(9, MIXED),
    lambda: GradedResponseModel(9, MIXED, n_factors=2),
    lambda: GradedResponseModel(9, MIXED, n_factors=3),
    lambda: GeneralizedPartialCredit(9, MIXED),
    lambda: GeneralizedPartialCredit(9, MIXED, n_factors=2),
    lambda: PartialCreditModel(9, MIXED),
    lambda: RatingScaleModel(9, 4),
    lambda: GradedRatingScaleModel(9, 5),
    lambda: NominalResponseModel(9, MIXED),
    lambda: NominalResponseModel(9, MIXED, n_factors=3),
]


def _theta(model: PolytomousItemModel, n_points: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(n_points)
    return rng.uniform(-3.0, 3.0, size=(n_points, model.n_factors))


def _per_item(
    model: PolytomousItemModel, theta: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.column_stack(
        [model.information(theta, item) for item in range(model.n_items)]
    )


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("n_points", [1, 37, 300])
def test_item_columns_match_per_item_information(
    factory: Callable[[], PolytomousItemModel], n_points: int
) -> None:
    model = _randomized(factory(), seed=n_points)
    theta = _theta(model, n_points)
    expected = _per_item(model, theta)

    columns = model._information_by_item(theta)

    assert columns.shape == (n_points, model.n_items)
    np.testing.assert_allclose(columns, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        model.information(theta), expected.sum(axis=1), rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        iteminfo(model, theta if model.n_factors > 1 else theta[:, 0]),
        expected,
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        information_utils.testinfo(
            model, theta if model.n_factors > 1 else theta[:, 0]
        ),
        expected.sum(axis=1),
        rtol=1e-12,
        atol=1e-14,
    )


@pytest.mark.parametrize("factory", FACTORIES)
def test_small_batches_use_one_all_item_kernel(
    factory: Callable[[], PolytomousItemModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _randomized(factory(), seed=3)
    theta = _theta(model, 5)
    expected = _per_item(model, theta)
    per_item = PolytomousItemModel._information_by_item
    calls = []

    def counted(self, values):
        calls.append(values.shape[0])
        return per_item(self, values)

    # Every family overrides the hook, so only the per-item fallback runs this.
    monkeypatch.setattr(PolytomousItemModel, "_information_by_item", counted)

    np.testing.assert_allclose(
        model._information_by_item(theta), expected, rtol=1e-12, atol=1e-14
    )
    assert calls == []

    monkeypatch.setattr(polytomous, "_MAX_VECTORIZED_INFORMATION_ROWS", 4)
    np.testing.assert_allclose(
        model._information_by_item(theta), expected, rtol=1e-12, atol=1e-14
    )
    assert calls == [5]


def test_vectorized_kernel_is_row_block_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _randomized(NominalResponseModel(9, MIXED, n_factors=2), seed=8)
    theta = _theta(model, 40)
    expected = model._information_by_item(theta)

    monkeypatch.setattr(polytomous, "_MAX_PROBABILITY_CHUNK_ENTRIES", 1)

    np.testing.assert_allclose(
        model._information_by_item(theta), expected, rtol=1e-13, atol=1e-15
    )


def test_subclass_item_information_overrides_are_respected() -> None:
    class DoubledGRM(GradedResponseModel):
        def _item_information(self, theta, item_idx):
            return 2.0 * super()._item_information(theta, item_idx)

    base = _randomized(GradedResponseModel(9, MIXED), seed=4)
    doubled = DoubledGRM(9, MIXED).set_parameters(**base.parameters)
    theta = _theta(base, 7)

    np.testing.assert_allclose(
        doubled._information_by_item(theta),
        2.0 * base._information_by_item(theta),
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        iteminfo(doubled, theta[:, 0]), 2.0 * iteminfo(base, theta[:, 0])
    )


def test_instance_curve_overrides_keep_per_item_information() -> None:
    model = _randomized(GeneralizedPartialCredit(9, MIXED), seed=6)
    reference = _randomized(GeneralizedPartialCredit(9, MIXED), seed=6)
    original = model.probability

    def shifted(theta, item_idx=None):
        return original(np.asarray(theta) + 0.25, item_idx)

    model.probability = shifted  # type: ignore[method-assign]
    theta = _theta(model, 6)

    np.testing.assert_allclose(
        model._information_by_item(theta),
        reference._information_by_item(theta + 0.25),
        rtol=1e-12,
    )


def test_replaced_class_curve_hooks_keep_per_item_information(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _randomized(GradedResponseModel(9, MIXED, n_factors=2), seed=12)
    theta = _theta(model, 6)
    expected = _per_item(model, theta)
    original = GradedResponseModel.probability

    def shifted(self, theta, item_idx=None):
        return original(self, np.asarray(theta) + 0.25, item_idx)

    # Per-item GRM information reads _category_probabilities, so the replaced
    # all-item curves must not leak into the columns.
    monkeypatch.setattr(GradedResponseModel, "probability", shifted)

    np.testing.assert_allclose(
        model._information_by_item(theta), expected, rtol=1e-12, atol=1e-14
    )


@pytest.mark.parametrize(
    ("factory", "hook"),
    [
        (FACTORIES[0], "_item_information"),
        (FACTORIES[4], "_item_information"),
        (FACTORIES[5], "_item_information"),
        (FACTORIES[6], "_category_probabilities"),
        (FACTORIES[7], "_category_probabilities"),
        (FACTORIES[8], "_item_information"),
        (FACTORIES[9], "probability"),
    ],
)
def test_class_level_hook_replacements_keep_per_item_information(
    factory: Callable[[], PolytomousItemModel],
    hook: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _randomized(factory(), seed=15)
    theta = _theta(model, 6)
    unchanged = _per_item(model, theta)
    owner = next(cls for cls in type(model).__mro__ if hook in vars(cls))
    original = vars(owner)[hook]

    def replaced(self, values, item_idx=None):
        if hook == "_item_information":
            return 2.0 * original(self, values, item_idx)
        return original(self, np.asarray(values) + 0.25, item_idx)

    # Replacing the defining class's hook must not leave the kernel stale.
    monkeypatch.setattr(owner, hook, replaced)
    expected = _per_item(model, theta)

    assert not np.allclose(expected, unchanged)
    np.testing.assert_allclose(
        model._information_by_item(theta), expected, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        model.information(theta), expected.sum(axis=1), rtol=1e-12, atol=1e-14
    )


@pytest.mark.parametrize("factory", [FACTORIES[1], FACTORIES[6], FACTORIES[8]])
def test_test_information_is_bounded_and_block_invariant(
    factory: Callable[[], PolytomousItemModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _randomized(factory(), seed=14)
    theta = _theta(model, 700)
    expected = _per_item(model, theta).sum(axis=1)
    calls = []
    original = model._information_by_item

    def counted(values):
        calls.append(values.shape[0])
        return original(values)

    monkeypatch.setattr(model, "_information_by_item", counted)
    monkeypatch.setattr(
        model_base, "_POLYTOMOUS_MAX_INFORMATION_VALUES", 300 * model.n_items
    )

    np.testing.assert_allclose(
        model.information(theta), expected, rtol=1e-12, atol=1e-14
    )
    assert calls == [300, 300, 100]


def test_ordinal_logit_test_information_rejects_empty_theta() -> None:
    model = SequentialResponseModel(3, [2, 4, 3])

    with pytest.raises(MirtValidationError, match="at least one value"):
        model.information(np.empty((0, 1)))


def test_iteminfo_keeps_overridden_information_methods() -> None:
    class ScaledGPCM(GeneralizedPartialCredit):
        def information(self, theta, item_idx=None):
            return 3.0 * super().information(theta, item_idx)

    base = _randomized(GeneralizedPartialCredit(9, MIXED), seed=9)
    scaled = ScaledGPCM(9, MIXED).set_parameters(**base.parameters)
    theta = np.linspace(-2.0, 2.0, 11)

    np.testing.assert_allclose(
        iteminfo(scaled, theta), 3.0 * iteminfo(base, theta), rtol=1e-14
    )


@pytest.mark.parametrize(
    "model_class",
    [SequentialResponseModel, ContinuationRatioModel, AdjacentCategoryModel],
)
def test_ordinal_logit_iteminfo_evaluates_items_once(
    model_class: type[SequentialResponseModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    model = model_class(6, [2, 5, 3, 4, 6, 3])
    model.set_parameters(discrimination=np.linspace(0.6, 1.7, 6))
    theta = np.linspace(-2.5, 2.5, 9)
    expected = _per_item(model, theta)
    original = model._all_item_information
    calls = []

    def counted(values):
        calls.append(values.shape)
        return original(values)

    monkeypatch.setattr(model, "_all_item_information", counted)

    np.testing.assert_allclose(iteminfo(model, theta), expected, rtol=1e-14)
    assert len(calls) == 1
    np.testing.assert_allclose(
        model.information(theta), expected.sum(axis=1), rtol=1e-14
    )
