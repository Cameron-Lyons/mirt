"""Mixed-format models that combine item families on one latent trait."""

import numpy as np
import pytest

import mirt
from mirt.exceptions import MirtDataError, MirtModelError, MirtValidationError
from mirt.models.dichotomous import (
    OneParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.mixed_format import MixedItemModel
from mirt.models.polytomous import GeneralizedPartialCredit, GradedResponseModel
from mirt.scoring._common import supports_row_batched_scoring

THETA = np.linspace(-3.0, 3.0, 13)[:, None]


def _three_pl(rng: np.random.Generator, n_items: int = 4) -> ThreeParameterLogistic:
    model = ThreeParameterLogistic(n_items)
    model.set_parameters(
        discrimination=rng.uniform(0.7, 2.0, n_items),
        difficulty=rng.normal(0.0, 1.0, n_items),
        guessing=rng.uniform(0.05, 0.25, n_items),
    )
    model._is_fitted = True
    return model


def _graded(rng: np.random.Generator) -> GradedResponseModel:
    model = GradedResponseModel(3, n_categories=[3, 4, 5])
    thresholds = np.sort(rng.normal(0.0, 1.0, (3, 4)), axis=1)
    thresholds[0, 2:] = 0.0
    thresholds[1, 3:] = 0.0
    model.set_parameters(discrimination=rng.uniform(0.7, 2.0, 3), thresholds=thresholds)
    model._is_fitted = True
    return model


@pytest.fixture
def parts() -> tuple[ThreeParameterLogistic, GradedResponseModel]:
    rng = np.random.default_rng(20261005)
    return _three_pl(rng), _graded(rng)


@pytest.fixture
def mixed(parts) -> MixedItemModel:
    three_pl, graded = parts
    # Interleaved positions exercise the item placement.
    return MixedItemModel([(three_pl, [0, 2, 4, 6]), (graded, [1, 3, 5])])


MC_ITEMS = [0, 2, 4, 6]
CR_ITEMS = [1, 3, 5]


def _responses(model: MixedItemModel, n_persons: int = 60) -> np.ndarray:
    rng = np.random.default_rng(7)
    responses = model.simulate(rng.standard_normal((n_persons, 1)), seed=8)
    responses[rng.random(responses.shape) < 0.1] = -1
    return responses


def test_structure_and_qualified_parameters(mixed, parts) -> None:
    three_pl, graded = parts

    assert mixed.n_items == 7
    assert mixed.item_types == ["3PL", "GRM", "3PL", "GRM", "3PL", "GRM", "3PL"]
    assert mixed.n_categories == [2, 3, 2, 4, 2, 5, 2]
    assert mixed.is_polytomous
    assert mixed.is_fitted
    assert mixed.component_names == ["3PL", "GRM"]
    assert list(mixed.parameters) == [
        "3PL.discrimination",
        "3PL.difficulty",
        "3PL.guessing",
        "GRM.discrimination",
        "GRM.thresholds",
    ]
    assert mixed.n_parameters == three_pl.n_parameters + graded.n_parameters
    np.testing.assert_array_equal(
        mixed.parameters["GRM.thresholds"], graded.parameters["thresholds"]
    )
    assert mixed.locate_item(5) == (1, 2)
    assert mixed.parameter_component("3PL.guessing") == (0, "guessing")
    assert "3PL: 4 items, GRM: 3 items" in repr(mixed)


def test_components_are_copied_and_live(mixed, parts) -> None:
    three_pl, _ = parts
    component = mixed.component_models[0]

    assert component is not three_pl
    mixed.set_parameters(**{"3PL.difficulty": np.zeros(4)})
    np.testing.assert_array_equal(component.difficulty, np.zeros(4))
    assert not np.array_equal(three_pl.difficulty, np.zeros(4))

    # Direct storage writes reach the component as for a plain model.
    mixed._parameters["3PL.guessing"] = np.full(4, 0.1)
    np.testing.assert_array_equal(component.guessing, np.full(4, 0.1))
    assert component.item_names == ["Item_0", "Item_2", "Item_4", "Item_6"]


def test_probability_places_component_curves(mixed, parts) -> None:
    three_pl, graded = parts
    probabilities = mixed.probability(THETA)

    assert probabilities.shape == (13, 7, 5)
    binary = three_pl.probability(THETA)
    np.testing.assert_allclose(probabilities[:, MC_ITEMS, 1], binary)
    np.testing.assert_allclose(probabilities[:, MC_ITEMS, 0], 1.0 - binary)
    np.testing.assert_array_equal(probabilities[:, MC_ITEMS, 2:], 0.0)
    np.testing.assert_allclose(probabilities[:, CR_ITEMS], graded.probability(THETA))
    np.testing.assert_allclose(probabilities.sum(axis=2), 1.0)

    np.testing.assert_allclose(
        mixed.probability(THETA, 2),
        np.column_stack((1.0 - binary[:, 1], binary[:, 1])),
    )
    np.testing.assert_allclose(
        mixed.probability(THETA, 3), graded.probability(THETA, 1)
    )
    np.testing.assert_allclose(
        mixed.category_probability(THETA, 5, 4), graded.probability(THETA, 2)[:, 4]
    )
    with pytest.raises(ValueError, match="out of range"):
        mixed.category_probability(THETA, 1, 3)


def test_probability_pairs_match_single_item_curves(mixed) -> None:
    rng = np.random.default_rng(3)
    items = rng.integers(0, mixed.n_items, size=40)
    theta = rng.normal(size=(40, 1))

    pairs = mixed.probability_pairs(theta, items)

    for row, item in enumerate(items):
        expected = mixed.probability(theta[row : row + 1], int(item))[0]
        np.testing.assert_allclose(pairs[row, : expected.size], expected)
        np.testing.assert_array_equal(pairs[row, expected.size :], 0.0)


def test_log_likelihood_batch_is_sum_of_component_batches(mixed, parts) -> None:
    three_pl, graded = parts
    responses = _responses(mixed)

    expected = three_pl.log_likelihood_batch(
        responses[:, MC_ITEMS], THETA
    ) + graded.log_likelihood_batch(responses[:, CR_ITEMS], THETA)

    np.testing.assert_allclose(mixed.log_likelihood_batch(responses, THETA), expected)
    rows = mixed.log_likelihood(responses[:13], THETA)
    np.testing.assert_allclose(rows, np.diag(expected[:13]))


def test_likelihood_rejects_codes_outside_item_categories(mixed) -> None:
    responses = np.zeros((2, 7), dtype=int)
    responses[0, 0] = 2

    with pytest.raises(MirtDataError, match="item 0"):
        mixed.log_likelihood_batch(responses, THETA)
    responses[0, 0] = 0
    responses[1, 5] = 5
    with pytest.raises(MirtDataError, match="item 5"):
        mixed.log_likelihood(responses, THETA[:2])


def test_information_and_expected_score_sum_components(mixed, parts) -> None:
    three_pl, graded = parts

    by_item = mixed._information_by_item(THETA)
    np.testing.assert_allclose(by_item[:, MC_ITEMS], three_pl.information(THETA))
    np.testing.assert_allclose(by_item[:, CR_ITEMS], graded._information_by_item(THETA))
    np.testing.assert_allclose(mixed.information(THETA), by_item.sum(axis=1))
    np.testing.assert_allclose(
        mixed.information(THETA, 6), three_pl.information(THETA, 3)
    )
    np.testing.assert_allclose(
        mixed.item_information_matrix(THETA, 3)[:, 0, 0], graded.information(THETA, 1)
    )
    np.testing.assert_allclose(
        mixed.expected_score(THETA),
        three_pl.expected_score(THETA) + graded.expected_score(THETA),
    )
    np.testing.assert_allclose(
        mixed.expected_score(THETA, 4), three_pl.probability(THETA, 2)
    )


def test_multidimensional_binary_information_matrix() -> None:
    slopes = np.array([[1.2, 0.4], [0.3, 1.5]])
    binary = TwoParameterLogistic(2, n_factors=2)
    binary.set_parameters(discrimination=slopes, difficulty=np.array([0.2, -0.4]))
    graded = GradedResponseModel(2, n_categories=3, n_factors=2)
    mixed = MixedItemModel([(binary, [0, 1]), (graded, [2, 3])])
    theta = np.array([[0.5, -0.2], [-1.0, 1.0]])

    p = binary.probability(theta, 1)
    expected = (p * (1 - p))[:, None, None] * np.outer(slopes[1], slopes[1])

    np.testing.assert_allclose(mixed.item_information_matrix(theta, 1), expected)
    np.testing.assert_allclose(
        mixed.item_information_matrix(theta, 3),
        graded.item_information_matrix(theta, 1),
    )


def test_get_and_set_item_parameters_use_the_owning_component(mixed, parts) -> None:
    _, graded = parts

    item = mixed.get_item_parameters(3)
    assert set(item) == {"discrimination", "thresholds"}
    np.testing.assert_array_equal(item["thresholds"], graded.thresholds[1])

    mixed.set_item_parameter(2, "guessing", 0.05)
    mixed.set_item_parameter(1, "GRM.discrimination", 1.7)
    assert mixed.get_item_parameters(2)["guessing"] == 0.05
    assert mixed.parameters["GRM.discrimination"][0] == 1.7
    with pytest.raises(MirtValidationError, match="Unknown parameter"):
        mixed.set_item_parameter(1, "guessing", 0.1)


def test_set_parameters_is_atomic(mixed) -> None:
    before = mixed.parameters

    with pytest.raises(MirtValidationError, match="guessing"):
        mixed.set_parameters(
            **{"GRM.discrimination": np.full(3, 2.0), "3PL.guessing": np.full(4, 2.0)}
        )
    with pytest.raises(MirtValidationError, match="Unknown parameter"):
        mixed.set_parameters(guessing=np.zeros(4))
    for name, values in mixed.parameters.items():
        np.testing.assert_array_equal(values, before[name])


def test_free_parameter_masks_are_component_masks() -> None:
    mixed = MixedItemModel(
        [
            (OneParameterLogistic(2), [0, 1]),
            (GradedResponseModel(2, n_categories=[3, 4]), [2, 3]),
        ]
    )
    masks = mixed.free_parameter_masks

    assert not masks["1PL.discrimination"].any()
    np.testing.assert_array_equal(
        masks["GRM.thresholds"], [[True, True, False], [True, True, True]]
    )
    assert mixed.n_parameters == 2 + 2 + 5

    fixed = np.array([False, True])
    mixed.set_free_parameter_masks({"GRM.discrimination": fixed})
    assert mixed._free_parameter_restrictions.keys() == {"GRM.discrimination"}
    np.testing.assert_array_equal(
        mixed.component_models[1].free_parameter_masks["discrimination"], fixed
    )
    assert mixed.n_parameters == 8
    assert mixed.copy().n_parameters == 8

    with pytest.raises(MirtValidationError, match="model-family fixed"):
        mixed.set_free_parameter_masks({"1PL.discrimination": np.ones(2, bool)})
    assert mixed.n_parameters == 8
    mixed.set_free_parameter_masks(None)
    assert mixed.n_parameters == 9


def test_copy_is_independent(mixed) -> None:
    duplicate = mixed.copy()
    duplicate.set_parameters(**{"3PL.difficulty": np.full(4, 3.0)})

    assert duplicate.is_fitted
    assert duplicate.item_names == mixed.item_names
    assert not np.allclose(mixed.parameters["3PL.difficulty"], 3.0)


def test_simulate_draws_valid_reproducible_categories(mixed) -> None:
    theta = np.random.default_rng(5).standard_normal((4000, 1))

    first = mixed.simulate(theta, seed=11)
    again = mixed.simulate(theta, seed=11, chunk_size=777)

    np.testing.assert_array_equal(first, again)
    assert first.shape == (4000, 7)
    assert np.all(first >= 0)
    assert np.all(first < np.asarray(mixed.n_categories))
    expected = np.stack(
        [mixed.expected_score(theta, item) for item in range(7)], axis=1
    )
    np.testing.assert_allclose(first.mean(axis=0), expected.mean(axis=0), atol=0.06)


def test_item_parameter_arrays_pad_missing_parameters(mixed, parts) -> None:
    three_pl, graded = parts
    arrays = mixed.item_parameter_arrays()

    assert list(arrays) == ["discrimination", "difficulty", "guessing", "thresholds"]
    np.testing.assert_array_equal(
        arrays["discrimination"][MC_ITEMS], three_pl.discrimination
    )
    np.testing.assert_array_equal(
        arrays["discrimination"][CR_ITEMS], graded.discrimination
    )
    assert np.all(np.isnan(arrays["guessing"][CR_ITEMS]))
    assert arrays["thresholds"].shape == (7, 4)
    assert np.all(np.isnan(arrays["thresholds"][MC_ITEMS]))

    errors = mixed.item_parameter_arrays({"3PL.guessing": np.full(4, 0.01)})
    np.testing.assert_array_equal(errors["guessing"][MC_ITEMS], 0.01)
    assert np.all(np.isnan(errors["discrimination"]))


def test_constructor_validation(parts) -> None:
    three_pl, graded = parts

    with pytest.raises(MirtValidationError, match="exactly once"):
        MixedItemModel([(three_pl, [0, 1, 2, 3]), (graded, [3, 4, 5])])
    with pytest.raises(MirtValidationError, match="positions"):
        MixedItemModel([(three_pl, [0, 1, 2]), (graded, [3, 4, 5])])
    with pytest.raises(MirtValidationError, match="non-empty"):
        MixedItemModel([])
    with pytest.raises(MirtValidationError, match="integer"):
        MixedItemModel([(three_pl, [0.0, 1.0, 2.0, 3.0])])
    with pytest.raises(MirtModelError, match="n_factors"):
        MixedItemModel(
            [(three_pl, range(4)), (GradedResponseModel(2, 3, n_factors=2), [4, 5])]
        )
    nested = MixedItemModel([(three_pl, range(4))])
    with pytest.raises(MirtModelError, match="nested"):
        MixedItemModel([(nested, range(4))])
    with pytest.raises(MirtValidationError, match="item_names"):
        MixedItemModel([(three_pl, range(4))], item_names=["a"])


def test_item_names_and_repeated_families() -> None:
    first = TwoParameterLogistic(2, item_names=["a", "b"])
    second = TwoParameterLogistic(1, item_names=["c"])
    mixed = MixedItemModel([(first, [0, 2]), (second, [1])])

    assert mixed.item_names == ["a", "c", "b"]
    assert mixed.component_names == ["2PL", "2PL_2"]
    assert "2PL_2.difficulty" in mixed.parameters

    clashing = MixedItemModel([(TwoParameterLogistic(1), [0]), (second.copy(), [1])])
    assert clashing.item_names == ["Item_0", "c"]
    default = MixedItemModel(
        [(TwoParameterLogistic(1), [0]), (TwoParameterLogistic(1), [1])]
    )
    assert default.item_names == ["Item_0", "Item_1"]


def test_from_itemtypes_groups_items_by_family() -> None:
    types = ["GRM", "3PL", "GPCM", "3PL", "GRM"]
    model = MixedItemModel.from_itemtypes(types, n_categories=[4, 2, 3, 2, 5])

    assert model.item_types == types
    assert model.component_names == ["GRM", "3PL", "GPCM"]
    assert model.n_categories == [4, 2, 3, 2, 5]
    np.testing.assert_array_equal(model.components[0][1], [0, 4])
    assert isinstance(model.component_models[2], GeneralizedPartialCredit)
    assert not model.is_fitted

    responses = np.array([[0, 1, 2, 0, 3], [3, 0, 0, 1, 0]])
    inferred = MixedItemModel.from_itemtypes(types, responses=responses)
    assert inferred.n_categories == [4, 2, 3, 2, 4]

    with pytest.raises(MirtValidationError, match="dichotomous"):
        MixedItemModel.from_itemtypes(types, n_categories=[4, 3, 3, 2, 5])
    with pytest.raises(MirtModelError, match="Unknown model"):
        MixedItemModel.from_itemtypes(["3PL", "BOGUS"], n_categories=3)
    with pytest.raises(MirtModelError, match="multidimensional"):
        MixedItemModel.from_itemtypes(["3PL", "GRM"], n_categories=3, n_factors=2)
    with pytest.raises(MirtDataError, match="dichotomous"):
        MixedItemModel.from_itemtypes(["3PL", "GRM"], responses=np.array([[2, 1]]))
    with pytest.raises(MirtValidationError, match="sequence"):
        MixedItemModel.from_itemtypes("3PL")


def test_one_component_scores_like_its_component(parts) -> None:
    _, graded = parts
    responses = _responses(MixedItemModel([(graded, range(3))]), n_persons=40)
    mixed = MixedItemModel([(graded, range(3))])

    for method in ("EAP", "MAP", "ML", "WLE", "EAPsum"):
        expected = mirt.fscores(graded, responses, method=method)
        actual = mirt.fscores(mixed, responses, method=method)
        np.testing.assert_allclose(actual.theta, expected.theta, atol=1e-6)
        np.testing.assert_allclose(
            actual.standard_error, expected.standard_error, atol=1e-6
        )


def test_mixed_pool_scoring_and_adaptive_testing(mixed) -> None:
    responses = _responses(mixed)

    # An instance-level likelihood keeps the per-pattern scoring path.
    per_pattern = mixed.copy()
    per_pattern.log_likelihood = per_pattern.log_likelihood
    for method in ("MAP", "ML", "WLE"):
        batched = mirt.fscores(mixed, responses, method=method)
        reference = mirt.fscores(per_pattern, responses, method=method)
        np.testing.assert_allclose(batched.theta, reference.theta, atol=1e-4)

    engine = mirt.CATEngine(mixed, seed=3)
    result = engine.run_simulation(0.5)
    assert result.n_items_administered > 0
    assert set(result.items_administered) <= set(range(mixed.n_items))


def test_parameter_items_locate_component_rows(mixed) -> None:
    np.testing.assert_array_equal(mixed.parameter_items("GRM.thresholds"), CR_ITEMS)
    items = mixed.parameter_items("3PL.guessing")
    np.testing.assert_array_equal(items, MC_ITEMS)

    with pytest.raises(ValueError, match="read-only"):
        items[0] = 3
    with pytest.raises(MirtValidationError, match="Unknown parameter"):
        mixed.parameter_items("guessing")


class _RowCoupledGraded(GradedResponseModel):
    """Likelihood that depends on every theta row it is given."""

    def log_likelihood(self, responses, theta):
        values = super().log_likelihood(responses, theta)
        return values - 0.5 * float(np.mean(theta))


def test_row_batched_scoring_requires_row_independent_components(parts) -> None:
    three_pl, graded = parts
    coupled = _RowCoupledGraded(3, n_categories=[3, 4, 5])
    coupled.set_parameters(**graded.parameters)
    coupled._is_fitted = True
    mixed = MixedItemModel([(three_pl, MC_ITEMS), (coupled, CR_ITEMS)])
    responses = _responses(mixed, n_persons=30)

    # Regression: stacking patterns changed the coupled component's likelihood.
    assert not supports_row_batched_scoring(mixed)
    per_pattern = mixed.copy()
    per_pattern.log_likelihood = per_pattern.log_likelihood
    np.testing.assert_allclose(
        mirt.fscores(mixed, responses, method="MAP").theta,
        mirt.fscores(per_pattern, responses, method="MAP").theta,
    )
