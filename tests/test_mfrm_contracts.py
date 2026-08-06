"""Behavioral and numerical contracts for many-facet Rasch models."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.models.mfrm import Facet, ManyFacetRaschModel, PolytomousMFRM

N_PERSONS = 11
N_ITEMS = 4
THETA = np.linspace(-2.5, 2.5, N_PERSONS)
ITEM_DIFFICULTY = np.array([-0.8, 0.1, 0.7, 1.2])
RATER_PARAMETERS = np.array([-0.6, 0.1, 0.8])
TASK_PARAMETERS = np.array([-0.25, 0.25])
RATER_ASSIGNMENTS = np.random.default_rng(17).integers(
    0,
    len(RATER_PARAMETERS),
    size=(N_PERSONS, N_ITEMS),
)
TASK_ASSIGNMENTS = np.random.default_rng(23).integers(
    0,
    len(TASK_PARAMETERS),
    size=N_PERSONS,
)
ASSIGNMENTS = {
    "rater": RATER_ASSIGNMENTS,
    "task": TASK_ASSIGNMENTS,
}


def _facets() -> list[Facet]:
    return [
        Facet("rater", len(RATER_PARAMETERS), is_anchored=False),
        Facet("task", len(TASK_PARAMETERS), is_anchored=False),
    ]


def _binary() -> ManyFacetRaschModel:
    return (
        ManyFacetRaschModel(N_ITEMS, _facets())
        .set_item_difficulty(ITEM_DIFFICULTY)
        .set_facet_parameters("rater", RATER_PARAMETERS)
        .set_facet_parameters("task", TASK_PARAMETERS)
    )


def _rating_scale() -> PolytomousMFRM:
    return (
        PolytomousMFRM(N_ITEMS, 5, _facets())
        .set_item_difficulty(ITEM_DIFFICULTY)
        .set_facet_parameters("rater", RATER_PARAMETERS)
        .set_facet_parameters("task", TASK_PARAMETERS)
        .set_thresholds(np.array([-1.1, -0.2, 0.4, 1.3]))
    )


def _partial_credit() -> PolytomousMFRM:
    return (
        PolytomousMFRM(
            N_ITEMS,
            5,
            _facets(),
            category_structure="partial_credit",
        )
        .set_item_difficulty(ITEM_DIFFICULTY)
        .set_facet_parameters("rater", RATER_PARAMETERS)
        .set_facet_parameters("task", TASK_PARAMETERS)
        .set_thresholds(
            np.array(
                [
                    [-1.4, -0.3, 0.6, 1.2],
                    [-0.9, -0.1, 0.7, 1.5],
                    [-1.2, 0.0, 0.5, 1.1],
                    [-0.7, 0.2, 0.8, 1.4],
                ]
            )
        )
    )


def test_binary_batch_matches_item_calls_and_manual_log_odds() -> None:
    model = _binary()
    expected = (
        THETA[:, None]
        - ITEM_DIFFICULTY[None, :]
        - RATER_PARAMETERS[RATER_ASSIGNMENTS]
        - TASK_PARAMETERS[TASK_ASSIGNMENTS, None]
    )

    assert_allclose(model.log_odds(THETA, facet_indices=ASSIGNMENTS), expected)
    probabilities = model.probability(THETA, facet_indices=ASSIGNMENTS)
    information = model.information(THETA, facet_indices=ASSIGNMENTS)

    for item_idx in range(N_ITEMS):
        assert_allclose(
            model.probability(THETA, item_idx, ASSIGNMENTS),
            probabilities[:, item_idx],
        )
        assert_allclose(
            model.information(THETA, item_idx, ASSIGNMENTS),
            information[:, item_idx],
        )

    assert_allclose(model.test_information(THETA, ASSIGNMENTS), information.sum(axis=1))


def test_person_level_facet_assignments_broadcast_across_items() -> None:
    model = _binary()
    person_assignments = {
        "rater": RATER_ASSIGNMENTS[:, 0],
        "task": TASK_ASSIGNMENTS,
    }
    response_assignments = {
        name: np.repeat(values[:, None], N_ITEMS, axis=1)
        for name, values in person_assignments.items()
    }

    assert_allclose(
        model.probability(THETA, facet_indices=person_assignments),
        model.probability(THETA, facet_indices=response_assignments),
    )


@pytest.mark.parametrize("factory", [_rating_scale, _partial_credit])
def test_polytomous_batch_matches_item_calls_and_normalizes(
    factory: Callable[[], PolytomousMFRM],
) -> None:
    model = factory()
    probabilities = model.probability(THETA, facet_indices=ASSIGNMENTS)

    assert probabilities.shape == (N_PERSONS, N_ITEMS, model.n_categories)
    assert np.all(np.isfinite(probabilities))
    assert_allclose(probabilities.sum(axis=-1), 1.0)
    assert_allclose(
        model.category_probability(THETA, None, 3, ASSIGNMENTS),
        probabilities[:, :, 3],
    )

    for item_idx in range(N_ITEMS):
        assert_allclose(
            model.probability(THETA, item_idx, ASSIGNMENTS),
            probabilities[:, item_idx],
        )
        for category in range(model.n_categories):
            assert_allclose(
                model.category_probability(
                    THETA,
                    item_idx,
                    category,
                    ASSIGNMENTS,
                ),
                probabilities[:, item_idx, category],
            )


@pytest.mark.parametrize("factory", [_rating_scale, _partial_credit])
def test_polytomous_probability_matches_partial_credit_definition(
    factory: Callable[[], PolytomousMFRM],
) -> None:
    model = factory()
    item_idx = 2
    measure = model.log_odds(THETA, item_idx, ASSIGNMENTS)
    thresholds = (
        model.thresholds
        if model.category_structure == "rating_scale"
        else model.thresholds[item_idx]
    )
    categories = np.arange(model.n_categories)
    cumulative_thresholds = np.concatenate(([0.0], np.cumsum(thresholds)))
    terms = np.exp(
        measure[:, None] * categories[None, :] - cumulative_thresholds[None, :]
    )
    expected = terms / terms.sum(axis=1, keepdims=True)

    assert_allclose(
        model.probability(THETA, item_idx, ASSIGNMENTS),
        expected,
    )


@pytest.mark.parametrize("factory", [_rating_scale, _partial_credit])
def test_polytomous_information_matches_probability_derivatives(
    factory: Callable[[], PolytomousMFRM],
) -> None:
    model = factory()
    step = 1e-5
    probabilities = model.probability(THETA, facet_indices=ASSIGNMENTS)
    derivative = (
        model.probability(THETA + step, facet_indices=ASSIGNMENTS)
        - model.probability(THETA - step, facet_indices=ASSIGNMENTS)
    ) / (2.0 * step)
    numerical_information = np.sum(derivative**2 / probabilities, axis=-1)

    actual = model.information(THETA, facet_indices=ASSIGNMENTS)
    assert actual.shape == (N_PERSONS, N_ITEMS)
    assert_allclose(actual, numerical_information, rtol=2e-9, atol=1e-11)
    assert_allclose(model.test_information(THETA, ASSIGNMENTS), actual.sum(axis=1))


@pytest.mark.parametrize("factory", [_rating_scale, _partial_credit])
def test_information_is_expected_score_derivative(
    factory: Callable[[], PolytomousMFRM],
) -> None:
    model = factory()
    step = 1e-5
    derivative = (
        model.expected_score(THETA + step, facet_indices=ASSIGNMENTS)
        - model.expected_score(THETA - step, facet_indices=ASSIGNMENTS)
    ) / (2.0 * step)

    assert_allclose(
        model.information(THETA, facet_indices=ASSIGNMENTS),
        derivative,
        rtol=2e-9,
        atol=1e-11,
    )


def test_extreme_measures_remain_finite_and_normalized() -> None:
    model = _partial_credit()
    extreme_theta = np.array([-1e6, 1e6])
    assignments = {
        "rater": np.array([0, 2]),
        "task": np.array([0, 1]),
    }

    probabilities = model.probability(extreme_theta, facet_indices=assignments)

    assert np.all(np.isfinite(probabilities))
    assert_allclose(probabilities.sum(axis=-1), 1.0)
    assert_allclose(probabilities[0, :, 0], 1.0)
    assert_allclose(probabilities[1, :, -1], 1.0)


def test_scalar_measure_and_models_without_facets_are_supported() -> None:
    binary = ManyFacetRaschModel(3, [])
    polytomous = PolytomousMFRM(3, 4, [])

    assert binary.probability(0.25).shape == (1, 3)
    assert polytomous.probability(0.25).shape == (1, 3, 4)


@pytest.mark.parametrize(
    "facet",
    [
        lambda: Facet("", 2),
        lambda: Facet("rater", 0),
        lambda: Facet("rater", 1.5),
        lambda: Facet("rater", True),
        lambda: Facet("rater", 2, labels=["same", "same"]),
        lambda: Facet("rater", 2, labels=["valid", ""]),
        lambda: Facet(1, 2),
        lambda: Facet("rater", 2, anchor_value=np.inf),
        lambda: Facet("rater", 2, anchor_value=True),
        lambda: Facet("rater", 2, anchor_value="invalid"),
    ],
)
def test_facet_definition_validation(facet: Callable[[], Facet]) -> None:
    with pytest.raises(ValueError):
        facet()


def test_model_structure_validation() -> None:
    with pytest.raises(ValueError, match="facet names must be unique"):
        ManyFacetRaschModel(2, [Facet("rater", 2), Facet("rater", 3)])
    with pytest.raises(TypeError, match="Facet instances"):
        ManyFacetRaschModel(2, ["rater"])
    with pytest.raises(ValueError, match="item_names length"):
        ManyFacetRaschModel(2, [], item_names=[])
    with pytest.raises(ValueError, match="category_structure"):
        PolytomousMFRM(2, 4, [], category_structure="nominal")


def test_model_owns_facet_definitions_and_parameter_arrays() -> None:
    labels = ["strict", "lenient"]
    facet = Facet("rater", 2, labels=labels, is_anchored=False)
    model = ManyFacetRaschModel(2, [facet])
    difficulty = np.array([-0.5, 0.5])
    severity = np.array([-0.2, 0.2])
    model.set_item_difficulty(difficulty).set_facet_parameters("rater", severity)

    labels[0] = "changed"
    facet.labels[0] = "also changed"
    returned_facet = model.facets[0]
    returned_facet.labels[0] = "changed again"
    difficulty[:] = 10.0
    severity[:] = 10.0

    assert model.get_facet("rater").labels == ["strict", "lenient"]
    assert_allclose(model.item_difficulty, [-0.5, 0.5])
    assert_allclose(model.facet_parameters["rater"], [-0.2, 0.2])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_parameter_setters_reject_nonfinite_values(value: float) -> None:
    binary = ManyFacetRaschModel(2, [Facet("rater", 2)])
    polytomous = PolytomousMFRM(2, 3, [Facet("rater", 2)])

    with pytest.raises(ValueError, match="difficulty values must be finite"):
        binary.set_item_difficulty(np.array([0.0, value]))
    with pytest.raises(ValueError, match="facet parameters must be finite"):
        binary.set_facet_parameters("rater", np.array([0.0, value]))
    with pytest.raises(ValueError, match="threshold values must be finite"):
        polytomous.set_thresholds(np.array([0.0, value]))


def test_facet_parameter_shape_is_validated() -> None:
    model = ManyFacetRaschModel(2, [Facet("rater", 2)])

    with pytest.raises(ValueError, match="parameters shape"):
        model.set_facet_parameters("rater", np.zeros(3))


def test_item_and_category_indices_are_bounds_checked() -> None:
    binary = _binary()
    polytomous = _rating_scale()

    for item_idx in (-1, N_ITEMS):
        with pytest.raises(IndexError, match="item_idx"):
            binary.probability(THETA, item_idx, ASSIGNMENTS)
    for category in (-1, polytomous.n_categories):
        with pytest.raises(IndexError, match="category"):
            polytomous.category_probability(THETA, 0, category, ASSIGNMENTS)
    with pytest.raises(TypeError, match="item_idx must be an integer"):
        binary.probability(THETA, 0.5, ASSIGNMENTS)
    with pytest.raises(TypeError, match="category must be an integer"):
        polytomous.category_probability(THETA, 0, 1.5, ASSIGNMENTS)


def test_facet_assignments_are_complete_integer_and_in_range() -> None:
    model = _binary()

    with pytest.raises(ValueError, match="Missing facet assignments"):
        model.probability(THETA, facet_indices={"rater": 0})
    with pytest.raises(ValueError, match="Unknown facet assignments"):
        model.probability(
            THETA,
            facet_indices={"rater": 0, "task": 0, "site": 0},
        )
    with pytest.raises(TypeError, match="indices must be integers"):
        model.probability(
            THETA,
            facet_indices={"rater": 0.0, "task": 0},
        )
    with pytest.raises(IndexError, match="index out of range"):
        model.probability(
            THETA,
            facet_indices={"rater": len(RATER_PARAMETERS), "task": 0},
        )
    with pytest.raises(ValueError, match="indices must have shape"):
        model.probability(
            THETA,
            facet_indices={"rater": np.zeros(3, dtype=int), "task": 0},
        )


def test_measure_validation_rejects_nonfinite_or_multidimensional_inputs() -> None:
    model = ManyFacetRaschModel(2, [])

    with pytest.raises(ValueError, match="one-dimensional"):
        model.probability(np.zeros((2, 1)))
    with pytest.raises(ValueError, match="theta values must be finite"):
        model.probability(np.array([0.0, np.nan]))
