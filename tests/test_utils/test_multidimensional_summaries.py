"""Contracts for multidimensional item-parameter summaries."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

import mirt
from mirt.models.bifactor import BifactorModel
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.multidimensional import MultidimensionalModel
from mirt.utils.multidimensional import (
    MDIFF,
    MDISC,
    composite_score_weights,
    direction_cosines,
)


def _slope_intercept_model() -> MultidimensionalModel:
    model = MultidimensionalModel(n_items=3, n_factors=2)
    model.set_parameters(
        slopes=np.array([[3.0, 4.0], [0.0, 2.0], [-1.0, 1.0]]),
        intercepts=np.array([-10.0, 1.0, np.sqrt(2.0)]),
    )
    return model


def test_slope_intercept_model_summaries_match_logistic_geometry() -> None:
    model = _slope_intercept_model()

    assert_allclose(MDISC(model), [5.0, 2.0, np.sqrt(2.0)])
    assert_allclose(MDIFF(model), [2.0, -0.5, -1.0])
    assert_allclose(
        direction_cosines(model),
        [[0.6, 0.8], [0.0, 1.0], [-1 / np.sqrt(2), 1 / np.sqrt(2)]],
    )

    directions = direction_cosines(model)
    half_probability_points = MDIFF(model)[:, None] * directions
    probabilities = np.array(
        [
            model.probability(half_probability_points[item : item + 1], item)[0]
            for item in range(model.n_items)
        ]
    )
    assert_allclose(probabilities, 0.5)


def test_discrimination_difficulty_parameterization_is_supported() -> None:
    model = TwoParameterLogistic(n_items=2, n_factors=2)
    model.set_parameters(
        discrimination=np.array([[3.0, 4.0], [1.0, 2.0]]),
        difficulty=np.array([2.0, -0.75]),
    )

    expected = np.array([14.0 / 5.0, -2.25 / np.sqrt(5.0)])
    assert_allclose(MDIFF(model), expected)

    half_probability_points = expected[:, None] * direction_cosines(model)
    probabilities = np.array(
        [
            model.probability(half_probability_points[item : item + 1], item)[0]
            for item in range(model.n_items)
        ]
    )
    assert_allclose(probabilities, 0.5)


def test_bifactor_loading_matrix_is_supported() -> None:
    model = BifactorModel(n_items=3, specific_factors=[4, 9, 4])
    model.set_parameters(
        general_loadings=np.array([3.0, 0.0, 5.0]),
        specific_loadings=np.array([4.0, 2.0, 12.0]),
        intercepts=np.array([-5.0, 1.0, 0.0]),
    )

    assert_allclose(MDISC(model), [5.0, 2.0, 13.0])
    assert_allclose(MDIFF(model), [1.0, -0.5, 0.0])
    assert_allclose(
        direction_cosines(model),
        [[0.6, 0.8, 0.0], [0.0, 0.0, 1.0], [5 / 13, 12 / 13, 0.0]],
    )

    directions = direction_cosines(model)
    half_probability_points = MDIFF(model)[:, None] * directions
    probabilities = np.array(
        [
            model.probability(half_probability_points[item : item + 1], item)[0]
            for item in range(model.n_items)
        ]
    )
    assert_allclose(probabilities, 0.5)


def test_stored_parameters_are_supported_without_named_attributes() -> None:
    model = SimpleNamespace(
        n_items=2,
        n_factors=2,
        parameters={
            "slopes": np.array([[1.0, 0.0], [0.0, 2.0]]),
            "intercepts": np.array([-1.5, 0.5]),
        },
    )

    assert_allclose(MDISC(model), [1.0, 2.0])
    assert_allclose(MDIFF(model), [1.5, -0.25])


def test_item_selection_preserves_order_and_item_axis() -> None:
    model = _slope_intercept_model()

    assert_allclose(MDISC(model, np.int64(1)), [2.0])
    assert_allclose(MDISC(model, [2, 0]), [np.sqrt(2.0), 5.0])
    assert direction_cosines(model, 1).shape == (1, 2)
    assert direction_cosines(model, []).shape == (0, 2)


@pytest.mark.parametrize("item_idx", [-1, 3, [0, 4]])
def test_item_selection_rejects_out_of_range_indices(item_idx: object) -> None:
    with pytest.raises(IndexError, match="out of range"):
        MDISC(_slope_intercept_model(), item_idx)  # type: ignore[arg-type]


@pytest.mark.parametrize("item_idx", [True, [0.5], [[0, 1]], "0"])
def test_item_selection_rejects_noninteger_indices(item_idx: object) -> None:
    with pytest.raises(TypeError, match="integer indices"):
        MDISC(_slope_intercept_model(), item_idx)  # type: ignore[arg-type]


def test_zero_discrimination_has_explicit_undefined_summaries() -> None:
    model = SimpleNamespace(
        n_items=2,
        n_factors=2,
        parameters={
            "discrimination": np.array([[0.0, 0.0], [3.0, 4.0]]),
            "difficulty": np.array([2.0, 1.0]),
        },
    )

    assert_allclose(MDISC(model), [0.0, 5.0])
    assert np.isnan(MDIFF(model)[0])
    assert np.isnan(direction_cosines(model)[0]).all()
    assert_allclose(direction_cosines(model)[1], [0.6, 0.8])


def test_discrimination_norms_remain_finite_near_float_limits() -> None:
    model = SimpleNamespace(
        n_items=1,
        n_factors=2,
        parameters={
            "slopes": np.array([[1e308, 1e308]]),
            "intercepts": np.array([0.0]),
        },
    )

    assert np.isfinite(MDISC(model)[0])
    assert_allclose(direction_cosines(model), [[1 / np.sqrt(2), 1 / np.sqrt(2)]])


def test_composite_weights_follow_normalized_reference_projection() -> None:
    model = _slope_intercept_model()

    weights = composite_score_weights(model, np.array([1.0, 0.0]))

    assert_allclose(weights, [1.5, 0.0, -0.5])
    assert_allclose(weights.sum(), 1.0)
    assert_allclose(composite_score_weights(model, np.array([10.0, 0.0])), weights)


@pytest.mark.parametrize(
    ("direction", "match"),
    [
        ([1.0], "shape"),
        ([1.0, 2.0, 3.0], "shape"),
        ([0.0, 0.0], "nonzero"),
        ([np.nan, 1.0], "finite"),
        ([np.inf, 1.0], "finite"),
    ],
)
def test_composite_weights_validate_reference_direction(
    direction: list[float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        composite_score_weights(_slope_intercept_model(), np.array(direction))


def test_composite_weights_reject_zero_total_projection() -> None:
    model = SimpleNamespace(
        n_items=2,
        n_factors=2,
        parameters={"slopes": np.array([[1.0, 0.0], [-1.0, 0.0]])},
    )

    with pytest.raises(ValueError, match="nonzero sum"):
        composite_score_weights(model, np.array([1.0, 0.0]))


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"slopes": np.ones((2, 3))}, "expected 2"),
        ({"slopes": np.ones((3, 2))}, "shape"),
        ({"slopes": np.array([[1.0, 0.0], [np.nan, 1.0]])}, "finite"),
    ],
)
def test_discrimination_contract_is_validated(
    parameters: dict[str, np.ndarray], message: str
) -> None:
    model = SimpleNamespace(n_items=2, n_factors=2, parameters=parameters)

    with pytest.raises(ValueError, match=message):
        MDISC(model)


def test_missing_discrimination_and_intercepts_raise_clear_errors() -> None:
    missing_discrimination = SimpleNamespace(n_items=2, n_factors=2, parameters={})
    missing_intercepts = SimpleNamespace(
        n_items=2,
        n_factors=2,
        parameters={"slopes": np.ones((2, 2))},
    )

    with pytest.raises(ValueError, match="discrimination"):
        MDISC(missing_discrimination)
    with pytest.raises(ValueError, match="intercepts or difficulty"):
        MDIFF(missing_intercepts)


@pytest.mark.parametrize(
    "parameters",
    [
        {"slopes": np.ones((2, 2)), "intercepts": np.ones((2, 1))},
        {"slopes": np.ones((2, 2)), "difficulty": np.ones((2, 1))},
        {"slopes": np.ones((2, 2)), "intercepts": np.array([0.0, np.inf])},
    ],
)
def test_location_parameter_contract_is_validated(
    parameters: dict[str, np.ndarray],
) -> None:
    model = SimpleNamespace(n_items=2, n_factors=2, parameters=parameters)

    with pytest.raises(ValueError, match="shape|finite"):
        MDIFF(model)


def test_all_multidimensional_summaries_are_top_level_exports() -> None:
    assert mirt.MDISC is MDISC
    assert mirt.MDIFF is MDIFF
    assert mirt.direction_cosines is direction_cosines
    assert mirt.composite_score_weights is composite_score_weights
