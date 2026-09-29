"""Regression coverage for selected-item information queries."""

import numpy as np
import pytest

from mirt import _information
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.polytomous import GeneralizedPartialCredit
from mirt.utils import information as information_utils
from mirt.utils.information import areainfo, expected_score, iteminfo, probtrace


def test_sparse_polytomous_selection_evaluates_only_requested_items(monkeypatch):
    model = GeneralizedPartialCredit(n_items=25, n_categories=4)
    theta = np.linspace(-3.0, 3.0, 101)
    selected = [24, 3, 0]
    original_information = model.information
    expected = np.column_stack(
        [original_information(theta[:, None], item_idx=index) for index in selected]
    )
    calls = []

    def tracked_information(theta_values, item_idx=None):
        calls.append(item_idx)
        return original_information(theta_values, item_idx=item_idx)

    monkeypatch.setattr(model, "information", tracked_information)

    actual = iteminfo(model, theta, selected)

    assert calls == selected
    np.testing.assert_allclose(actual, expected)


def test_empty_polytomous_selection_avoids_model_evaluation(monkeypatch):
    model = GeneralizedPartialCredit(n_items=5, n_categories=3)
    theta = np.array([-1.0, 0.0, 1.0])

    def unexpected_information(theta_values, item_idx=None):
        raise AssertionError("empty selections must not evaluate the model")

    monkeypatch.setattr(model, "information", unexpected_information)

    assert iteminfo(model, theta, []).shape == (theta.size, 0)


def test_dichotomous_selection_reuses_full_information_matrix(monkeypatch):
    model = TwoParameterLogistic(n_items=8)
    theta = np.linspace(-2.0, 2.0, 17)
    selected = [7, 1, 0]
    original_information = model.information
    expected = original_information(theta[:, None])[:, selected]
    calls = []

    def tracked_information(theta_values, item_idx=None):
        calls.append(item_idx)
        return original_information(theta_values, item_idx=item_idx)

    monkeypatch.setattr(model, "information", tracked_information)

    actual = iteminfo(model, theta, selected)

    assert calls == [None]
    np.testing.assert_allclose(actual, expected)


def test_areainfo_returns_selected_item_areas_in_order():
    model = GeneralizedPartialCredit(n_items=4, n_categories=[2, 3, 4, 5])
    selected = [3, 0, 2]

    actual = areainfo(
        model,
        theta_range=(-2.0, 2.0),
        n_points=81,
        item_idx=selected,
    )
    expected = np.array(
        [
            areainfo(
                model,
                theta_range=(-2.0, 2.0),
                n_points=81,
                item_idx=index,
            )
            for index in selected
        ]
    )

    assert actual.shape == (len(selected),)
    np.testing.assert_allclose(actual, expected)


def test_areainfo_preserves_scalar_and_empty_selection_shapes():
    model = GeneralizedPartialCredit(n_items=3, n_categories=4)

    assert isinstance(areainfo(model), float)
    assert isinstance(areainfo(model, item_idx=1), float)
    assert areainfo(model, item_idx=[]).shape == (0,)


@pytest.mark.parametrize("kind", ["binary", "ordinal"])
@pytest.mark.parametrize("function", [iteminfo, expected_score])
def test_numpy_indices_preserve_scalar_selection_and_duplicate_order(kind, function):
    model = (
        TwoParameterLogistic(4)
        if kind == "binary"
        else GeneralizedPartialCredit(4, [2, 3, 4, 5])
    )
    if kind == "binary":
        model.set_parameters(
            discrimination=np.array([0.6, 0.9, 1.3, 1.8]),
            difficulty=np.array([-1.5, -0.5, 0.25, 1.0]),
        )
    theta = np.linspace(-2.0, 2.0, 11)
    method = model.information if function is iteminfo else model.expected_score
    selected = np.array([3, 1, 3, 0], dtype=np.int64)
    selected.flags.writeable = False
    expected = np.column_stack([method(theta[:, None], int(i)) for i in selected])

    np.testing.assert_allclose(function(model, theta, selected), expected)
    scalar = function(model, theta, np.int64(1))
    assert scalar.shape == theta.shape
    np.testing.assert_allclose(scalar, expected[:, 1])
    assert function(model, theta, np.array([], dtype=int)).shape == (theta.size, 0)
    np.testing.assert_array_equal(selected, [3, 1, 3, 0])


@pytest.mark.parametrize(
    "function", [information_utils.testinfo, iteminfo, expected_score, probtrace]
)
def test_one_multidimensional_vector_matches_one_matrix_row(function):
    model = TwoParameterLogistic(4, n_factors=2)
    point = np.array([0.5, -0.25])
    point.flags.writeable = False

    actual = function(model, point)
    expected = function(model, point[None, :])

    assert actual.shape[0] == 1
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(point, [0.5, -0.25])


@pytest.mark.parametrize("function", [iteminfo, expected_score, areainfo])
@pytest.mark.parametrize(
    "selection", [True, np.bool_(False), [-1], [3], [0.5], [[1]], [False]]
)
def test_invalid_item_selection_is_rejected_before_model_evaluation(
    function, selection, monkeypatch
):
    model = TwoParameterLogistic(3)

    def unexpected(*args, **kwargs):
        raise AssertionError("invalid indices must not reach model evaluation")

    monkeypatch.setattr(model, "information", unexpected)
    monkeypatch.setattr(model, "expected_score", unexpected)
    with pytest.raises((ValueError, IndexError), match="item_idx"):
        function(
            model,
            item_idx=selection,
            **({} if function is areainfo else {"theta": [0.0]}),
        )


@pytest.mark.parametrize(
    "function", [information_utils.testinfo, iteminfo, expected_score, probtrace]
)
@pytest.mark.parametrize("theta", [np.nan, [0.0, np.inf], [[[0.0]]], [[0.0, 1.0]]])
def test_information_utilities_validate_theta(function, theta):
    with pytest.raises(ValueError, match="theta"):
        function(TwoParameterLogistic(3), theta)


@pytest.mark.parametrize(
    "function", [information_utils.testinfo, iteminfo, expected_score]
)
@pytest.mark.parametrize("n_factors", [1, 2])
def test_empty_theta_preserves_vectorized_shapes(function, n_factors, monkeypatch):
    model = TwoParameterLogistic(4, n_factors=n_factors)

    def unexpected(*args, **kwargs):
        raise AssertionError("empty cohorts must not evaluate the model")

    monkeypatch.setattr(model, "information", unexpected)
    monkeypatch.setattr(model, "expected_score", unexpected)
    expected_shape = (0, 4) if function is iteminfo else (0,)
    assert function(model, []).shape == expected_shape
    assert function(model, np.empty((0, n_factors))).shape == expected_shape


def test_sparse_binary_selection_avoids_evaluating_unselected_items(monkeypatch):
    model = TwoParameterLogistic(100)
    model.set_parameters(
        discrimination=np.linspace(0.5, 2.0, 100),
        difficulty=np.linspace(-2.0, 2.0, 100),
    )
    theta = np.linspace(-2.0, 2.0, 35)
    selected = [90, 2, 90]
    original = model.information
    expected = original(theta[:, None])[:, selected]
    calls = []

    def tracked(points, item_idx=None):
        assert item_idx in selected
        calls.append(item_idx)
        return original(points, item_idx)

    monkeypatch.setattr(model, "information", tracked)
    np.testing.assert_allclose(iteminfo(model, theta, selected), expected)
    assert calls == [90, 2]


@pytest.mark.parametrize(
    "function", [information_utils.testinfo, iteminfo, expected_score]
)
def test_large_binary_curves_are_evaluated_in_bounded_blocks(function, monkeypatch):
    model = TwoParameterLogistic(10)
    theta = np.linspace(-2.0, 2.0, 41)
    expected = function(model, theta)
    method_name = "expected_score" if function is expected_score else "information"
    original = getattr(model, method_name)
    calls = []

    def tracked(points, item_idx=None):
        assert len(points) * model.n_items <= 34
        calls.append(points.copy())
        return original(points, item_idx)

    monkeypatch.setattr(_information, "_INFORMATION_CHUNK_ELEMENTS", 34)
    monkeypatch.setattr(information_utils, "_EXPECTED_SCORE_CHUNK_ELEMENTS", 34)
    monkeypatch.setattr(model, method_name, tracked)
    np.testing.assert_allclose(function(model, theta), expected, atol=1e-14)
    np.testing.assert_array_equal(np.concatenate(calls)[:, 0], theta)
    assert len(calls) > 1


def test_expected_score_selection_evaluates_duplicates_once(monkeypatch):
    model = GeneralizedPartialCredit(4, [2, 3, 4, 5])
    original = model.expected_score
    calls = []

    def tracked(theta, item_idx=None):
        calls.append(item_idx)
        return original(theta, item_idx)

    monkeypatch.setattr(model, "expected_score", tracked)
    actual = expected_score(model, [-1.0, 0.0, 1.0], [3, 1, 3])
    assert calls == [3, 1]
    np.testing.assert_array_equal(actual[:, 0], actual[:, 2])


def test_total_information_model_falls_back_to_selected_item_queries(monkeypatch):
    model = TwoParameterLogistic(4)
    original = model.information
    theta = np.linspace(-1.0, 1.0, 9)
    expected = original(theta[:, None])[:, [3, 1, 3]]
    calls = []

    def total_information(points, item_idx=None):
        calls.append(item_idx)
        values = original(points, item_idx)
        return values.sum(axis=1) if item_idx is None else values

    monkeypatch.setattr(model, "information", total_information)
    actual = iteminfo(model, theta, [3, 1, 3])
    np.testing.assert_allclose(actual, expected)
    assert calls == [None, 3, 1]


@pytest.mark.parametrize("function", [information_utils.testinfo, iteminfo])
@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1])
def test_information_queries_reject_invalid_values(function, value, monkeypatch):
    model = TwoParameterLogistic(3)
    monkeypatch.setattr(
        model, "information", lambda theta: np.full((len(theta), 3), value)
    )
    with pytest.raises(ValueError, match="information"):
        function(model, [0.0, 1.0])


def test_numpy_item_selections_are_supported_by_area_queries():
    model = GeneralizedPartialCredit(4, [2, 3, 4, 5])
    selection = np.array([3, 0, 3])
    expected = [areainfo(model, item_idx=int(index)) for index in selection]
    actual = areainfo(model, item_idx=selection)
    np.testing.assert_allclose(actual, expected)
    assert areainfo(model, item_idx=np.array([], dtype=int)).shape == (0,)
    assert isinstance(areainfo(model, item_idx=np.int64(3)), float)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_points": 1},
        {"n_points": True},
        {"theta_range": (1.0, -1.0)},
        {"theta_range": (np.nan, 1.0)},
    ],
)
def test_area_queries_reject_invalid_integration_inputs(kwargs):
    with pytest.raises(ValueError, match="n_points|theta_range"):
        areainfo(TwoParameterLogistic(3), **kwargs)


@pytest.mark.parametrize("polytomous", [False, True])
def test_probability_only_models_produce_bounded_expected_scores(
    polytomous, monkeypatch
):
    class ProbabilityModel:
        n_factors = 1
        n_items = 3
        n_categories = 4
        is_polytomous = polytomous

        def probability(self, theta, item_idx=None):
            assert len(theta) <= (2 if item_idx is None else 6)
            if polytomous:
                one = np.tile([0.1, 0.2, 0.3, 0.4], (len(theta), 1))
                return (
                    one if item_idx is not None else np.tile(one[:, None, :], (1, 3, 1))
                )
            one = np.full(len(theta), 0.25)
            return one if item_idx is not None else np.tile(one[:, None], (1, 3))

    monkeypatch.setattr(
        information_utils, "_EXPECTED_SCORE_CHUNK_ELEMENTS", 24 if polytomous else 6
    )
    model = ProbabilityModel()
    expected = 2.0 if polytomous else 0.25
    np.testing.assert_allclose(expected_score(model, np.arange(17)), 3 * expected)
    np.testing.assert_allclose(
        expected_score(model, np.arange(17), [2, 0, 2]), expected
    )
