"""Regression coverage for selected-item information queries."""

import numpy as np

from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.polytomous import GeneralizedPartialCredit
from mirt.utils.information import areainfo, iteminfo


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
