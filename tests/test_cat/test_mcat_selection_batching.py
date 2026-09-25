"""Batched MCAT selection agrees with independent covariance updates."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.cat import AOptimality, BayesianMCAT, COptimality, DOptimality
from mirt.cat.mcat_selection import _compute_item_information_matrix
from mirt.models import (
    MultidimensionalModel,
    NoncompensatoryModel,
    TwoParameterLogistic,
)


@pytest.fixture(
    params=[MultidimensionalModel, NoncompensatoryModel, TwoParameterLogistic]
)
def model(request):
    rng = np.random.default_rng(321)
    model = request.param(n_items=17, n_factors=3)
    slopes = rng.uniform(0.3, 2.0, size=(17, 3))
    if request.param is MultidimensionalModel:
        model.set_parameters(slopes=slopes, intercepts=rng.normal(size=17))
    elif request.param is TwoParameterLogistic:
        model.set_parameters(discrimination=slopes, difficulty=rng.normal(size=17))
    else:
        model.set_parameters(discrimination=slopes, difficulty=rng.normal(size=(17, 3)))
    return model


@pytest.mark.parametrize(
    "strategy",
    [
        DOptimality(),
        AOptimality(),
        COptimality(np.array([0.7, -0.2, 0.5])),
        BayesianMCAT(),
    ],
)
def test_batched_criteria_match_scalar_reference(model, strategy):
    theta = np.array([0.3, -0.6, 0.8])
    covariance = np.array([[0.7, -0.1, 0.2], [-0.1, 0.8, 0.1], [0.2, 0.1, 0.9]])
    available = {0, 2, 6, 9, 13, 16}
    expected = {}
    for item in available:
        info = _compute_item_information_matrix(model, theta, item)
        precision = np.linalg.inv(covariance + np.eye(3) * 1e-8)
        updated = np.linalg.inv(precision + info + np.eye(3) * 1e-8)
        expected[item] = strategy._criterion_from_post_cov(updated)

    actual = strategy.get_item_criteria(model, theta, covariance, available)
    assert actual.keys() == expected.keys()
    for item in available:
        assert_allclose(actual[item], expected[item], rtol=1e-13, atol=1e-14)
    assert strategy.select_item(model, theta, covariance, available) == max(
        expected, key=expected.__getitem__
    )


def test_inverts_prior_once_and_updates_candidates_together(model, monkeypatch):
    original = np.linalg.inv
    shapes = []

    def capture(matrix):
        shapes.append(matrix.shape)
        return original(matrix)

    monkeypatch.setattr(np.linalg, "inv", capture)
    DOptimality().select_item(model, np.zeros(3), np.eye(3), set(range(17)))
    assert shapes == [(3, 3), (17, 3, 3)]


def test_batched_updates_bound_memory_and_preserve_candidate_order(model, monkeypatch):
    theta = np.zeros(3)
    covariance = np.eye(3)
    items = {0, 2, 6, 9, 13}
    expected = DOptimality().get_item_criteria(model, theta, covariance, items)
    monkeypatch.setattr("mirt.cat.mcat_selection._SELECTION_WORKING_BYTES", 32 * 9 * 2)
    original = np.linalg.inv
    shapes = []

    def capture(matrix):
        shapes.append(matrix.shape)
        return original(matrix)

    monkeypatch.setattr(np.linalg, "inv", capture)
    actual = DOptimality().get_item_criteria(model, theta, covariance, items)
    assert shapes == [(3, 3), (2, 3, 3), (2, 3, 3), (1, 3, 3)]
    assert list(actual) == sorted(items)
    assert_allclose(list(actual.values()), list(expected.values()), rtol=1e-13)


def test_empty_criteria_and_selection_contract(model):
    strategy = DOptimality()
    assert strategy.get_item_criteria(model, np.zeros(3), np.eye(3), set()) == {}
    with pytest.raises(ValueError, match="No available items"):
        strategy.select_item(model, np.zeros(3), np.eye(3), set())


def test_equal_criteria_choose_smallest_item():
    model = MultidimensionalModel(n_items=20, n_factors=2)
    selected = DOptimality().select_item(model, np.zeros(2), np.eye(2), {19, 3, 11})
    assert selected == 3


def test_nonfinite_criteria_fail_instead_of_returning_invalid_item(model):
    class InvalidCriterion(DOptimality):
        def _criterion_from_post_cov(self, covariance):
            return np.nan

    with pytest.raises(ValueError, match="criteria must be finite"):
        InvalidCriterion().select_item(model, np.zeros(3), np.eye(3), {0, 1})
