import numpy as np
import pytest
from numpy.testing import assert_allclose

import mirt.diagnostics.residuals as residuals_module
from mirt.diagnostics.residuals import analyze_residuals, compute_residuals
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.polytomous import GradedResponseModel


class CountingTwoParameterLogistic(TwoParameterLogistic):
    def __init__(self, n_items):
        super().__init__(n_items)
        self.probability_calls = []

    def probability(self, theta, item_idx=None):
        self.probability_calls.append(item_idx)
        return super().probability(theta, item_idx)


class CountingGradedResponseModel(GradedResponseModel):
    def __init__(self, n_items, n_categories):
        super().__init__(n_items, n_categories=n_categories)
        self.probability_calls = []

    def probability(self, theta, item_idx=None):
        self.probability_calls.append(item_idx)
        return super().probability(theta, item_idx)


@pytest.mark.parametrize(
    "residual_type", ["raw", "standardized", "pearson", "deviance"]
)
def test_binary_batch_matches_itemwise_path(monkeypatch, residual_type):
    model = CountingTwoParameterLogistic(3)
    model.set_parameters(
        discrimination=np.array([0.7, 1.1, 1.6]),
        difficulty=np.array([-0.8, 0.2, 1.0]),
    )
    theta = np.linspace(-2.0, 2.0, 11)
    responses = np.array(
        [
            [0, 0, -1],
            [0, 1, 0],
            [1, 0, 0],
            [0, -1, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
    )

    monkeypatch.setattr(residuals_module, "_RESIDUAL_MAX_PROBABILITY_VALUES", 0)
    expected = compute_residuals(model, responses, theta, residual_type)
    assert model.probability_calls == [0, 1, 2]

    model.probability_calls.clear()
    monkeypatch.setattr(
        residuals_module,
        "_RESIDUAL_MAX_PROBABILITY_VALUES",
        responses.size,
    )
    actual = compute_residuals(model, responses, theta, residual_type)

    assert model.probability_calls == [None]
    assert_allclose(actual, expected, rtol=1e-14, atol=1e-14, equal_nan=True)


@pytest.mark.parametrize(
    "residual_type", ["raw", "standardized", "pearson", "deviance"]
)
def test_polytomous_batch_matches_itemwise_path(monkeypatch, residual_type):
    model = CountingGradedResponseModel(3, n_categories=[3, 4, 5])
    model.set_parameters(
        discrimination=np.array([0.8, 1.1, 1.4]),
        thresholds=np.array(
            [
                [-1.0, 0.7, 0.0, 0.0],
                [-1.2, 0.0, 1.1, 0.0],
                [-1.4, -0.4, 0.5, 1.3],
            ]
        ),
    )
    theta = np.linspace(-2.5, 2.5, 9)
    responses = np.array(
        [
            [0, 0, 0],
            [0, 1, -1],
            [1, 0, 1],
            [2, 2, 1],
            [1, 3, 2],
            [2, -1, 3],
            [1, 2, 4],
            [2, 3, 3],
            [2, 3, 4],
        ]
    )

    monkeypatch.setattr(residuals_module, "_RESIDUAL_MAX_PROBABILITY_VALUES", 0)
    expected = compute_residuals(model, responses, theta, residual_type)
    assert model.probability_calls == [0, 1, 2]

    model.probability_calls.clear()
    monkeypatch.setattr(
        residuals_module,
        "_RESIDUAL_MAX_PROBABILITY_VALUES",
        responses.size * 5,
    )
    actual = compute_residuals(model, responses, theta, residual_type)

    assert model.probability_calls == [None]
    assert_allclose(actual, expected, rtol=1e-14, atol=1e-14, equal_nan=True)


def test_analysis_computes_all_residuals_in_one_batch():
    model = CountingTwoParameterLogistic(2)
    theta = np.linspace(-1.5, 1.5, 8)
    responses = np.array(
        [[0, 0], [0, 1], [0, -1], [1, 0], [1, 1], [1, 0], [1, 1], [1, 1]]
    )

    result = analyze_residuals(model, responses, theta)

    assert model.probability_calls == [None]
    assert result.raw_residuals.shape == responses.shape
    assert result.expected_values.shape == responses.shape


def test_probability_budget_retains_itemwise_fallback(monkeypatch):
    model = CountingGradedResponseModel(2, n_categories=3)
    theta = np.array([-1.0, 0.0, 1.0])
    responses = np.array([[0, 0], [1, 1], [2, 2]])
    monkeypatch.setattr(
        residuals_module,
        "_RESIDUAL_MAX_PROBABILITY_VALUES",
        responses.size * 3 - 1,
    )

    compute_residuals(model, responses, theta)

    assert model.probability_calls == [0, 1]
