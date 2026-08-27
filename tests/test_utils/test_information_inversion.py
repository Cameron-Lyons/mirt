"""Regression tests for batched test-score inversion."""

import numpy as np
import pytest

from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.multidimensional import MultidimensionalModel
from mirt.models.polytomous import GeneralizedPartialCredit
from mirt.utils.information import expected_score, theta_for_score


@pytest.mark.parametrize(
    "model",
    [
        TwoParameterLogistic(n_items=8),
        GeneralizedPartialCredit(n_items=8, n_categories=4),
    ],
    ids=["dichotomous", "polytomous"],
)
def test_theta_for_score_round_trips_batches_and_preserves_shape(model):
    theta = np.array([[-3.5, -2.0, -0.5], [0.5, 2.0, 3.5]])
    targets = expected_score(model, theta.ravel()).reshape(theta.shape)

    actual = theta_for_score(model, targets, theta_range=(-4.0, 4.0))

    assert isinstance(actual, np.ndarray)
    assert actual.shape == theta.shape
    np.testing.assert_allclose(actual, theta, atol=5e-7, rtol=0.0)


def test_theta_for_score_preserves_scalar_return_type():
    model = TwoParameterLogistic(n_items=5)
    target = float(expected_score(model, np.array([0.75]))[0])

    actual = theta_for_score(model, target)

    assert isinstance(actual, float)
    assert actual == pytest.approx(0.75, abs=5e-7)


def test_theta_for_score_clamps_targets_to_theta_bounds():
    model = TwoParameterLogistic(n_items=4)

    actual = theta_for_score(model, [-1.0, 5.0], theta_range=(-2.0, 2.0))

    np.testing.assert_array_equal(actual, [-2.0, 2.0])


class _CountingScoreModel:
    n_factors = 1
    n_items = 100

    def __init__(self, *, direction: float = 1.0) -> None:
        self.direction = direction
        self.calls = 0

    def expected_score(self, theta, item_idx=None):
        del item_idx
        self.calls += 1
        return self.direction * np.asarray(theta)[:, 0]


def test_theta_for_score_vectorizes_model_evaluations_across_targets():
    model = _CountingScoreModel()
    targets = np.linspace(-5.0, 5.0, 2_000)

    actual = theta_for_score(model, targets)

    np.testing.assert_allclose(actual, targets, atol=5e-7, rtol=0.0)
    assert model.calls < 30


def test_theta_for_score_supports_decreasing_characteristic_curves():
    model = _CountingScoreModel(direction=-1.0)

    actual = theta_for_score(model, [1.5, 0.0, -1.5], theta_range=(-2.0, 2.0))

    np.testing.assert_allclose(actual, [-1.5, 0.0, 1.5], atol=5e-7, rtol=0.0)


class _NonmonotonicScoreModel:
    n_factors = 1
    n_items = 1

    @staticmethod
    def expected_score(theta, item_idx=None):
        del item_idx
        values = np.asarray(theta)[:, 0]
        return values**2


def test_theta_for_score_rejects_nonmonotonic_curves():
    with pytest.raises(ValueError, match="must be monotonic"):
        theta_for_score(_NonmonotonicScoreModel(), 1.0)


@pytest.mark.parametrize(
    ("target", "theta_range", "message"),
    [
        (np.nan, (-6.0, 6.0), "only finite"),
        ("invalid", (-6.0, 6.0), "numeric values"),
        (1.0, (2.0, 2.0), "lower < upper"),
        (1.0, (2.0, -2.0), "lower < upper"),
        (1.0, (-np.inf, 2.0), "finite bounds"),
    ],
)
def test_theta_for_score_validates_inputs(target, theta_range, message):
    model = TwoParameterLogistic(n_items=2)

    with pytest.raises(ValueError, match=message):
        theta_for_score(model, target, theta_range=theta_range)


def test_theta_for_score_rejects_multidimensional_models():
    model = MultidimensionalModel(n_items=3, n_factors=2)

    with pytest.raises(ValueError, match="unidimensional"):
        theta_for_score(model, 1.0)


def test_theta_for_score_accepts_empty_target_arrays():
    model = TwoParameterLogistic(n_items=2)

    actual = theta_for_score(model, np.empty((2, 0)))

    assert actual.shape == (2, 0)
    assert actual.dtype == np.float64
