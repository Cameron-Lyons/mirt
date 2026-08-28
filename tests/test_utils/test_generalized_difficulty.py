"""Contracts for generalized item-difficulty inversion."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit, ndtr, ndtri

from mirt.models.dichotomous import (
    FiveParameterLogistic,
    FourParameterLogistic,
    OneParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
    UnipolarLogLogistic,
)
from mirt.models.multidimensional import MultidimensionalModel
from mirt.models.polytomous import GeneralizedPartialCredit
from mirt.utils.information import gen_difficulty


def _configured_logistic(model):
    parameters = {"difficulty": np.array([-1.2, 0.1, 1.4])}
    if model.model_name != "1PL":
        parameters["discrimination"] = np.array([0.7, 1.3, 2.1])
    if model.model_name in {"3PL", "4PL", "5PL"}:
        parameters["guessing"] = np.array([0.10, 0.20, 0.30])
    if model.model_name in {"4PL", "5PL"}:
        parameters["upper"] = np.array([0.85, 0.90, 0.95])
    if model.model_name == "5PL":
        parameters["asymmetry"] = np.array([0.65, 1.0, 1.8])
    return model.set_parameters(**parameters)


@pytest.mark.parametrize(
    "model",
    [
        OneParameterLogistic(3),
        TwoParameterLogistic(3),
        ThreeParameterLogistic(3),
        FourParameterLogistic(3),
        FiveParameterLogistic(3),
    ],
    ids=["1pl", "2pl", "3pl", "4pl", "5pl"],
)
def test_standard_logistic_families_invert_all_items_exactly(model):
    model = _configured_logistic(model)
    targets = np.array([0.40, 0.60, 0.75])

    theta = gen_difficulty(model, target_prob=targets, theta_range=(-8.0, 8.0))

    probabilities = model.probability(theta[:, None])
    np.testing.assert_allclose(np.diag(probabilities), targets, atol=2e-14, rtol=0.0)


def test_analytic_path_does_not_evaluate_probability(monkeypatch):
    model = _configured_logistic(FiveParameterLogistic(3))

    def unexpected_probability(*args, **kwargs):
        raise AssertionError("standard logistic inversion must be closed form")

    monkeypatch.setattr(model, "probability", unexpected_probability)

    theta = gen_difficulty(model, target_prob=[0.4, 0.6, 0.75])

    assert theta.shape == (3,)
    assert np.all(np.isfinite(theta))


def test_item_selection_supports_per_item_targets_and_scalar_returns():
    model = TwoParameterLogistic(4).set_parameters(
        discrimination=np.array([0.5, 1.0, 1.5, 2.0]),
        difficulty=np.array([-1.0, -0.5, 0.5, 1.0]),
    )
    items = np.array([3, 1])
    targets = np.array([0.25, 0.75])

    selected = gen_difficulty(model, item_idx=items, target_prob=targets)
    scalar = gen_difficulty(model, item_idx=np.int64(2), target_prob=0.8)

    expected = (
        model.difficulty[items]
        + np.log(targets / (1.0 - targets)) / model.discrimination[items]
    )
    np.testing.assert_allclose(selected, expected, atol=1e-14, rtol=0.0)
    assert isinstance(scalar, float)
    assert scalar == pytest.approx(
        model.difficulty[2] + np.log(4.0) / model.discrimination[2]
    )


def test_empty_item_selection_returns_an_empty_float_array():
    result = gen_difficulty(TwoParameterLogistic(3), item_idx=[])

    assert result.shape == (0,)
    assert result.dtype == np.float64


def test_item_asymptotes_clamp_to_requested_theta_bounds():
    model = FourParameterLogistic(2).set_parameters(
        discrimination=np.ones(2),
        difficulty=np.zeros(2),
        guessing=np.array([0.2, 0.1]),
        upper=np.array([0.8, 0.9]),
    )

    result = gen_difficulty(
        model,
        target_prob=np.array([0.1, 0.95]),
        theta_range=(-3.0, 4.0),
    )

    np.testing.assert_array_equal(result, [-3.0, 4.0])


class _ProbitModel:
    n_items = 3
    n_factors = 1
    is_polytomous = False
    model_name = "probit"

    def __init__(self, *, decreasing: bool = False) -> None:
        self.discrimination = np.array([0.8, 1.2, 1.7])
        self.difficulty = np.array([-0.6, 0.2, 1.1])
        self.direction = -1.0 if decreasing else 1.0

    def probability(self, theta, item_idx=None):
        values = np.asarray(theta, dtype=np.float64)[:, 0]
        z = (
            self.direction
            * self.discrimination[None, :]
            * (values[:, None] - self.difficulty[None, :])
        )
        probabilities = ndtr(z)
        return probabilities if item_idx is None else probabilities[:, item_idx]


def test_custom_monotonic_model_uses_validated_numerical_fallback():
    model = _ProbitModel()
    items = [2, 0]
    targets = np.array([0.3, 0.8])

    actual = gen_difficulty(
        model,
        item_idx=items,
        target_prob=targets,
        theta_range=(-5.0, 5.0),
    )

    expected = model.difficulty[items] + ndtri(targets) / model.discrimination[items]
    np.testing.assert_allclose(actual, expected, atol=5e-7, rtol=0.0)


def test_custom_decreasing_curve_is_inverted_in_the_correct_direction():
    model = _ProbitModel(decreasing=True)
    target = 0.8

    actual = gen_difficulty(model, item_idx=1, target_prob=target)

    expected = model.difficulty[1] - ndtri(target) / model.discrimination[1]
    assert actual == pytest.approx(expected, abs=5e-7)


def test_custom_targets_outside_finite_curve_range_clamp_by_direction():
    increasing = _ProbitModel()
    decreasing = _ProbitModel(decreasing=True)

    np.testing.assert_array_equal(
        gen_difficulty(
            increasing,
            item_idx=[0, 1],
            target_prob=[0.001, 0.9999],
            theta_range=(-3.0, 3.0),
        ),
        [-3.0, 3.0],
    )
    np.testing.assert_array_equal(
        gen_difficulty(
            decreasing,
            item_idx=[0, 1],
            target_prob=[0.999, 0.0001],
            theta_range=(-3.0, 3.0),
        ),
        [-3.0, 3.0],
    )


def test_nonmonotonic_custom_curve_is_rejected():
    model = UnipolarLogLogistic(2)

    with pytest.raises(ValueError, match="must be monotonic and vary"):
        gen_difficulty(model, target_prob=0.2, theta_range=(-4.0, 4.0))


class _ConstantProbabilityModel(_ProbitModel):
    def probability(self, theta, item_idx=None):
        shape = (len(theta), self.n_items) if item_idx is None else (len(theta),)
        return np.full(shape, 0.5)


def test_constant_custom_curve_is_rejected_as_ambiguous():
    with pytest.raises(ValueError, match="must be monotonic and vary"):
        gen_difficulty(_ConstantProbabilityModel(), item_idx=0)


@pytest.mark.parametrize(
    ("item_idx", "message", "error"),
    [
        (True, "integer item indices", ValueError),
        (1.5, "integer item indices", ValueError),
        ([[0]], "one-dimensional", ValueError),
        ([0.0, 1.0], "integer item indices", ValueError),
        ([0, 4], "out of range", IndexError),
    ],
)
def test_item_selection_is_validated(item_idx, message, error):
    with pytest.raises(error, match=message):
        gen_difficulty(TwoParameterLogistic(3), item_idx=item_idx)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (0.0, "strictly between"),
        (1.0, "strictly between"),
        (np.nan, "strictly between"),
        ("invalid", "numeric"),
        ([0.2, 0.4, 0.6], "broadcast"),
    ],
)
def test_target_probabilities_are_validated(target, message):
    with pytest.raises(ValueError, match=message):
        gen_difficulty(TwoParameterLogistic(2), target_prob=target)


@pytest.mark.parametrize(
    "theta_range",
    [(0.0, 0.0), (1.0, -1.0), (-np.inf, 2.0), (0.0,), "invalid"],
)
def test_theta_range_is_validated(theta_range):
    with pytest.raises(ValueError, match="theta_range"):
        gen_difficulty(TwoParameterLogistic(2), theta_range=theta_range)


def test_polytomous_and_multidimensional_models_are_rejected():
    with pytest.raises(ValueError, match="dichotomous models only"):
        gen_difficulty(GeneralizedPartialCredit(2, n_categories=4))

    with pytest.raises(ValueError, match="unidimensional"):
        gen_difficulty(MultidimensionalModel(2, n_factors=2))


class _InvalidProbabilityModel(_ProbitModel):
    def probability(self, theta, item_idx=None):
        probabilities = super().probability(theta, item_idx)
        return np.full_like(probabilities, np.nan)


def test_custom_model_probabilities_are_validated():
    with pytest.raises(ValueError, match="finite and lie"):
        gen_difficulty(_InvalidProbabilityModel(), item_idx=0)


class _LogisticCustomModel(_ProbitModel):
    model_name = "custom-logistic"

    def probability(self, theta, item_idx=None):
        values = np.asarray(theta, dtype=np.float64)[:, 0]
        z = (
            self.direction
            * self.discrimination[None, :]
            * (values[:, None] - self.difficulty[None, :])
        )
        probabilities = expit(z)
        return probabilities if item_idx is None else probabilities[:, item_idx]


def test_custom_logistic_name_uses_numerical_fallback():
    model = _LogisticCustomModel()

    actual = gen_difficulty(model, item_idx=2, target_prob=0.7)

    expected = model.difficulty[2] + np.log(0.7 / 0.3) / model.discrimination[2]
    assert actual == pytest.approx(expected, abs=5e-7)
