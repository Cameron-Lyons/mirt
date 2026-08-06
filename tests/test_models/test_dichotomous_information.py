"""Analytic and numerical contracts for dichotomous information curves."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.models.dichotomous import (
    ComplementaryLogLog,
    FiveParameterLogistic,
    FourParameterLogistic,
    NegativeLogLog,
    ThreeParameterLogistic,
    UnipolarLogLogistic,
)


def _three_parameter() -> ThreeParameterLogistic:
    model = ThreeParameterLogistic(3)
    return model.set_parameters(
        discrimination=np.array([0.7, 1.2, 2.0]),
        difficulty=np.array([-1.0, 0.0, 1.0]),
        guessing=np.array([0.1, 0.2, 0.3]),
    )


def _four_parameter() -> FourParameterLogistic:
    model = FourParameterLogistic(3)
    return model.set_parameters(
        discrimination=np.array([0.7, 1.2, 2.0]),
        difficulty=np.array([-1.0, 0.0, 1.0]),
        guessing=np.array([0.1, 0.2, 0.3]),
        upper=np.array([0.95, 0.9, 0.85]),
    )


def _five_parameter() -> FiveParameterLogistic:
    model = FiveParameterLogistic(3)
    return model.set_parameters(
        discrimination=np.array([0.7, 1.2, 2.0]),
        difficulty=np.array([-1.0, 0.0, 1.0]),
        guessing=np.array([0.1, 0.2, 0.3]),
        upper=np.array([0.95, 0.9, 0.85]),
        asymmetry=np.array([0.6, 1.3, 2.0]),
    )


@pytest.mark.parametrize(
    "factory",
    [
        _three_parameter,
        _four_parameter,
        _five_parameter,
        lambda: UnipolarLogLogistic(3),
        lambda: ComplementaryLogLog(3),
        lambda: NegativeLogLog(3),
    ],
)
def test_information_matches_probability_derivative(
    factory: Callable[[], object],
) -> None:
    model = factory()
    theta = np.linspace(-2.0, 2.0, 161)
    step = 1e-5

    probability = model.probability(theta)
    derivative = (model.probability(theta + step) - model.probability(theta - step)) / (
        2.0 * step
    )
    denominator = probability * (1.0 - probability)
    numerical = np.divide(
        derivative**2,
        denominator,
        out=np.zeros_like(probability),
        where=denominator > 0,
    )
    analytic = model.information(theta)

    stable = (probability > 1e-4) & (probability < 1.0 - 1e-4)
    assert np.any(stable)
    assert_allclose(analytic[stable], numerical[stable], rtol=2e-5, atol=1e-10)


@pytest.mark.parametrize(
    "factory", [_three_parameter, _four_parameter, _five_parameter]
)
def test_single_item_information_matches_all_item_column(
    factory: Callable[[], object],
) -> None:
    model = factory()
    theta = np.linspace(-3.0, 3.0, 51)

    assert_allclose(
        model.information(theta, item_idx=1),
        model.information(theta)[:, 1],
    )


def test_three_parameter_information_is_four_parameter_limit() -> None:
    three = _three_parameter()
    four = FourParameterLogistic(3).set_parameters(
        discrimination=three.discrimination,
        difficulty=three.difficulty,
        guessing=three.guessing,
        upper=np.ones(3),
    )
    theta = np.linspace(-8.0, 8.0, 501)

    assert_allclose(three.probability(theta), four.probability(theta))
    assert_allclose(three.information(theta), four.information(theta))


def test_five_parameter_information_is_four_parameter_limit() -> None:
    five = FiveParameterLogistic(3).set_parameters(
        discrimination=np.array([0.7, 1.2, 2.0]),
        difficulty=np.array([-1.0, 0.0, 1.0]),
        guessing=np.array([0.1, 0.2, 0.3]),
        upper=np.array([0.95, 0.9, 0.85]),
        asymmetry=np.ones(3),
    )
    four = _four_parameter()
    theta = np.linspace(-8.0, 8.0, 501)

    assert_allclose(five.probability(theta), four.probability(theta))
    assert_allclose(five.information(theta), four.information(theta))


def test_unipolar_information_is_zero_at_curve_peak() -> None:
    model = UnipolarLogLogistic(2).set_parameters(
        discrimination=np.array([0.8, 1.5]),
        difficulty=np.array([-0.5, 1.0]),
    )

    probability = model.probability(model.difficulty)
    information = np.array(
        [
            model.information(np.array([difficulty]), item_idx=item_idx)[0]
            for item_idx, difficulty in enumerate(model.difficulty)
        ]
    )

    assert_allclose(np.diag(probability), np.full(2, 0.25))
    assert_allclose(information, np.zeros(2), atol=1e-15)


@pytest.mark.parametrize("model", [ComplementaryLogLog(2), NegativeLogLog(2)])
def test_double_exponential_links_are_finite_without_warnings(model: object) -> None:
    theta = np.array([-1_000.0, -100.0, 0.0, 100.0, 1_000.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        probability = model.probability(theta)
        information = model.information(theta)

    assert np.all(np.isfinite(probability))
    assert np.all(np.isfinite(information))
    assert np.all((probability >= 0) & (probability <= 1))
    assert np.all(information >= 0)
    assert_allclose(information[[0, -1]], 0.0, atol=1e-15)
