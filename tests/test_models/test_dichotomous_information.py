"""Analytic and numerical contracts for dichotomous information curves."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.models import dichotomous
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


@pytest.mark.parametrize("guessing,upper", [(0.0, 1.0), (0.2, 0.95)])
def test_five_parameter_extreme_shapes_preserve_midpoint_probability_and_information(
    guessing: float, upper: float
) -> None:
    shape = np.array([1e-20, 1e-3, 1.0, 1e10, 1e20, 1e308])
    slope = np.array([1e20, 1e3, 1.0, 5.0, 10.0, 100.0])
    log_half = np.log(0.5) / shape
    theta = (log_half - np.log(-np.expm1(log_half))) / slope
    model = FiveParameterLogistic(len(shape)).set_parameters(
        discrimination=slope,
        guessing=np.full(shape.size, guessing),
        upper=np.full(shape.size, upper),
        asymmetry=shape,
    )
    target = guessing + (upper - guessing) * 0.5
    derivative = slope * (shape * -np.expm1(log_half)) * (upper - guessing) * 0.5
    expected_information = derivative**2 / (target * (1.0 - target))
    indices = np.arange(shape.size)
    theta.flags.writeable = False

    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        probabilities = model.probability(theta)
        information = model.information(theta)
        paired = model.probability_pairs(theta[:, None], indices)
        individual = np.array(
            [model.probability([point], item_idx=i)[0] for i, point in enumerate(theta)]
        )
        individual_information = np.array(
            [model.information([point], item_idx=i)[0] for i, point in enumerate(theta)]
        )

    assert_allclose(np.diag(probabilities), target, rtol=3e-13, atol=0.0)
    assert_allclose(paired, target, rtol=3e-13, atol=0.0)
    assert_allclose(individual, paired, rtol=1e-14, atol=0.0)
    assert_allclose(np.diag(information), expected_information, rtol=3e-12, atol=0.0)
    assert_allclose(individual_information, expected_information, rtol=3e-12, atol=0.0)
    assert np.all(np.isfinite(information))


def test_five_parameter_information_preserves_both_logistic_tails() -> None:
    model = FiveParameterLogistic(1).set_parameters(guessing=np.array([0.0]))
    theta = np.array([-710.0, -400.0, 40.0, 400.0, 710.0])
    exponential = np.exp(-np.abs(theta))
    expected = exponential / (1.0 + exponential) ** 2

    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        actual = model.information(theta, item_idx=0)

    assert_allclose(actual, expected, rtol=1e-12, atol=0.0)


def test_five_parameter_large_shape_recovers_underflowing_sigmoid_tail() -> None:
    model = FiveParameterLogistic(1).set_parameters(
        guessing=np.array([0.0]), asymmetry=np.array([1e308])
    )
    theta = np.array([710.0, 800.0, 1000.0])
    exponent = np.exp(np.log(1e308) - theta)
    expected_probability = np.exp(-exponent)
    expected_information = exponent**2 / np.expm1(exponent)

    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        probability = model.probability(theta, item_idx=0)
        information = model.information(theta, item_idx=0)

    assert_allclose(probability, expected_probability, rtol=1e-14)
    assert_allclose(information, expected_information, rtol=3e-12, atol=0.0)


def test_five_parameter_subnormal_shape_preserves_information_scale() -> None:
    shape = np.nextafter(0.0, 1.0)
    slope = 1e308
    model = FiveParameterLogistic(1).set_parameters(
        discrimination=np.array([slope]),
        guessing=np.array([0.0]),
        asymmetry=np.array([shape]),
    )
    expected = np.exp(2.0 * np.log(slope) + np.log(shape) - np.log(4.0 * np.log(2.0)))

    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        information = model.information([0.0], item_idx=0)

    assert_allclose(information, expected, rtol=3e-12, atol=0.0)


def test_subnormal_power_does_not_lose_precision_before_slope_rescaling() -> None:
    slope = 1e170
    shape = 1e308
    theta = (np.log(shape) - np.log(740.0)) / slope
    exponent = np.exp(np.log(shape) - slope * theta)
    log_derivative = np.log(slope) + np.log(0.8) + np.log(exponent) - exponent
    expected = np.exp(2.0 * log_derivative - np.log(0.2 * 0.8))
    model = FiveParameterLogistic(1).set_parameters(
        discrimination=np.array([slope]), asymmetry=np.array([shape])
    )
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        actual = model.information([theta], item_idx=0)
    assert_allclose(actual, expected, rtol=3e-12, atol=0.0)


@pytest.mark.parametrize(
    "slope,guessing,upper", [(0.0, 0.2, 0.95), (1.0, 0.3, 0.3), (-1.0, 0.0, 1.0)]
)
def test_five_parameter_flat_and_decreasing_curves_are_supported(
    slope, guessing, upper
):
    model = FiveParameterLogistic(1).set_parameters(
        discrimination=np.array([slope]),
        guessing=np.array([guessing]),
        upper=np.array([upper]),
    )
    theta = np.array([-400.0, 0.0, 400.0])
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        actual = model.information(theta, item_idx=0)
    if slope == 0.0 or guessing == upper:
        assert_allclose(actual, 0.0, atol=0.0)
    else:
        exponential = np.exp(-np.abs(theta))
        assert_allclose(
            actual, exponential / (1.0 + exponential) ** 2, rtol=1e-12, atol=0.0
        )


def test_five_parameter_information_bounds_blocks_and_limits_log_fallback(monkeypatch):
    model = _five_parameter()
    theta = np.concatenate(([-400.0], np.linspace(-2.0, 2.0, 21), [400.0]))
    expected = np.column_stack([model.information(theta, i) for i in range(3)])
    original = dichotomous._five_pl_information
    original_log = dichotomous._five_pl_log_information
    block_sizes = []
    fallback_sizes = []

    def tracked(logits, *parameters):
        assert logits.size <= 31
        block_sizes.append(logits.size)
        return original(logits, *parameters)

    def tracked_log(logits, *parameters):
        assert np.all(np.abs(logits) > 100.0)
        fallback_sizes.append(logits.size)
        return original_log(logits, *parameters)

    monkeypatch.setattr(dichotomous, "_UNIDIMENSIONAL_CURVE_CHUNK_ELEMENTS", 31)
    monkeypatch.setattr(dichotomous, "_five_pl_information", tracked)
    monkeypatch.setattr(dichotomous, "_five_pl_log_information", tracked_log)
    actual = model.information(theta)

    assert_allclose(actual, expected, rtol=2e-13, atol=0.0)
    assert sum(block_sizes) == theta.size * model.n_items
    assert 0 < sum(fallback_sizes) < sum(block_sizes)
