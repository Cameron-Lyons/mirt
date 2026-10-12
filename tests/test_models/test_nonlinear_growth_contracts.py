import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.exceptions import MirtValidationError
from mirt.models.dynamic import NonlinearGrowthModel


def _scalar_reference(growth_type, times, asymptotes, rates, inflections):
    trajectories = []
    for asymptote, rate, inflection in zip(asymptotes, rates, inflections, strict=True):
        if growth_type == "exponential":
            trajectory = asymptote * (1.0 - np.exp(-rate * times))
        elif growth_type == "logistic":
            trajectory = asymptote / (1.0 + np.exp(-rate * (times - inflection)))
        else:
            trajectory = asymptote * np.exp(-np.exp(-rate * (times - inflection)))
        trajectories.append(trajectory)
    return np.asarray(trajectories)


@pytest.mark.parametrize("growth_type", ["exponential", "logistic", "gompertz"])
def test_compute_theta_matches_scalar_reference(growth_type):
    times = np.linspace(-2.0, 8.0, 17)
    asymptotes = np.array([0.8, 1.2, 2.0])
    rates = np.array([0.2, 0.75, 1.4])
    inflections = np.array([-0.5, 2.0, 5.0])
    model = NonlinearGrowthModel(growth_type=growth_type)

    actual = model.compute_theta(times, asymptotes, rates, inflections)

    expected = _scalar_reference(growth_type, times, asymptotes, rates, inflections)
    assert actual.shape == (3, 17)
    assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_compute_theta_broadcasts_shared_parameters():
    times = np.array([-1.0, 0.0, 1.0, 3.0])
    asymptotes = np.array([1.0, 1.5, 2.0])
    inflections = np.array([-0.5, 0.5, 1.5])
    model = NonlinearGrowthModel(growth_type="logistic")

    actual = model.compute_theta(
        times,
        asymptote=asymptotes,
        rate=0.7,
        inflection=inflections,
    )

    expected = _scalar_reference(
        "logistic",
        times,
        asymptotes,
        np.full(3, 0.7),
        inflections,
    )
    assert_allclose(actual, expected)


def test_compute_theta_preserves_scalar_and_single_person_shapes():
    model = NonlinearGrowthModel()

    assert np.asarray(model.compute_theta(0.0)).shape == ()
    assert model.compute_theta(np.array([0.0, 1.0])).shape == (2,)
    assert model.compute_theta(0.0, asymptote=np.array([1.0, 2.0])).shape == (2,)


@pytest.mark.parametrize(
    ("time_values", "parameters", "message"),
    [
        (np.array([]), {}, "time_values"),
        (np.ones((2, 2)), {}, "time_values"),
        (np.array([0.0, np.nan]), {}, "time_values"),
        (np.array([0.0]), {"asymptote": np.array([])}, "asymptote"),
        (np.array([0.0]), {"rate": np.ones((1, 2))}, "rate"),
        (np.array([0.0]), {"inflection": np.array([np.inf])}, "inflection"),
    ],
)
def test_compute_theta_rejects_invalid_inputs(time_values, parameters, message):
    with pytest.raises(ValueError, match=message):
        NonlinearGrowthModel().compute_theta(time_values, **parameters)


def test_compute_theta_rejects_incompatible_person_counts():
    with pytest.raises(ValueError, match="incompatible person counts"):
        NonlinearGrowthModel().compute_theta(
            np.array([0.0, 1.0]),
            asymptote=np.array([1.0, 2.0]),
            rate=np.array([0.5, 0.75, 1.0]),
        )


def test_compute_theta_rejects_unknown_growth_type():
    model = NonlinearGrowthModel()
    model.growth_type = "linear"

    with pytest.raises(ValueError, match="Unknown growth type"):
        model.compute_theta(np.array([0.0, 1.0]))


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"growth_type": "linear"}, "growth_type"),
        ({"rate": np.nan}, "rate must be finite"),
        ({"asymptote": np.array([1.0, 2.0])}, "asymptote must be a finite scalar"),
        ({"inflection": "early"}, "inflection must be a finite scalar"),
        ({"residual_variance": -0.1}, "residual_variance must be finite and non"),
        ({"rate_var": np.inf}, "rate_var must be finite and non"),
    ],
)
def test_constructor_rejects_invalid_options(options, message):
    with pytest.raises(MirtValidationError, match=message):
        NonlinearGrowthModel(**options)


@pytest.mark.parametrize("growth_type", ["logistic", "gompertz"])
def test_growth_velocity_is_finite_and_exact_in_saturated_tails(growth_type):
    model = NonlinearGrowthModel(growth_type=growth_type, asymptote=2.0, rate=50.0)
    times = np.array([-20.0, -15.0, -0.01, 0.0, 0.01, 15.0, 20.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        velocity = model.growth_velocity(times)

    step = 1e-7
    with np.errstate(over="ignore"):
        central = (
            model.compute_theta(times + step) - model.compute_theta(times - step)
        ) / (2.0 * step)
    assert_allclose(velocity, central, rtol=1e-7, atol=1e-9)
    assert_allclose(velocity[[0, 1, 5, 6]], 0.0, atol=1e-300)
    if growth_type == "logistic":
        assert velocity[3] == pytest.approx(2.0 * 50.0 / 4.0)
