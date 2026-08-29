import numpy as np
import pytest
from numpy.testing import assert_allclose

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
    model = NonlinearGrowthModel(growth_type="linear")

    with pytest.raises(ValueError, match="Unknown growth type"):
        model.compute_theta(np.array([0.0, 1.0]))
