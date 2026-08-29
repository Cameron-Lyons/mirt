"""Contract tests for piecewise linear growth trajectories."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.models.dynamic import PiecewiseGrowthModel


def _reference_trajectories(
    times: np.ndarray,
    intercepts: np.ndarray,
    slopes: np.ndarray,
    changepoints: np.ndarray,
) -> np.ndarray:
    """Evaluate the original scalar recurrence for parity checks."""
    trajectories = np.empty((intercepts.size, times.size))
    for person_index, person_slopes in enumerate(slopes):
        for time_index, time_value in enumerate(times):
            piece_index = int(np.sum(time_value > changepoints))
            value = intercepts[person_index]
            time_remaining = time_value
            for piece in range(piece_index + 1):
                if piece < piece_index:
                    segment_length = (
                        changepoints[piece]
                        if piece == 0
                        else changepoints[piece] - changepoints[piece - 1]
                    )
                    value += person_slopes[piece] * segment_length
                    time_remaining -= segment_length
                else:
                    value += person_slopes[piece] * time_remaining
            trajectories[person_index, time_index] = value
    return trajectories


@pytest.mark.parametrize("n_pieces", [1, 2, 3, 5])
def test_compute_theta_matches_scalar_recurrence(n_pieces: int) -> None:
    rng = np.random.default_rng(1200 + n_pieces)
    changepoints = np.sort(rng.uniform(-2.0, 3.0, size=n_pieces - 1))
    model = PiecewiseGrowthModel(n_pieces=n_pieces, changepoints=changepoints)
    times = rng.uniform(-4.0, 6.0, size=31)
    intercepts = rng.normal(size=7)
    slopes = rng.normal(size=(7, n_pieces))

    actual = model.compute_theta(times, intercepts, slopes)
    expected = _reference_trajectories(times, intercepts, slopes, changepoints)

    assert actual.shape == (7, 31)
    assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_compute_theta_broadcasts_shared_parameters() -> None:
    model = PiecewiseGrowthModel(n_pieces=2, changepoints=np.array([1.5]))
    times = np.array([0.0, 1.5, 3.0])

    shared_slopes = model.compute_theta(
        times,
        intercept=np.array([-1.0, 2.0]),
        slopes=np.array([0.5, -0.25]),
    )
    shared_intercept = model.compute_theta(
        times,
        intercept=1.0,
        slopes=np.array([[0.5, -0.25], [1.0, 0.25]]),
    )

    assert shared_slopes.shape == (2, 3)
    assert_allclose(shared_slopes[1] - shared_slopes[0], 3.0)
    assert shared_intercept.shape == (2, 3)
    assert_allclose(shared_intercept[:, 0], 1.0)


def test_compute_theta_preserves_documented_shapes() -> None:
    model = PiecewiseGrowthModel(n_pieces=1)

    one_person = model.compute_theta(2.0, intercept=1.0, slopes=np.array([0.5]))
    many_people = model.compute_theta(
        2.0,
        intercept=np.array([1.0, 2.0]),
        slopes=np.array([[0.5], [1.0]]),
    )

    assert one_person.shape == (1,)
    assert many_people.shape == (2, 1)


def test_compute_theta_rejects_incompatible_person_counts() -> None:
    model = PiecewiseGrowthModel(n_pieces=1)

    with pytest.raises(ValueError, match="incompatible person counts"):
        model.compute_theta(
            np.array([0.0, 1.0]),
            intercept=np.array([0.0, 1.0]),
            slopes=np.array([[0.1], [0.2], [0.3]]),
        )


@pytest.mark.parametrize(
    ("time_values", "message"),
    [
        (np.array([]), "non-empty scalar or vector"),
        (np.zeros((1, 2)), "non-empty scalar or vector"),
        (np.array([0.0, np.nan]), "finite values"),
    ],
)
def test_compute_theta_rejects_invalid_times(
    time_values: np.ndarray, message: str
) -> None:
    model = PiecewiseGrowthModel(n_pieces=1)

    with pytest.raises(ValueError, match=message):
        model.compute_theta(time_values, intercept=0.0, slopes=np.array([0.1]))


@pytest.mark.parametrize(
    "slopes",
    [
        np.array([]),
        np.zeros((2, 3)),
        np.zeros((1, 1, 2)),
    ],
)
def test_compute_theta_rejects_invalid_slope_shapes(slopes: np.ndarray) -> None:
    model = PiecewiseGrowthModel(n_pieces=2, changepoints=np.array([1.0]))

    with pytest.raises(ValueError, match="slopes must have shape"):
        model.compute_theta(np.array([0.0]), intercept=0.0, slopes=slopes)


def test_compute_theta_rejects_nonfinite_person_parameters() -> None:
    model = PiecewiseGrowthModel(n_pieces=1)

    with pytest.raises(ValueError, match="intercept.*finite"):
        model.compute_theta(np.array([0.0]), intercept=np.inf, slopes=np.array([0.1]))
    with pytest.raises(ValueError, match="slopes.*finite"):
        model.compute_theta(np.array([0.0]), intercept=0.0, slopes=np.array([np.nan]))


@pytest.mark.parametrize("n_pieces", [0, -1, 1.5, True])
def test_constructor_rejects_invalid_piece_count(n_pieces: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        PiecewiseGrowthModel(n_pieces=n_pieces)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("n_pieces", "kwargs", "message"),
    [
        (2, {"changepoints": np.array([[1.0]])}, "one-dimensional"),
        (2, {"changepoints": np.array([np.inf])}, "finite values"),
        (3, {"changepoints": np.array([2.0, 1.0])}, "strictly increasing"),
        (1, {"slope_means": np.array([0.1, 0.2])}, "slope_means length"),
        (1, {"slope_means": np.array([np.nan])}, "slope_means.*finite"),
        (1, {"slope_vars": np.array([0.1, 0.2])}, "slope_vars length"),
        (1, {"slope_vars": np.array([-0.1])}, "slope_vars.*non-negative"),
        (1, {"intercept_mean": np.inf}, "intercept_mean.*finite scalar"),
        (1, {"intercept_var": -0.1}, "intercept_var.*non-negative"),
        (1, {"residual_variance": -0.1}, "residual_variance.*non-negative"),
    ],
)
def test_constructor_rejects_invalid_parameters(
    n_pieces: int, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PiecewiseGrowthModel(n_pieces=n_pieces, **kwargs)  # type: ignore[arg-type]


def test_constructor_detaches_parameter_arrays() -> None:
    changepoints = np.array([1.0])
    slope_means = np.array([0.1, 0.2])
    slope_vars = np.array([0.01, 0.02])
    model = PiecewiseGrowthModel(
        n_pieces=2,
        changepoints=changepoints,
        slope_means=slope_means,
        slope_vars=slope_vars,
    )

    changepoints[0] = 9.0
    slope_means[:] = 9.0
    slope_vars[:] = 9.0

    assert_allclose(model.changepoints, [1.0])
    assert_allclose(model.slope_means, [0.1, 0.2])
    assert_allclose(model.slope_vars, [0.01, 0.02])


@pytest.mark.parametrize("n_persons", [0, -1, 1.5, True])
def test_simulate_rejects_invalid_person_count(n_persons: object) -> None:
    model = PiecewiseGrowthModel(n_pieces=1)

    with pytest.raises(ValueError, match="positive integer"):
        model.simulate(n_persons, np.array([0.0]))  # type: ignore[arg-type]
