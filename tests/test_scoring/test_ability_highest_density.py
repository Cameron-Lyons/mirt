"""Contracts for exact highest-density ability intervals."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mirt.exceptions import MirtValidationError
from mirt.results.ability_posterior import AbilityPosteriorResult


def _small_result() -> AbilityPosteriorResult:
    return AbilityPosteriorResult(
        points=np.array([-1.0, 0.0, 2.0]),
        weights=np.array([[0.2, 0.3, 0.5], [0.8, 0.2, 0.0]]),
        log_marginal_likelihood=np.array([-2.0, -3.0]),
    )


def _reference_interval(
    coordinates: np.ndarray,
    weights: np.ndarray,
    level: float,
) -> tuple[float, float]:
    unique_coordinates, inverse = np.unique(coordinates, return_inverse=True)
    marginal_weights = np.zeros(len(unique_coordinates), dtype=np.float64)
    np.add.at(marginal_weights, inverse, weights)
    marginal_weights /= np.sum(marginal_weights)

    candidates: list[tuple[float, float, float, float]] = []
    for start in range(len(unique_coordinates)):
        mass = 0.0
        for end in range(start, len(unique_coordinates)):
            mass += marginal_weights[end]
            if mass >= level:
                lower = float(unique_coordinates[start])
                upper = float(unique_coordinates[end])
                candidates.append((upper - lower, -mass, lower, upper))
                break
    _, _, lower, upper = min(candidates)
    return lower, upper


def test_skewed_intervals_are_shorter_than_equal_tail_intervals() -> None:
    result = _small_result()

    lower, upper = result.highest_density_intervals(level=0.8)
    equal_tail_lower, equal_tail_upper = result.credible_intervals(level=0.8)

    assert_array_equal(lower, [0.0, -1.0])
    assert_array_equal(upper, [2.0, -1.0])
    assert upper[0] - lower[0] < equal_tail_upper[0] - equal_tail_lower[0]


def test_point_intervals_prefer_greater_mass_then_lower_coordinate() -> None:
    greater_mass = AbilityPosteriorResult(
        points=np.array([0.0, 1.0]),
        weights=np.array([[0.4, 0.6]]),
        log_marginal_likelihood=np.array([-1.0]),
    )
    equal_mass = AbilityPosteriorResult(
        points=np.array([-1.0, 1.0]),
        weights=np.array([[0.5, 0.5]]),
        log_marginal_likelihood=np.array([-1.0]),
    )

    assert_array_equal(
        greater_mass.highest_density_intervals(level=0.35),
        (np.array([1.0]), np.array([1.0])),
    )
    assert_array_equal(
        equal_mass.highest_density_intervals(level=0.4),
        (np.array([-1.0]), np.array([-1.0])),
    )


def test_multidimensional_intervals_combine_duplicate_coordinates() -> None:
    result = AbilityPosteriorResult(
        points=np.array([[-1.0, -2.0], [-1.0, 1.0], [2.0, 1.0]]),
        weights=np.array([[0.2, 0.3, 0.5], [0.6, 0.2, 0.2]]),
        log_marginal_likelihood=np.array([-1.0, -2.0]),
    )

    lower, upper = result.highest_density_intervals(level=0.5)

    assert lower.shape == upper.shape == (2, 2)
    assert_array_equal(lower, [[-1.0, 1.0], [-1.0, -2.0]])
    assert_array_equal(upper, [[-1.0, 1.0], [-1.0, -2.0]])


def test_random_intervals_match_exhaustive_reference() -> None:
    rng = np.random.default_rng(812)
    points = rng.integers(-4, 5, size=(31, 3)).astype(np.float64)
    weights = rng.dirichlet(np.linspace(0.5, 2.0, len(points)), size=23)
    result = AbilityPosteriorResult(
        points=points,
        weights=weights,
        log_marginal_likelihood=np.zeros(len(weights)),
    )
    level = 0.73

    lower, upper = result.highest_density_intervals(level=level, batch_size=5)

    for person in range(result.n_persons):
        for factor in range(result.n_factors):
            expected = _reference_interval(
                points[:, factor],
                weights[person],
                level,
            )
            assert (lower[person, factor], upper[person, factor]) == expected
            enclosed = (points[:, factor] >= lower[person, factor]) & (
                points[:, factor] <= upper[person, factor]
            )
            assert np.sum(weights[person, enclosed]) >= level - 1e-14


def test_batch_size_does_not_change_intervals() -> None:
    rng = np.random.default_rng(91)
    points = np.linspace(-4.0, 4.0, 51)
    weights = rng.dirichlet(np.ones(len(points)), size=17)
    result = AbilityPosteriorResult(points, weights, np.zeros(len(weights)))
    expected = result.highest_density_intervals(level=0.9, batch_size=1)

    for batch_size in (2, 8, 100):
        actual = result.highest_density_intervals(
            level=0.9,
            batch_size=batch_size,
        )
        assert_array_equal(actual, expected)


@pytest.mark.parametrize("level", [0.0, 1.0, np.nan, np.inf, True, "0.9"])
def test_rejects_invalid_levels(level: object) -> None:
    with pytest.raises(MirtValidationError, match="level"):
        _small_result().highest_density_intervals(level=level)  # type: ignore[arg-type]


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True, "2"])
def test_rejects_invalid_batch_sizes(batch_size: object) -> None:
    with pytest.raises(MirtValidationError, match="batch_size"):
        _small_result().highest_density_intervals(
            batch_size=batch_size,  # type: ignore[arg-type]
        )


def test_empty_posterior_returns_empty_interval_arrays() -> None:
    result = AbilityPosteriorResult(
        points=np.array([-1.0, 0.0, 1.0]),
        weights=np.empty((0, 3)),
        log_marginal_likelihood=np.empty(0),
    )

    lower, upper = result.highest_density_intervals()

    assert lower.shape == upper.shape == (0,)
