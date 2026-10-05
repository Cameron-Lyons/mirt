"""Integration contracts shared by linked equating and bounded recursion."""

from itertools import product

import numpy as np
import pytest
from scipy.special import expit

import mirt
from mirt.equating import (
    LinkingConstants,
    LinkingResult,
    observed_score_equating,
    true_score_equating,
)
from mirt.models import TwoParameterLogistic


def _pattern_distribution(probabilities, weights):
    distribution = np.zeros(probabilities.shape[1] + 1)
    for responses in product((0, 1), repeat=probabilities.shape[1]):
        likelihood = np.ones(len(weights))
        for item, response in enumerate(responses):
            likelihood *= (
                probabilities[:, item] if response else 1 - probabilities[:, item]
            )
        distribution[sum(responses)] += weights @ likelihood / weights.sum()
    return distribution


def _kolen_brennan_equivalents(old, new):
    """Transcribe Kolen and Brennan (2014, eqs. 2.14-2.18) score by score."""
    cumulative_new = np.cumsum(new)
    equivalents = []
    for score in range(len(old)):
        rank = np.sum(old[:score]) + old[score] / 2
        (above,) = np.nonzero(cumulative_new > rank)
        cell = above[0]
        below = cumulative_new[cell - 1] if cell else 0.0
        equivalents.append(cell - 0.5 + (rank - below) / new[cell])
    return np.array(equivalents)


@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_positional_linking_and_streaming_preserve_zero_weight_population_batches(
    backend,
):
    if backend == "rust" and not mirt.is_rust_available():
        pytest.skip("compiled backend unavailable")
    A, B = 1.6, -0.7
    old, new = TwoParameterLogistic(2), TwoParameterLogistic(3)
    old_slopes, old_locations = np.array([0.6, 1.1]), np.array([-0.4, 0.6])
    new_slopes, new_locations = np.array([1.2, 0.9, 1.7]), np.array([-0.9, 0.2, 1.0])
    old.set_parameters(discrimination=old_slopes, difficulty=old_locations)
    new.set_parameters(
        discrimination=A * new_slopes, difficulty=(new_locations - B) / A
    )
    linking = LinkingResult(LinkingConstants(A, B), [])
    theta = np.array([-2.0, -0.8, 0.1, 0.7, 2.1])
    weights = np.array([0.0, 0.0, 1.0, 4.0, 2.0])
    theta.setflags(write=False)
    weights.setflags(write=False)
    old_distribution = _pattern_distribution(
        expit(old_slopes * (theta[:, None] - old_locations)), weights
    )
    new_distribution = _pattern_distribution(
        expit(new_slopes * (theta[:, None] - new_locations)), weights
    )
    expected = _kolen_brennan_equivalents(old_distribution, new_distribution)

    previous_backend = mirt.get_backend()
    mirt.set_backend(backend)
    try:
        result = observed_score_equating(
            old,
            new,
            weights,
            theta,
            61,
            [1, 0],
            [2, 0, 1],
            "none",
            linking,
            batch_size=2,
        )
    finally:
        mirt.set_backend(previous_backend)

    np.testing.assert_allclose(result.new_scores, expected, atol=1e-14)
    np.testing.assert_array_equal(result.theta, theta)
    np.testing.assert_array_equal(weights, [0.0, 0.0, 1.0, 4.0, 2.0])


@pytest.mark.parametrize("equate", [true_score_equating, observed_score_equating])
def test_invalid_linking_is_rejected_before_custom_curves_are_evaluated(
    equate,
    monkeypatch,
):
    old, new = TwoParameterLogistic(2), TwoParameterLogistic(3)

    def unexpected_curve(*args, **kwargs):
        pytest.fail("invalid linking constants reached model probability callbacks")

    monkeypatch.setattr(old, "probability", unexpected_curve)
    monkeypatch.setattr(new, "probability", unexpected_curve)
    linking = LinkingResult(LinkingConstants(0.0, 0.0), [])
    with pytest.raises(ValueError, match="A must be finite and positive"):
        equate(old, new, linking_result=linking)
