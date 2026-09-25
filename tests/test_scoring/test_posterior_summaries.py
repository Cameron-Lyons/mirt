"""Numerical and memory-batching contracts for posterior summaries and draws."""

from __future__ import annotations

import tracemalloc
from fractions import Fraction

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import mirt.results.ability_posterior as posterior_module
from mirt import AbilityPosteriorResult, TwoParameterLogistic, ability_posterior
from mirt.exceptions import MirtValidationError


def _posterior(n_factors: int = 2) -> AbilityPosteriorResult:
    # Unsorted points, tied factor coordinates, and zero weights exercise the
    # discrete CDF boundaries independently of any particular model.
    points = np.array([[2.0, -1.0], [-1.0, 2.0], [0.0, -1.0], [2.0, 3.0]])
    weights = np.array([[0.0, 0.2, 0.3, 0.5], [0.5, 0.5, 0.0, 0.0]])
    return AbilityPosteriorResult(
        points=points[:, :n_factors],
        weights=weights,
        log_marginal_likelihood=np.array([-1.0, -2.0]),
    )


@pytest.mark.parametrize("n_factors", [1, 2])
@pytest.mark.parametrize("working_bytes", [1, 8 * 1024 * 1024])
def test_quantiles_match_independent_discrete_cdf(
    n_factors, working_bytes, monkeypatch
):
    monkeypatch.setattr(posterior_module, "_SUMMARY_WORKING_BYTES", working_bytes)
    result = _posterior(n_factors)
    probabilities = np.array([0.9, 0.2, 0.5, 0.2])
    expected = np.empty((len(probabilities), result.n_persons, n_factors))
    for person, weights in enumerate(result.weights):
        for factor in range(n_factors):
            coordinates = np.unique(result.points[:, factor])
            masses = np.array(
                [
                    weights[result.points[:, factor] == point].sum()
                    for point in coordinates
                ]
            )
            indices = np.searchsorted(masses.cumsum(), probabilities, side="left")
            expected[:, person, factor] = coordinates[indices]

    if n_factors == 1:
        expected = expected[:, :, 0]
    assert_array_equal(result.quantile(probabilities), expected)
    assert_array_equal(result.quantile(np.float64(0.5)), expected[2])
    assert_array_equal(result.quantile(Fraction(1, 2)), expected[2])
    assert_array_equal(result.quantile([0.5]), expected[2:3])
    assert_array_equal(result.median, expected[2])
    lower, upper = result.credible_intervals(0.6)
    assert_array_equal(lower, expected[1])
    assert_array_equal(upper, result.quantile(0.8))


@pytest.mark.parametrize(
    "probability",
    [
        [],
        [[0.5]],
        [[0.2], [0.3, 0.5]],
        True,
        np.bool_(False),
        "0.5",
        [0.2, np.nan],
        [0.0, 0.5],
        [0.5, 1.0],
        [np.inf],
        [None],
        0.5j,
    ],
)
def test_quantiles_reject_invalid_probabilities(probability):
    with pytest.raises(MirtValidationError, match="probability"):
        _posterior().quantile(probability)


@pytest.mark.parametrize("n_factors", [1, 2])
def test_batched_summaries_match_full_array_reference_without_mutation(
    n_factors, monkeypatch
):
    result = _posterior(n_factors)
    weights_before = result.weights.copy()
    cuts = (
        np.array([-1.0, 2.0])
        if n_factors == 1
        else np.array([[-1.0, 2.0], [2.0, -1.0]])
    )
    cuts_2d = cuts[:, None] if n_factors == 1 else cuts
    expected_probabilities = np.array(
        [
            [
                weights[result.points[:, factor] > cuts_2d[person, factor]].sum()
                for factor in range(n_factors)
            ]
            for person, weights in enumerate(result.weights)
        ]
    )
    expected_entropy = np.array(
        [
            -np.sum(weights[weights > 0] * np.log(weights[weights > 0]))
            for weights in result.weights
        ]
    )
    monkeypatch.setattr(posterior_module, "_SUMMARY_WORKING_BYTES", 1)

    with np.errstate(all="raise"):
        assert_allclose(result.entropy, expected_entropy)
        actual = result.classification_probabilities(cuts)
        expected = (
            expected_probabilities.ravel() if n_factors == 1 else expected_probabilities
        )
        assert_allclose(actual, expected)
        assert_allclose(result.classification_probabilities(10.0), 0.0)
        assert_allclose(result.classification_probabilities(-10.0), 1.0)
        result.quantile([0.1, 0.5, 0.9])
        result.sample(seed=42)
    assert_array_equal(result.weights, weights_before)


@pytest.mark.parametrize("n_factors", [1, 2])
def test_empty_posterior_summaries_and_sampling(n_factors):
    result = AbilityPosteriorResult(
        points=np.ones((3, n_factors)),
        weights=np.empty((0, 3)),
        log_marginal_likelihood=np.empty(0),
    )
    shape = (0,) if n_factors == 1 else (0, n_factors)
    assert result.quantile(0.5).shape == shape
    assert result.quantile([0.25, 0.75]).shape == (2, *shape)
    assert result.entropy.shape == (0,)
    assert result.classification_probabilities().shape == shape
    assert result.sample(3, seed=10).shape == (0, n_factors, 3)


def test_samples_match_seeded_inverse_cdf_and_preserve_joint_points(monkeypatch):
    result = _posterior()
    rng = np.random.default_rng(42)
    expected = np.stack(
        [
            result.points[
                np.searchsorted(weights.cumsum(), rng.random(25), side="right")
            ].T
            for weights in result.weights
        ]
    )
    assert_array_equal(result.sample(25, seed=42), expected)
    monkeypatch.setattr(posterior_module, "_SUMMARY_WORKING_BYTES", 1)
    assert_array_equal(result.sample(np.int64(25), seed=42), expected)
    assert not np.array_equal(result.sample(25, seed=41), expected)


def test_sampling_recovers_probabilities_and_never_draws_zero_mass_points():
    result = _posterior()
    draws = result.sample(20_000, seed=12)
    for person, weights in enumerate(result.weights):
        for point, weight in zip(result.points, weights, strict=True):
            frequency = np.all(draws[person].T == point, axis=1).mean()
            if weight == 0.0:
                assert frequency == 0.0
            else:
                assert frequency == pytest.approx(weight, abs=0.015)


@pytest.mark.parametrize("n_draws", [0, -1, True, np.bool_(True), 1.5, "5"])
def test_sampling_requires_a_positive_integer(n_draws):
    with pytest.raises(MirtValidationError, match="n_draws"):
        _posterior().sample(n_draws)


def test_unidimensional_samples_keep_factor_axis():
    assert _posterior(1).sample(seed=1).shape == (2, 1, 5)


def test_person_identifiers_are_validated_before_likelihood_evaluation(monkeypatch):
    model = TwoParameterLogistic(n_items=2)
    model._is_fitted = True

    def unexpected_likelihood(*args, **kwargs):
        raise AssertionError("invalid identifiers should fail before model evaluation")

    monkeypatch.setattr(model, "log_likelihood_batch", unexpected_likelihood)
    with pytest.raises(MirtValidationError, match="person_ids"):
        ability_posterior(model, [[0, 1], [1, 0]], person_ids=["only one"])


@pytest.mark.performance
@pytest.mark.parametrize(
    "operation", ["quantiles", "intervals", "entropy", "classification", "sample"]
)
def test_summary_temporary_memory_is_bounded(operation, monkeypatch):
    result = AbilityPosteriorResult(
        points=np.linspace(-3.0, 3.0, 513),
        weights=np.full((512, 513), 1.0 / 513),
        log_marginal_likelihood=np.zeros(512),
    )
    monkeypatch.setattr(posterior_module, "_SUMMARY_WORKING_BYTES", 64 * 1024)
    operations = {
        "quantiles": lambda: result.quantile([0.1, 0.5, 0.9]),
        "intervals": result.credible_intervals,
        "entropy": lambda: result.entropy,
        "classification": result.classification_probabilities,
        "sample": result.sample,
    }
    tracemalloc.start()
    try:
        operations[operation]()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Generous allowance for outputs and NumPy overhead, still far below one
    # copy of the full respondent-by-grid matrix.
    assert peak < result.weights.nbytes // 4
