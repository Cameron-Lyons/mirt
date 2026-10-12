"""Parity and reporting coverage for vectorized testlet Q3 diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

import mirt.models.testlet as testlet_module
from mirt.models.testlet import RandomTestletEffectsModel, compute_testlet_q3


def _pairwise_reference(
    responses: np.ndarray,
    theta: np.ndarray,
    discrimination: np.ndarray,
    difficulty: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(responses, dtype=np.float64)
    observed = np.isfinite(values) & (values >= 0.0)
    codes = np.where(observed, values, 0.0)
    expected = 1.0 / (
        1.0 + np.exp(-(theta[:, None] * discrimination[None, :] - difficulty[None, :]))
    )
    residuals = np.where(observed, codes - expected, np.nan)
    n_items = values.shape[1]
    correlations = np.full((n_items, n_items), np.nan)
    pair_counts = observed.astype(np.intp).T @ observed.astype(np.intp)

    for first in range(n_items):
        for second in range(first, n_items):
            pair_observed = observed[:, first] & observed[:, second]
            if np.count_nonzero(pair_observed) < 2:
                continue
            first_values = residuals[pair_observed, first]
            second_values = residuals[pair_observed, second]
            if np.std(first_values) == 0.0 or np.std(second_values) == 0.0:
                continue
            correlation = np.corrcoef(first_values, second_values)[0, 1]
            correlations[first, second] = correlation
            correlations[second, first] = correlation

    return correlations, pair_counts


def _q3_inputs() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(2026)
    n_persons, n_items = 41, 7
    theta = rng.normal(size=n_persons)
    discrimination = rng.lognormal(mean=0.0, sigma=0.25, size=n_items)
    difficulty = rng.normal(size=n_items)
    probabilities = 1.0 / (
        1.0 + np.exp(-(theta[:, None] * discrimination[None, :] - difficulty[None, :]))
    )
    responses = (rng.random(probabilities.shape) < probabilities).astype(np.float64)
    responses[rng.random(responses.shape) < 0.18] = -1.0
    responses[0, 0] = np.nan
    responses[:39, 6] = -1.0
    membership = np.array([0, 0, 0, 3, 3, -1, -1])
    return responses, theta, discrimination, difficulty, membership


def test_q3_matches_pairwise_reference_and_reports_sample_sizes() -> None:
    responses, theta, discrimination, difficulty, membership = _q3_inputs()

    result = compute_testlet_q3(
        responses,
        theta,
        discrimination,
        difficulty,
        membership,
    )
    expected_q3, expected_counts = _pairwise_reference(
        responses,
        theta,
        discrimination,
        difficulty,
    )

    np.testing.assert_allclose(
        result["q3_matrix"],
        expected_q3,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    np.testing.assert_array_equal(result["pair_counts"], expected_counts)

    rows, columns = np.triu_indices(len(membership), k=1)
    expected_values = expected_q3[rows, columns]
    finite = np.isfinite(expected_values)
    within = (membership[rows] >= 0) & (membership[rows] == membership[columns])
    expected_within = expected_values[finite & within]
    expected_between = expected_values[finite & ~within]

    assert result["n_within_pairs"] == expected_within.size
    assert result["n_between_pairs"] == expected_between.size
    assert result["within_testlet_mean"] == pytest.approx(np.mean(expected_within))
    assert result["between_testlet_mean"] == pytest.approx(np.mean(expected_between))
    assert result["within_testlet_max"] == pytest.approx(np.max(expected_within))
    assert result["between_testlet_max"] == pytest.approx(np.max(expected_between))


def test_q3_row_chunking_preserves_results(monkeypatch: pytest.MonkeyPatch) -> None:
    responses, theta, discrimination, difficulty, membership = _q3_inputs()
    expected = compute_testlet_q3(
        responses,
        theta,
        discrimination,
        difficulty,
        membership,
    )
    monkeypatch.setattr(testlet_module, "_PAIRWISE_CORRELATION_TARGET_ELEMENTS", 7)

    actual = compute_testlet_q3(
        responses,
        theta,
        discrimination,
        difficulty,
        membership,
    )

    np.testing.assert_allclose(
        actual["q3_matrix"],
        expected["q3_matrix"],
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    np.testing.assert_array_equal(actual["pair_counts"], expected["pair_counts"])
    for name in (
        "within_testlet_mean",
        "between_testlet_mean",
        "within_testlet_max",
        "between_testlet_max",
    ):
        assert actual[name] == pytest.approx(expected[name])
    assert actual["n_within_pairs"] == expected["n_within_pairs"]
    assert actual["n_between_pairs"] == expected["n_between_pairs"]


def _pairwise_moment(
    responses: np.ndarray,
    theta: np.ndarray,
    model: RandomTestletEffectsModel,
    items: np.ndarray,
    sigma: float,
) -> float:
    """Observed minus implied within-testlet covariance, pair by pair."""
    nodes, weights = np.polynomial.hermite.hermgauss(model.n_quadpts)
    nodes, weights = nodes * np.sqrt(2.0), weights / np.sqrt(np.pi)
    values = np.asarray(responses, dtype=np.float64)
    observed = np.isfinite(values) & (values >= 0.0)
    a, d, b = (
        model.discrimination[items],
        model.testlet_loadings[items],
        model.difficulty[items],
    )
    curves = 1.0 / (
        1.0
        + np.exp(
            -(
                theta[:, None, None] * a[None, :, None]
                - b[None, :, None]
                + sigma * d[None, :, None] * nodes[None, None, :]
            )
        )
    )
    marginal = curves @ weights
    total = 0.0
    for first in range(items.size):
        for second in range(first + 1, items.size):
            rows = observed[:, items[first]] & observed[:, items[second]]
            residual = (values[rows, items[first]] - marginal[rows, first]) * (
                values[rows, items[second]] - marginal[rows, second]
            )
            joint = (curves[rows, first] * curves[rows, second]) @ weights
            implied = joint - marginal[rows, first] * marginal[rows, second]
            total += float(np.sum(residual - implied))
    return total


def test_testlet_variance_estimates_solve_the_pairwise_moment_condition() -> None:
    rng = np.random.default_rng(2026)
    membership = np.array([0, 0, 0, 4, 4, 4, -1])
    model = RandomTestletEffectsModel(7, membership)
    model.set_parameters(
        discrimination=rng.lognormal(0.0, 0.25, 7),
        testlet_loadings=np.r_[np.ones(6), 0.0],
        difficulty=rng.normal(size=7),
        testlet_variances=np.array([1.5, 0.6]),
    )
    theta = rng.normal(size=600)
    full = np.column_stack([theta, rng.normal(size=(600, 2))])
    probabilities = model.probability(full)
    responses = (rng.random(probabilities.shape) < probabilities).astype(np.float64)
    responses[rng.random(responses.shape) < 0.18] = -1.0
    responses[0, 0] = np.nan

    actual = model.estimate_testlet_variances(responses, theta)

    assert np.all((actual > 0.0) & (actual < 25.0))
    for position, label in enumerate(model.testlet_labels):
        items = np.flatnonzero(membership == label)
        sigma = float(np.sqrt(actual[position]))
        below = _pairwise_moment(responses, theta, model, items, 0.999 * sigma)
        above = _pairwise_moment(responses, theta, model, items, 1.001 * sigma)
        assert below > 0.0 > above
