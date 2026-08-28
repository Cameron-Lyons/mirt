"""Regression coverage for binary diagnostic backend fallbacks."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import mirt.backends.rust._helpers as backend_helpers
import mirt.backends.rust.diagnostics as diagnostics
from mirt._core import sigmoid
from mirt.constants import PROB_EPSILON


def _reference_residuals(
    responses: np.ndarray,
    theta: np.ndarray,
    discrimination: np.ndarray,
    difficulty: np.ndarray,
) -> np.ndarray:
    result = np.full(responses.shape, np.nan)
    for item in range(responses.shape[1]):
        probability = np.clip(
            sigmoid(discrimination[item] * (theta - difficulty[item])),
            PROB_EPSILON,
            1.0 - PROB_EPSILON,
        )
        valid = responses[:, item] >= 0
        variance = probability * (1.0 - probability)
        result[valid, item] = (responses[valid, item] - probability[valid]) / np.sqrt(
            variance[valid] + PROB_EPSILON
        )
    return result


def _reference_q3(
    responses: np.ndarray,
    theta: np.ndarray,
    discrimination: np.ndarray,
    difficulty: np.ndarray,
) -> np.ndarray:
    residuals = _reference_residuals(
        responses,
        theta,
        discrimination,
        difficulty,
    )
    result = np.zeros((responses.shape[1], responses.shape[1]))
    for left in range(responses.shape[1]):
        for right in range(left + 1, responses.shape[1]):
            valid = (
                (responses[:, left] >= 0)
                & (responses[:, right] >= 0)
                & np.isfinite(residuals[:, left])
                & np.isfinite(residuals[:, right])
            )
            if np.count_nonzero(valid) > 2:
                value = np.corrcoef(
                    residuals[valid, left],
                    residuals[valid, right],
                )[0, 1]
                result[left, right] = value
                result[right, left] = value
    return result


def _reference_ld_chi2(
    responses: np.ndarray,
    theta: np.ndarray,
    discrimination: np.ndarray,
    difficulty: np.ndarray,
) -> np.ndarray:
    n_items = responses.shape[1]
    result = np.full((n_items, n_items), np.nan)
    for left in range(n_items):
        for right in range(left + 1, n_items):
            valid = (responses[:, left] >= 0) & (responses[:, right] >= 0)
            if np.count_nonzero(valid) < 10:
                continue

            response_left = responses[valid, left] > 0
            response_right = responses[valid, right] > 0
            theta_valid = theta[valid]
            probability_left = sigmoid(
                discrimination[left] * (theta_valid - difficulty[left])
            )
            probability_right = sigmoid(
                discrimination[right] * (theta_valid - difficulty[right])
            )
            observed = np.array(
                [
                    np.count_nonzero(~response_left & ~response_right),
                    np.count_nonzero(~response_left & response_right),
                    np.count_nonzero(response_left & ~response_right),
                    np.count_nonzero(response_left & response_right),
                ]
            )
            expected = np.maximum(
                np.array(
                    [
                        np.sum((1.0 - probability_left) * (1.0 - probability_right)),
                        np.sum((1.0 - probability_left) * probability_right),
                        np.sum(probability_left * (1.0 - probability_right)),
                        np.sum(probability_left * probability_right),
                    ]
                ),
                0.5,
            )
            value = np.sum((observed - expected) ** 2 / expected)
            result[left, right] = value
            result[right, left] = value
    return result


def _reference_fit_statistics(
    responses: np.ndarray,
    theta: np.ndarray,
    discrimination: np.ndarray,
    difficulty: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_persons, n_items = responses.shape
    squared = np.full((n_persons, n_items), np.nan)
    variance = np.full((n_persons, n_items), np.nan)
    for item in range(n_items):
        probability = np.clip(
            sigmoid(discrimination[item] * (theta - difficulty[item])),
            PROB_EPSILON,
            1.0 - PROB_EPSILON,
        )
        item_variance = probability * (1.0 - probability)
        valid = responses[:, item] >= 0
        raw = responses[valid, item] - probability[valid]
        squared[valid, item] = raw**2 / (item_variance[valid] + PROB_EPSILON)
        variance[valid, item] = item_variance[valid]

    return (
        np.nanmean(squared, axis=0),
        np.nansum(squared * variance, axis=0) / np.nansum(variance, axis=0),
        np.nanmean(squared, axis=1),
        np.nansum(squared * variance, axis=1) / np.nansum(variance, axis=1),
    )


@pytest.fixture
def binary_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(818)
    n_persons, n_items = 73, 9
    theta = rng.normal(size=n_persons)
    discrimination = rng.uniform(0.4, 2.1, size=n_items)
    difficulty = rng.normal(size=n_items)
    probability = sigmoid(
        discrimination[None, :] * (theta[:, None] - difficulty[None, :])
    )
    responses = (rng.random(probability.shape) < probability).astype(np.int32)
    responses[rng.random(responses.shape) < 0.18] = -1
    responses[:71, -1] = -1
    return responses, theta, discrimination, difficulty


def test_numpy_fallbacks_match_scalar_references(
    binary_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses, theta, discrimination, difficulty = binary_data
    monkeypatch.setattr(diagnostics, "rust_enabled", lambda: False)

    assert_allclose(
        diagnostics.compute_standardized_residuals(
            responses,
            theta,
            discrimination,
            difficulty,
        ),
        _reference_residuals(responses, theta, discrimination, difficulty),
        rtol=1e-13,
        atol=1e-13,
        equal_nan=True,
    )
    assert_allclose(
        diagnostics.compute_q3_matrix(
            responses,
            theta,
            discrimination,
            difficulty,
        ),
        _reference_q3(responses, theta, discrimination, difficulty),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    assert_allclose(
        diagnostics.compute_ld_chi2_matrix(
            responses,
            theta,
            discrimination,
            difficulty,
        ),
        _reference_ld_chi2(responses, theta, discrimination, difficulty),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    for actual, expected in zip(
        diagnostics.compute_fit_statistics(
            responses,
            theta,
            discrimination,
            difficulty,
        ),
        _reference_fit_statistics(responses, theta, discrimination, difficulty),
        strict=True,
    ):
        assert_allclose(actual, expected, rtol=1e-13, atol=1e-13, equal_nan=True)


def test_pairwise_fallbacks_match_references_across_small_chunks(
    binary_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses, theta, discrimination, difficulty = binary_data
    monkeypatch.setattr(diagnostics, "rust_enabled", lambda: False)
    monkeypatch.setattr(backend_helpers, "_MAX_VECTOR_CHUNK_ENTRIES", 20)

    assert_allclose(
        diagnostics.compute_q3_matrix(
            responses,
            theta,
            discrimination,
            difficulty,
        ),
        _reference_q3(responses, theta, discrimination, difficulty),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    assert_allclose(
        diagnostics.compute_ld_chi2_matrix(
            responses,
            theta,
            discrimination,
            difficulty,
        ),
        _reference_ld_chi2(responses, theta, discrimination, difficulty),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


def test_missing_encodings_and_column_theta_are_normalized(
    binary_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses, theta, discrimination, difficulty = binary_data
    alternative = responses.astype(np.float64)
    alternative[responses == -1] = np.nan
    alternative[0, -1] = -9.0
    expected_responses = np.where(
        np.isfinite(alternative) & (alternative >= 0), alternative, -1
    )
    monkeypatch.setattr(diagnostics, "rust_enabled", lambda: False)

    calls: tuple[
        Callable[..., Any],
        Callable[..., Any],
        Callable[..., Any],
        Callable[..., Any],
    ] = (
        diagnostics.compute_standardized_residuals,
        diagnostics.compute_q3_matrix,
        diagnostics.compute_ld_chi2_matrix,
        diagnostics.compute_fit_statistics,
    )
    for function in calls:
        actual = function(alternative, theta[:, None], discrimination, difficulty)
        expected = function(expected_responses, theta, discrimination, difficulty)
        if isinstance(actual, tuple):
            for actual_value, expected_value in zip(actual, expected, strict=True):
                assert_allclose(actual_value, expected_value, equal_nan=True)
        else:
            assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize(
    ("responses", "theta", "discrimination", "difficulty", "match"),
    [
        (np.zeros(4), np.zeros(4), np.ones(1), np.zeros(1), "two-dimensional"),
        (np.empty((0, 2)), np.empty(0), np.ones(2), np.zeros(2), "non-empty"),
        (np.zeros((4, 2)), np.zeros(3), np.ones(2), np.zeros(2), "theta"),
        (np.zeros((4, 2)), np.zeros(4), np.ones(3), np.zeros(2), "discrimination"),
        (np.zeros((4, 2)), np.zeros(4), np.ones(2), np.zeros(3), "difficulty"),
        (
            np.zeros((4, 2)),
            np.array([0.0, 1.0, np.inf, 2.0]),
            np.ones(2),
            np.zeros(2),
            "theta",
        ),
        (np.full((4, 2), np.inf), np.zeros(4), np.ones(2), np.zeros(2), "responses"),
        (np.full((4, 2), 0.5), np.zeros(4), np.ones(2), np.zeros(2), "0 or 1"),
        (np.zeros((4, 2)), np.zeros(4), [1.0, np.nan], np.zeros(2), "discrimination"),
        (np.zeros((4, 2)), np.zeros(4), np.ones(2), [0.0, np.inf], "difficulty"),
    ],
)
def test_direct_diagnostic_inputs_are_validated(
    responses: np.ndarray,
    theta: np.ndarray,
    discrimination: np.ndarray,
    difficulty: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        diagnostics.compute_standardized_residuals(
            responses,
            theta,
            discrimination,
            difficulty,
        )


@pytest.mark.parametrize(
    "function",
    [
        diagnostics.compute_standardized_residuals,
        diagnostics.compute_q3_matrix,
        diagnostics.compute_ld_chi2_matrix,
        diagnostics.compute_fit_statistics,
    ],
)
def test_every_entry_point_validates_before_dispatch(
    function: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "rust_enabled", lambda: True)
    with pytest.raises(ValueError, match="0 or 1"):
        function(
            np.full((5, 2), 2),
            np.zeros(5),
            np.ones(2),
            np.zeros(2),
        )


def test_native_dispatch_receives_contiguous_normalized_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads: list[tuple[np.ndarray, ...]] = []

    def record_matrix(*values: np.ndarray) -> np.ndarray:
        payloads.append(values)
        return np.zeros((values[0].shape[1], values[0].shape[1]))

    def record_residuals(*values: np.ndarray) -> np.ndarray:
        payloads.append(values)
        return np.zeros(values[0].shape)

    def record_fit(*values: np.ndarray) -> tuple[np.ndarray, ...]:
        payloads.append(values)
        return tuple(
            np.zeros(size)
            for size in (values[0].shape[1],) * 2 + (values[0].shape[0],) * 2
        )

    monkeypatch.setattr(diagnostics, "rust_enabled", lambda: True)
    monkeypatch.setattr(
        diagnostics,
        "mirt_rs",
        SimpleNamespace(
            compute_standardized_residuals=record_residuals,
            compute_q3_matrix=record_matrix,
            compute_ld_chi2_matrix=record_matrix,
            compute_fit_statistics=record_fit,
        ),
    )
    responses = np.array([[1.0, np.nan, 0.0], [-4.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    theta = np.array([-0.5, 0.0, 0.5])
    discrimination = np.array([0.8, 1.0, 1.2])
    difficulty = np.array([-0.2, 0.0, 0.2])

    diagnostics.compute_standardized_residuals(
        responses,
        theta,
        discrimination,
        difficulty,
    )
    diagnostics.compute_q3_matrix(responses, theta, discrimination, difficulty)
    diagnostics.compute_ld_chi2_matrix(responses, theta, discrimination, difficulty)
    diagnostics.compute_fit_statistics(responses, theta, discrimination, difficulty)

    assert len(payloads) == 4
    for (
        response_values,
        theta_values,
        discrimination_values,
        difficulty_values,
    ) in payloads:
        assert response_values.dtype == np.int32
        assert theta_values.dtype == np.float64
        assert discrimination_values.dtype == np.float64
        assert difficulty_values.dtype == np.float64
        assert response_values.flags.c_contiguous
        assert theta_values.flags.c_contiguous
        assert_array_equal(
            response_values,
            np.array([[1, -1, 0], [-1, 1, 0], [0, 1, 1]], dtype=np.int32),
        )
