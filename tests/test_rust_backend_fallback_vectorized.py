"""Tests for vectorized Python fallback paths in ``mirt._rust_backend``."""

from __future__ import annotations

import numpy as np
import pytest

import mirt
import mirt._rust_backend as rb
import mirt.backends.rust._helpers as helpers
from mirt._core import sigmoid
from mirt.constants import PROB_EPSILON


def _slow_log_likelihoods_grm(
    responses: np.ndarray,
    quad_points: np.ndarray,
    discrimination: np.ndarray,
    thresholds: np.ndarray,
    n_categories: np.ndarray,
) -> np.ndarray:
    n_persons, n_items = responses.shape
    log_likes = np.zeros((n_persons, len(quad_points)))

    for q, theta in enumerate(quad_points):
        for i in range(n_persons):
            for j in range(n_items):
                response = responses[i, j]
                if response < 0:
                    continue
                n_cat = n_categories[j]
                if response == 0:
                    probability = 1.0 - sigmoid(
                        discrimination[j] * (theta - thresholds[j, 0])
                    )
                elif response == n_cat - 1:
                    probability = sigmoid(
                        discrimination[j] * (theta - thresholds[j, response - 1])
                    )
                else:
                    probability = sigmoid(
                        discrimination[j] * (theta - thresholds[j, response - 1])
                    ) - sigmoid(discrimination[j] * (theta - thresholds[j, response]))
                log_likes[i, q] += np.log(max(probability, PROB_EPSILON))

    return log_likes


def _slow_log_likelihoods_gpcm(
    responses: np.ndarray,
    quad_points: np.ndarray,
    discrimination: np.ndarray,
    steps: np.ndarray,
    n_categories: np.ndarray,
) -> np.ndarray:
    n_persons, n_items = responses.shape
    log_likes = np.zeros((n_persons, len(quad_points)))

    for q, theta in enumerate(quad_points):
        for i in range(n_persons):
            for j in range(n_items):
                response = responses[i, j]
                if response < 0:
                    continue
                n_cat = n_categories[j]
                numerators = np.zeros(n_cat)
                for category in range(1, n_cat):
                    numerators[category] = numerators[category - 1] + (
                        discrimination[j] * (theta - steps[j, category])
                    )
                max_numerator = np.max(numerators)
                log_denominator = max_numerator + np.log(
                    np.sum(np.exp(numerators - max_numerator))
                )
                probability = np.exp(numerators[response] - log_denominator)
                log_likes[i, q] += np.log(max(probability, PROB_EPSILON))

    return log_likes


def _with_numpy_backend_and_chunk(chunk: int):
    previous = mirt.get_backend()
    old_chunk = helpers._MAX_VECTOR_CHUNK_ENTRIES
    mirt.set_backend("numpy")
    helpers._MAX_VECTOR_CHUNK_ENTRIES = chunk
    return previous, old_chunk


def _restore(previous: str, old_chunk: int) -> None:
    helpers._MAX_VECTOR_CHUNK_ENTRIES = old_chunk
    mirt.set_backend(previous)


def test_grm_vectorized_fallback_matches_reference() -> None:
    previous, old_chunk = _with_numpy_backend_and_chunk(30)
    try:
        rng = np.random.default_rng(717)
        n_categories = np.array([2, 3, 5, 4, 3, 5], dtype=np.int32)
        responses = np.column_stack(
            [rng.integers(0, n_cat, size=22, dtype=np.int32) for n_cat in n_categories]
        )
        responses[rng.random(size=responses.shape) < 0.18] = -1
        responses[:, 2] = -1
        quad_points = np.linspace(-4.0, 4.0, 19)
        discrimination = rng.uniform(0.5, 2.0, size=len(n_categories))
        thresholds = np.sort(
            rng.normal(size=(len(n_categories), max(n_categories) - 1)),
            axis=1,
        )

        expected = _slow_log_likelihoods_grm(
            responses,
            quad_points,
            discrimination,
            thresholds,
            n_categories,
        )
        actual = rb.compute_log_likelihoods_grm(
            responses,
            quad_points,
            discrimination,
            thresholds,
            n_categories,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    finally:
        _restore(previous, old_chunk)


def test_gpcm_vectorized_fallback_matches_reference() -> None:
    previous, old_chunk = _with_numpy_backend_and_chunk(30)
    try:
        rng = np.random.default_rng(818)
        n_categories = np.array([2, 3, 5, 4, 3, 5], dtype=np.int32)
        responses = np.column_stack(
            [rng.integers(0, n_cat, size=22, dtype=np.int32) for n_cat in n_categories]
        )
        responses[rng.random(size=responses.shape) < 0.18] = -1
        responses[:, 4] = -1
        quad_points = np.linspace(-4.0, 4.0, 19)
        discrimination = rng.uniform(0.5, 2.0, size=len(n_categories))
        steps = rng.normal(size=(len(n_categories), max(n_categories)))

        expected = _slow_log_likelihoods_gpcm(
            responses,
            quad_points,
            discrimination,
            steps,
            n_categories,
        )
        actual = rb.compute_log_likelihoods_gpcm(
            responses,
            quad_points,
            discrimination,
            steps,
            n_categories,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    finally:
        _restore(previous, old_chunk)


def test_polytomous_fallbacks_skip_fully_missing_item_groups() -> None:
    previous, old_chunk = _with_numpy_backend_and_chunk(12)
    try:
        responses = np.full((4, 2), -1, dtype=np.int32)
        quad_points = np.array([-1.0, 0.0, 1.0])
        discrimination = np.array([0.8, 1.2])
        n_categories = np.array([3, 5], dtype=np.int32)
        thresholds = np.zeros((2, 4))
        steps = np.zeros((2, 5))

        grm = rb.compute_log_likelihoods_grm(
            responses,
            quad_points,
            discrimination,
            thresholds,
            n_categories,
        )
        gpcm = rb.compute_log_likelihoods_gpcm(
            responses,
            quad_points,
            discrimination,
            steps,
            n_categories,
        )

        np.testing.assert_array_equal(grm, np.zeros((4, 3)))
        np.testing.assert_array_equal(gpcm, np.zeros((4, 3)))
    finally:
        _restore(previous, old_chunk)


def test_polytomous_fallbacks_reject_out_of_range_categories() -> None:
    previous, old_chunk = _with_numpy_backend_and_chunk(12)
    try:
        responses = np.array([[3]], dtype=np.int32)
        quad_points = np.array([0.0])
        discrimination = np.array([1.0])
        n_categories = np.array([3], dtype=np.int32)

        for function, parameters in (
            (rb.compute_log_likelihoods_grm, np.zeros((1, 2))),
            (rb.compute_log_likelihoods_gpcm, np.zeros((1, 3))),
        ):
            with pytest.raises(IndexError, match="outside the item category range"):
                function(
                    responses,
                    quad_points,
                    discrimination,
                    parameters,
                    n_categories,
                )
    finally:
        _restore(previous, old_chunk)


_POLYTOMOUS_BACKENDS = [
    "numpy",
    pytest.param(
        "rust",
        marks=pytest.mark.skipif(
            not helpers.RUST_AVAILABLE, reason="Rust extension not built"
        ),
    ),
]


@pytest.fixture
def polytomous_backend(request: pytest.FixtureRequest):
    previous = mirt.get_backend()
    mirt.set_backend(request.param)
    try:
        yield request.param
    finally:
        mirt.set_backend(previous)


def _polytomous_cases() -> list[tuple[object, object, np.ndarray]]:
    thresholds = np.array([[-0.5, 0.5], [-1.0, 0.8], [0.2, 1.1]])
    steps = np.column_stack([np.zeros(3), thresholds])
    return [
        (rb.compute_log_likelihoods_grm, _slow_log_likelihoods_grm, thresholds),
        (rb.compute_log_likelihoods_gpcm, _slow_log_likelihoods_gpcm, steps),
    ]


@pytest.mark.parametrize("polytomous_backend", _POLYTOMOUS_BACKENDS, indirect=True)
def test_polytomous_likelihoods_treat_nan_as_missing(polytomous_backend: str) -> None:
    # Regression: the NumPy path dropped every item whose column held a NaN,
    # also for the persons who answered it.
    responses = np.array([[0.0, 1.0, 2.0], [np.nan, 1.0, 2.0], [2.0, np.nan, 0.0]])
    coded = np.where(np.isnan(responses), -1, responses).astype(np.int64)
    quad_points = np.array([-2.0, 0.0, 2.0])
    discrimination = np.array([0.8, 1.4, 2.0])
    n_categories = np.array([3, 3, 3], dtype=np.int32)

    for function, reference, locations in _polytomous_cases():
        expected = reference(
            coded, quad_points, discrimination, locations, n_categories
        )
        actual = function(
            responses, quad_points, discrimination, locations, n_categories
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("polytomous_backend", _POLYTOMOUS_BACKENDS, indirect=True)
def test_polytomous_likelihoods_do_not_wrap_large_codes(
    polytomous_backend: str,
) -> None:
    # Regression: the native path cast int64 codes to int32, reading 2**33 as
    # category 0 and -(2**32) + 1 as category 1.
    quad_points = np.array([-1.0, 1.0])
    discrimination = np.array([1.0, 1.5])
    n_categories = np.array([3, 3], dtype=np.int32)
    for function, reference, locations in _polytomous_cases():
        locations = locations[:2]
        for code in (2**31, 2**33):
            with pytest.raises(IndexError, match="outside the item category range"):
                function(
                    np.array([[1, code]], dtype=np.int64),
                    quad_points,
                    discrimination,
                    locations,
                    n_categories,
                )
        missing = np.array([[1, -(2**32) + 1]], dtype=np.int64)
        expected = reference(
            np.array([[1, -1]]), quad_points, discrimination, locations, n_categories
        )
        actual = function(missing, quad_points, discrimination, locations, n_categories)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("polytomous_backend", _POLYTOMOUS_BACKENDS, indirect=True)
def test_polytomous_likelihoods_propagate_nan_parameters(
    polytomous_backend: str,
) -> None:
    # Regression: the native kernels floored NaN probabilities at 1e-10.
    responses = np.array([[0, 1, 2], [-1, 1, 2]])
    quad_points = np.array([-1.0, 1.0])
    discrimination = np.array([1.0, 1.2, 0.9])
    n_categories = np.array([3, 3, 3], dtype=np.int32)
    for function, _, locations in _polytomous_cases():
        locations = locations.copy()
        locations[0, -1] = np.nan
        with np.errstate(invalid="ignore"):
            actual = function(
                responses, quad_points, discrimination, locations, n_categories
            )
        assert np.all(np.isnan(actual[0]))


@pytest.mark.skipif(not helpers.RUST_AVAILABLE, reason="Rust extension unavailable")
@pytest.mark.parametrize("parameter", ["discrimination", "location"])
def test_native_polytomous_kernels_propagate_nan_to_the_persons_it_affects(
    parameter: str,
) -> None:
    # Regression: f64::max(NaN, 1e-10) floored NaN category probabilities, so
    # the kernels returned log(1e-10) where the parameter was NaN.
    from mirt import mirt_rs

    responses = np.array([[2, 1, 0], [-1, 1, 2]], dtype=np.int32)
    quad_points = np.array([-1.0, 1.0])
    n_categories = np.array([3, 3, 3], dtype=np.int32)
    kernels = (
        mirt_rs.compute_log_likelihoods_grm,
        mirt_rs.compute_log_likelihoods_gpcm,
    )
    for kernel, (_, _, locations) in zip(kernels, _polytomous_cases(), strict=True):
        discrimination = np.array([1.0, 1.2, 0.9])
        locations = locations.copy()
        if parameter == "discrimination":
            discrimination[0] = np.nan
        else:
            locations[0, -1] = np.nan
        actual = kernel(responses, quad_points, discrimination, locations, n_categories)
        # Only the first person answered item 0, in its top category.
        assert np.all(np.isnan(actual[0]))
        assert np.all(np.isfinite(actual[1]))
