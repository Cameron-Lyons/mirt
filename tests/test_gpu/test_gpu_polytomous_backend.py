"""Contract and parity tests for tensor polytomous likelihood kernels."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import expit, logsumexp

from mirt._gpu_backend import (
    compute_log_likelihoods_gpcm_gpu,
    compute_log_likelihoods_grm_gpu,
    is_torch_available,
)
from mirt.constants import PROB_EPSILON

pytestmark = pytest.mark.skipif(
    not is_torch_available(),
    reason="PyTorch not installed",
)

Kernel = Callable[
    [
        NDArray[np.int_],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    NDArray[np.float64],
]
PolytomousInputs = tuple[
    NDArray[np.int_],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]


def _aggregate_reference(
    responses: NDArray[np.int_],
    log_category_probabilities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Aggregate a small reference problem without tensor-specific operations."""
    result = np.zeros(
        (responses.shape[0], log_category_probabilities.shape[0]),
        dtype=np.float64,
    )
    for person, response_pattern in enumerate(responses):
        for item, category in enumerate(response_pattern):
            if category >= 0:
                result[person] += log_category_probabilities[:, item, category]
    return result


def _grm_reference(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    thresholds: NDArray[np.float64],
) -> NDArray[np.float64]:
    boundary_logits = discrimination[None, :, None] * (
        quad_points[:, None, None] - thresholds[None, :, :]
    )
    boundary_probabilities = expit(boundary_logits)
    shape = (quad_points.size, responses.shape[1], 1)
    cumulative_probabilities = np.concatenate(
        (np.ones(shape), boundary_probabilities, np.zeros(shape)),
        axis=2,
    )
    category_probabilities = -np.diff(cumulative_probabilities, axis=2)
    log_probabilities = np.log(np.clip(category_probabilities, PROB_EPSILON, 1.0))
    return _aggregate_reference(responses, log_probabilities)


def _gpcm_reference(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    thresholds: NDArray[np.float64],
) -> NDArray[np.float64]:
    step_logits = discrimination[None, :, None] * (
        quad_points[:, None, None] - thresholds[None, :, :]
    )
    zero_logits = np.zeros((quad_points.size, responses.shape[1], 1))
    category_logits = np.concatenate(
        (zero_logits, np.cumsum(step_logits, axis=2)),
        axis=2,
    )
    log_probabilities = category_logits - logsumexp(
        category_logits,
        axis=2,
        keepdims=True,
    )
    log_probabilities = np.maximum(log_probabilities, np.log(PROB_EPSILON))
    return _aggregate_reference(responses, log_probabilities)


@pytest.fixture
def polytomous_inputs() -> PolytomousInputs:
    rng = np.random.default_rng(20260828)
    n_persons, n_items, n_categories = 17, 8, 5
    responses = rng.integers(
        0,
        n_categories,
        size=(n_persons, n_items),
        dtype=np.int_,
    )
    responses[rng.random(responses.shape) < 0.2] = -1
    quad_points = np.linspace(-3.5, 3.5, 13)
    discrimination = rng.uniform(0.5, 2.0, n_items)
    grm_thresholds = np.sort(
        rng.uniform(-2.0, 2.0, size=(n_items, n_categories - 1)),
        axis=1,
    )
    gpcm_thresholds = rng.uniform(
        -2.0,
        2.0,
        size=(n_items, n_categories - 1),
    )
    return (
        responses,
        quad_points,
        discrimination,
        grm_thresholds,
        gpcm_thresholds,
    )


def test_grm_kernel_matches_numpy_reference_with_missing_values(
    polytomous_inputs: PolytomousInputs,
) -> None:
    responses, points, discrimination, thresholds, _ = polytomous_inputs
    expected = _grm_reference(responses, points, discrimination, thresholds)

    actual = compute_log_likelihoods_grm_gpu(
        responses,
        points,
        discrimination,
        thresholds,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_gpcm_kernel_matches_numpy_reference_with_missing_values(
    polytomous_inputs: PolytomousInputs,
) -> None:
    responses, points, discrimination, _, thresholds = polytomous_inputs
    expected = _gpcm_reference(responses, points, discrimination, thresholds)

    actual = compute_log_likelihoods_gpcm_gpu(
        responses,
        points,
        discrimination,
        thresholds,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "kernel",
    [compute_log_likelihoods_grm_gpu, compute_log_likelihoods_gpcm_gpu],
)
def test_polytomous_kernels_ignore_missing_responses(kernel: Kernel) -> None:
    responses = np.full((3, 2), -1, dtype=np.int_)
    points = np.array([-1.0, 0.0, 1.0])
    discrimination = np.array([0.8, 1.2])
    thresholds = np.array([[-1.0, 0.0, 1.0], [-0.75, 0.25, 1.25]])

    result = kernel(responses, points, discrimination, thresholds)

    np.testing.assert_array_equal(result, np.zeros((3, 3)))


def test_gpcm_kernel_remains_finite_for_extreme_logits() -> None:
    result = compute_log_likelihoods_gpcm_gpu(
        np.array([[4]], dtype=np.int_),
        np.array([1000.0]),
        np.array([10.0]),
        np.array([[-100.0, -50.0, 0.0, 50.0]]),
    )

    assert np.isfinite(result).all()
    assert result[0, 0] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "kernel",
    [compute_log_likelihoods_grm_gpu, compute_log_likelihoods_gpcm_gpu],
)
def test_polytomous_kernels_reject_unknown_categories(kernel: Kernel) -> None:
    with pytest.raises(ValueError, match="between 0 and 2"):
        kernel(
            np.array([[3]], dtype=np.int_),
            np.array([0.0]),
            np.array([1.0]),
            np.array([[-0.5, 0.5]]),
        )


def test_grm_kernel_rejects_unordered_thresholds() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        compute_log_likelihoods_grm_gpu(
            np.array([[1]], dtype=np.int_),
            np.array([0.0]),
            np.array([1.0]),
            np.array([[0.5, -0.5]]),
        )


@pytest.mark.parametrize(
    ("responses", "points", "discrimination", "thresholds", "message"),
    [
        (
            np.array([0, 1], dtype=np.int_),
            np.array([0.0]),
            np.ones(2),
            np.zeros((2, 2)),
            "two-dimensional",
        ),
        (
            np.array([[0, 1]], dtype=np.int_),
            np.array([], dtype=np.float64),
            np.ones(2),
            np.zeros((2, 2)),
            "non-empty",
        ),
        (
            np.array([[0, 1]], dtype=np.int_),
            np.array([0.0]),
            np.ones(1),
            np.zeros((2, 2)),
            "discrimination",
        ),
        (
            np.array([[0.5]]),
            np.array([0.0]),
            np.ones(1),
            np.zeros((1, 2)),
            "finite integers",
        ),
        (
            np.array([[0]], dtype=np.int_),
            np.array([np.nan]),
            np.ones(1),
            np.zeros((1, 2)),
            "must be finite",
        ),
    ],
)
def test_polytomous_kernel_input_contracts(
    responses: NDArray[np.int_],
    points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    thresholds: NDArray[np.float64],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_log_likelihoods_gpcm_gpu(
            responses,
            points,
            discrimination,
            thresholds,
        )
