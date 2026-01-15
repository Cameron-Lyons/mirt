"""Tests for GPU backend functions.

These tests verify numerical accuracy by comparing GPU results against
NumPy/Rust implementations.
"""

import numpy as np
import pytest

from mirt._gpu_backend import (
    GPU_AVAILABLE,
    is_gpu_available,
    is_torch_available,
)

pytestmark = pytest.mark.skipif(
    not is_torch_available(),
    reason="PyTorch not installed",
)


class TestGPUAvailability:
    def test_is_gpu_available_returns_bool(self) -> None:
        result = is_gpu_available()
        assert isinstance(result, bool)

    def test_is_torch_available_returns_bool(self) -> None:
        result = is_torch_available()
        assert isinstance(result, bool)

    def test_gpu_available_constant(self) -> None:
        assert isinstance(GPU_AVAILABLE, bool)


class TestConversionUtilities:
    def test_to_torch_and_back(self) -> None:
        from mirt._gpu_backend import to_numpy, to_torch

        arr = np.random.randn(100, 50).astype(np.float64)
        tensor = to_torch(arr)
        result = to_numpy(tensor)

        np.testing.assert_allclose(arr, result, rtol=1e-10, atol=1e-10)

    def test_to_torch_shape_preserved(self) -> None:
        from mirt._gpu_backend import to_torch

        for shape in [(10,), (10, 20), (10, 20, 30)]:
            arr = np.random.randn(*shape).astype(np.float64)
            tensor = to_torch(arr)
            assert tensor.shape == arr.shape


class TestLogLikelihoods2PL:
    def test_compute_log_likelihoods_2pl_gpu_matches_numpy(self) -> None:
        from mirt._gpu_backend import compute_log_likelihoods_2pl_gpu
        from mirt._rust_backend import compute_log_likelihoods_2pl

        rng = np.random.default_rng(42)
        n_persons, n_items, n_quad = 100, 20, 21

        responses = rng.integers(0, 2, size=(n_persons, n_items)).astype(np.int32)
        responses[rng.random((n_persons, n_items)) < 0.1] = -1

        quad_points = np.linspace(-4, 4, n_quad)
        discrimination = rng.uniform(0.5, 2.0, size=n_items)
        difficulty = rng.uniform(-2, 2, size=n_items)

        result_numpy = compute_log_likelihoods_2pl(
            responses, quad_points, discrimination, difficulty
        )
        result_gpu = compute_log_likelihoods_2pl_gpu(
            responses, quad_points, discrimination, difficulty
        )

        np.testing.assert_allclose(result_numpy, result_gpu, rtol=1e-5, atol=1e-8)

    def test_compute_log_likelihoods_2pl_gpu_shape(self) -> None:
        from mirt._gpu_backend import compute_log_likelihoods_2pl_gpu

        rng = np.random.default_rng(42)
        n_persons, n_items, n_quad = 50, 10, 15

        responses = rng.integers(0, 2, size=(n_persons, n_items)).astype(np.int32)
        quad_points = np.linspace(-3, 3, n_quad)
        discrimination = rng.uniform(0.5, 2.0, size=n_items)
        difficulty = rng.uniform(-2, 2, size=n_items)

        result = compute_log_likelihoods_2pl_gpu(
            responses, quad_points, discrimination, difficulty
        )

        assert result.shape == (n_persons, n_quad)

    def test_compute_log_likelihoods_2pl_gpu_handles_missing(self) -> None:
        from mirt._gpu_backend import compute_log_likelihoods_2pl_gpu

        n_persons, n_items, n_quad = 10, 5, 7
        responses = np.full((n_persons, n_items), -1, dtype=np.int32)

        quad_points = np.linspace(-3, 3, n_quad)
        discrimination = np.ones(n_items)
        difficulty = np.zeros(n_items)

        result = compute_log_likelihoods_2pl_gpu(
            responses, quad_points, discrimination, difficulty
        )

        np.testing.assert_allclose(result, 0.0, rtol=1e-10, atol=1e-10)


class TestLogLikelihoods3PL:
    def test_compute_log_likelihoods_3pl_gpu_matches_numpy(self) -> None:
        from mirt._gpu_backend import compute_log_likelihoods_3pl_gpu
        from mirt._rust_backend import compute_log_likelihoods_3pl

        rng = np.random.default_rng(42)
        n_persons, n_items, n_quad = 100, 20, 21

        responses = rng.integers(0, 2, size=(n_persons, n_items)).astype(np.int32)
        responses[rng.random((n_persons, n_items)) < 0.1] = -1

        quad_points = np.linspace(-4, 4, n_quad)
        discrimination = rng.uniform(0.5, 2.0, size=n_items)
        difficulty = rng.uniform(-2, 2, size=n_items)
        guessing = rng.uniform(0.1, 0.3, size=n_items)

        result_numpy = compute_log_likelihoods_3pl(
            responses, quad_points, discrimination, difficulty, guessing
        )
        result_gpu = compute_log_likelihoods_3pl_gpu(
            responses, quad_points, discrimination, difficulty, guessing
        )

        np.testing.assert_allclose(result_numpy, result_gpu, rtol=1e-5, atol=1e-8)


class TestEStepComplete:
    def test_e_step_complete_gpu_matches_numpy(self) -> None:
        from mirt._gpu_backend import e_step_complete_gpu
        from mirt._rust_backend import e_step_complete

        rng = np.random.default_rng(42)
        n_persons, n_items, n_quad = 100, 20, 21

        responses = rng.integers(0, 2, size=(n_persons, n_items)).astype(np.int32)
        responses[rng.random((n_persons, n_items)) < 0.1] = -1

        quad_points = np.linspace(-4, 4, n_quad)
        quad_weights = np.ones(n_quad) / n_quad
        discrimination = rng.uniform(0.5, 2.0, size=n_items)
        difficulty = rng.uniform(-2, 2, size=n_items)

        posterior_numpy, marginal_numpy = e_step_complete(
            responses, quad_points, quad_weights, discrimination, difficulty
        )
        posterior_gpu, marginal_gpu = e_step_complete_gpu(
            responses, quad_points, quad_weights, discrimination, difficulty
        )

        np.testing.assert_allclose(posterior_numpy, posterior_gpu, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(marginal_numpy, marginal_gpu, rtol=1e-5, atol=1e-8)

    def test_e_step_complete_gpu_posterior_sums_to_one(self) -> None:
        from mirt._gpu_backend import e_step_complete_gpu

        rng = np.random.default_rng(42)
        n_persons, n_items, n_quad = 50, 10, 15

        responses = rng.integers(0, 2, size=(n_persons, n_items)).astype(np.int32)
        quad_points = np.linspace(-3, 3, n_quad)
        quad_weights = np.ones(n_quad) / n_quad
        discrimination = rng.uniform(0.5, 2.0, size=n_items)
        difficulty = rng.uniform(-2, 2, size=n_items)

        posterior, _ = e_step_complete_gpu(
            responses, quad_points, quad_weights, discrimination, difficulty
        )

        row_sums = posterior.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5, atol=1e-8)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA GPU not available")
class TestGPUSpecific:
    def test_large_batch_performance(self) -> None:
        from mirt._gpu_backend import compute_log_likelihoods_2pl_gpu

        rng = np.random.default_rng(42)
        n_persons, n_items, n_quad = 1000, 50, 21

        responses = rng.integers(0, 2, size=(n_persons, n_items)).astype(np.int32)
        quad_points = np.linspace(-4, 4, n_quad)
        discrimination = rng.uniform(0.5, 2.0, size=n_items)
        difficulty = rng.uniform(-2, 2, size=n_items)

        result = compute_log_likelihoods_2pl_gpu(
            responses, quad_points, discrimination, difficulty
        )

        assert result.shape == (n_persons, n_quad)
        assert np.all(np.isfinite(result))


class TestGRMLogLikelihoods:
    def test_compute_log_likelihoods_grm_gpu_shape(self) -> None:
        from mirt._gpu_backend import compute_log_likelihoods_grm_gpu

        rng = np.random.default_rng(42)
        n_persons, n_items, n_quad = 50, 10, 15
        n_categories = 5

        responses = rng.integers(0, n_categories, size=(n_persons, n_items)).astype(
            np.int32
        )
        quad_points = np.linspace(-3, 3, n_quad)
        discrimination = rng.uniform(0.5, 2.0, size=n_items)
        thresholds = np.sort(
            rng.uniform(-2, 2, size=(n_items, n_categories - 1)), axis=1
        )

        result = compute_log_likelihoods_grm_gpu(
            responses, quad_points, discrimination, thresholds
        )

        assert result.shape == (n_persons, n_quad)
        assert np.all(np.isfinite(result))


class TestGPCMLogLikelihoods:
    def test_compute_log_likelihoods_gpcm_gpu_shape(self) -> None:
        from mirt._gpu_backend import compute_log_likelihoods_gpcm_gpu

        rng = np.random.default_rng(42)
        n_persons, n_items, n_quad = 50, 10, 15
        n_categories = 5

        responses = rng.integers(0, n_categories, size=(n_persons, n_items)).astype(
            np.int32
        )
        quad_points = np.linspace(-3, 3, n_quad)
        discrimination = rng.uniform(0.5, 2.0, size=n_items)
        thresholds = rng.uniform(-2, 2, size=(n_items, n_categories - 1))

        result = compute_log_likelihoods_gpcm_gpu(
            responses, quad_points, discrimination, thresholds
        )

        assert result.shape == (n_persons, n_quad)
        assert np.all(np.isfinite(result))
