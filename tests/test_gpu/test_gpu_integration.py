"""Integration tests for GPU-accelerated estimation.

These tests verify that models fit with GPU acceleration produce
results consistent with CPU implementations.
"""

import numpy as np
import pytest

from mirt._gpu_backend import is_torch_available
from mirt.models.dichotomous import TwoParameterLogistic

pytestmark = pytest.mark.skipif(
    not is_torch_available(),
    reason="PyTorch not installed",
)


class TestEMEstimatorGPU:
    def test_em_estimator_with_gpu_fits_2pl(self) -> None:
        from mirt.estimation.em import EMEstimator

        rng = np.random.default_rng(42)
        n_persons, n_items = 200, 10

        true_theta = rng.standard_normal(n_persons)
        true_a = rng.uniform(0.8, 1.5, n_items)
        true_b = rng.uniform(-1.5, 1.5, n_items)

        z = true_a[None, :] * (true_theta[:, None] - true_b[None, :])
        probs = 1 / (1 + np.exp(-z))
        responses = (rng.random((n_persons, n_items)) < probs).astype(np.int32)

        model = TwoParameterLogistic(n_items=n_items)
        estimator = EMEstimator(
            n_quadpts=21,
            max_iter=100,
            tol=1e-4,
            use_gpu=True,
        )

        result = estimator.fit(model, responses)

        assert result.converged or result.n_iterations == 100
        assert np.isfinite(result.log_likelihood)

        est_a = result.model.parameters["discrimination"]
        est_b = result.model.parameters["difficulty"]

        assert est_a.shape == (n_items,)
        assert est_b.shape == (n_items,)
        assert np.all(est_a > 0)

    def test_em_estimator_gpu_vs_cpu_consistency(self) -> None:
        from mirt.estimation.em import EMEstimator

        rng = np.random.default_rng(123)
        n_persons, n_items = 150, 8

        true_theta = rng.standard_normal(n_persons)
        true_a = rng.uniform(0.8, 1.5, n_items)
        true_b = rng.uniform(-1.5, 1.5, n_items)

        z = true_a[None, :] * (true_theta[:, None] - true_b[None, :])
        probs = 1 / (1 + np.exp(-z))
        responses = (rng.random((n_persons, n_items)) < probs).astype(np.int32)

        model_cpu = TwoParameterLogistic(n_items=n_items)
        estimator_cpu = EMEstimator(
            n_quadpts=21,
            max_iter=50,
            tol=1e-4,
            use_gpu=False,
        )
        result_cpu = estimator_cpu.fit(model_cpu, responses)

        model_gpu = TwoParameterLogistic(n_items=n_items)
        estimator_gpu = EMEstimator(
            n_quadpts=21,
            max_iter=50,
            tol=1e-4,
            use_gpu=True,
        )
        result_gpu = estimator_gpu.fit(model_gpu, responses)

        np.testing.assert_allclose(
            result_cpu.model.parameters["discrimination"],
            result_gpu.model.parameters["discrimination"],
            rtol=0.1,
            atol=0.1,
        )
        np.testing.assert_allclose(
            result_cpu.model.parameters["difficulty"],
            result_gpu.model.parameters["difficulty"],
            rtol=0.1,
            atol=0.1,
        )


class TestGVEMEstimatorGPU:
    def test_gvem_estimator_with_gpu_fits_2pl(self) -> None:
        from mirt.estimation.gvem import GVEMEstimator

        rng = np.random.default_rng(42)
        n_persons, n_items = 200, 10

        true_theta = rng.standard_normal(n_persons)
        true_a = rng.uniform(0.8, 1.5, n_items)
        true_b = rng.uniform(-1.5, 1.5, n_items)

        z = true_a[None, :] * (true_theta[:, None] - true_b[None, :])
        probs = 1 / (1 + np.exp(-z))
        responses = (rng.random((n_persons, n_items)) < probs).astype(np.int32)

        model = TwoParameterLogistic(n_items=n_items)
        estimator = GVEMEstimator(
            max_iter=100,
            tol=1e-4,
            use_gpu=True,
        )

        result = estimator.fit(model, responses)

        assert result.converged or result.n_iterations == 100
        assert np.isfinite(result.log_likelihood)

        est_a = result.model.parameters["discrimination"]
        est_b = result.model.parameters["difficulty"]

        assert est_a.shape == (n_items,)
        assert est_b.shape == (n_items,)


class TestBackendSelection:
    def test_set_backend_to_gpu(self) -> None:
        import mirt

        if mirt.GPU_AVAILABLE:
            mirt.set_backend("gpu")
            assert mirt.get_backend() == "gpu"
            mirt.set_backend("auto")
        else:
            with pytest.raises(ValueError, match="GPU backend requested"):
                mirt.set_backend("gpu")

    def test_get_backend_info(self) -> None:
        import mirt

        info = mirt.get_backend_info()

        assert "current_backend" in info
        assert "effective_backend" in info
        assert "gpu_available" in info
        assert "rust_available" in info
        assert "torch_available" in info

        assert info["torch_available"] is True

    def test_auto_backend_selection(self) -> None:
        import mirt

        mirt.set_backend("auto")
        info = mirt.get_backend_info()

        if info["gpu_available"]:
            assert info["effective_backend"] == "gpu"
        elif info["rust_available"]:
            assert info["effective_backend"] == "rust"
        else:
            assert info["effective_backend"] == "numpy"
