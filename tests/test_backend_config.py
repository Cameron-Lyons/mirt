"""Tests for backend selection helpers."""

from __future__ import annotations

import pytest

import mirt
from mirt.backends.rust._helpers import rust_required


@pytest.fixture
def restore_backend():
    previous = mirt.get_backend()
    yield
    mirt.set_backend(previous)


def test_set_backend_numpy_and_auto(restore_backend) -> None:
    mirt.set_backend("numpy")
    assert mirt.get_backend() == "numpy"
    assert mirt.should_use_rust(True) is False

    mirt.set_backend("auto")
    assert mirt.get_backend() == "auto"
    assert mirt.should_use_rust(False) is False
    if mirt.is_rust_available():
        assert mirt.should_use_rust(True) is True


def test_set_backend_invalid() -> None:
    with pytest.raises(ValueError, match="Invalid backend"):
        mirt.set_backend("jax")


def test_set_backend_rust_when_unavailable(monkeypatch, restore_backend) -> None:
    monkeypatch.setattr("mirt._backend_config.RUST_AVAILABLE", False)
    with pytest.raises(ValueError, match="Rust backend requested"):
        mirt.set_backend("rust")


def test_get_backend_info_keys(restore_backend) -> None:
    info = mirt.get_backend_info()
    assert "current_backend" in info
    assert "effective_backend" in info
    assert "rust_available" in info
    assert info["rust_available"] is mirt.is_rust_available()


def test_rust_required_message() -> None:
    with pytest.raises(RuntimeError, match="Rust backend required for demo_fn"):
        rust_required("demo_fn")


@pytest.fixture
def fake_gpu(monkeypatch):
    """Report an available GPU to every dispatch decision."""
    import mirt.estimation.em as em_module
    import mirt.estimation.gvem as gvem_module

    for module in (em_module, gvem_module, mirt._backend_config):
        monkeypatch.setattr(module, "is_gpu_available", lambda: True)


@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_cpu_backend_preferences_disable_gpu_dispatch(
    backend, fake_gpu, restore_backend
) -> None:
    # Regression: set_backend("numpy"/"rust") still ran EM and GVEM on the GPU.
    from mirt.estimation.em import EMEstimator
    from mirt.estimation.gvem import GVEMEstimator

    if backend == "rust" and not mirt.is_rust_available():
        pytest.skip("Rust extension not built")
    mirt.set_backend(backend)

    assert EMEstimator()._should_use_gpu is False
    assert GVEMEstimator()._should_use_gpu is False
    assert mirt.get_backend_info()["effective_backend"] == backend
    # An explicit per-call request takes precedence over the preference.
    assert EMEstimator(use_gpu=True)._should_use_gpu is True
    assert GVEMEstimator(use_gpu=True)._should_use_gpu is True


@pytest.mark.parametrize("backend", ["auto", "gpu"])
def test_gpu_backend_preferences_allow_gpu_dispatch(
    backend, fake_gpu, restore_backend
) -> None:
    from mirt.estimation.em import EMEstimator
    from mirt.estimation.gvem import GVEMEstimator

    mirt.set_backend(backend)

    assert EMEstimator()._should_use_gpu is True
    assert GVEMEstimator()._should_use_gpu is True
    assert mirt.get_backend_info()["effective_backend"] == "gpu"
    assert EMEstimator(use_gpu=False)._should_use_gpu is False
    assert GVEMEstimator(use_gpu=False)._should_use_gpu is False


def test_fit_mirt_forwards_use_gpu(monkeypatch, fake_gpu, restore_backend) -> None:
    import numpy as np

    import mirt.estimation.em as em_module
    from mirt.exceptions import MirtValidationError

    calls = []

    def kernel(responses, points, discrimination, difficulty):
        calls.append(1)
        logits = discrimination * (points[:, None] - difficulty)
        log_p, log_q = -np.logaddexp(0.0, -logits), -np.logaddexp(0.0, logits)
        return (responses == 1) @ log_p.T + (responses == 0) @ log_q.T

    monkeypatch.setattr(em_module, "compute_log_likelihoods_2pl_gpu", kernel)
    data = mirt.simdata(n_persons=100, n_items=5, seed=2)
    options = {"model": "2PL", "max_iter": 3, "compute_standard_errors": False}

    mirt.fit_mirt(data, use_gpu=False, **options)
    assert not calls
    mirt.set_backend("numpy")
    mirt.fit_mirt(data, **options)
    assert not calls
    mirt.fit_mirt(data, use_gpu=True, **options)
    assert calls
    with pytest.raises(MirtValidationError, match="use_gpu"):
        mirt.fit_mirt(data, use_gpu="cuda", **options)
