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
        mirt.set_backend("jax")  # type: ignore[arg-type]


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
