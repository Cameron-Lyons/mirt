"""Contract tests for E-step expected-count backends."""

import numpy as np
import pytest

import mirt.backends.rust.estep as estep_module


def _posterior_weights(n_persons: int, n_quad: int = 5) -> np.ndarray:
    values = np.arange(1, n_persons * n_quad + 1, dtype=np.float64).reshape(
        n_persons,
        n_quad,
    )
    return values / values.sum(axis=1, keepdims=True)


def _binary_reference(
    responses: np.ndarray,
    posterior_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r_k = np.zeros(posterior_weights.shape[1])
    n_k = np.zeros(posterior_weights.shape[1])
    for response, weights in zip(responses, posterior_weights):
        if response < 0:
            continue
        n_k += weights
        if response == 1:
            r_k += weights
    return r_k, n_k


def _polytomous_reference(
    responses: np.ndarray,
    posterior_weights: np.ndarray,
    n_categories: int,
) -> np.ndarray:
    r_kc = np.zeros((posterior_weights.shape[1], n_categories))
    for response, weights in zip(responses, posterior_weights):
        if 0 <= response < n_categories:
            r_kc[:, response] += weights
    return r_kc


def _parallel_reference(
    responses: np.ndarray,
    posterior_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_items = responses.shape[1]
    n_quad = posterior_weights.shape[1]
    r_k = np.zeros((n_items, n_quad))
    n_k = np.zeros((n_items, n_quad))
    for item_index in range(n_items):
        r_k[item_index], n_k[item_index] = _binary_reference(
            responses[:, item_index],
            posterior_weights,
        )
    return r_k, n_k


def test_binary_expected_counts_numpy_matches_scalar_reference(monkeypatch):
    """The binary fallback preserves missing and exact-success semantics."""
    monkeypatch.setattr(estep_module, "rust_enabled", lambda: False)
    monkeypatch.setattr(estep_module, "_entry_chunk_size", lambda *_: 2)
    responses = np.array([1, 0, -1, 2, 1, -9, 0], dtype=np.int64)
    posterior_weights = _posterior_weights(len(responses))

    actual = estep_module.compute_expected_counts(responses, posterior_weights)
    expected = _binary_reference(responses, posterior_weights)

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_polytomous_expected_counts_numpy_matches_scalar_reference(monkeypatch):
    """The polytomous fallback skips missing and out-of-range categories."""
    monkeypatch.setattr(estep_module, "rust_enabled", lambda: False)
    monkeypatch.setattr(estep_module, "_entry_chunk_size", lambda *_: 2)
    responses = np.array([0, 4, 1, -1, 2, 5, 3, -9, 1], dtype=np.int64)
    posterior_weights = _posterior_weights(len(responses))

    actual = estep_module.compute_expected_counts_polytomous(
        responses,
        posterior_weights,
        n_categories=5,
    )
    expected = _polytomous_reference(responses, posterior_weights, 5)

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_parallel_expected_counts_numpy_matches_scalar_reference(monkeypatch):
    """The all-item fallback preserves semantics across person chunks."""
    monkeypatch.setattr(estep_module, "rust_enabled", lambda: False)
    monkeypatch.setattr(estep_module, "_entry_chunk_size", lambda *_: 2)
    responses = np.array(
        [
            [1, 0, -1, 2],
            [0, 1, 1, -9],
            [-1, 1, 0, 1],
            [1, 2, 1, 0],
            [0, -1, 1, 1],
            [1, 0, 0, 1],
        ],
        dtype=np.int64,
    )
    posterior_weights = _posterior_weights(len(responses))

    actual = estep_module.compute_expected_counts_parallel(
        responses,
        posterior_weights,
    )
    expected = _parallel_reference(responses, posterior_weights)

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.skipif(estep_module.mirt_rs is None, reason="native backend unavailable")
def test_polytomous_expected_counts_native_matches_numpy(monkeypatch):
    """The optimized NumPy and native polytomous paths remain equivalent."""
    responses = np.array([0, 4, 1, -1, 2, 5, 3, -9, 1], dtype=np.int64)
    posterior_weights = _posterior_weights(len(responses))

    monkeypatch.setattr(estep_module, "rust_enabled", lambda: True)
    native = estep_module.compute_expected_counts_polytomous(
        responses,
        posterior_weights,
        n_categories=5,
    )
    monkeypatch.setattr(estep_module, "rust_enabled", lambda: False)
    numpy_result = estep_module.compute_expected_counts_polytomous(
        responses,
        posterior_weights,
        n_categories=5,
    )

    np.testing.assert_allclose(numpy_result, native, rtol=1e-14, atol=1e-14)


@pytest.mark.skipif(estep_module.mirt_rs is None, reason="native backend unavailable")
def test_parallel_expected_counts_native_matches_numpy(monkeypatch):
    """The optimized NumPy and native all-item paths remain equivalent."""
    responses = np.array(
        [
            [1, 0, -1, 2],
            [0, 1, 1, -9],
            [-1, 1, 0, 1],
            [1, 2, 1, 0],
            [0, -1, 1, 1],
            [1, 0, 0, 1],
        ],
        dtype=np.int64,
    )
    posterior_weights = _posterior_weights(len(responses))

    monkeypatch.setattr(estep_module, "rust_enabled", lambda: True)
    native = estep_module.compute_expected_counts_parallel(
        responses,
        posterior_weights,
    )
    monkeypatch.setattr(estep_module, "rust_enabled", lambda: False)
    numpy_result = estep_module.compute_expected_counts_parallel(
        responses,
        posterior_weights,
    )

    np.testing.assert_allclose(numpy_result, native, rtol=1e-14, atol=1e-14)
