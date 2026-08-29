"""Parity tests for batched tensor GVEM kernels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import mirt._gpu_backend as gpu_backend

pytestmark = pytest.mark.skipif(
    not gpu_backend.is_torch_available(),
    reason="PyTorch not installed",
)


@pytest.fixture
def cpu_tensor_runtime(monkeypatch: pytest.MonkeyPatch) -> Any:
    import torch

    monkeypatch.setattr(gpu_backend, "_torch", torch)
    monkeypatch.setattr(gpu_backend, "_torch_import_error", None)
    monkeypatch.setattr(gpu_backend, "_gpu_available", True)
    monkeypatch.setattr(gpu_backend, "_gpu_device", torch.device("cpu"))
    return gpu_backend


@pytest.fixture
def gvem_inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    n_persons, n_items, n_factors = 31, 7, 3
    responses = rng.integers(0, 2, size=(n_persons, n_items), dtype=np.int32)
    responses[rng.random(responses.shape) < 0.2] = -1
    responses[0] = -1
    responses[:, 0] = -1
    slopes = rng.normal(size=(n_items, n_factors))
    intercepts = rng.normal(size=n_items)
    prior_cov = np.array(
        [
            [1.2, 0.1, -0.05],
            [0.1, 0.9, 0.08],
            [-0.05, 0.08, 1.1],
        ]
    )
    prior_mean = np.array([0.2, -0.1, 0.3])
    mu = rng.normal(size=(n_persons, n_factors))
    raw_covariances = rng.normal(size=(n_persons, n_factors, n_factors))
    sigma = np.einsum(
        "nij,nkj->nik",
        raw_covariances,
        raw_covariances,
        optimize=True,
    )
    sigma += 0.5 * np.eye(n_factors)[None, :, :]
    xi = rng.uniform(0.2, 2.0, size=(n_persons, n_items))
    xi[1, 1] = 0.0
    return {
        "responses": responses,
        "slopes": slopes,
        "intercepts": intercepts,
        "prior_mean": prior_mean,
        "prior_cov": prior_cov,
        "mu": mu,
        "sigma": sigma,
        "xi": xi,
    }


def _jj_lambda(xi: np.ndarray) -> np.ndarray:
    absolute = np.abs(xi)
    values = np.full_like(absolute, 0.125)
    regular = absolute >= 1e-6
    values[regular] = np.tanh(absolute[regular] / 2.0) / (4.0 * absolute[regular])
    return values


def _reference_e_step(
    responses: np.ndarray,
    slopes: np.ndarray,
    intercepts: np.ndarray,
    prior_cov_inv: np.ndarray,
    xi: np.ndarray,
    n_inner_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = responses >= 0
    updated_xi = xi.copy()
    for _ in range(n_inner_iter):
        lam = _jj_lambda(updated_xi)
        weights = np.where(valid, 2.0 * lam, 0.0)
        precision = prior_cov_inv + np.einsum(
            "ij,jf,jg->ifg",
            weights,
            slopes,
            slopes,
            optimize=True,
        )
        sigma = np.linalg.inv(precision)
        coefficients = np.where(
            valid,
            responses - 0.5 - 2.0 * lam * intercepts,
            0.0,
        )
        natural_mean = coefficients @ slopes
        mu = np.einsum("ifg,ig->if", sigma, natural_mean, optimize=True)
        eta_mean = mu @ slopes.T + intercepts
        eta_variance = np.einsum(
            "jf,ifg,jg->ij",
            slopes,
            sigma,
            slopes,
            optimize=True,
        )
        candidate_xi = np.sqrt(np.maximum(eta_variance + eta_mean**2, 0.0))
        updated_xi = np.where(valid, candidate_xi, updated_xi)
    return mu, sigma, updated_xi


def test_batched_e_step_matches_array_reference(
    cpu_tensor_runtime: Any,
    gvem_inputs: dict[str, np.ndarray],
) -> None:
    prior_cov_inv = np.linalg.inv(gvem_inputs["prior_cov"])
    actual = cpu_tensor_runtime.gvem_e_step_gpu(
        gvem_inputs["responses"],
        gvem_inputs["slopes"],
        gvem_inputs["intercepts"],
        prior_cov_inv,
        gvem_inputs["mu"],
        gvem_inputs["sigma"],
        gvem_inputs["xi"],
        n_inner_iter=2,
    )
    expected = _reference_e_step(
        gvem_inputs["responses"],
        gvem_inputs["slopes"],
        gvem_inputs["intercepts"],
        prior_cov_inv,
        gvem_inputs["xi"],
        2,
    )

    assert actual is not None
    for actual_values, expected_values in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            actual_values,
            expected_values,
            rtol=1e-12,
            atol=1e-12,
        )


def test_batched_elbo_matches_array_reference(
    cpu_tensor_runtime: Any,
    gvem_inputs: dict[str, np.ndarray],
) -> None:
    responses = gvem_inputs["responses"]
    slopes = gvem_inputs["slopes"]
    intercepts = gvem_inputs["intercepts"]
    prior_mean = gvem_inputs["prior_mean"]
    prior_cov = gvem_inputs["prior_cov"]
    mu = gvem_inputs["mu"]
    sigma = gvem_inputs["sigma"]
    xi = gvem_inputs["xi"]

    actual = cpu_tensor_runtime.gvem_compute_elbo_gpu(
        responses,
        slopes,
        intercepts,
        prior_mean,
        prior_cov,
        mu,
        sigma,
        xi,
    )

    valid = responses >= 0
    lam = _jj_lambda(xi)
    eta_mean = mu @ slopes.T + intercepts
    eta_variance = np.einsum(
        "jf,ifg,jg->ij",
        slopes,
        sigma,
        slopes,
        optimize=True,
    )
    likelihood_terms = (
        -np.logaddexp(0.0, -xi)
        + (responses - 0.5) * eta_mean
        - 0.5 * xi
        - lam * (eta_variance + eta_mean**2 - xi**2)
    )
    expected_log_likelihood = np.sum(likelihood_terms[valid])

    prior_cov_inv = np.linalg.inv(prior_cov)
    difference = mu - prior_mean
    kl_mean = 0.5 * np.einsum(
        "if,fg,ig->i",
        difference,
        prior_cov_inv,
        difference,
        optimize=True,
    )
    kl_trace = 0.5 * np.einsum(
        "fg,igf->i",
        prior_cov_inv,
        sigma,
        optimize=True,
    )
    kl = (
        kl_mean
        + kl_trace
        + 0.5 * (np.linalg.slogdet(prior_cov)[1] - np.linalg.slogdet(sigma)[1])
        - 0.5 * slopes.shape[1]
    )
    expected = float(expected_log_likelihood - np.sum(kl))

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_batched_m_step_matches_array_reference_and_retains_empty_items(
    cpu_tensor_runtime: Any,
    gvem_inputs: dict[str, np.ndarray],
) -> None:
    responses = gvem_inputs["responses"]
    mu = gvem_inputs["mu"]
    sigma = gvem_inputs["sigma"]
    xi = gvem_inputs["xi"]
    current_slopes = gvem_inputs["slopes"]
    current_intercepts = gvem_inputs["intercepts"]

    actual = cpu_tensor_runtime.gvem_m_step_gpu(
        responses,
        mu,
        sigma,
        xi,
        current_slopes,
        current_intercepts,
    )

    valid = responses >= 0
    lam = _jj_lambda(xi)
    second_moments = sigma + np.einsum("if,ig->ifg", mu, mu, optimize=True)
    expected_slopes = current_slopes.copy()
    expected_intercepts = current_intercepts.copy()
    for item_index in range(responses.shape[1]):
        valid_persons = valid[:, item_index]
        if not np.any(valid_persons):
            continue
        weights = 2.0 * lam[valid_persons, item_index]
        system = np.einsum(
            "i,ifg->fg",
            weights,
            second_moments[valid_persons],
            optimize=True,
        )
        system += 1e-6 * np.eye(mu.shape[1])
        coefficients = (
            responses[valid_persons, item_index]
            - 0.5
            - weights * current_intercepts[item_index]
        )
        right_hand_side = np.einsum(
            "i,if->f",
            coefficients,
            mu[valid_persons],
            optimize=True,
        )
        expected_slopes[item_index] = np.linalg.solve(system, right_hand_side)

        linear_terms = mu[valid_persons] @ expected_slopes[item_index]
        numerator = np.sum(
            responses[valid_persons, item_index] - 0.5 - weights * linear_terms
        )
        denominator = np.sum(weights)
        expected_intercepts[item_index] = np.clip(
            numerator / denominator if denominator > 1e-12 else 0.0,
            -10.0,
            10.0,
        )

    assert actual is not None
    np.testing.assert_allclose(actual[0], expected_slopes, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual[1], expected_intercepts, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(actual[0][0], current_slopes[0])
    assert actual[1][0] == current_intercepts[0]
