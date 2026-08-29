"""GPU backend for MIRT using PyTorch.

This module provides GPU-accelerated implementations of computationally
intensive operations. It follows the same fallback pattern as _rust_backend.py,
automatically detecting GPU availability and falling back gracefully.
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON, REGULARIZATION_EPSILON

_torch: ModuleType | None = None
_torch_import_error: ImportError | None = None
_gpu_available: bool | None = None
_gpu_device: Any = None


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available.

    Returns
    -------
    bool
        True if PyTorch is installed and CUDA is available.
    """
    try:
        _load_torch()
    except ImportError:
        return False
    return bool(_gpu_available)


def is_torch_available() -> bool:
    """Check if PyTorch is installed (regardless of CUDA).

    Returns
    -------
    bool
        True if PyTorch is installed.
    """
    if _torch is not None:
        return True
    if _torch_import_error is not None:
        return False
    try:
        return find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


def _load_torch() -> ModuleType:
    """Import and initialize the optional PyTorch runtime on first use."""
    global _gpu_available, _gpu_device, _torch, _torch_import_error

    if _torch is not None:
        return _torch
    if _torch_import_error is not None:
        raise ImportError(
            "PyTorch is not installed. Install with: pip install torch"
        ) from _torch_import_error

    try:
        _torch = import_module("torch")
    except ImportError as exc:
        _torch_import_error = exc
        raise ImportError(
            "PyTorch is not installed. Install with: pip install torch"
        ) from exc

    _gpu_available = bool(_torch.cuda.is_available())
    _gpu_device = _torch.device("cuda" if _gpu_available else "cpu")
    return _torch


def _load_torch_runtime() -> tuple[ModuleType, Any]:
    """Return the initialized PyTorch module and preferred device."""
    torch = _load_torch()
    return torch, _gpu_device


def __getattr__(name: str) -> Any:
    """Resolve legacy availability constants without eager optional imports."""
    if name == "GPU_AVAILABLE":
        return is_gpu_available()
    if name == "GPU_DEVICE":
        try:
            _, device = _load_torch_runtime()
        except ImportError:
            return None
        return device
    if name == "_TORCH_AVAILABLE":
        return is_torch_available()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_gpu_device_name() -> str | None:
    """Get the name of the available GPU device.

    Returns
    -------
    str or None
        GPU device name, or None if no GPU is available.
    """
    if is_gpu_available():
        torch = _load_torch()
        return str(torch.cuda.get_device_name(0))
    return None


def get_gpu_memory_info() -> dict[str, int] | None:
    """Get GPU memory information.

    Returns
    -------
    dict or None
        Dictionary with 'total', 'allocated', and 'free' memory in bytes,
        or None if no GPU is available.
    """
    if is_gpu_available():
        torch = _load_torch()
        return {
            "total": torch.cuda.get_device_properties(0).total_memory,
            "allocated": torch.cuda.memory_allocated(0),
            "free": torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated(0),
        }
    return None


def to_torch(
    arr: NDArray[np.floating],
    device: Any = None,
    dtype: Any = None,
) -> Any:
    """Convert NumPy array to PyTorch tensor on specified device.

    Parameters
    ----------
    arr : ndarray
        NumPy array to convert.
    device : torch.device, optional
        Target device. Defaults to GPU_DEVICE if available, else CPU.
    dtype : torch.dtype, optional
        Target dtype. Defaults to torch.float64.

    Returns
    -------
    torch.Tensor
        PyTorch tensor on the specified device.
    """
    torch, default_device = _load_torch_runtime()

    if device is None:
        device = default_device
    if dtype is None:
        dtype = torch.float64

    return torch.from_numpy(np.asarray(arr)).to(device=device, dtype=dtype)


def to_numpy(tensor: Any) -> NDArray[np.float64]:
    """Convert PyTorch tensor to NumPy array.

    Parameters
    ----------
    tensor : torch.Tensor
        PyTorch tensor to convert.

    Returns
    -------
    ndarray
        NumPy array with the same data.
    """
    return tensor.detach().cpu().numpy()


def compute_log_likelihoods_2pl_gpu(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> NDArray[np.float64]:
    """GPU-accelerated log-likelihood computation for 2PL model.

    Computes log P(X|theta) for all persons at all quadrature points using
    fully vectorized PyTorch operations.

    Parameters
    ----------
    responses : ndarray of shape (n_persons, n_items)
        Response matrix. Missing responses coded as negative values.
    quad_points : ndarray of shape (n_quad,)
        Quadrature points (theta values).
    discrimination : ndarray of shape (n_items,)
        Item discrimination parameters (a).
    difficulty : ndarray of shape (n_items,)
        Item difficulty parameters (b).

    Returns
    -------
    ndarray of shape (n_persons, n_quad)
        Log-likelihoods for each person at each quadrature point.
    """
    torch, device = _load_torch_runtime()

    resp = torch.from_numpy(responses.astype(np.float64)).to(device)
    theta = torch.from_numpy(quad_points.astype(np.float64)).to(device)
    a = torch.from_numpy(discrimination.astype(np.float64)).to(device)
    b = torch.from_numpy(difficulty.astype(np.float64)).to(device)

    z = a[None, None, :] * (theta[None, :, None] - b[None, None, :])
    probs = torch.sigmoid(z)
    probs = probs.clamp(PROB_EPSILON, 1 - PROB_EPSILON)

    valid = (resp >= 0).unsqueeze(1)
    resp_exp = resp.unsqueeze(1)

    log_p1 = torch.log(probs)
    log_p0 = torch.log(1 - probs)

    ll_correct = resp_exp * log_p1
    ll_incorrect = (1 - resp_exp) * log_p0
    ll_per_item = ll_correct + ll_incorrect

    ll_per_item = torch.where(valid, ll_per_item, torch.zeros_like(ll_per_item))
    log_likes = ll_per_item.sum(dim=2)

    return to_numpy(log_likes)


def compute_log_likelihoods_3pl_gpu(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    guessing: NDArray[np.float64],
) -> NDArray[np.float64]:
    """GPU-accelerated log-likelihood computation for 3PL model.

    Parameters
    ----------
    responses : ndarray of shape (n_persons, n_items)
        Response matrix. Missing responses coded as negative values.
    quad_points : ndarray of shape (n_quad,)
        Quadrature points (theta values).
    discrimination : ndarray of shape (n_items,)
        Item discrimination parameters (a).
    difficulty : ndarray of shape (n_items,)
        Item difficulty parameters (b).
    guessing : ndarray of shape (n_items,)
        Item guessing parameters (c).

    Returns
    -------
    ndarray of shape (n_persons, n_quad)
        Log-likelihoods for each person at each quadrature point.
    """
    torch, device = _load_torch_runtime()

    resp = torch.from_numpy(responses.astype(np.float64)).to(device)
    theta = torch.from_numpy(quad_points.astype(np.float64)).to(device)
    a = torch.from_numpy(discrimination.astype(np.float64)).to(device)
    b = torch.from_numpy(difficulty.astype(np.float64)).to(device)
    c = torch.from_numpy(guessing.astype(np.float64)).to(device)

    z = a[None, None, :] * (theta[None, :, None] - b[None, None, :])
    p_star = torch.sigmoid(z)
    probs = c[None, None, :] + (1 - c[None, None, :]) * p_star
    probs = probs.clamp(PROB_EPSILON, 1 - PROB_EPSILON)

    valid = (resp >= 0).unsqueeze(1)
    resp_exp = resp.unsqueeze(1)

    log_p1 = torch.log(probs)
    log_p0 = torch.log(1 - probs)

    ll_per_item = resp_exp * log_p1 + (1 - resp_exp) * log_p0
    ll_per_item = torch.where(valid, ll_per_item, torch.zeros_like(ll_per_item))
    log_likes = ll_per_item.sum(dim=2)

    return to_numpy(log_likes)


def compute_log_likelihoods_mirt_gpu(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> NDArray[np.float64]:
    """GPU-accelerated log-likelihood for multidimensional 2PL model.

    Parameters
    ----------
    responses : ndarray of shape (n_persons, n_items)
        Response matrix. Missing responses coded as negative values.
    quad_points : ndarray of shape (n_quad, n_factors)
        Quadrature points in multidimensional space.
    discrimination : ndarray of shape (n_items, n_factors)
        Item discrimination parameters (factor loadings).
    difficulty : ndarray of shape (n_items,)
        Item difficulty parameters.

    Returns
    -------
    ndarray of shape (n_persons, n_quad)
        Log-likelihoods for each person at each quadrature point.
    """
    torch, device = _load_torch_runtime()

    resp = torch.from_numpy(responses.astype(np.float64)).to(device)
    theta = torch.from_numpy(quad_points.astype(np.float64)).to(device)
    a = torch.from_numpy(discrimination.astype(np.float64)).to(device)
    b = torch.from_numpy(difficulty.astype(np.float64)).to(device)

    a_sum = a.sum(dim=1)
    z = torch.einsum("qf,jf->qj", theta, a) - a_sum[None, :] * b[None, :]

    probs = torch.sigmoid(z)
    probs = probs.clamp(PROB_EPSILON, 1 - PROB_EPSILON)

    valid = resp >= 0
    log_p1 = torch.log(probs)
    log_p0 = torch.log(1 - probs)

    ll_per_item = (
        resp[:, None, :] * log_p1[None, :, :]
        + (1 - resp[:, None, :]) * log_p0[None, :, :]
    )
    ll_per_item = torch.where(
        valid[:, None, :], ll_per_item, torch.zeros_like(ll_per_item)
    )
    log_likes = ll_per_item.sum(dim=2)

    return to_numpy(log_likes)


def e_step_complete_gpu(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    prior_mean: float = 0.0,
    prior_var: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """GPU-accelerated complete E-step for 2PL model.

    Computes posterior weights and marginal likelihoods in a single pass.

    Parameters
    ----------
    responses : ndarray of shape (n_persons, n_items)
        Response matrix. Missing responses coded as negative values.
    quad_points : ndarray of shape (n_quad,)
        Quadrature points.
    quad_weights : ndarray of shape (n_quad,)
        Quadrature weights.
    discrimination : ndarray of shape (n_items,)
        Item discrimination parameters.
    difficulty : ndarray of shape (n_items,)
        Item difficulty parameters.
    prior_mean : float
        Mean of the prior distribution.
    prior_var : float
        Variance of the prior distribution.

    Returns
    -------
    posterior_weights : ndarray of shape (n_persons, n_quad)
        Posterior probability weights for each person at each quadrature point.
    marginal_ll : ndarray of shape (n_persons,)
        Marginal likelihood for each person.
    """
    torch, device = _load_torch_runtime()

    resp = torch.from_numpy(responses.astype(np.float64)).to(device)
    theta = torch.from_numpy(quad_points.astype(np.float64)).to(device)
    weights = torch.from_numpy(quad_weights.astype(np.float64)).to(device)
    a = torch.from_numpy(discrimination.astype(np.float64)).to(device)
    b = torch.from_numpy(difficulty.astype(np.float64)).to(device)

    z = a[None, None, :] * (theta[None, :, None] - b[None, None, :])
    probs = torch.sigmoid(z)
    probs = probs.clamp(PROB_EPSILON, 1 - PROB_EPSILON)

    valid = (resp >= 0).unsqueeze(1)
    resp_exp = resp.unsqueeze(1)

    ll_per_item = resp_exp * torch.log(probs) + (1 - resp_exp) * torch.log(1 - probs)
    ll_per_item = torch.where(valid, ll_per_item, torch.zeros_like(ll_per_item))
    log_likes = ll_per_item.sum(dim=2)

    log_reference = -0.5 * (np.log(2 * np.pi) + theta**2)
    log_target = (
        -0.5 * np.log(2 * np.pi * prior_var)
        - 0.5 * ((theta - prior_mean) ** 2) / prior_var
    )
    log_prior_mass = torch.log(weights + 1e-300) + log_target - log_reference
    log_prior_mass -= torch.logsumexp(log_prior_mass, dim=0)

    log_joint = log_likes + log_prior_mass[None, :]

    log_marginal = torch.logsumexp(log_joint, dim=1, keepdim=True)
    log_posterior = log_joint - log_marginal

    posterior_weights = torch.exp(log_posterior)
    marginal_ll = torch.exp(log_marginal.squeeze(1))

    return to_numpy(posterior_weights), to_numpy(marginal_ll)


def compute_expected_counts_gpu(
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """GPU-accelerated expected count computation for dichotomous items.

    Parameters
    ----------
    responses : ndarray of shape (n_persons,)
        Item responses for a single item. Missing coded as negative.
    posterior_weights : ndarray of shape (n_persons, n_quad)
        Posterior weights from E-step.

    Returns
    -------
    r_k : ndarray of shape (n_quad,)
        Expected number of correct responses at each quadrature point.
    n_k : ndarray of shape (n_quad,)
        Expected number of responses at each quadrature point.
    """
    torch, device = _load_torch_runtime()

    resp = torch.from_numpy(responses.astype(np.float64)).to(device)
    weights = torch.from_numpy(posterior_weights.astype(np.float64)).to(device)

    valid = resp >= 0
    correct = resp == 1

    r_k = (weights[correct]).sum(dim=0)
    n_k = (weights[valid]).sum(dim=0)

    return to_numpy(r_k), to_numpy(n_k)


def _torch_jj_lambda(torch: ModuleType, xi: Any) -> Any:
    """Evaluate the Jaakkola--Jordan coefficient without zero divisions."""
    xi_abs = torch.abs(xi)
    safe_xi = torch.clamp(xi_abs, min=1e-6)
    values = torch.tanh(safe_xi / 2.0) / (4.0 * safe_xi)
    return torch.where(xi_abs < 1e-6, torch.full_like(xi_abs, 0.125), values)


def gvem_e_step_gpu(
    responses: NDArray[np.int_],
    slopes: NDArray[np.float64],
    intercepts: NDArray[np.float64],
    prior_cov_inv: NDArray[np.float64],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    xi: NDArray[np.float64],
    n_inner_iter: int = 3,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None:
    """GPU-accelerated E-step for GVEM estimator.

    Uses the Jaakkola-Jordan bound for closed-form variational updates.

    Parameters
    ----------
    responses : ndarray of shape (n_persons, n_items)
        Response matrix. Missing responses coded as negative values.
    slopes : ndarray of shape (n_items, n_factors)
        Item slope parameters (factor loadings).
    intercepts : ndarray of shape (n_items,)
        Item intercept parameters.
    prior_cov_inv : ndarray of shape (n_factors, n_factors)
        Inverse of prior covariance matrix.
    mu : ndarray of shape (n_persons, n_factors)
        Current variational means.
    sigma : ndarray of shape (n_persons, n_factors, n_factors)
        Current variational covariances.
    xi : ndarray of shape (n_persons, n_items)
        Current variational parameters for JJ bound.
    n_inner_iter : int
        Number of inner iterations for variational updates.

    Returns
    -------
    tuple or None
        Updated (mu, sigma, xi) or None if GPU not available.
    """
    if not is_gpu_available():
        return None

    torch, device = _load_torch_runtime()

    resp = torch.from_numpy(responses.astype(np.float64)).to(device)
    a = torch.from_numpy(slopes.astype(np.float64)).to(device)
    d = torch.from_numpy(intercepts.astype(np.float64)).to(device)
    prior_inv = torch.from_numpy(prior_cov_inv.astype(np.float64)).to(device)
    mu_t = torch.from_numpy(mu.astype(np.float64)).to(device)
    sigma_t = torch.from_numpy(sigma.astype(np.float64)).to(device)
    xi_t = torch.from_numpy(xi.astype(np.float64)).to(device)

    valid_mask = resp >= 0

    for _ in range(n_inner_iter):
        lam = _torch_jj_lambda(torch, xi_t)
        weights = torch.where(valid_mask, 2.0 * lam, 0.0)
        precision = prior_inv.unsqueeze(0) + torch.einsum(
            "ij,jf,jg->ifg",
            weights,
            a,
            a,
        )
        sigma_t = torch.linalg.inv(precision)

        coefficients = torch.where(
            valid_mask,
            resp - 0.5 - 2.0 * lam * d.unsqueeze(0),
            0.0,
        )
        natural_mean = coefficients @ a
        mu_t = torch.einsum("ifg,ig->if", sigma_t, natural_mean)

        eta_mean = mu_t @ a.T + d
        eta_variance = torch.einsum("jf,ifg,jg->ij", a, sigma_t, a)
        updated_xi = torch.sqrt(torch.clamp(eta_variance + eta_mean**2, min=0.0))
        xi_t = torch.where(valid_mask, updated_xi, xi_t)

    return to_numpy(mu_t), to_numpy(sigma_t), to_numpy(xi_t)


def gvem_compute_elbo_gpu(
    responses: NDArray[np.int_],
    slopes: NDArray[np.float64],
    intercepts: NDArray[np.float64],
    prior_mean: NDArray[np.float64],
    prior_cov: NDArray[np.float64],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    xi: NDArray[np.float64],
) -> float | None:
    """GPU-accelerated ELBO computation for GVEM.

    Parameters
    ----------
    responses : ndarray of shape (n_persons, n_items)
        Response matrix.
    slopes : ndarray of shape (n_items, n_factors)
        Item slope parameters.
    intercepts : ndarray of shape (n_items,)
        Item intercept parameters.
    prior_mean : ndarray of shape (n_factors,)
        Prior mean.
    prior_cov : ndarray of shape (n_factors, n_factors)
        Prior covariance.
    mu : ndarray of shape (n_persons, n_factors)
        Variational means.
    sigma : ndarray of shape (n_persons, n_factors, n_factors)
        Variational covariances.
    xi : ndarray of shape (n_persons, n_items)
        Variational parameters.

    Returns
    -------
    float or None
        ELBO value or None if GPU not available.
    """
    if not is_gpu_available():
        return None

    torch, device = _load_torch_runtime()

    resp = torch.from_numpy(responses.astype(np.float64)).to(device)
    a = torch.from_numpy(slopes.astype(np.float64)).to(device)
    d = torch.from_numpy(intercepts.astype(np.float64)).to(device)
    prior_mu = torch.from_numpy(prior_mean.astype(np.float64)).to(device)
    prior_cov_t = torch.from_numpy(prior_cov.astype(np.float64)).to(device)
    mu_t = torch.from_numpy(mu.astype(np.float64)).to(device)
    sigma_t = torch.from_numpy(sigma.astype(np.float64)).to(device)
    xi_t = torch.from_numpy(xi.astype(np.float64)).to(device)

    n_factors = slopes.shape[1]

    valid_mask = resp >= 0

    prior_cov_inv = torch.linalg.inv(prior_cov_t)
    prior_log_det = torch.linalg.slogdet(prior_cov_t)[1]

    lam = _torch_jj_lambda(torch, xi_t)
    eta_mean = mu_t @ a.T + d
    eta_variance = torch.einsum("jf,ifg,jg->ij", a, sigma_t, a)
    eta_second = eta_variance + eta_mean**2
    likelihood_terms = (
        -torch.logaddexp(torch.zeros_like(xi_t), -xi_t)
        + (resp - 0.5) * eta_mean
        - 0.5 * xi_t
        - lam * (eta_second - xi_t**2)
    )
    expected_log_likelihood = likelihood_terms[valid_mask].sum()

    diff = mu_t - prior_mu
    kl_mean = 0.5 * torch.einsum("if,fg,ig->i", diff, prior_cov_inv, diff)
    kl_trace = 0.5 * torch.einsum("fg,igf->i", prior_cov_inv, sigma_t)
    log_det_q = torch.linalg.slogdet(sigma_t)[1]
    kl = kl_mean + kl_trace + 0.5 * (prior_log_det - log_det_q) - 0.5 * n_factors

    return float((expected_log_likelihood - kl.sum()).item())


def gvem_m_step_gpu(
    responses: NDArray[np.int_],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    xi: NDArray[np.float64],
    current_slopes: NDArray[np.float64],
    current_intercepts: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """GPU-accelerated M-step for GVEM estimator.

    Solves weighted least squares problems for item parameters.

    Parameters
    ----------
    responses : ndarray of shape (n_persons, n_items)
        Response matrix.
    mu : ndarray of shape (n_persons, n_factors)
        Variational means.
    sigma : ndarray of shape (n_persons, n_factors, n_factors)
        Variational covariances.
    xi : ndarray of shape (n_persons, n_items)
        Variational parameters.
    current_slopes : ndarray of shape (n_items, n_factors)
        Current slope estimates.
    current_intercepts : ndarray of shape (n_items,)
        Current intercept estimates.

    Returns
    -------
    tuple or None
        Updated (slopes, intercepts) or None if GPU not available.
    """
    if not is_gpu_available():
        return None

    torch, device = _load_torch_runtime()

    resp = torch.from_numpy(responses.astype(np.float64)).to(device)
    mu_t = torch.from_numpy(mu.astype(np.float64)).to(device)
    sigma_t = torch.from_numpy(sigma.astype(np.float64)).to(device)
    xi_t = torch.from_numpy(xi.astype(np.float64)).to(device)

    n_factors = mu.shape[1]

    valid_mask = resp >= 0

    lam = _torch_jj_lambda(torch, xi_t)
    weights = torch.where(valid_mask, 2.0 * lam, 0.0)
    second_moments = sigma_t + torch.einsum("if,ig->ifg", mu_t, mu_t)
    slope_system = torch.einsum("ij,ifg->jfg", weights, second_moments)
    slope_system += REGULARIZATION_EPSILON * torch.eye(
        n_factors,
        device=device,
        dtype=torch.float64,
    ).unsqueeze(0)

    current_d = torch.as_tensor(
        current_intercepts,
        device=device,
        dtype=torch.float64,
    )
    slope_coefficients = torch.where(
        valid_mask,
        resp - 0.5 - weights * current_d.unsqueeze(0),
        0.0,
    )
    slope_rhs = torch.einsum("ij,if->jf", slope_coefficients, mu_t)
    new_slopes = torch.linalg.solve(slope_system, slope_rhs.unsqueeze(2)).squeeze(2)

    linear_terms = mu_t @ new_slopes.T
    intercept_numerator = torch.where(
        valid_mask,
        resp - 0.5 - weights * linear_terms,
        0.0,
    ).sum(dim=0)
    intercept_denominator = weights.sum(dim=0)
    new_intercepts = torch.where(
        intercept_denominator > PROB_EPSILON,
        intercept_numerator / torch.clamp(intercept_denominator, min=PROB_EPSILON),
        0.0,
    ).clamp(-10.0, 10.0)

    valid_items = valid_mask.any(dim=0)
    current_a = torch.as_tensor(current_slopes, device=device, dtype=torch.float64)
    new_slopes = torch.where(valid_items[:, None], new_slopes, current_a)
    new_intercepts = torch.where(valid_items, new_intercepts, current_d)

    return to_numpy(new_slopes), to_numpy(new_intercepts)


def _validate_polytomous_likelihood_inputs(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    thresholds: NDArray[np.float64],
) -> int:
    """Validate shared polytomous likelihood inputs and return category count."""
    response_values = np.asarray(responses)
    point_values = np.asarray(quad_points)
    discrimination_values = np.asarray(discrimination)
    threshold_values = np.asarray(thresholds)

    if response_values.ndim != 2:
        raise ValueError("responses must be a two-dimensional matrix")
    n_items = response_values.shape[1]
    if point_values.ndim != 1 or point_values.size == 0:
        raise ValueError("quad_points must be a non-empty one-dimensional array")
    if discrimination_values.shape != (n_items,):
        raise ValueError(f"discrimination must have shape ({n_items},)")
    if (
        threshold_values.ndim != 2
        or threshold_values.shape[0] != n_items
        or threshold_values.shape[1] == 0
    ):
        raise ValueError("thresholds must have shape (n_items, n_categories - 1)")
    if not np.all(np.isfinite(response_values)) or np.any(
        response_values != np.floor(response_values)
    ):
        raise ValueError("response category codes must be finite integers")
    if (
        not np.all(np.isfinite(point_values))
        or not np.all(np.isfinite(discrimination_values))
        or not np.all(np.isfinite(threshold_values))
    ):
        raise ValueError("model parameters and quadrature points must be finite")

    n_categories = threshold_values.shape[1] + 1
    observed = response_values >= 0
    if np.any(response_values[observed] >= n_categories):
        raise ValueError(
            "responses must be negative for missing values or between "
            f"0 and {n_categories - 1}"
        )
    return n_categories


def _aggregate_polytomous_log_likelihoods(
    torch: ModuleType,
    responses: Any,
    log_category_probabilities: Any,
) -> Any:
    """Sum selected item-category log probabilities for every person."""
    valid = responses >= 0
    safe_responses = responses.clamp_min(0)
    indicators = torch.nn.functional.one_hot(
        safe_responses,
        num_classes=log_category_probabilities.shape[2],
    ).to(dtype=log_category_probabilities.dtype)
    indicators *= valid.unsqueeze(-1)
    return torch.einsum(
        "pik,qik->pq",
        indicators,
        log_category_probabilities,
    )


def compute_log_likelihoods_grm_gpu(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    thresholds: NDArray[np.float64],
) -> NDArray[np.float64]:
    """GPU-accelerated log-likelihood for Graded Response Model.

    Parameters
    ----------
    responses : ndarray of shape (n_persons, n_items)
        Response matrix (0 to n_categories-1). Missing coded as negative.
    quad_points : ndarray of shape (n_quad,)
        Quadrature points.
    discrimination : ndarray of shape (n_items,)
        Item discrimination parameters.
    thresholds : ndarray of shape (n_items, n_categories-1)
        Category threshold parameters.

    Returns
    -------
    ndarray of shape (n_persons, n_quad)
        Log-likelihoods.
    """
    n_categories = _validate_polytomous_likelihood_inputs(
        responses,
        quad_points,
        discrimination,
        thresholds,
    )
    if np.any(np.diff(thresholds, axis=1) <= 0.0):
        raise ValueError("GRM thresholds must be strictly increasing within each item")

    torch, device = _load_torch_runtime()
    resp = torch.as_tensor(responses, device=device, dtype=torch.int64)
    theta = torch.as_tensor(quad_points, device=device, dtype=torch.float64)
    a = torch.as_tensor(discrimination, device=device, dtype=torch.float64)
    b = torch.as_tensor(thresholds, device=device, dtype=torch.float64)

    boundary_logits = a[None, :, None] * (theta[:, None, None] - b[None, :, :])
    boundary_probabilities = torch.sigmoid(boundary_logits)
    shape = (theta.shape[0], responses.shape[1], 1)
    cumulative_probabilities = torch.cat(
        (
            torch.ones(shape, device=device, dtype=torch.float64),
            boundary_probabilities,
            torch.zeros(shape, device=device, dtype=torch.float64),
        ),
        dim=2,
    )
    category_probabilities = (
        cumulative_probabilities[:, :, :n_categories]
        - cumulative_probabilities[:, :, 1:]
    ).clamp(PROB_EPSILON, 1.0)
    log_likes = _aggregate_polytomous_log_likelihoods(
        torch,
        resp,
        torch.log(category_probabilities),
    )
    return to_numpy(log_likes)


def compute_log_likelihoods_gpcm_gpu(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    thresholds: NDArray[np.float64],
) -> NDArray[np.float64]:
    """GPU-accelerated log-likelihood for Generalized Partial Credit Model.

    Parameters
    ----------
    responses : ndarray of shape (n_persons, n_items)
        Response matrix (0 to n_categories-1). Missing coded as negative.
    quad_points : ndarray of shape (n_quad,)
        Quadrature points.
    discrimination : ndarray of shape (n_items,)
        Item discrimination parameters.
    thresholds : ndarray of shape (n_items, n_categories-1)
        Step difficulty parameters.

    Returns
    -------
    ndarray of shape (n_persons, n_quad)
        Log-likelihoods.
    """
    _validate_polytomous_likelihood_inputs(
        responses,
        quad_points,
        discrimination,
        thresholds,
    )
    torch, device = _load_torch_runtime()
    resp = torch.as_tensor(responses, device=device, dtype=torch.int64)
    theta = torch.as_tensor(quad_points, device=device, dtype=torch.float64)
    a = torch.as_tensor(discrimination, device=device, dtype=torch.float64)
    b = torch.as_tensor(thresholds, device=device, dtype=torch.float64)

    step_logits = a[None, :, None] * (theta[:, None, None] - b[None, :, :])
    zero_logits = torch.zeros(
        (theta.shape[0], responses.shape[1], 1),
        device=device,
        dtype=torch.float64,
    )
    category_logits = torch.cat(
        (zero_logits, torch.cumsum(step_logits, dim=2)),
        dim=2,
    )
    log_category_probabilities = torch.log_softmax(category_logits, dim=2).clamp_min(
        float(np.log(PROB_EPSILON))
    )
    log_likes = _aggregate_polytomous_log_likelihoods(
        torch,
        resp,
        log_category_probabilities,
    )
    return to_numpy(log_likes)
