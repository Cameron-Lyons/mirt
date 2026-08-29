"""Rust backend: plausible.

Fallback mode: numpy. All functions provide NumPy fallbacks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.backends.rust._helpers import (
    _entry_chunk_size,
    mirt_rs,
    rust_enabled,
)
from mirt.constants import PROB_EPSILON

FALLBACK_MODE = "numpy"
_MAX_SEED = int(np.iinfo(np.int64).max)


def _integer_argument(name: str, value: int, *, allow_zero: bool) -> int:
    """Validate an integer count without accepting booleans or truncating."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    minimum = 0 if allow_zero else 1
    if result < minimum:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _finite_scale(name: str, value: float, *, allow_zero: bool) -> float:
    """Validate a finite sampling scale."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not np.isfinite(result) or result < 0.0 or (result == 0.0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _resolve_seed(seed: int | None) -> int:
    """Return a backend-safe nonnegative seed."""
    if seed is None:
        return int(np.random.default_rng().integers(0, 2**31))
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer or None")
    result = int(seed)
    if result < 0 or result > _MAX_SEED:
        raise ValueError(f"seed must be between 0 and {_MAX_SEED}")
    return result


def _response_matrix(
    responses: NDArray[np.int_],
    *,
    binary: bool,
    missing_code: int | None = None,
) -> NDArray[np.int_]:
    """Validate response storage shared by native and NumPy paths."""
    result = np.asarray(responses)
    if result.ndim != 2:
        raise ValueError("responses must be a two-dimensional matrix")
    if not np.issubdtype(result.dtype, np.integer):
        raise TypeError("responses must contain integers")
    if result.size:
        limits = np.iinfo(np.int32)
        if np.min(result) < limits.min or np.max(result) > limits.max:
            raise ValueError("responses must fit in signed 32-bit integers")
    if binary:
        supported = (result < 0) | (result == 0) | (result == 1)
        if missing_code is not None:
            supported |= result == missing_code
        if not np.all(supported):
            raise ValueError("observed responses must be coded as 0 or 1")
    return result


def _float_vector(
    name: str,
    values: NDArray[np.float64],
    *,
    length: int | None = None,
    flatten: bool = False,
) -> NDArray[np.float64]:
    """Validate a finite float vector and optionally its length."""
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numbers") from exc
    if flatten:
        result = result.ravel()
    elif result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if length is not None and result.size != length:
        raise ValueError(f"{name} must have length {length}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _validate_2pl_inputs(
    responses: NDArray[np.int_],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> tuple[
    NDArray[np.int_],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Validate binary responses and aligned 2PL item parameters."""
    response_matrix = _response_matrix(responses, binary=True)
    n_items = response_matrix.shape[1]
    discrimination_vector = _float_vector(
        "discrimination", discrimination, length=n_items
    )
    difficulty_vector = _float_vector("difficulty", difficulty, length=n_items)
    return response_matrix, discrimination_vector, difficulty_vector


def _quadrature_inputs(
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate aligned quadrature nodes and nonnegative finite mass."""
    point_vector = _float_vector("quad_points", quad_points)
    if point_vector.size == 0:
        raise ValueError("quad_points must not be empty")
    weight_vector = _float_vector(
        "quad_weights", quad_weights, length=point_vector.size
    )
    weight_sum = float(np.sum(weight_vector))
    if np.any(weight_vector < 0.0) or not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("quad_weights must be nonnegative with positive total mass")
    return point_vector, weight_vector


def _binary_log_likelihood_grid(
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate every response pattern at every unidimensional theta."""
    linear_predictor = discrimination[None, :] * (theta[:, None] - difficulty[None, :])
    probabilities = np.clip(sigmoid(linear_predictor), PROB_EPSILON, 1.0 - PROB_EPSILON)
    correct = (responses == 1).astype(np.float64)
    incorrect = (responses == 0).astype(np.float64)
    return correct @ np.log(probabilities).T + incorrect @ np.log1p(-probabilities).T


def _sample_posterior_grid(
    cumulative_posterior: NDArray[np.float64],
    quad_points: NDArray[np.float64],
    n_plausible: int,
    jitter_sd: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Draw every posterior-grid sample with bounded inverse-CDF chunks."""
    n_persons, n_quad = cumulative_posterior.shape
    plausible_values = np.empty((n_persons, n_plausible), dtype=np.float64)
    flat_values = plausible_values.ravel()
    n_samples = flat_values.size
    chunk_size = _entry_chunk_size(
        n_samples,
        n_quad,
    )

    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        person_indices = np.arange(start, stop, dtype=np.intp) // n_plausible
        uniforms = rng.random(stop - start)
        indices = np.count_nonzero(
            uniforms[:, None] >= cumulative_posterior[person_indices],
            axis=1,
        )
        draws = quad_points[indices]
        if jitter_sd > 0.0:
            draws += rng.normal(0.0, jitter_sd, size=stop - start)
        flat_values[start:stop] = draws

    return plausible_values


def _binary_log_posterior(
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate independent standard-normal 2PL log posteriors."""
    linear_predictor = discrimination[None, :] * (theta[:, None] - difficulty[None, :])
    probabilities = np.clip(sigmoid(linear_predictor), PROB_EPSILON, 1.0 - PROB_EPSILON)
    log_likelihood = np.sum(
        (responses == 1) * np.log(probabilities)
        + (responses == 0) * np.log1p(-probabilities),
        axis=1,
    )
    return log_likelihood - 0.5 * theta**2


def _missing_code(value: int) -> int:
    """Validate a missing response code accepted by the native backend."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("missing_code must be an integer")
    result = int(value)
    limits = np.iinfo(np.int32)
    if result < limits.min or result > limits.max:
        raise ValueError("missing_code must fit in a signed 32-bit integer")
    return result


def _impute_numpy(
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    missing_code: int,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Impute all missing cells in row-major order with one vectorized kernel."""
    imputed = responses.copy()
    person_indices, item_indices = np.nonzero(responses == missing_code)
    if person_indices.size == 0:
        return imputed
    linear_predictor = discrimination[item_indices] * (
        theta[person_indices] - difficulty[item_indices]
    )
    probabilities = sigmoid(linear_predictor)
    imputed[person_indices, item_indices] = (
        rng.random(person_indices.size) < probabilities
    ).astype(imputed.dtype)
    return imputed


def generate_plausible_values_posterior(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    n_plausible: int = 5,
    jitter_sd: float = 0.3,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate plausible values using posterior sampling."""
    responses, discrimination, difficulty = _validate_2pl_inputs(
        responses, discrimination, difficulty
    )
    quad_points, quad_weights = _quadrature_inputs(quad_points, quad_weights)
    n_plausible = _integer_argument("n_plausible", n_plausible, allow_zero=False)
    jitter_sd = _finite_scale("jitter_sd", jitter_sd, allow_zero=True)
    seed = _resolve_seed(seed)

    if rust_enabled():
        return mirt_rs.generate_plausible_values_posterior(
            np.ascontiguousarray(responses, dtype=np.int32),
            np.ascontiguousarray(quad_points),
            np.ascontiguousarray(quad_weights),
            np.ascontiguousarray(discrimination),
            np.ascontiguousarray(difficulty),
            n_plausible,
            jitter_sd,
            seed,
        )

    rng = np.random.default_rng(seed)
    log_weights = np.log(quad_weights + 1e-300)
    posterior = _binary_log_likelihood_grid(
        responses, quad_points, discrimination, difficulty
    )
    posterior += log_weights[None, :]
    posterior -= np.max(posterior, axis=1, keepdims=True)
    np.exp(posterior, out=posterior)
    posterior /= np.sum(posterior, axis=1, keepdims=True)
    np.cumsum(posterior, axis=1, out=posterior)
    posterior[:, -1] = 1.0

    return _sample_posterior_grid(
        posterior,
        quad_points,
        n_plausible,
        jitter_sd,
        rng,
    )


def generate_plausible_values_mcmc(
    responses: NDArray[np.int_],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    n_plausible: int = 5,
    n_iter: int = 500,
    proposal_sd: float = 0.5,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate plausible values using MCMC."""
    responses, discrimination, difficulty = _validate_2pl_inputs(
        responses, discrimination, difficulty
    )
    n_plausible = _integer_argument("n_plausible", n_plausible, allow_zero=False)
    n_iter = _integer_argument("n_iter", n_iter, allow_zero=False)
    proposal_sd = _finite_scale("proposal_sd", proposal_sd, allow_zero=False)
    seed = _resolve_seed(seed)

    if rust_enabled():
        return mirt_rs.generate_plausible_values_mcmc(
            np.ascontiguousarray(responses, dtype=np.int32),
            np.ascontiguousarray(discrimination),
            np.ascontiguousarray(difficulty),
            n_plausible,
            n_iter,
            proposal_sd,
            seed,
        )

    rng = np.random.default_rng(seed)
    n_persons = responses.shape[0]
    pvs = np.empty((n_persons, n_plausible), dtype=np.float64)
    theta = np.zeros(n_persons, dtype=np.float64)
    current_log_posterior = _binary_log_posterior(
        responses, theta, discrimination, difficulty
    )
    for draw in range(n_plausible):
        for _ in range(n_iter):
            proposal = theta + rng.normal(0.0, proposal_sd, size=n_persons)
            proposal_log_posterior = _binary_log_posterior(
                responses, proposal, discrimination, difficulty
            )
            accepted = np.log(rng.random(n_persons)) < (
                proposal_log_posterior - current_log_posterior
            )
            theta[accepted] = proposal[accepted]
            current_log_posterior[accepted] = proposal_log_posterior[accepted]
        pvs[:, draw] = theta

    return pvs


def compute_observed_margins(
    responses: NDArray[np.int_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute observed univariate and bivariate margins."""
    responses = _response_matrix(responses, binary=False)
    if rust_enabled():
        return mirt_rs.compute_observed_margins(
            np.ascontiguousarray(responses, dtype=np.int32)
        )

    valid = responses >= 0
    values = np.where(valid, responses, 0).astype(np.float64, copy=False)
    valid_float = valid.astype(np.float64)
    univariate_counts = np.sum(valid_float, axis=0)
    obs_uni = np.divide(
        np.sum(values, axis=0),
        univariate_counts,
        out=np.zeros(responses.shape[1], dtype=np.float64),
        where=univariate_counts > 0.0,
    )
    pair_counts = valid_float.T @ valid_float
    obs_bi = np.divide(
        values.T @ values,
        pair_counts,
        out=np.zeros_like(pair_counts),
        where=pair_counts > 0.0,
    )
    np.fill_diagonal(obs_bi, 0.0)

    return obs_uni, obs_bi


def compute_expected_margins(
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute expected margins under the model."""
    quad_points, quad_weights = _quadrature_inputs(quad_points, quad_weights)
    discrimination = _float_vector("discrimination", discrimination)
    difficulty = _float_vector("difficulty", difficulty, length=discrimination.size)
    if rust_enabled():
        return mirt_rs.compute_expected_margins(
            np.ascontiguousarray(quad_points),
            np.ascontiguousarray(quad_weights),
            np.ascontiguousarray(discrimination),
            np.ascontiguousarray(difficulty),
        )

    probs = sigmoid(
        discrimination[:, None] * (quad_points[None, :] - difficulty[:, None])
    )
    exp_uni = np.sum(probs * quad_weights, axis=1)
    exp_bi = (probs * quad_weights[None, :]) @ probs.T
    np.fill_diagonal(exp_bi, 0.0)

    return exp_uni, exp_bi


def generate_bootstrap_indices(
    n_persons: int,
    n_bootstrap: int,
    seed: int | None = None,
) -> NDArray[np.int64]:
    """Generate bootstrap sample indices."""
    n_persons = _integer_argument("n_persons", n_persons, allow_zero=True)
    n_bootstrap = _integer_argument("n_bootstrap", n_bootstrap, allow_zero=True)
    seed = _resolve_seed(seed)

    if rust_enabled():
        return mirt_rs.generate_bootstrap_indices(n_persons, n_bootstrap, seed)

    rng = np.random.default_rng(seed)
    if n_persons == 0:
        return np.empty((n_bootstrap, 0), dtype=np.int64)
    return rng.integers(0, n_persons, size=(n_bootstrap, n_persons))


def resample_responses(
    responses: NDArray[np.int_],
    indices: NDArray[np.int64],
) -> NDArray[np.int_]:
    """Resample responses matrix."""
    responses = _response_matrix(responses, binary=False)
    indices = np.asarray(indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("indices must be a one-dimensional integer array")
    if np.any(indices < 0) or np.any(indices >= responses.shape[0]):
        raise IndexError("indices must refer to existing response rows")
    if rust_enabled():
        return mirt_rs.resample_responses(
            np.ascontiguousarray(responses, dtype=np.int32),
            np.ascontiguousarray(indices, dtype=np.int64),
        )

    return responses[indices]


def impute_from_probabilities(
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    missing_code: int = -1,
    seed: int | None = None,
) -> NDArray[np.int_]:
    """Impute missing responses using model probabilities."""
    missing_code = _missing_code(missing_code)
    responses = _response_matrix(responses, binary=True, missing_code=missing_code)
    n_persons, n_items = responses.shape
    theta = _float_vector("theta", theta, length=n_persons, flatten=True)
    discrimination = _float_vector("discrimination", discrimination, length=n_items)
    difficulty = _float_vector("difficulty", difficulty, length=n_items)
    seed = _resolve_seed(seed)

    if rust_enabled():
        return mirt_rs.impute_from_probabilities(
            np.ascontiguousarray(responses, dtype=np.int32),
            np.ascontiguousarray(theta),
            np.ascontiguousarray(discrimination),
            np.ascontiguousarray(difficulty),
            missing_code,
            seed,
        )

    rng = np.random.default_rng(seed)
    return _impute_numpy(
        responses, theta, discrimination, difficulty, missing_code, rng
    )


def multiple_imputation(
    responses: NDArray[np.int_],
    theta_mean: NDArray[np.float64],
    theta_se: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    missing_code: int = -1,
    n_imputations: int = 5,
    seed: int | None = None,
) -> NDArray[np.int_]:
    """Multiple imputation in parallel."""
    missing_code = _missing_code(missing_code)
    responses = _response_matrix(responses, binary=True, missing_code=missing_code)
    n_persons, n_items = responses.shape
    theta_mean = _float_vector("theta_mean", theta_mean, length=n_persons, flatten=True)
    theta_se = _float_vector("theta_se", theta_se, length=n_persons, flatten=True)
    if np.any(theta_se < 0.0):
        raise ValueError("theta_se must be nonnegative")
    discrimination = _float_vector("discrimination", discrimination, length=n_items)
    difficulty = _float_vector("difficulty", difficulty, length=n_items)
    n_imputations = _integer_argument("n_imputations", n_imputations, allow_zero=False)
    seed = _resolve_seed(seed)

    if rust_enabled():
        return mirt_rs.multiple_imputation(
            np.ascontiguousarray(responses, dtype=np.int32),
            np.ascontiguousarray(theta_mean),
            np.ascontiguousarray(theta_se),
            np.ascontiguousarray(discrimination),
            np.ascontiguousarray(difficulty),
            missing_code,
            n_imputations,
            seed,
        )

    rng = np.random.default_rng(seed)
    imputations = np.empty((n_imputations, n_persons, n_items), dtype=responses.dtype)

    for m in range(n_imputations):
        theta_draw = theta_mean + rng.standard_normal(n_persons) * theta_se
        imputations[m] = _impute_numpy(
            responses,
            theta_draw,
            discrimination,
            difficulty,
            missing_code,
            np.random.default_rng(seed + m),
        )

    return imputations
