"""Rust backend: joint response-accuracy and response-time inference.

Fallback mode: numpy. All functions provide vectorized NumPy fallbacks and
honor the global backend preference.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.backends.rust._helpers import mirt_rs, rust_enabled
from mirt.constants import PROB_EPSILON

FALLBACK_MODE = "numpy"
_LOG_2PI = float(np.log(2.0 * np.pi))
_SAMPLE_LIKELIHOOD_TARGET_ELEMENTS = 2_000_000


def _as_f64(values: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return a native-safe contiguous float64 array."""
    return np.ascontiguousarray(values, dtype=np.float64)


def _as_i32(values: NDArray[np.integer]) -> NDArray[np.int32]:
    """Return a native-safe contiguous int32 array."""
    return np.ascontiguousarray(values, dtype=np.int32)


def rt_joint_log_likelihood(
    responses: NDArray[np.integer],
    log_rt: NDArray[np.float64],
    theta: NDArray[np.float64],
    tau: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    time_discrimination: NDArray[np.float64],
    time_intensity: NDArray[np.float64],
    guessing: NDArray[np.float64] | None = None,
    *,
    use_rust: bool = True,
) -> NDArray[np.float64]:
    """Compute conditional joint log likelihoods for every person."""
    if use_rust and rust_enabled():
        arguments = (
            _as_i32(responses),
            _as_f64(log_rt),
            _as_f64(theta),
            _as_f64(tau),
            _as_f64(discrimination),
            _as_f64(difficulty),
        )
        timing_arguments = (
            _as_f64(time_discrimination),
            _as_f64(time_intensity),
        )
        if guessing is None:
            return np.asarray(
                mirt_rs.rt_joint_log_likelihood(
                    *arguments,
                    *timing_arguments,
                ),
                dtype=np.float64,
            )
        return np.asarray(
            mirt_rs.rt_joint_log_likelihood_3pl(
                *arguments,
                _as_f64(guessing),
                *timing_arguments,
            ),
            dtype=np.float64,
        )

    responses = np.asarray(responses)
    log_rt = np.asarray(log_rt, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    discrimination = np.asarray(discrimination, dtype=np.float64)
    difficulty = np.asarray(difficulty, dtype=np.float64)
    time_discrimination = np.asarray(time_discrimination, dtype=np.float64)
    time_intensity = np.asarray(time_intensity, dtype=np.float64)

    logits = discrimination[None, :] * (theta[:, None] - difficulty[None, :])
    probability = sigmoid(logits)
    if guessing is not None:
        guessing_values = np.asarray(guessing, dtype=np.float64)
        probability = (
            guessing_values[None, :] + (1.0 - guessing_values[None, :]) * probability
        )
    probability = np.clip(probability, PROB_EPSILON, 1.0 - PROB_EPSILON)

    observed_accuracy = responses >= 0
    correct = responses == 1
    accuracy = np.where(
        observed_accuracy,
        np.where(correct, np.log(probability), np.log1p(-probability)),
        0.0,
    )

    observed_timing = ~np.isnan(log_rt)
    timing_values = np.where(observed_timing, log_rt, 0.0)
    alpha = time_discrimination[None, :]
    residual = timing_values - (time_intensity[None, :] - tau[:, None])
    timing = np.log(alpha) - 0.5 * _LOG_2PI - 0.5 * (alpha * residual) ** 2
    timing = np.where(observed_timing, timing, 0.0)
    return np.sum(accuracy + timing, axis=1)


def rt_joint_log_likelihood_samples(
    responses: NDArray[np.integer],
    log_rt: NDArray[np.float64],
    theta: NDArray[np.float64],
    tau: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    time_discrimination: NDArray[np.float64],
    time_intensity: NDArray[np.float64],
    guessing: NDArray[np.float64] | None = None,
    *,
    use_rust: bool = True,
    sample_chunk_size: int | None = None,
) -> NDArray[np.float64]:
    """Compute paired posterior log likelihoods for every sample and person."""
    if use_rust and rust_enabled():
        return np.asarray(
            mirt_rs.rt_joint_log_likelihood_samples(
                _as_i32(responses),
                _as_f64(log_rt),
                _as_f64(theta),
                _as_f64(tau),
                _as_f64(discrimination),
                _as_f64(difficulty),
                None if guessing is None else _as_f64(guessing),
                _as_f64(time_discrimination),
                _as_f64(time_intensity),
            ),
            dtype=np.float64,
        )

    responses = np.asarray(responses)
    log_rt = np.asarray(log_rt, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    discrimination = np.asarray(discrimination, dtype=np.float64)
    difficulty = np.asarray(difficulty, dtype=np.float64)
    time_discrimination = np.asarray(time_discrimination, dtype=np.float64)
    time_intensity = np.asarray(time_intensity, dtype=np.float64)

    n_samples, n_persons = theta.shape
    n_items = responses.shape[1]
    if sample_chunk_size is None:
        elements_per_sample = max(1, n_persons * n_items)
        sample_chunk_size = max(
            1,
            _SAMPLE_LIKELIHOOD_TARGET_ELEMENTS // elements_per_sample,
        )

    observed_accuracy = responses >= 0
    correct = responses == 1
    observed_timing = ~np.isnan(log_rt)
    timing_values = np.where(observed_timing, log_rt, 0.0)
    alpha = time_discrimination[None, None, :]
    log_alpha_normalizer = (np.log(time_discrimination) - 0.5 * _LOG_2PI)[None, None, :]
    result = np.empty((n_samples, n_persons), dtype=np.float64)

    for start in range(0, n_samples, sample_chunk_size):
        stop = min(start + sample_chunk_size, n_samples)
        logits = discrimination[None, None, :] * (
            theta[start:stop, :, None] - difficulty[None, None, :]
        )
        probability = np.asarray(sigmoid(logits), dtype=np.float64)
        if guessing is not None:
            guessing_values = np.asarray(guessing, dtype=np.float64)[None, None, :]
            probability *= 1.0 - guessing_values
            probability += guessing_values
        np.clip(probability, PROB_EPSILON, 1.0 - PROB_EPSILON, out=probability)

        log_probability = np.log(probability)
        np.subtract(1.0, probability, out=probability)
        np.log(probability, out=probability)
        accuracy = np.where(correct[None, :, :], log_probability, probability)
        accuracy *= observed_accuracy[None, :, :]

        residual = timing_values[None, :, :] - (
            time_intensity[None, None, :] - tau[start:stop, :, None]
        )
        residual *= alpha
        np.square(residual, out=residual)
        residual *= -0.5
        residual += log_alpha_normalizer
        residual *= observed_timing[None, :, :]
        accuracy += residual
        result[start:stop] = np.sum(accuracy, axis=2)

    return result


def rt_accept_person_proposals(
    responses: NDArray[np.integer],
    log_rt: NDArray[np.float64],
    theta: NDArray[np.float64],
    tau: NDArray[np.float64],
    theta_proposed: NDArray[np.float64],
    tau_proposed: NDArray[np.float64],
    log_uniform: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    time_discrimination: NDArray[np.float64],
    time_intensity: NDArray[np.float64],
    mean: NDArray[np.float64],
    covariance_inverse: NDArray[np.float64],
    log_determinant: float,
    guessing: NDArray[np.float64] | None = None,
    *,
    use_rust: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Accept or reject pre-generated joint ability and speed proposals."""
    if use_rust and rust_enabled():
        new_theta, new_tau, accepted = mirt_rs.rt_accept_person_proposals(
            _as_i32(responses),
            _as_f64(log_rt),
            _as_f64(theta),
            _as_f64(tau),
            _as_f64(theta_proposed),
            _as_f64(tau_proposed),
            _as_f64(log_uniform),
            _as_f64(discrimination),
            _as_f64(difficulty),
            _as_f64(time_discrimination),
            _as_f64(time_intensity),
            _as_f64(mean),
            _as_f64(covariance_inverse),
            float(log_determinant),
            None if guessing is None else _as_f64(guessing),
        )
        return (
            np.asarray(new_theta, dtype=np.float64),
            np.asarray(new_tau, dtype=np.float64),
            np.asarray(accepted, dtype=np.bool_),
        )

    current_likelihood = rt_joint_log_likelihood(
        responses,
        log_rt,
        theta,
        tau,
        discrimination,
        difficulty,
        time_discrimination,
        time_intensity,
        guessing,
        use_rust=False,
    )
    proposed_likelihood = rt_joint_log_likelihood(
        responses,
        log_rt,
        theta_proposed,
        tau_proposed,
        discrimination,
        difficulty,
        time_discrimination,
        time_intensity,
        guessing,
        use_rust=False,
    )

    current_centered = np.column_stack((theta, tau)) - mean
    proposed_centered = np.column_stack((theta_proposed, tau_proposed)) - mean
    current_prior = -0.5 * np.einsum(
        "ni,ij,nj->n",
        current_centered,
        covariance_inverse,
        current_centered,
        optimize=True,
    )
    proposed_prior = -0.5 * np.einsum(
        "ni,ij,nj->n",
        proposed_centered,
        covariance_inverse,
        proposed_centered,
        optimize=True,
    )
    # The shared normalizing constant, including ``log_determinant``, cancels
    # from the Metropolis-Hastings ratio. Keep the argument in the interface so
    # the compiled and NumPy signatures remain identical.
    del log_determinant

    log_acceptance = (proposed_likelihood + proposed_prior) - (
        current_likelihood + current_prior
    )
    accepted = np.asarray(log_uniform) < log_acceptance
    return (
        np.where(accepted, theta_proposed, theta),
        np.where(accepted, tau_proposed, tau),
        accepted,
    )
