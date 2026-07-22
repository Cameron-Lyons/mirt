"""Rust backend: dynamic.

Fallback mode: optional. Returns None when Rust is unavailable; callers own Python paths.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt.backends.rust._helpers import (
    mirt_rs,
    rust_enabled,
)

FALLBACK_MODE = "optional"


def bkt_forward(
    responses: NDArray[np.int_],
    skill_assignments: NDArray[np.int_],
    p_init: NDArray[np.float64],
    p_learn: NDArray[np.float64],
    p_forget: NDArray[np.float64],
    p_slip: NDArray[np.float64],
    p_guess: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """BKT forward algorithm using Rust backend.

    Parameters
    ----------
    responses : NDArray
        Response sequence (n_trials,)
    skill_assignments : NDArray
        Skill index for each trial (n_trials,)
    p_init : NDArray
        Initial knowledge probability per skill (n_skills,)
    p_learn : NDArray
        Learning probability per skill (n_skills,)
    p_forget : NDArray
        Forgetting probability per skill (n_skills,)
    p_slip : NDArray
        Slip probability per skill (n_skills,)
    p_guess : NDArray
        Guess probability per skill (n_skills,)

    Returns
    -------
    tuple or None
        (alpha, scaling) or None if Rust unavailable
    """
    if rust_enabled():
        return mirt_rs.bkt_forward(
            responses.astype(np.int32),
            skill_assignments.astype(np.int32),
            p_init.astype(np.float64),
            p_learn.astype(np.float64),
            p_forget.astype(np.float64),
            p_slip.astype(np.float64),
            p_guess.astype(np.float64),
        )

    return None


def bkt_backward(
    responses: NDArray[np.int_],
    skill_assignments: NDArray[np.int_],
    scaling: NDArray[np.float64],
    p_learn: NDArray[np.float64],
    p_forget: NDArray[np.float64],
    p_slip: NDArray[np.float64],
    p_guess: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """BKT backward algorithm using Rust backend.

    Parameters
    ----------
    responses : NDArray
        Response sequence (n_trials,)
    skill_assignments : NDArray
        Skill index for each trial (n_trials,)
    scaling : NDArray
        Scaling factors from forward pass (n_trials,)
    p_learn : NDArray
        Learning probability per skill (n_skills,)
    p_forget : NDArray
        Forgetting probability per skill (n_skills,)
    p_slip : NDArray
        Slip probability per skill (n_skills,)
    p_guess : NDArray
        Guess probability per skill (n_skills,)

    Returns
    -------
    NDArray or None
        beta array or None if Rust unavailable
    """
    if rust_enabled():
        return mirt_rs.bkt_backward(
            responses.astype(np.int32),
            skill_assignments.astype(np.int32),
            scaling.astype(np.float64),
            p_learn.astype(np.float64),
            p_forget.astype(np.float64),
            p_slip.astype(np.float64),
            p_guess.astype(np.float64),
        )

    return None


def bkt_forward_backward_batch(
    responses: NDArray[np.int_],
    skill_assignments: NDArray[np.int_],
    p_init: NDArray[np.float64],
    p_learn: NDArray[np.float64],
    p_forget: NDArray[np.float64],
    p_slip: NDArray[np.float64],
    p_guess: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """BKT forward-backward for multiple persons in parallel.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_trials)
    skill_assignments : NDArray
        Skill index for each trial (n_trials,)
    p_init : NDArray
        Initial knowledge probability per skill (n_skills,)
    p_learn : NDArray
        Learning probability per skill (n_skills,)
    p_forget : NDArray
        Forgetting probability per skill (n_skills,)
    p_slip : NDArray
        Slip probability per skill (n_skills,)
    p_guess : NDArray
        Guess probability per skill (n_skills,)

    Returns
    -------
    tuple or None
        (gamma, log_likelihoods) or None if Rust unavailable
    """
    if rust_enabled():
        return mirt_rs.bkt_forward_backward_batch(
            responses.astype(np.int32),
            skill_assignments.astype(np.int32),
            p_init.astype(np.float64),
            p_learn.astype(np.float64),
            p_forget.astype(np.float64),
            p_slip.astype(np.float64),
            p_guess.astype(np.float64),
        )

    return None


def bkt_viterbi(
    responses: NDArray[np.int_],
    skill_assignments: NDArray[np.int_],
    p_init: NDArray[np.float64],
    p_learn: NDArray[np.float64],
    p_forget: NDArray[np.float64],
    p_slip: NDArray[np.float64],
    p_guess: NDArray[np.float64],
) -> NDArray[np.int_] | None:
    """BKT Viterbi algorithm for most likely state sequence.

    Parameters
    ----------
    responses : NDArray
        Response sequence (n_trials,)
    skill_assignments : NDArray
        Skill index for each trial (n_trials,)
    p_init : NDArray
        Initial knowledge probability per skill (n_skills,)
    p_learn : NDArray
        Learning probability per skill (n_skills,)
    p_forget : NDArray
        Forgetting probability per skill (n_skills,)
    p_slip : NDArray
        Slip probability per skill (n_skills,)
    p_guess : NDArray
        Guess probability per skill (n_skills,)

    Returns
    -------
    NDArray or None
        Most likely state sequence or None if Rust unavailable
    """
    if rust_enabled():
        return mirt_rs.bkt_viterbi(
            responses.astype(np.int32),
            skill_assignments.astype(np.int32),
            p_init.astype(np.float64),
            p_learn.astype(np.float64),
            p_forget.astype(np.float64),
            p_slip.astype(np.float64),
            p_guess.astype(np.float64),
        )

    return None


def bkt_ffbs(
    responses: NDArray[np.int_],
    skill_assignments: NDArray[np.int_],
    p_init: NDArray[np.float64],
    p_learn: NDArray[np.float64],
    p_forget: NDArray[np.float64],
    p_slip: NDArray[np.float64],
    p_guess: NDArray[np.float64],
    seed: int,
) -> NDArray[np.int_] | None:
    """Forward-filtering backward-sampling for BKT.

    Parameters
    ----------
    responses : NDArray
        Response sequence (n_trials,)
    skill_assignments : NDArray
        Skill index for each trial (n_trials,)
    p_init : NDArray
        Initial knowledge probability per skill (n_skills,)
    p_learn : NDArray
        Learning probability per skill (n_skills,)
    p_forget : NDArray
        Forgetting probability per skill (n_skills,)
    p_slip : NDArray
        Slip probability per skill (n_skills,)
    p_guess : NDArray
        Guess probability per skill (n_skills,)
    seed : int
        Random seed

    Returns
    -------
    NDArray or None
        Sampled state sequence or None if Rust unavailable
    """
    if rust_enabled():
        return mirt_rs.bkt_ffbs(
            responses.astype(np.int32),
            skill_assignments.astype(np.int32),
            p_init.astype(np.float64),
            p_learn.astype(np.float64),
            p_forget.astype(np.float64),
            p_slip.astype(np.float64),
            p_guess.astype(np.float64),
            int(seed),
        )

    return None


def bkt_ffbs_batch(
    responses: NDArray[np.int_],
    skill_assignments: NDArray[np.int_],
    p_init: NDArray[np.float64],
    p_learn: NDArray[np.float64],
    p_forget: NDArray[np.float64],
    p_slip: NDArray[np.float64],
    p_guess: NDArray[np.float64],
    seed: int,
) -> NDArray[np.int_] | None:
    """Batch FFBS for multiple persons.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_trials)
    skill_assignments : NDArray
        Skill index for each trial (n_trials,)
    p_init : NDArray
        Initial knowledge probability per skill (n_skills,)
    p_learn : NDArray
        Learning probability per skill (n_skills,)
    p_forget : NDArray
        Forgetting probability per skill (n_skills,)
    p_slip : NDArray
        Slip probability per skill (n_skills,)
    p_guess : NDArray
        Guess probability per skill (n_skills,)
    seed : int
        Random seed

    Returns
    -------
    NDArray or None
        Sampled state sequences or None if Rust unavailable
    """
    if rust_enabled():
        return mirt_rs.bkt_ffbs_batch(
            responses.astype(np.int32),
            skill_assignments.astype(np.int32),
            p_init.astype(np.float64),
            p_learn.astype(np.float64),
            p_forget.astype(np.float64),
            p_slip.astype(np.float64),
            p_guess.astype(np.float64),
            int(seed),
        )

    return None


def longitudinal_log_likelihood(
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> float | None:
    """Compute log-likelihood for longitudinal IRT data.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_obs, n_items)
    theta : NDArray
        Ability estimates (n_obs,)
    discrimination : NDArray
        Item discrimination parameters (n_items,)
    difficulty : NDArray
        Item difficulty parameters (n_items,)

    Returns
    -------
    float or None
        Log-likelihood or None if Rust unavailable
    """
    if rust_enabled():
        return mirt_rs.longitudinal_log_likelihood(
            responses.astype(np.int32),
            theta.astype(np.float64),
            discrimination.astype(np.float64),
            difficulty.astype(np.float64),
        )

    return None


def compute_growth_trajectory(
    growth_factors: NDArray[np.float64],
    time_values: NDArray[np.float64],
    growth_model: str,
) -> NDArray[np.float64] | None:
    """Compute growth curve predictions.

    Parameters
    ----------
    growth_factors : NDArray
        Growth factors (n_persons, n_growth_factors)
    time_values : NDArray
        Time values for each occasion (n_timepoints,)
    growth_model : str
        Growth model type ("linear" or "quadratic")

    Returns
    -------
    NDArray or None
        Trajectory predictions or None if Rust unavailable
    """
    if rust_enabled():
        return mirt_rs.compute_growth_trajectory(
            growth_factors.astype(np.float64),
            time_values.astype(np.float64),
            growth_model,
        )

    return None
