"""Rust backend: dynamic.

Fallback mode: optional. Returns None when Rust is unavailable; callers own Python paths.

Every wrapper raises :class:`ValueError` for inconsistent lengths or skill
indices before calling the kernel, which would otherwise panic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt.backends.rust._helpers import (
    _call_native,
    _ensure_f64,
    _ensure_i32,
    mirt_rs,
    rust_enabled,
)

FALLBACK_MODE = "optional"


def _validate_bkt_inputs(
    responses: NDArray[np.int_],
    skill_assignments: NDArray[np.int_],
    skill_parameters: dict[str, NDArray[np.float64]],
    *,
    batch: bool = False,
    scaling: NDArray[np.float64] | None = None,
) -> None:
    """Check the shapes and skill indices that the BKT kernels index by.

    Raises
    ------
    ValueError
        If ``responses`` is not a sequence (a person-by-trial matrix when
        ``batch``), ``skill_assignments`` or ``scaling`` does not have one
        entry per trial, the skill parameters differ in length, or a skill
        index is outside ``[0, n_skills)``.
    """
    response_values = np.asarray(responses)
    if response_values.ndim != (2 if batch else 1):
        expected = "a persons-by-trials matrix" if batch else "one-dimensional"
        raise ValueError(f"responses must be {expected}")
    n_trials = response_values.shape[-1]
    skills = np.asarray(skill_assignments)
    if skills.shape != (n_trials,):
        raise ValueError(
            f"skill_assignments must have shape ({n_trials},), got {skills.shape}"
        )
    if scaling is not None and np.shape(scaling) != (n_trials,):
        raise ValueError(
            f"scaling must have shape ({n_trials},), got {np.shape(scaling)}"
        )
    shapes = {name: np.shape(values) for name, values in skill_parameters.items()}
    n_skills_shape = next(iter(shapes.values()))
    if len(n_skills_shape) != 1 or any(
        shape != n_skills_shape for shape in shapes.values()
    ):
        raise ValueError(
            "skill parameters must be one-dimensional with one entry per "
            f"skill, got shapes {shapes}"
        )
    if skills.size and (
        skills.dtype.kind not in "iu"
        or int(skills.min()) < 0
        or int(skills.max()) >= n_skills_shape[0]
    ):
        raise ValueError(
            "skill_assignments must be integer skill indices in "
            f"[0, {n_skills_shape[0]})"
        )


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
        _validate_bkt_inputs(
            responses,
            skill_assignments,
            {
                "p_init": p_init,
                "p_learn": p_learn,
                "p_forget": p_forget,
                "p_slip": p_slip,
                "p_guess": p_guess,
            },
        )
        return _call_native(
            mirt_rs.bkt_forward,
            _ensure_i32(responses),
            _ensure_i32(skill_assignments),
            _ensure_f64(p_init),
            _ensure_f64(p_learn),
            _ensure_f64(p_forget),
            _ensure_f64(p_slip),
            _ensure_f64(p_guess),
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
        _validate_bkt_inputs(
            responses,
            skill_assignments,
            {
                "p_learn": p_learn,
                "p_forget": p_forget,
                "p_slip": p_slip,
                "p_guess": p_guess,
            },
            scaling=scaling,
        )
        return _call_native(
            mirt_rs.bkt_backward,
            _ensure_i32(responses),
            _ensure_i32(skill_assignments),
            _ensure_f64(scaling),
            _ensure_f64(p_learn),
            _ensure_f64(p_forget),
            _ensure_f64(p_slip),
            _ensure_f64(p_guess),
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
        _validate_bkt_inputs(
            responses,
            skill_assignments,
            {
                "p_init": p_init,
                "p_learn": p_learn,
                "p_forget": p_forget,
                "p_slip": p_slip,
                "p_guess": p_guess,
            },
            batch=True,
        )
        return _call_native(
            mirt_rs.bkt_forward_backward_batch,
            _ensure_i32(responses),
            _ensure_i32(skill_assignments),
            _ensure_f64(p_init),
            _ensure_f64(p_learn),
            _ensure_f64(p_forget),
            _ensure_f64(p_slip),
            _ensure_f64(p_guess),
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
        _validate_bkt_inputs(
            responses,
            skill_assignments,
            {
                "p_init": p_init,
                "p_learn": p_learn,
                "p_forget": p_forget,
                "p_slip": p_slip,
                "p_guess": p_guess,
            },
        )
        return _call_native(
            mirt_rs.bkt_viterbi,
            _ensure_i32(responses),
            _ensure_i32(skill_assignments),
            _ensure_f64(p_init),
            _ensure_f64(p_learn),
            _ensure_f64(p_forget),
            _ensure_f64(p_slip),
            _ensure_f64(p_guess),
        )

    return None
