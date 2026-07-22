"""Rust backend: multigroup.

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


def multigroup_e_step_2pl(
    responses_list: list[NDArray[np.int_]],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    disc_list: list[NDArray[np.float64]],
    diff_list: list[NDArray[np.float64]],
    prior_means: NDArray[np.float64],
    prior_vars: NDArray[np.float64],
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]] | None:
    """Compute multigroup E-step for 2PL models using Rust backend.

    Processes all groups in parallel using Rayon.

    Parameters
    ----------
    responses_list : list of NDArray
        Response matrices, one per group (n_persons_g, n_items)
    quad_points : NDArray
        Quadrature points (n_quad,)
    quad_weights : NDArray
        Quadrature weights (n_quad,)
    disc_list : list of NDArray
        Discrimination arrays, one per group (n_items,)
    diff_list : list of NDArray
        Difficulty arrays, one per group (n_items,)
    prior_means : NDArray
        Prior means per group (n_groups,)
    prior_vars : NDArray
        Prior variances per group (n_groups,)

    Returns
    -------
    tuple or None
        (posterior_weights, group_log_likelihoods) or None if Rust unavailable
        - posterior_weights: list of (n_persons_g, n_quad) arrays
        - group_log_likelihoods: (n_groups,) array
    """
    if rust_enabled():
        responses_int = [r.astype(np.int32) for r in responses_list]
        disc_float = [d.astype(np.float64) for d in disc_list]
        diff_float = [d.astype(np.float64) for d in diff_list]

        return mirt_rs.multigroup_e_step_2pl(
            responses_int,
            quad_points.astype(np.float64),
            quad_weights.astype(np.float64),
            disc_float,
            diff_float,
            prior_means.astype(np.float64),
            prior_vars.astype(np.float64),
        )

    return None


def multigroup_e_step_3pl(
    responses_list: list[NDArray[np.int_]],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    disc_list: list[NDArray[np.float64]],
    diff_list: list[NDArray[np.float64]],
    guess_list: list[NDArray[np.float64]],
    prior_means: NDArray[np.float64],
    prior_vars: NDArray[np.float64],
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]] | None:
    """Compute multigroup E-step for 3PL models using Rust backend.

    Parameters
    ----------
    responses_list : list of NDArray
        Response matrices, one per group (n_persons_g, n_items)
    quad_points : NDArray
        Quadrature points (n_quad,)
    quad_weights : NDArray
        Quadrature weights (n_quad,)
    disc_list : list of NDArray
        Discrimination arrays, one per group (n_items,)
    diff_list : list of NDArray
        Difficulty arrays, one per group (n_items,)
    guess_list : list of NDArray
        Guessing arrays, one per group (n_items,)
    prior_means : NDArray
        Prior means per group (n_groups,)
    prior_vars : NDArray
        Prior variances per group (n_groups,)

    Returns
    -------
    tuple or None
        (posterior_weights, group_log_likelihoods) or None if Rust unavailable
    """
    if rust_enabled():
        responses_int = [r.astype(np.int32) for r in responses_list]
        disc_float = [d.astype(np.float64) for d in disc_list]
        diff_float = [d.astype(np.float64) for d in diff_list]
        guess_float = [g.astype(np.float64) for g in guess_list]

        return mirt_rs.multigroup_e_step_3pl(
            responses_int,
            quad_points.astype(np.float64),
            quad_weights.astype(np.float64),
            disc_float,
            diff_float,
            guess_float,
            prior_means.astype(np.float64),
            prior_vars.astype(np.float64),
        )

    return None


def multigroup_e_step_grm(
    responses_list: list[NDArray[np.int_]],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    disc_list: list[NDArray[np.float64]],
    thresh_list: list[NDArray[np.float64]],
    n_categories_list: list[NDArray[np.int_]],
    prior_means: NDArray[np.float64],
    prior_vars: NDArray[np.float64],
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]] | None:
    """Compute multigroup E-step for GRM models using Rust backend.

    Processes all groups in parallel using Rayon.

    Parameters
    ----------
    responses_list : list of NDArray
        Response matrices, one per group (n_persons_g, n_items)
    quad_points : NDArray
        Quadrature points (n_quad,)
    quad_weights : NDArray
        Quadrature weights (n_quad,)
    disc_list : list of NDArray
        Discrimination arrays, one per group (n_items,)
    thresh_list : list of NDArray
        Threshold matrices, one per group (n_items, max_categories-1)
    n_categories_list : list of NDArray
        Number of categories per item, one per group (n_items,)
    prior_means : NDArray
        Prior means per group (n_groups,)
    prior_vars : NDArray
        Prior variances per group (n_groups,)

    Returns
    -------
    tuple or None
        (posterior_weights, group_log_likelihoods) or None if Rust unavailable
    """
    if rust_enabled():
        responses_int = [r.astype(np.int32) for r in responses_list]
        disc_float = [d.astype(np.float64) for d in disc_list]
        thresh_float = [t.astype(np.float64) for t in thresh_list]
        n_cats_int = [n.astype(np.int32) for n in n_categories_list]

        return mirt_rs.multigroup_e_step_grm(
            responses_int,
            quad_points.astype(np.float64),
            quad_weights.astype(np.float64),
            disc_float,
            thresh_float,
            n_cats_int,
            prior_means.astype(np.float64),
            prior_vars.astype(np.float64),
        )

    return None


def multigroup_e_step_gpcm(
    responses_list: list[NDArray[np.int_]],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    disc_list: list[NDArray[np.float64]],
    steps_list: list[NDArray[np.float64]],
    n_categories_list: list[NDArray[np.int_]],
    prior_means: NDArray[np.float64],
    prior_vars: NDArray[np.float64],
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]] | None:
    """Compute multigroup E-step for GPCM models using Rust backend.

    Processes all groups in parallel using Rayon.

    Parameters
    ----------
    responses_list : list of NDArray
        Response matrices, one per group (n_persons_g, n_items)
    quad_points : NDArray
        Quadrature points (n_quad,)
    quad_weights : NDArray
        Quadrature weights (n_quad,)
    disc_list : list of NDArray
        Discrimination arrays, one per group (n_items,)
    steps_list : list of NDArray
        Step matrices, one per group (n_items, max_categories)
    n_categories_list : list of NDArray
        Number of categories per item, one per group (n_items,)
    prior_means : NDArray
        Prior means per group (n_groups,)
    prior_vars : NDArray
        Prior variances per group (n_groups,)

    Returns
    -------
    tuple or None
        (posterior_weights, group_log_likelihoods) or None if Rust unavailable
    """
    if rust_enabled():
        responses_int = [r.astype(np.int32) for r in responses_list]
        disc_float = [d.astype(np.float64) for d in disc_list]
        steps_float = [s.astype(np.float64) for s in steps_list]
        n_cats_int = [n.astype(np.int32) for n in n_categories_list]

        return mirt_rs.multigroup_e_step_gpcm(
            responses_int,
            quad_points.astype(np.float64),
            quad_weights.astype(np.float64),
            disc_float,
            steps_float,
            n_cats_int,
            prior_means.astype(np.float64),
            prior_vars.astype(np.float64),
        )

    return None


def multigroup_e_step_nrm(
    responses_list: list[NDArray[np.int_]],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    slopes_list: list[NDArray[np.float64]],
    intercepts_list: list[NDArray[np.float64]],
    n_categories_list: list[NDArray[np.int_]],
    prior_means: NDArray[np.float64],
    prior_vars: NDArray[np.float64],
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]] | None:
    """Compute multigroup E-step for NRM models using Rust backend.

    Processes all groups in parallel using Rayon.

    Parameters
    ----------
    responses_list : list of NDArray
        Response matrices, one per group (n_persons_g, n_items)
    quad_points : NDArray
        Quadrature points (n_quad,)
    quad_weights : NDArray
        Quadrature weights (n_quad,)
    slopes_list : list of NDArray
        Slope matrices, one per group (n_items, max_categories)
    intercepts_list : list of NDArray
        Intercept matrices, one per group (n_items, max_categories)
    n_categories_list : list of NDArray
        Number of categories per item, one per group (n_items,)
    prior_means : NDArray
        Prior means per group (n_groups,)
    prior_vars : NDArray
        Prior variances per group (n_groups,)

    Returns
    -------
    tuple or None
        (posterior_weights, group_log_likelihoods) or None if Rust unavailable
    """
    if rust_enabled():
        responses_int = [r.astype(np.int32) for r in responses_list]
        slopes_float = [s.astype(np.float64) for s in slopes_list]
        intercepts_float = [i.astype(np.float64) for i in intercepts_list]
        n_cats_int = [n.astype(np.int32) for n in n_categories_list]

        return mirt_rs.multigroup_e_step_nrm(
            responses_int,
            quad_points.astype(np.float64),
            quad_weights.astype(np.float64),
            slopes_float,
            intercepts_float,
            n_cats_int,
            prior_means.astype(np.float64),
            prior_vars.astype(np.float64),
        )

    return None


def multigroup_expected_counts(
    responses_list: list[NDArray[np.int_]],
    posterior_weights_list: list[NDArray[np.float64]],
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]] | None:
    """Compute expected counts for all groups in parallel.

    Parameters
    ----------
    responses_list : list of NDArray
        Response matrices, one per group (n_persons_g, n_items)
    posterior_weights_list : list of NDArray
        Posterior weights, one per group (n_persons_g, n_quad)

    Returns
    -------
    tuple or None
        (r_k_list, n_k_list) or None if Rust unavailable
        - r_k_list: list of (n_items, n_quad) expected correct counts
        - n_k_list: list of (n_items, n_quad) expected total counts
    """
    if rust_enabled():
        responses_int = [r.astype(np.int32) for r in responses_list]
        weights_float = [w.astype(np.float64) for w in posterior_weights_list]

        return mirt_rs.multigroup_expected_counts(responses_int, weights_float)

    return None
