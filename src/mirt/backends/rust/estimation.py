"""Rust backend: estimation.

Fallback mode: mixed. mhrm_fit_2pl / bootstrap_fit_2pl are required; em_iteration_3pl is optional.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mirt.backends.rust._helpers import (
    _call_native,
    _category_codes,
    _ensure_f64,
    mirt_rs,
    rust_enabled,
    rust_required,
)

FALLBACK_MODE = "mixed"


def _validate_positive_integer(value: int, message: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(message)
    return int(value)


def _validate_positive_float(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be positive")

    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _binary_codes(responses: NDArray[np.int_]) -> NDArray[np.int32]:
    """Return a 0/1 response matrix as int32 codes with missing entries -1.

    The 2PL kernels score every observed code other than 1 as incorrect in
    their likelihoods but use it as a count in their gradients.

    Raises
    ------
    ValueError
        If ``responses`` is not a matrix or has an observed code other than
        0 or 1.
    """
    try:
        codes = _category_codes(responses)
    except IndexError as error:
        raise ValueError("observed responses must contain only 0 or 1") from error
    if codes.ndim != 2:
        raise ValueError("responses must be a two-dimensional array")
    if np.any(codes > 1):
        raise ValueError("observed responses must contain only 0 or 1")
    return codes


def _validate_em_controls(
    n_quadpts: int,
    max_iter: int,
    tol: float,
) -> tuple[int, int, float]:
    return (
        _validate_positive_integer(n_quadpts, "n_quadpts must be positive"),
        _validate_positive_integer(max_iter, "max_iter must be at least 1"),
        _validate_positive_float(tol, "tol"),
    )


def _validate_chain_schedule(
    length: int,
    burnin: int,
    length_name: str,
) -> tuple[int, int]:
    length = _validate_positive_integer(length, f"{length_name} must be at least 1")
    if (
        isinstance(burnin, (bool, np.bool_))
        or not isinstance(burnin, (int, np.integer))
        or burnin < 0
    ):
        raise ValueError("burnin must be a non-negative integer")
    return length, int(burnin)


def mhrm_fit_2pl(
    responses: NDArray[np.int_],
    n_cycles: int = 2000,
    burnin: int = 500,
    proposal_sd: float = 0.5,
    seed: int | None = None,
    gain_sequence: str = "standard",
    *,
    disc_bounds: tuple[float, float] | None = None,
    diff_bounds: tuple[float, float] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Fit 2PL model using MHRM algorithm in Rust.

    The kernel runs the algorithm of ``MHRMEstimator`` (Cai, 2010): every
    cycle samples abilities and takes a Robbins-Monro step on each item,
    preconditioned by its running complete-data information. Burn-in cycles
    use unit gains; the estimates average the iterates after ``burnin`` (the
    final iterate when ``burnin >= n_cycles``).

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), missing coded as negative.
    n_cycles : int
        Number of MHRM cycles.
    burnin : int
        Number of initial unit-gain cycles, excluded from the parameter average.
    proposal_sd : float
        Standard deviation of the ability random-walk proposal.
    seed : int, optional
        Random seed.
    gain_sequence : {"standard", "adaptive"}
        Gain ``1 / (t + 1)`` or ``min(1, 10 / (t + 10))`` for the ``t``-th
        cycle after burn-in.
    disc_bounds, diff_bounds : tuple of float, optional
        Boxes of the slopes and difficulties; by default the EM boxes of a
        unidimensional 2PL, ``(0.1, 20)`` and ``(-30, 30)``.

    Returns
    -------
    tuple
        (discrimination, difficulty, log_likelihood), where the log-likelihood
        is evaluated at the final ability draw.
    """
    n_cycles, burnin = _validate_chain_schedule(n_cycles, burnin, "n_cycles")
    proposal_sd = _validate_positive_float(proposal_sd, "proposal_sd")
    if gain_sequence not in ("standard", "adaptive"):
        raise ValueError("gain_sequence must be 'standard' or 'adaptive'")
    default_disc, default_diff = _default_2pl_bounds()
    disc_box = _bound_pair(default_disc if disc_bounds is None else disc_bounds)
    diff_box = _bound_pair(default_diff if diff_bounds is None else diff_bounds)
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31)

    if rust_enabled():
        return mirt_rs.mhrm_fit_2pl(
            _binary_codes(responses),
            n_cycles,
            burnin,
            proposal_sd,
            int(seed),
            disc_box,
            diff_box,
            gain_sequence,
        )

    rust_required("mhrm_fit_2pl")


def _default_2pl_bounds() -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the EM boxes of a unidimensional 2PL slope and difficulty."""
    from mirt.estimation.base import _parameter_bounds
    from mirt.models.dichotomous import TwoParameterLogistic

    model = TwoParameterLogistic(1)
    return (
        _parameter_bounds(model, "discrimination"),
        _parameter_bounds(model, "difficulty"),
    )


def bootstrap_fit_2pl(
    responses: NDArray[np.int_],
    n_bootstrap: int = 100,
    n_quadpts: int = 21,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int | None = None,
    initial_discrimination: NDArray[np.float64] | None = None,
    initial_difficulty: NDArray[np.float64] | None = None,
    *,
    quad_points: NDArray[np.float64] | None = None,
    quad_weights: NDArray[np.float64] | None = None,
    disc_bounds: tuple[float, float] | None = None,
    diff_bounds: tuple[float, float] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run parallel bootstrap for 2PL model in Rust.

    Every replicate resamples persons with replacement and fits the 2PL by
    marginal maximum likelihood, stopping once the log-likelihood changes by
    less than ``tol``. Its M-steps are bounded Fisher-scoring ascents that end
    at the box-constrained optimum of each item. Replicate ``b`` draws from
    its own random stream, mixed from ``seed`` and ``b``.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), missing coded as negative.
    n_bootstrap : int
        Number of replicates.
    n_quadpts : int
        Gauss-Hermite points of the standard normal grid, used when
        ``quad_points`` is not given.
    max_iter : int
        Largest number of EM iterations per replicate.
    tol : float
        Log-likelihood change at which a replicate stops.
    seed : int, optional
        Random seed.
    initial_discrimination, initial_difficulty : NDArray, optional
        Starting values of every replicate, clamped into the boxes; by
        default replicates start from unit slopes and logit difficulties.
    quad_points, quad_weights : NDArray, optional
        Integration grid and weights against the latent density, such as the
        grid of the fit being bootstrapped.
    disc_bounds, diff_bounds : tuple of float, optional
        Boxes of the slopes and difficulties; by default the EM boxes of a
        unidimensional 2PL, ``(0.1, 20)`` and ``(-30, 30)``.

    Returns
    -------
    tuple
        (disc_samples, diff_samples) - arrays of shape (n_bootstrap, n_items)
    """
    n_bootstrap = _validate_positive_integer(
        n_bootstrap, "n_bootstrap must be at least 1"
    )
    n_quadpts, max_iter, tol = _validate_em_controls(n_quadpts, max_iter, tol)
    if (initial_discrimination is None) != (initial_difficulty is None):
        raise ValueError(
            "initial_discrimination and initial_difficulty must be provided together"
        )
    if (quad_points is None) != (quad_weights is None):
        raise ValueError("quad_points and quad_weights must be provided together")

    response_values = np.asarray(responses)
    if response_values.ndim != 2:
        raise ValueError("responses must be a two-dimensional array")
    n_items = response_values.shape[1]
    if initial_discrimination is not None and initial_difficulty is not None:
        initial_discrimination = np.asarray(initial_discrimination, dtype=np.float64)
        initial_difficulty = np.asarray(initial_difficulty, dtype=np.float64)
        if initial_discrimination.shape != (n_items,) or initial_difficulty.shape != (
            n_items,
        ):
            raise ValueError("initial parameters must contain one value per item")
        if not np.all(np.isfinite(initial_discrimination)) or not np.all(
            np.isfinite(initial_difficulty)
        ):
            raise ValueError("initial parameters must be finite")

    if quad_points is None or quad_weights is None:
        from mirt.estimation.quadrature import GaussHermiteQuadrature

        grid = GaussHermiteQuadrature(n_points=n_quadpts)
        quad_points, quad_weights = grid.nodes.ravel(), grid.weights
    points = np.asarray(quad_points, dtype=np.float64)
    weights = np.asarray(quad_weights, dtype=np.float64)
    if points.ndim != 1 or points.size == 0 or weights.shape != points.shape:
        raise ValueError(
            "quad_points and quad_weights must be non-empty vectors of equal length"
        )
    default_disc, default_diff = _default_2pl_bounds()
    disc_box = _bound_pair(default_disc if disc_bounds is None else disc_bounds)
    diff_box = _bound_pair(default_diff if diff_bounds is None else diff_bounds)

    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31)

    if rust_enabled():
        return mirt_rs.bootstrap_fit_2pl(
            _binary_codes(responses),
            n_bootstrap,
            _ensure_f64(points),
            _ensure_f64(weights),
            max_iter,
            tol,
            int(seed),
            disc_box,
            diff_box,
            _ensure_f64(initial_discrimination),
            _ensure_f64(initial_difficulty),
        )

    rust_required("bootstrap_fit_2pl")


def _bound_pair(pair: tuple[float, float]) -> tuple[float, float]:
    """Return a ``(lower, upper)`` box as floats, rejecting malformed pairs."""
    try:
        lower, upper = (float(value) for value in pair)
    except (TypeError, ValueError) as error:
        raise ValueError("bounds must be (lower, upper) pairs") from error
    return lower, upper


def _validate_3pl_iteration_inputs(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    item_parameters: dict[str, NDArray[np.float64]],
    prior_mean: float,
    prior_var: float,
    max_m_iter: int,
    bounds: dict[str, tuple[float, float]],
) -> tuple[float, float, int]:
    """Check what the native 3PL iteration asserts or indexes by.

    Returns the prior mean and variance as floats and ``max_m_iter`` as an
    ``int``.

    Raises
    ------
    ValueError
        If ``responses`` is not a matrix, the quadrature points and weights
        are empty or differ in length, an item parameter does not have one
        entry per item, the prior is not finite with a positive variance,
        ``max_m_iter`` is not a positive integer, or a bound pair is NaN or
        has its lower bound above its upper bound.
    """
    response_shape = np.shape(responses)
    if len(response_shape) != 2:
        raise ValueError("responses must be a two-dimensional array")
    n_items = response_shape[1]
    point_shape = np.shape(quad_points)
    if len(point_shape) != 1 or point_shape[0] < 1:
        raise ValueError("quad_points must be a non-empty one-dimensional array")
    if np.shape(quad_weights) != point_shape:
        raise ValueError(
            f"quad_weights must have shape {point_shape}, got {np.shape(quad_weights)}"
        )
    for name, values in item_parameters.items():
        if np.shape(values) != (n_items,):
            raise ValueError(
                f"{name} must have shape ({n_items},), got {np.shape(values)}"
            )
    try:
        mean, variance = float(prior_mean), float(prior_var)
    except (TypeError, ValueError) as error:
        raise ValueError("prior_mean and prior_var must be real numbers") from error
    if not np.isfinite(mean) or not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("prior_mean must be finite and prior_var finite and positive")
    iterations = _validate_positive_integer(
        max_m_iter, "max_m_iter must be a positive integer"
    )
    for name, pair in bounds.items():
        try:
            lower, upper = (float(value) for value in pair)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a (lower, upper) pair") from error
        # ``f64::clamp`` panics on NaN or inverted bounds; infinite ones work.
        if not lower <= upper:
            raise ValueError(f"{name} must satisfy lower <= upper")
    return mean, variance, iterations


def em_iteration_3pl(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    guessing: NDArray[np.float64],
    prior_mean: float = 0.0,
    prior_var: float = 1.0,
    max_m_iter: int = 10,
    m_tol: float = 1e-4,
    disc_bounds: tuple[float, float] = (0.1, 5.0),
    diff_bounds: tuple[float, float] = (-6.0, 6.0),
    guess_bounds: tuple[float, float] = (0.0, 0.35),
    damping_ab: float | None = None,
    damping_c: float | None = None,
    regularization: float = 0.01,
    regularization_c: float = 0.1,
    frequencies: NDArray[np.float64] | None = None,
    n_jobs: int = 1,
    pool: Any = None,
    priors: Mapping[str, Any] | None = None,
) -> (
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        float,
    ]
    | None
):
    """Single EM iteration for 3PL model (batched E+M step).

    Performs both E-step and M-step in a single call to reduce FFI overhead.
    Each item's M-step is a bounded Fisher-scoring ascent with backtracking:
    parameters on (or within 1e-3 of) a bound whose score points outside it
    move onto that bound while the others solve the reduced system, so the
    iteration reaches the bounded optimum of the expected log-likelihood. The
    ridge terms only stabilize the steps and do not change that optimum.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), missing coded as negative
    quad_points : NDArray
        Quadrature points (n_quad,)
    quad_weights : NDArray
        Quadrature weights (n_quad,)
    discrimination : NDArray
        Current discrimination parameters (n_items,)
    difficulty : NDArray
        Current difficulty parameters (n_items,)
    guessing : NDArray
        Current guessing parameters (n_items,)
    prior_mean : float
        Prior mean for theta distribution
    prior_var : float
        Prior variance for theta distribution
    max_m_iter : int
        Maximum Newton-Raphson iterations in M-step
    m_tol : float
        Convergence tolerance for M-step
    disc_bounds : tuple
        (min, max) bounds for discrimination
    diff_bounds : tuple
        (min, max) bounds for difficulty
    guess_bounds : tuple
        (min, max) bounds for guessing
    damping_ab, damping_c : float, optional
        Deprecated and ignored; the backtracking line search replaces the
        fixed step damping.
    regularization : float
        Ridge added to the a and b Fisher information in the Newton steps
    regularization_c : float
        Ridge added to the c Fisher information in the Newton steps
    frequencies : NDArray, optional
        Positive row weights, such as response-pattern counts
    n_jobs : int
        Worker count of ``pool``.
    pool : EMThreadPool, optional
        Pool of ``min(n_jobs, n_items)`` workers on which the item M-steps
        run, reused across iterations, such as
        ``EMFitContext.native_pool(n_jobs)``. Without it the M-steps, like the
        E-step, run on the native global pool. The results do not depend on
        the pool.
    priors : mapping of str to Prior, optional
        ``NormalPrior``, ``LogNormalPrior`` or ``BetaPrior`` objects by
        parameter name (``"discrimination"``, ``"difficulty"``,
        ``"guessing"``). Each M-step then ascends the expected log-posterior
        and ends at its box-constrained mode; the returned log-likelihood
        excludes the log-prior.

    Returns
    -------
    tuple or None
        (new_discrimination, new_difficulty, new_guessing, posterior_weights, log_likelihood)
        Returns None if Rust unavailable

    Raises
    ------
    ValueError
        If the arrays disagree in shape, a response is coded other than 0, 1
        or missing, the prior variance is not positive, ``max_m_iter`` is not
        a positive integer or a bound pair is inverted.
    """
    if damping_ab is not None or damping_c is not None:
        warnings.warn(
            "damping_ab and damping_c are deprecated and ignored; the 3PL M-step "
            "uses a backtracking line search",
            DeprecationWarning,
            stacklevel=2,
        )
    if rust_enabled():
        prior_mean, prior_var, max_m_iter = _validate_3pl_iteration_inputs(
            responses,
            quad_points,
            quad_weights,
            {
                "discrimination": discrimination,
                "difficulty": difficulty,
                "guessing": guessing,
            },
            prior_mean,
            prior_var,
            max_m_iter,
            {
                "disc_bounds": disc_bounds,
                "diff_bounds": diff_bounds,
                "guess_bounds": guess_bounds,
            },
        )
        return _call_native(
            mirt_rs.em_iteration_3pl,
            _binary_codes(responses),
            _ensure_f64(quad_points),
            _ensure_f64(quad_weights),
            _ensure_f64(discrimination),
            _ensure_f64(difficulty),
            _ensure_f64(guessing),
            prior_mean,
            prior_var,
            max_m_iter,
            m_tol,
            disc_bounds,
            diff_bounds,
            guess_bounds,
            regularization,
            regularization_c,
            _ensure_f64(frequencies),
            _validate_positive_integer(n_jobs, "n_jobs must be a positive integer"),
            pool,
            _prior_table_or_raise(priors),
        )

    return None


# Columns of a native prior table row: family code and two parameters.
_PRIOR_CODES = {"NormalPrior": 1.0, "LogNormalPrior": 2.0, "BetaPrior": 3.0}
_3PL_PARAMETERS = ("discrimination", "difficulty", "guessing")


def native_3pl_prior_table(
    priors: Mapping[str, Any] | None,
) -> NDArray[np.float64] | None:
    """Encode 3PL item priors as the native ``(3, 3)`` prior table.

    Rows hold ``(code, first, second)`` for discrimination, difficulty and
    guessing: code 0 for no prior, 1 for ``NormalPrior(mu, sigma)``, 2 for
    ``LogNormalPrior(mu, sigma)`` and 3 for ``BetaPrior(alpha, beta)``.
    Returns ``None`` when a prior has another type, including subclasses, or
    names another parameter, so that callers keep their NumPy path.
    """
    from mirt.estimation.priors import BetaPrior, LogNormalPrior, NormalPrior

    table = np.zeros((3, 3), dtype=np.float64)
    for name, prior in (priors or {}).items():
        if name not in _3PL_PARAMETERS:
            return None
        row = _3PL_PARAMETERS.index(name)
        if type(prior) in (NormalPrior, LogNormalPrior):
            table[row] = (_PRIOR_CODES[type(prior).__name__], prior._mu, prior._sigma)
        elif type(prior) is BetaPrior:
            table[row] = (_PRIOR_CODES["BetaPrior"], prior._alpha, prior._beta)
        else:
            return None
    return table


def _prior_table_or_raise(
    priors: Mapping[str, Any] | None,
) -> NDArray[np.float64] | None:
    """Return the native prior table of ``priors``, or None without priors."""
    if not priors:
        return None
    table = native_3pl_prior_table(priors)
    if table is None:
        raise ValueError(
            "priors must map discrimination, difficulty or guessing to "
            "NormalPrior, LogNormalPrior or BetaPrior objects"
        )
    return table
