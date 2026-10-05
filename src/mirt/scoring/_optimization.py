"""Configuration helpers and row-batched optimizers for ability scoring."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from math import sqrt

import numpy as np
from numpy.typing import NDArray

# Constants copied from scipy.optimize._optimize._minimize_scalar_bounded so
# every row follows exactly the sequence of trial points that SciPy would use.
_SQRT_EPS = sqrt(2.2e-16)
_GOLDEN_MEAN = 0.5 * (3.0 - sqrt(5.0))

RowObjective = Callable[[NDArray[np.intp], NDArray[np.float64]], NDArray[np.float64]]


def validate_theta_bounds(bounds: object) -> tuple[float, float]:
    """Return finite, increasing theta bounds as native floats."""
    if isinstance(bounds, (str, bytes)) or not isinstance(bounds, Iterable):
        raise ValueError("bounds must contain exactly two finite values")
    values: tuple[object, ...] = tuple(bounds)
    if len(values) != 2:
        raise ValueError("bounds must contain exactly two finite values")
    try:
        lower, upper = (float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("bounds must contain exactly two finite values") from exc
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("bounds must contain finite values with lower < upper")
    return lower, upper


def bounded_scalar_minimize(
    objective: RowObjective,
    n_rows: int,
    lower: float,
    upper: float,
    *,
    xatol: float = 1e-5,
    maxiter: int = 500,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Minimize independent scalar objectives with row-batched Brent search.

    This is a masked, row-vectorized port of SciPy's bounded Brent method
    (``minimize_scalar(method="bounded")``). Each row takes the same golden
    section and parabolic steps that SciPy would take for that row alone, so
    results match SciPy's per-row optimum for deterministic objectives.

    Parameters
    ----------
    objective : callable
        ``objective(rows, x)`` returns objective values for the requested
        ``rows`` evaluated at the matching entries of ``x``.
    n_rows : int
        Number of independent objectives.
    lower, upper : float
        Finite search interval shared by every row.
    xatol : float, default=1e-5
        Absolute tolerance on the minimizer.
    maxiter : int, default=500
        Maximum number of objective evaluations per row.

    Returns
    -------
    x : ndarray of shape (n_rows,)
        Minimizers.
    fun : ndarray of shape (n_rows,)
        Objective values at ``x``.
    """
    a = np.full(n_rows, lower, dtype=np.float64)
    b = np.full(n_rows, upper, dtype=np.float64)
    xf = a + _GOLDEN_MEAN * (b - a)
    if n_rows == 0:
        return xf, np.empty(0, dtype=np.float64)

    fx = np.array(
        objective(np.arange(n_rows, dtype=np.intp), xf.copy()),
        dtype=np.float64,
        copy=True,
    )
    nfc, fulc = xf.copy(), xf.copy()
    fnfc, ffulc = fx.copy(), fx.copy()
    rat = np.zeros(n_rows, dtype=np.float64)
    e = np.zeros(n_rows, dtype=np.float64)
    xm = 0.5 * (a + b)
    tol1 = _SQRT_EPS * np.abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1
    active = np.flatnonzero(np.abs(xf - xm) > (tol2 - 0.5 * (b - a)))
    n_evaluations = 1

    while active.size:
        a_i, b_i, xm_i = a[active], b[active], xm[active]
        xf_i, fx_i = xf[active], fx[active]
        nfc_i, fnfc_i = nfc[active], fnfc[active]
        fulc_i, ffulc_i = fulc[active], ffulc[active]
        e_i, rat_i = e[active], rat[active]
        tol1_i, tol2_i = tol1[active], tol2[active]

        # Parabolic interpolation through the three best points.
        parabolic = np.abs(e_i) > tol1_i
        r = (xf_i - nfc_i) * (fx_i - ffulc_i)
        q = (xf_i - fulc_i) * (fx_i - fnfc_i)
        p = (xf_i - fulc_i) * q - (xf_i - nfc_i) * r
        q = 2.0 * (q - r)
        p = np.where(q > 0.0, -p, p)
        q = np.abs(q)
        accept = (
            parabolic
            & (np.abs(p) < np.abs(0.5 * q * e_i))
            & (p > q * (a_i - xf_i))
            & (p < q * (b_i - xf_i))
        )
        e_i = np.where(parabolic, rat_i, e_i)
        parabolic_rat = np.divide(p + 0.0, q, out=np.zeros_like(p), where=accept)
        trial = xf_i + parabolic_rat
        near_bound = accept & (((trial - a_i) < tol2_i) | ((b_i - trial) < tol2_i))
        toward_middle = np.sign(xm_i - xf_i) + ((xm_i - xf_i) == 0)
        parabolic_rat = np.where(near_bound, tol1_i * toward_middle, parabolic_rat)

        # Golden-section step for rows without an acceptable parabola.
        golden = ~accept
        golden_e = np.where(xf_i >= xm_i, a_i - xf_i, b_i - xf_i)
        e_i = np.where(golden, golden_e, e_i)
        rat_i = np.where(golden, _GOLDEN_MEAN * golden_e, parabolic_rat)

        direction = np.sign(rat_i) + (rat_i == 0)
        x = xf_i + direction * np.maximum(np.abs(rat_i), tol1_i)
        fu = np.asarray(objective(active, x), dtype=np.float64)
        n_evaluations += 1

        improved = fu <= fx_i
        worse = ~improved
        above = x >= xf_i
        below = x < xf_i
        a_i = np.where(improved & above, xf_i, np.where(worse & below, x, a_i))
        b_i = np.where(improved & ~above, xf_i, np.where(worse & ~below, x, b_i))

        replace_second = worse & ((fu <= fnfc_i) | (nfc_i == xf_i))
        replace_third = (
            worse
            & ~replace_second
            & ((fu <= ffulc_i) | (fulc_i == xf_i) | (fulc_i == nfc_i))
        )
        shift = improved | replace_second
        fulc[active] = np.where(shift, nfc_i, np.where(replace_third, x, fulc_i))
        ffulc[active] = np.where(shift, fnfc_i, np.where(replace_third, fu, ffulc_i))
        nfc[active] = np.where(improved, xf_i, np.where(replace_second, x, nfc_i))
        fnfc[active] = np.where(improved, fx_i, np.where(replace_second, fu, fnfc_i))
        xf_i = np.where(improved, x, xf_i)
        xf[active] = xf_i
        fx[active] = np.where(improved, fu, fx_i)
        a[active], b[active] = a_i, b_i
        e[active], rat[active] = e_i, rat_i

        xm_i = 0.5 * (a_i + b_i)
        tol1_i = _SQRT_EPS * np.abs(xf_i) + xatol / 3.0
        tol2_i = 2.0 * tol1_i
        xm[active], tol1[active], tol2[active] = xm_i, tol1_i, tol2_i
        active = active[np.abs(xf_i - xm_i) > (tol2_i - 0.5 * (b_i - a_i))]
        if n_evaluations >= maxiter:
            break

    return xf, fx


def finite_difference_derivatives(
    objective: RowObjective,
    rows: NDArray[np.intp],
    x: NDArray[np.float64],
    *,
    h: float,
    center: NDArray[np.float64] | None = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
]:
    """Row-wise central-difference gradient and Hessian.

    The stencil, step sizes ``h * max(1, |x|)`` and arithmetic order match
    :func:`mirt.utils.numeric.compute_hessian_se`, so one row reproduces its
    Hessian. ``objective(rows, x)`` evaluates each row at its own point.

    Returns
    -------
    center, gradient, hessian, finite
        Objective values at ``x``, gradients ``(n, d)``, Hessians
        ``(n, d, d)``, and whether every stencil value of a row was finite.
    """
    n_rows, n_dims = x.shape
    steps = h * np.maximum(1.0, np.abs(x))
    f_center = objective(rows, x) if center is None else center
    finite = np.isfinite(f_center)
    gradient = np.empty((n_rows, n_dims), dtype=np.float64)
    hessian = np.empty((n_rows, n_dims, n_dims), dtype=np.float64)

    def shifted(*offsets: tuple[int, float]) -> NDArray[np.float64]:
        point = x.copy()
        for dim, sign in offsets:
            point[:, dim] += sign * steps[:, dim]
        values = np.asarray(objective(rows, point), dtype=np.float64)
        finite[:] &= np.isfinite(values)
        return values

    # Rows with non-finite stencil values are reported through ``finite``.
    with np.errstate(invalid="ignore", over="ignore"):
        for row in range(n_dims):
            f_plus = shifted((row, 1.0))
            f_minus = shifted((row, -1.0))
            gradient[:, row] = (f_plus - f_minus) / (2.0 * steps[:, row])
            hessian[:, row, row] = (f_plus - 2.0 * f_center + f_minus) / (
                steps[:, row] ** 2
            )
            for column in range(row + 1, n_dims):
                cross = (
                    shifted((row, 1.0), (column, 1.0))
                    - shifted((row, 1.0), (column, -1.0))
                    - shifted((row, -1.0), (column, 1.0))
                    + shifted((row, -1.0), (column, -1.0))
                ) / (4.0 * steps[:, row] * steps[:, column])
                hessian[:, row, column] = cross
                hessian[:, column, row] = cross

    return f_center, gradient, hessian, finite


def batched_hessian_se(
    objective: RowObjective,
    x: NDArray[np.float64],
    *,
    h: float = 1e-5,
) -> NDArray[np.float64]:
    """Apply :func:`mirt.utils.numeric.compute_hessian_se` to every row.

    Rows whose finite-difference Hessian is numerically singular, with an
    eigenvalue at or below ``sqrt(eps)`` times its largest magnitude (at least
    one), receive NaN standard errors, as in the single-row helper.
    """
    n_rows, n_dims = x.shape
    _, _, hessian, finite = finite_difference_derivatives(
        objective, np.arange(n_rows, dtype=np.intp), x, h=h
    )
    if not np.all(finite):
        raise ValueError("func must return finite scalar values near x")

    standard_error = np.full((n_rows, n_dims), np.nan, dtype=np.float64)
    if n_rows == 0:
        return standard_error
    eigenvalues = np.linalg.eigvalsh(hessian)
    scale = np.maximum(np.max(np.abs(eigenvalues), axis=1), 1.0)
    tolerance = np.sqrt(np.finfo(np.float64).eps) * scale
    regular = np.flatnonzero(np.all(eigenvalues > tolerance[:, None], axis=1))
    for row in regular:
        try:
            covariance = np.linalg.inv(hessian[row])
        except np.linalg.LinAlgError:
            continue
        variances = np.diag(covariance)
        if np.all(np.isfinite(variances)) and np.all(variances > 0.0):
            standard_error[row] = np.sqrt(variances)
    return standard_error


def batched_newton_minimize(
    objective: RowObjective,
    x0: NDArray[np.float64],
    lower: float,
    upper: float,
    *,
    max_iter: int = 100,
    xtol: float = 1e-9,
    h: float = 1e-4,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Minimize independent smooth objectives in a shared box, row-batched.

    Each row takes projected Newton steps with finite-difference derivatives:
    coordinates on a bound whose gradient points outward are held fixed, the
    remaining Hessian block is shifted to be positive definite when needed,
    and a backtracking Armijo search keeps every accepted step descending.

    Parameters
    ----------
    objective : callable
        ``objective(rows, x)`` returns values for the requested rows at the
        matching rows of ``x`` with shape ``(len(rows), n_dims)``.
    x0 : ndarray of shape (n_rows, n_dims)
        Starting points; they are projected into ``[lower, upper]``.
    lower, upper : float
        Bounds shared by every coordinate.
    max_iter : int, default=100
        Maximum Newton iterations per row.
    xtol : float, default=1e-9
        A row converges once its Newton step is no longer than this.
    h : float, default=1e-4
        Relative finite-difference step for derivatives.

    Returns
    -------
    x : ndarray of shape (n_rows, n_dims)
        Final points.
    fun : ndarray of shape (n_rows,)
        Objective values at ``x``.
    converged : ndarray of shape (n_rows,)
        Whether a row met the convergence test. Callers should re-solve the
        remaining rows with a more conservative method.
    """
    x = np.clip(np.array(x0, dtype=np.float64, copy=True), lower, upper)
    n_rows, n_dims = x.shape
    fun = np.asarray(
        objective(np.arange(n_rows, dtype=np.intp), x), dtype=np.float64
    ).copy()
    converged = np.zeros(n_rows, dtype=np.bool_)
    active = np.flatnonzero(np.isfinite(fun))
    identity = np.eye(n_dims)

    for _ in range(max_iter):
        if active.size == 0:
            break
        x_active, f_active = x[active], fun[active]
        _, gradient, hessian, finite = finite_difference_derivatives(
            objective, active, x_active, h=h, center=f_active
        )
        fixed = ((x_active <= lower) & (gradient > 0.0)) | (
            (x_active >= upper) & (gradient < 0.0)
        )
        free = ~fixed
        reduced = np.where(free[:, :, None] & free[:, None, :], hessian, identity)
        reduced_gradient = np.where(free, gradient, 0.0)
        # Shift indefinite or nearly singular blocks so the step descends.
        finite &= np.all(np.isfinite(reduced), axis=(1, 2))
        reduced[~finite] = identity
        eigenvalues = np.linalg.eigvalsh(reduced)
        floor = 1e-8 * np.maximum(np.max(np.abs(eigenvalues), axis=1), 1.0)
        shift = np.maximum(floor - eigenvalues[:, 0], 0.0)
        reduced += shift[:, None, None] * identity
        direction = -np.linalg.solve(reduced, reduced_gradient[..., None])[..., 0]
        step_norm = np.max(np.abs(direction), axis=1)

        done = finite & (step_norm <= xtol)
        converged[active[done]] = True
        searching = np.flatnonzero(finite & ~done)

        step_size = np.ones(searching.size, dtype=np.float64)
        pending = np.arange(searching.size)
        for _ in range(40):
            if pending.size == 0:
                break
            local = searching[pending]
            trial = np.clip(
                x_active[local] + step_size[pending, None] * direction[local],
                lower,
                upper,
            )
            f_trial = np.asarray(objective(active[local], trial), dtype=np.float64)
            # Projection can bend a descent step; never accept an increase.
            predicted = np.minimum(
                np.sum(gradient[local] * (trial - x_active[local]), axis=1), 0.0
            )
            accept = np.isfinite(f_trial) & (
                f_trial <= f_active[local] + 1e-4 * predicted
            )
            moved = np.max(np.abs(trial - x_active[local]), axis=1)
            accepted_rows = active[local[accept]]
            x[accepted_rows] = trial[accept]
            fun[accepted_rows] = f_trial[accept]
            converged[accepted_rows[moved[accept] <= xtol]] = True
            pending = pending[~accept]
            step_size[pending] *= 0.5

        # Without an acceptable step, a row has stalled at the noise floor of
        # its finite-difference derivatives only if the Newton step was tiny.
        stalled = searching[pending]
        converged[active[stalled[step_norm[stalled] <= 1e-6]]] = True
        keep = finite & ~done
        keep[stalled] = False
        active = active[keep & ~converged[active]]

    return x, fun, converged
