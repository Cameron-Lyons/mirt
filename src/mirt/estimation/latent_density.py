"""Latent density specifications for IRT models.

This module provides various latent trait distribution options:
- Gaussian (standard normal, default)
- Empirical histogram (nonparametric)
- Davidian curves (semi-parametric)
- Custom density functions

With free item parameters the location and scale of an estimated density
are not identified. Woods' histogram, Davidian curves and Gaussian mixtures
are therefore held at mean 0 and variance 1 in every EM cycle, so the item
parameters stay on the scale of a standard-normal fit.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from math import factorial
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import logsumexp, roots_hermitenorm

from mirt._prior_mass import log_density_quadrature_mass, normalize_log_mass
from mirt.constants import PROB_EPSILON, REGULARIZATION_EPSILON

if TYPE_CHECKING:
    pass


def _validate_count(value: Any, *, name: str, allow_zero: bool = False) -> int:
    """Return a validated integer count."""
    minimum = 0 if allow_zero else 1
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < minimum
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return int(value)


def _point_count(theta: NDArray[np.float64], *, name: str = "theta") -> int:
    """Return the number of points in a one- or two-dimensional array."""
    points = np.asarray(theta, dtype=np.float64)
    if points.ndim == 0:
        return 1
    if points.ndim not in (1, 2):
        raise ValueError(f"{name} must be a one- or two-dimensional array")
    return int(points.shape[0])


def _as_univariate_points(
    theta: NDArray[np.float64],
    *,
    name: str = "theta",
) -> NDArray[np.float64]:
    """Return univariate points as an owned one-dimensional float array."""
    points = np.asarray(theta, dtype=np.float64)
    if points.ndim == 0:
        points = points.reshape(1)
    elif points.ndim == 2 and points.shape[1] == 1:
        points = points[:, 0]
    elif points.ndim != 1:
        raise ValueError(f"{name} must contain univariate points")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite values")
    return points.copy()


def _as_multivariate_points(
    theta: NDArray[np.float64],
    *,
    n_dimensions: int,
    name: str = "theta",
) -> NDArray[np.float64]:
    """Return points as an owned ``(n_points, n_dimensions)`` array."""
    points = np.asarray(theta, dtype=np.float64)
    if points.ndim == 0:
        if n_dimensions != 1:
            raise ValueError(f"{name} must have {n_dimensions} columns")
        points = points.reshape(1, 1)
    elif points.ndim == 1:
        if n_dimensions == 1:
            points = points.reshape(-1, 1)
        elif points.size == n_dimensions:
            points = points.reshape(1, -1)
        else:
            raise ValueError(f"{name} must have {n_dimensions} columns")
    elif points.ndim != 2 or points.shape[1] != n_dimensions:
        raise ValueError(f"{name} must have shape (n_points, {n_dimensions})")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite values")
    return points.copy()


def _normalize_weights(
    weights: NDArray[np.float64],
    *,
    n_points: int,
    name: str = "weights",
) -> NDArray[np.float64]:
    """Validate and normalize non-negative point weights."""
    normalized = np.asarray(weights, dtype=np.float64)
    if normalized.ndim != 1 or len(normalized) != n_points:
        raise ValueError(f"{name} must have shape ({n_points},)")
    if not np.all(np.isfinite(normalized)) or np.any(normalized < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    scale = float(np.max(normalized, initial=0.0))
    if scale <= 0.0:
        raise ValueError(f"{name} must have positive total mass")
    scaled = normalized / scale
    return scaled / scaled.sum()


def _normalize_probabilities(
    probabilities: NDArray[np.float64],
    *,
    name: str,
    expected_length: int | None = None,
) -> NDArray[np.float64]:
    """Validate and normalize a one-dimensional probability vector."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if expected_length is not None and len(values) != expected_length:
        raise ValueError(f"{name} must have length {expected_length}")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    scale = float(np.max(values, initial=0.0))
    if scale <= 0.0:
        raise ValueError(f"{name} must have positive total mass")
    scaled = values.copy() / scale
    return scaled / scaled.sum()


# Densities whose mean and variance are this close to 0 and 1 count as
# standardized. Normal masses on the rectangular grid of [-6, 6] miss about
# 4e-8 of the unit variance in the truncated tails.
_STANDARD_MOMENT_TOL = 1e-6
# Floor for histogram heights before taking logarithms.
_TINY = 1e-300


def _is_standard(mean: float, variance: float) -> bool:
    """Return whether moments are those of a standardized density."""
    return (
        abs(mean) <= _STANDARD_MOMENT_TOL
        and abs(variance - 1.0) <= _STANDARD_MOMENT_TOL
    )


def _grid_moments(
    nodes: NDArray[np.float64], masses: NDArray[np.float64]
) -> tuple[float, float]:
    """Return the mean and variance of discrete masses on ``nodes``."""
    mean = float(masses @ nodes)
    return mean, float(masses @ (nodes - mean) ** 2)


def _woods_standardize(
    nodes: NDArray[np.float64], masses: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Standardize a histogram by Woods' (2007) interpolation-extrapolation.

    The masses on the sorted ``nodes`` are read as density heights over
    cells reaching halfway to the neighbouring nodes (the end cells are as
    wide as the adjacent spacing). Moving every height to its standardized
    node ``(node - mean) / sd`` standardizes the density. The moved heights
    are interpolated linearly at the original nodes and, beyond the outermost
    moved nodes, extrapolated with the log-linear decay of the two outermost
    heights, never increasing outward. On an equally spaced grid this is the
    ``standardizeQuadrature`` step of R's ``mirt``, with the interpolation
    fractions taken over the rescaled spacing.
    """
    mean, variance = _grid_moments(nodes, masses)
    if not variance > 0.0:
        return masses.copy()
    widths = np.empty_like(nodes)
    widths[1:-1] = (nodes[2:] - nodes[:-2]) / 2.0
    widths[0] = nodes[1] - nodes[0]
    widths[-1] = nodes[-1] - nodes[-2]
    heights = masses / widths
    log_heights = np.log(np.maximum(heights, _TINY))
    moved = (nodes - mean) / np.sqrt(variance)

    standardized = np.interp(nodes, moved, heights)
    for end, inner, outside in (
        (0, 1, nodes < moved[0]),
        (-1, -2, nodes > moved[-1]),
    ):
        decay = max(log_heights[inner] - log_heights[end], 0.0) / abs(
            moved[inner] - moved[end]
        )
        distance = np.abs(nodes[outside] - moved[end])
        standardized[outside] = np.exp(log_heights[end] - decay * distance)
    standardized *= widths
    return standardized / standardized.sum()


def _tilt_to_standard(
    nodes: NDArray[np.float64],
    masses: NDArray[np.float64],
    *,
    max_iter: int = 100,
) -> NDArray[np.float64] | None:
    """Return the masses closest to ``masses`` with mean 0 and variance 1.

    Minimizes the Kullback-Leibler divergence from ``masses`` under the two
    moment constraints. The solution tilts the masses exponentially,
    ``p ~ masses * exp(g @ lam)`` with the moment functions
    ``g = (x, x**2 - 1)``, and damped Newton steps find ``lam`` by minimizing
    the convex dual ``log(sum(masses * exp(g @ lam)))``. Returns ``None`` when
    no distribution on the support of ``masses`` has these moments.
    """
    support = masses > 0.0
    moments = np.column_stack([nodes[support], nodes[support] ** 2 - 1.0])
    log_base = np.log(masses[support])
    multipliers = np.zeros(2)
    dual = float(logsumexp(log_base))
    log_tilted = log_base - dual
    for _ in range(max_iter):
        tilted = np.exp(log_tilted)
        residual = tilted @ moments
        if np.max(np.abs(residual)) < 1e-12:
            fitted = np.zeros_like(masses)
            fitted[support] = tilted / tilted.sum()
            return fitted
        centered = moments - residual
        hessian = (centered * tilted[:, None]).T @ centered
        try:
            step = np.linalg.solve(hessian, residual)
        except np.linalg.LinAlgError:
            return None
        length = 1.0
        while length > 1e-12:
            trial = log_base + moments @ (multipliers - length * step)
            trial_dual = float(logsumexp(trial))
            if trial_dual <= dual + 1e-14 * max(1.0, abs(dual)):
                break
            length *= 0.5
        else:
            return None
        multipliers = multipliers - length * step
        dual = trial_dual
        log_tilted = trial - trial_dual
    return None


def _best_on_segment(
    counts: NDArray[np.float64],
    start: NDArray[np.float64],
    end: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the masses on the segment from ``start`` to ``end`` that
    maximize ``sum(counts * log(masses))``.

    The objective is concave along the segment, so bisection on its slope
    finds the maximizer. The result fits at least as well as ``start``.
    """
    observed = counts > 0.0
    weight, low_end, high_end = counts[observed], start[observed], end[observed]
    change = high_end - low_end

    def slope(step: float) -> float:
        mixed = np.maximum(low_end + step * change, _TINY)
        return float(weight @ (change / mixed))

    if slope(1.0) >= 0.0:
        return end.copy()
    if slope(0.0) <= 0.0:
        return start.copy()
    low, high = 0.0, 1.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        if slope(middle) > 0.0:
            low = middle
        else:
            high = middle
    return start + low * (end - start)


def _solve_quadratic_constraints(
    coefficients: NDArray[np.float64],
    matrices: list[NDArray[np.float64]],
    targets: list[float],
    *,
    max_iter: int = 20,
    tol: float = 1e-13,
) -> NDArray[np.float64] | None:
    """Move ``coefficients`` onto ``c @ M @ c = target`` for every pair.

    Gauss-Newton steps take the minimum-norm correction of the linearized
    constraints. Returns ``None`` when the constraints are not met within
    ``tol`` after ``max_iter`` steps.
    """
    current = np.asarray(coefficients, dtype=np.float64)
    if not np.all(np.isfinite(current)):
        return None
    for iteration in range(max_iter + 1):
        residual = np.array(
            [
                float(current @ matrix @ current) - target
                for matrix, target in zip(matrices, targets, strict=True)
            ]
        )
        if np.max(np.abs(residual)) <= tol:
            return current.copy()
        if iteration == max_iter:
            break
        jacobian = np.array([2.0 * (matrix @ current) for matrix in matrices])
        try:
            correction = np.linalg.solve(jacobian @ jacobian.T, residual)
        except np.linalg.LinAlgError:
            return None
        current = current - jacobian.T @ correction
        if not np.all(np.isfinite(current)):
            return None
    return None


class LatentDensity(ABC):
    """Abstract base class for latent density specifications."""

    @abstractmethod
    def log_density(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute log density at theta points.

        Parameters
        ----------
        theta : ndarray
            Theta values, shape (n_points,) or (n_points, n_dims)

        Returns
        -------
        ndarray
            Log density values, shape (n_points,)
        """
        pass

    def density(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute density at theta points."""
        return np.exp(self.log_density(theta))

    def log_quadrature_mass(
        self,
        theta: NDArray[np.float64],
        quadrature_weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return normalized prior masses on standard-normal GH nodes.

        ``GaussHermiteQuadrature`` weights already integrate against a
        standard-normal reference measure. A target density therefore enters
        through its density ratio to that reference, not as an additional
        density factor.
        """
        points = np.asarray(theta, dtype=np.float64)
        if points.ndim == 1:
            points = points[:, None]
        return log_density_quadrature_mass(
            points,
            quadrature_weights,
            self.log_density(points),
        )

    @abstractmethod
    def update(
        self,
        theta_points: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        """Update density parameters based on posterior weights.

        Parameters
        ----------
        theta_points : ndarray
            Quadrature points, shape (n_quad, n_dims)
        weights : ndarray
            Posterior weights summed across persons, shape (n_quad,)
        """
        pass

    @property
    @abstractmethod
    def n_parameters(self) -> int:
        """Number of estimated parameters in the density."""
        pass


class GaussianDensity(LatentDensity):
    """Multivariate Gaussian latent density.

    This is the standard assumption in most IRT models.

    Parameters
    ----------
    mean : ndarray, optional
        Mean vector. Default is zeros.
    cov : ndarray, optional
        Covariance matrix. Default is identity.
    estimate_mean : bool
        Whether to estimate mean during EM. Default False.
    estimate_cov : bool
        Whether to estimate covariance during EM. Default False.
    """

    def __init__(
        self,
        mean: NDArray[np.float64] | None = None,
        cov: NDArray[np.float64] | None = None,
        estimate_mean: bool = False,
        estimate_cov: bool = False,
        n_dimensions: int | None = None,
    ) -> None:
        mean_array: NDArray[np.float64] | None = None
        if mean is not None:
            mean_array = np.asarray(mean, dtype=np.float64)
            if mean_array.ndim != 1 or len(mean_array) == 0:
                raise ValueError("mean must be a non-empty one-dimensional array")
            if not np.all(np.isfinite(mean_array)):
                raise ValueError("mean must contain only finite values")

        cov_array: NDArray[np.float64] | None = None
        if cov is not None:
            cov_array = np.asarray(cov, dtype=np.float64)
            if (
                cov_array.ndim != 2
                or cov_array.shape[0] == 0
                or cov_array.shape[0] != cov_array.shape[1]
            ):
                raise ValueError("cov must be a non-empty square matrix")

        inferred_dimensions = (
            len(mean_array)
            if mean_array is not None
            else cov_array.shape[0]
            if cov_array is not None
            else 1
        )
        if n_dimensions is None:
            self.n_dimensions = int(inferred_dimensions)
        else:
            self.n_dimensions = _validate_count(
                n_dimensions,
                name="n_dimensions",
            )
            if self.n_dimensions != inferred_dimensions and (
                mean_array is not None or cov_array is not None
            ):
                raise ValueError(
                    "n_dimensions must match the supplied mean and covariance"
                )

        if mean_array is not None and len(mean_array) != self.n_dimensions:
            raise ValueError("mean length must match the covariance dimensions")
        if cov_array is not None and cov_array.shape != (
            self.n_dimensions,
            self.n_dimensions,
        ):
            raise ValueError("cov shape must match the mean dimensions")

        self.mean = (
            np.zeros(self.n_dimensions, dtype=np.float64)
            if mean_array is None
            else mean_array.copy()
        )
        self.cov = (
            np.eye(self.n_dimensions, dtype=np.float64)
            if cov_array is None
            else cov_array.copy()
        )

        if not isinstance(estimate_mean, (bool, np.bool_)):
            raise TypeError("estimate_mean must be a boolean")
        if not isinstance(estimate_cov, (bool, np.bool_)):
            raise TypeError("estimate_cov must be a boolean")
        self.estimate_mean = bool(estimate_mean)
        self.estimate_cov = bool(estimate_cov)

        self._update_precision()

    def _update_precision(self) -> None:
        """Update precision matrix and normalizing constant."""
        if not np.all(np.isfinite(self.cov)):
            raise ValueError("cov must contain only finite values")
        if not np.allclose(self.cov, self.cov.T, rtol=1e-10, atol=1e-12):
            raise ValueError("cov must be symmetric")
        try:
            np.linalg.cholesky(self.cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError("cov must be positive definite") from exc
        self._precision = np.linalg.solve(self.cov, np.eye(self.n_dimensions))
        self._log_det = float(np.linalg.slogdet(self.cov)[1])
        self._log_norm = -0.5 * (self.n_dimensions * np.log(2 * np.pi) + self._log_det)

    def log_density(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        points = _as_multivariate_points(theta, n_dimensions=self.n_dimensions)
        points -= self.mean
        mahal = np.einsum("ij,ij->i", points @ self._precision, points)
        mahal *= -0.5
        mahal += self._log_norm
        return mahal

    def update(
        self,
        theta_points: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        points = _as_multivariate_points(
            theta_points,
            n_dimensions=self.n_dimensions,
            name="theta_points",
        )
        normalized_weights = _normalize_weights(weights, n_points=len(points))

        if not self.estimate_mean and not self.estimate_cov:
            return

        if self.estimate_mean:
            self.mean = normalized_weights @ points

        if self.estimate_cov:
            # Validation owns this scratch array; center it without copying inputs.
            points -= self.mean
            self.cov = points.T @ (normalized_weights[:, None] * points)
            self.cov = (self.cov + self.cov.T) / 2
            self.cov += REGULARIZATION_EPSILON * np.eye(self.n_dimensions)

        self._update_precision()

    @property
    def n_parameters(self) -> int:
        n = 0
        if self.estimate_mean:
            n += self.n_dimensions
        if self.estimate_cov:
            n += self.n_dimensions * (self.n_dimensions + 1) // 2
        return n


def _gaussian_deviance(
    cov: NDArray[np.float64], second_moment: NDArray[np.float64]
) -> float:
    """Return ``log|cov| + tr(cov^-1 S)``, or infinity when ``cov`` is not PD."""
    try:
        cholesky = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        return np.inf
    log_det = 2.0 * float(np.sum(np.log(np.diag(cholesky))))
    return log_det + float(np.trace(np.linalg.solve(cov, second_moment)))


def _is_block_pattern(free: NDArray[np.bool_]) -> bool:
    """Return whether ``free`` links factors into blocks with every pair free."""
    linked = free | np.eye(len(free), dtype=np.bool_)
    closure = linked.astype(np.int64)
    for _ in range(max(1, len(free).bit_length())):
        closure = np.minimum(closure @ closure, 1)
    return bool(np.array_equal(closure.astype(np.bool_), linked))


def _constrained_gaussian_covariance(
    second_moment: NDArray[np.float64],
    start: NDArray[np.float64],
    free: NDArray[np.bool_],
    *,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> NDArray[np.float64]:
    """Maximize a zero-mean Gaussian likelihood over patterned covariances.

    Maximizes ``-log|C| - tr(C^{-1} S)`` over symmetric matrices ``C`` that
    equal ``start`` outside the symmetric mask ``free``. When every variance
    is free, the fixed covariances are zero and the free entries form complete
    blocks, the maximizer is ``S`` restricted to those blocks. Otherwise Fisher
    scoring with step halving keeps every iterate positive definite and never
    decreases the likelihood.

    Parameters
    ----------
    second_moment : ndarray of shape (K, K)
        Second-moment matrix ``S`` about the zero mean.
    start : ndarray of shape (K, K)
        Positive definite starting covariance; supplies the fixed entries.
    free : ndarray of bool, shape (K, K)
        Symmetric mask of the entries to estimate.
    max_iter : int, default=100
        Maximum number of scoring steps.
    tol : float, default=1e-10
        Stop when no free entry moves by more than this amount.

    Returns
    -------
    ndarray of shape (K, K)
        The constrained maximizer.
    """
    n_dimensions = len(start)
    off_diagonal = ~np.eye(n_dimensions, dtype=np.bool_)
    if (
        np.all(np.diag(free))
        and not np.any(start[off_diagonal & ~free])
        and _is_block_pattern(free)
    ):
        return np.where(free, second_moment, 0.0)

    rows, cols = np.nonzero(np.triu(free))
    current = np.array(start, dtype=np.float64, copy=True)
    if rows.size == 0:
        return current
    # Free entry (i, j) moves C along w * (e_i e_j' + e_j e_i').
    weight = np.where(rows == cols, 0.5, 1.0)
    value = _gaussian_deviance(current, second_moment)
    for _ in range(max_iter):
        precision = np.linalg.inv(current)
        residual = precision - precision @ second_moment @ precision
        gradient = 2.0 * weight * residual[rows, cols]
        information = (
            2.0
            * np.outer(weight, weight)
            * (
                precision[np.ix_(cols, rows)] * precision[np.ix_(rows, cols)]
                + precision[np.ix_(cols, cols)] * precision[np.ix_(rows, rows)]
            )
        )
        step = np.linalg.solve(information, gradient)
        direction = np.zeros_like(current)
        direction[rows, cols] = step
        direction[cols, rows] = step
        length = 1.0
        while length > 1e-8:
            candidate = current - length * direction
            candidate_value = _gaussian_deviance(candidate, second_moment)
            if candidate_value <= value:
                break
            length *= 0.5
        else:
            break
        current, value = candidate, candidate_value
        if length * np.max(np.abs(step)) < tol:
            break
    return current


class FactorCovarianceDensity(GaussianDensity):
    """Zero-mean Gaussian density with estimated factor correlations.

    Factor variances stay at their starting values (one by default) unless
    freed, so the density holds a correlation matrix whose free entries are
    estimated by EM while the other covariances stay at their starting
    values (zero by default).

    Each EM iteration maximizes the expected complete-data log-likelihood
    over the free entries with the item parameters held at their new values
    (an ECM step, Meng and Rubin, 1993), so EM keeps increasing the
    likelihood, or the posterior when item priors are used. Complete blocks
    of free entries with free variances have a closed form; other patterns
    use Fisher scoring with step halving.

    Parameters
    ----------
    n_dimensions : int
        Number of factors.
    free : array_like of bool, shape (n_dimensions, n_dimensions), optional
        Symmetric mask of estimated covariance entries. ``True`` off the
        diagonal estimates a correlation; ``True`` on the diagonal frees a
        variance, which is identified only when fixed slopes set the scale of
        that factor. By default every correlation is free and the variances
        are fixed.
    cov : ndarray of shape (n_dimensions, n_dimensions), optional
        Positive definite starting covariance, identity by default. Entries
        outside ``free`` stay at their starting values.

    Attributes
    ----------
    cov : ndarray
        Current factor covariance; a correlation matrix with the default
        unit variances.

    References
    ----------
    Meng, X.-L., & Rubin, D. B. (1993). Maximum likelihood estimation via the
    ECM algorithm: A general framework. *Biometrika*, 80(2), 267-278.
    """

    def __init__(
        self,
        n_dimensions: int,
        free: NDArray[np.bool_] | None = None,
        cov: NDArray[np.float64] | None = None,
    ) -> None:
        n_dimensions = _validate_count(n_dimensions, name="n_dimensions")
        super().__init__(cov=cov, estimate_cov=True, n_dimensions=n_dimensions)
        if free is None:
            mask = ~np.eye(n_dimensions, dtype=np.bool_)
        else:
            mask = np.asarray(free)
            if mask.dtype != np.bool_ or mask.shape != (n_dimensions, n_dimensions):
                raise ValueError(
                    f"free must be a Boolean array of shape "
                    f"({n_dimensions}, {n_dimensions})"
                )
            if not np.array_equal(mask, mask.T):
                raise ValueError("free must be symmetric")
        self._free = mask.copy()

    @property
    def free(self) -> NDArray[np.bool_]:
        """Symmetric mask of the estimated covariance entries."""
        return self._free.copy()

    @property
    def correlation(self) -> NDArray[np.float64]:
        """Factor correlation matrix implied by :attr:`cov`."""
        scale = np.sqrt(np.diag(self.cov))
        correlation = self.cov / np.outer(scale, scale)
        np.fill_diagonal(correlation, 1.0)
        return correlation

    @property
    def n_parameters(self) -> int:
        return int(np.count_nonzero(np.triu(self._free)))

    def update(
        self,
        theta_points: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        """Maximize over the free entries with the other entries held fixed."""
        points = _as_multivariate_points(
            theta_points, n_dimensions=self.n_dimensions, name="theta_points"
        )
        normalized = _normalize_weights(weights, n_points=len(points))
        moment = points.T @ (normalized[:, None] * points)
        cov = _constrained_gaussian_covariance(
            (moment + moment.T) / 2, self.cov, self._free
        )
        self.cov = (cov + cov.T) / 2
        self._update_precision()


class EmpiricalHistogram(LatentDensity):
    """Empirical histogram density (nonparametric).

    Estimates the latent density as a discrete distribution over
    quadrature points, with probabilities updated during EM.

    Parameters
    ----------
    n_bins : int
        Number of histogram bins (equals n_quadpts typically)
    """

    def __init__(
        self,
        n_bins: int | None = None,
        initial_probs: NDArray[np.float64] | None = None,
    ) -> None:
        self.n_bins = None if n_bins is None else _validate_count(n_bins, name="n_bins")
        if initial_probs is not None:
            self._probs = _normalize_probabilities(
                initial_probs,
                name="initial_probs",
                expected_length=self.n_bins,
            )
            self.n_bins = len(self._probs)
        else:
            self._probs = None

    def _initialize(self, n_points: int) -> None:
        """Initialize with uniform distribution."""
        if self.n_bins is not None and self.n_bins != n_points:
            raise ValueError(
                f"density has {self.n_bins} bins but received {n_points} points"
            )
        if self._probs is None:
            self.n_bins = n_points
            self._probs = np.ones(n_points) / n_points

    def log_density(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._probs is None:
            raise ValueError("Histogram not initialized. Call update() first.")
        n_points = _point_count(theta)
        if n_points != len(self._probs):
            raise ValueError(
                f"theta contains {n_points} points, expected {len(self._probs)}"
            )
        return np.log(np.clip(self._probs, 1e-300, None))

    def log_quadrature_mass(
        self,
        theta: NDArray[np.float64],
        quadrature_weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the histogram's discrete masses without GH reweighting."""
        n_points = _point_count(theta)
        _normalize_weights(
            quadrature_weights,
            n_points=n_points,
            name="quadrature_weights",
        )
        self._initialize(n_points)
        return normalize_log_mass(self.log_density(theta))

    def update(
        self,
        theta_points: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        n_points = _point_count(theta_points, name="theta_points")
        normalized_weights = _normalize_weights(weights, n_points=n_points)
        self._initialize(n_points)
        self._probs = normalized_weights
        self._probs = np.clip(self._probs, PROB_EPSILON, None)
        self._probs = self._probs / self._probs.sum()

    @property
    def probabilities(self) -> NDArray[np.float64] | None:
        """Return an owned copy of the discrete probability masses."""
        return None if self._probs is None else self._probs.copy()

    @property
    def n_parameters(self) -> int:
        if self._probs is None:
            return 0
        return len(self._probs) - 1


class EmpiricalHistogramWoods(LatentDensity):
    """Empirical histogram density standardized by Woods' method.

    Like :class:`EmpiricalHistogram`, the density is a discrete distribution
    over the quadrature nodes estimated from the posterior counts of each EM
    cycle. With free item parameters the location and scale of a histogram
    are not identified, so each update standardizes the posterior histogram
    to mean 0 and variance 1 by the interpolation-extrapolation method of
    Woods (2007), as R's ``mirt`` does for ``dentype="EHW"``: the nodes are
    rescaled to the standardized scale, the histogram heights are
    interpolated linearly back at the original nodes, and the tails beyond
    the rescaled range are extrapolated with a log-linear decay. A final
    exponential tilt, the smallest reweighting in Kullback-Leibler
    divergence, makes the mean and variance exact.

    Each update is a conditional maximization step. When the standardized
    histogram would fit the posterior counts worse than the current one, the
    update moves from the current histogram only as far toward it as
    improves the fit; every histogram on the way has mean 0 and variance 1.
    The marginal likelihood therefore never decreases across EM cycles. The
    first E-step uses the standard-normal masses of the quadrature rule
    unless ``initial_probs`` are given.

    Parameters
    ----------
    n_bins : int, optional
        Number of histogram bins, which must equal the number of quadrature
        nodes. Defaults to the number of nodes of the first evaluation.
    extrapolation_factor : float, optional
        Deprecated and ignored; Woods' standardization has no blending
        factor. Passing it emits a ``DeprecationWarning``.
    initial_probs : ndarray, optional
        Starting probability masses, normalized to sum to one.

    Notes
    -----
    On an equally spaced grid the rescaling is R's ``standardizeQuadrature``,
    except that interpolation and extrapolation use the spacing of the
    rescaled nodes; R uses the original spacing, which leaves the rescaled
    mean off zero whenever the standard deviation differs from one.

    The mean and variance constraints leave ``n_bins - 3`` free masses, the
    count of :attr:`n_parameters` and of R's ``mirt``. Grids with fewer than
    three distinct nodes cannot be rescaled and keep the posterior histogram.

    Like any empirical histogram, the estimate converges slowly when response
    patterns such as perfect scores pull mass toward the outermost nodes;
    such fits may need a larger ``max_iter``.

    References
    ----------
    Woods, C. M. (2007). Empirical histograms in item response theory
        with ordinal data. Educational and Psychological Measurement,
        67, 73-87.
    """

    def __init__(
        self,
        n_bins: int | None = None,
        extrapolation_factor: float | None = None,
        initial_probs: NDArray[np.float64] | None = None,
    ) -> None:
        self.n_bins = None if n_bins is None else _validate_count(n_bins, name="n_bins")
        if extrapolation_factor is not None:
            if isinstance(extrapolation_factor, (bool, np.bool_)) or not isinstance(
                extrapolation_factor, Real
            ):
                raise TypeError("extrapolation_factor must be a real number")
            extrapolation_factor = float(extrapolation_factor)
            if not np.isfinite(extrapolation_factor) or extrapolation_factor < 0.0:
                raise ValueError("extrapolation_factor must be finite and non-negative")
            warnings.warn(
                "extrapolation_factor is deprecated and ignored: Woods' "
                "standardization has no blending factor",
                DeprecationWarning,
                stacklevel=2,
            )
        self.extrapolation_factor: float | None = extrapolation_factor
        self._probs: NDArray[np.float64] | None = None

        if initial_probs is not None:
            self._probs = _normalize_probabilities(
                initial_probs,
                name="initial_probs",
                expected_length=self.n_bins,
            )
            self.n_bins = len(self._probs)

    def _initialize(
        self, n_points: int, reference: NDArray[np.float64] | None = None
    ) -> None:
        """Initialize with the reference masses, or uniformly without them."""
        if self.n_bins is not None and self.n_bins != n_points:
            raise ValueError(
                f"density has {self.n_bins} bins but received {n_points} points"
            )
        if self._probs is None:
            self.n_bins = n_points
            self._probs = (
                np.ones(n_points) / n_points if reference is None else reference.copy()
            )

    def log_density(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._probs is None:
            raise ValueError("Histogram not initialized. Call update() first.")
        n_points = _point_count(theta)
        if n_points != len(self._probs):
            raise ValueError(
                f"theta contains {n_points} points, expected {len(self._probs)}"
            )
        return np.log(np.clip(self._probs, 1e-300, None))

    def log_quadrature_mass(
        self,
        theta: NDArray[np.float64],
        quadrature_weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the histogram's discrete masses without GH reweighting.

        An uninitialized histogram starts from the normalized quadrature
        weights, the standard-normal masses on the quadrature nodes.
        """
        n_points = _point_count(theta)
        reference = _normalize_weights(
            quadrature_weights,
            n_points=n_points,
            name="quadrature_weights",
        )
        self._initialize(n_points, reference)
        return normalize_log_mass(self.log_density(theta))

    def update(
        self,
        theta_points: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        """Set the histogram to the standardized posterior histogram."""
        nodes = _as_univariate_points(theta_points, name="theta_points")
        counts = _normalize_weights(weights, n_points=len(nodes))
        previous = self._probs
        self._initialize(len(nodes))

        order = np.argsort(nodes, kind="stable")
        sorted_nodes = nodes[order]
        if len(nodes) < 3 or np.any(np.diff(sorted_nodes) <= 0.0):
            self._probs = counts
            return
        fitted = self._standardize(
            sorted_nodes,
            counts[order],
            None if previous is None else previous[order],
        )
        self._probs = np.empty_like(fitted)
        self._probs[order] = fitted

    @staticmethod
    def _standardize(
        nodes: NDArray[np.float64],
        counts: NDArray[np.float64],
        previous: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        """Return standardized masses that fit ``counts`` at least as well as
        a standardized ``previous`` histogram."""
        woods = _woods_standardize(nodes, counts)
        candidate = _tilt_to_standard(nodes, woods)
        if previous is None or not _is_standard(*_grid_moments(nodes, previous)):
            return woods if candidate is None else candidate
        if candidate is None:
            return previous
        # Both ends hold mean 0 and variance 1, and so does every point
        # between them; the best of these fits the counts at least as well
        # as the current histogram.
        return _best_on_segment(counts, previous, candidate)

    @property
    def probabilities(self) -> NDArray[np.float64] | None:
        """Return an owned copy of the standardized probability masses."""
        return None if self._probs is None else self._probs.copy()

    @property
    def n_parameters(self) -> int:
        """Free masses: one fewer than the bins, less the two moments."""
        if self._probs is None:
            return 0
        return max(len(self._probs) - 3, 0)


class DavidianCurve(LatentDensity):
    """Davidian curve semi-parametric density.

    Uses a polynomial transformation of the standard normal to create
    flexible density shapes while maintaining smoothness.

    The density is: f(theta) = phi(theta) * [sum_k c_k * H_k(theta)]^2

    where phi is standard normal and H_k are Hermite polynomials.

    Each update maximizes the expected complete-data log-likelihood of the
    posterior counts at the quadrature nodes over the normalized
    coefficients. With ``standardize=True`` the maximization is restricted to
    curves with mean 0 and variance 1, the identification of the
    Davidian-curve IRT of Woods and Lin (2009), so the latent location and
    scale cannot drift into the item parameters.

    Parameters
    ----------
    degree : int
        Degree of the polynomial (number of c coefficients).
        Higher values allow more flexible shapes.
    coefficients : ndarray, optional
        Initial coefficients. Default starts at standard normal.
    standardize : bool, default=True
        Hold the density at mean 0 and variance 1. Disable it only when
        fixed item parameters identify the latent location and scale.

    Notes
    -----
    The constrained maximization uses sequential quadratic programming, and
    an update that would fit the counts worse than the current curve is
    skipped. The analytic mean and variance of the curve are held at 0 and
    1. On Gauss-Hermite nodes with at least ``degree + 2`` points the
    normalization and moments of the quadrature masses are exact as well,
    so every update is a conditional maximization step and the marginal
    likelihood never decreases; on the fine rectangular grid they agree to
    about ``1e-6``.

    The mean and variance constraints leave ``degree - 2`` free coefficients,
    the count of :attr:`n_parameters`; curves of degree 2 or less are then
    held at the standard normal. R's ``mirt`` (``dentype="Davidian-k"``)
    instead refits the curve to standardized posterior counts, which
    standardizes it only approximately, and counts ``k`` parameters.

    References
    ----------
    Davidian, M., & Gallant, A. R. (1993). The nonlinear mixed effects
    model with a smooth random effects density. Biometrika, 80(3), 475-488.

    Woods, C. M., & Lin, N. (2009). Item response theory with estimation
    of the latent density using Davidian curves. Applied Psychological
    Measurement, 33(2), 102-117.
    """

    def __init__(
        self,
        degree: int = 4,
        coefficients: NDArray[np.float64] | None = None,
        standardize: bool = True,
    ) -> None:
        self.degree = _validate_count(degree, name="degree", allow_zero=True)
        if not isinstance(standardize, (bool, np.bool_)):
            raise TypeError("standardize must be a boolean")
        self.standardize = bool(standardize)

        if coefficients is not None:
            self._coeffs = np.asarray(coefficients, dtype=np.float64)
            if self._coeffs.ndim != 1 or len(self._coeffs) != self.degree + 1:
                raise ValueError(f"coefficients must have shape ({self.degree + 1},)")
            if not np.all(np.isfinite(self._coeffs)):
                raise ValueError("coefficients must contain only finite values")
            self._coeffs = self._coeffs.copy()
        else:
            self._coeffs = np.zeros(self.degree + 1)
            self._coeffs[0] = 1.0

        self._normalize_coefficients()
        self._moments = self._moment_matrices()

    def _normalize_coefficients(self) -> None:
        """Normalize coefficients so density integrates to 1.

        For Hermite polynomial expansion f(x) = phi(x) * g(x)^2 where
        g(x) = sum_k c_k * H_k(x), the integral is sum_k c_k^2 * k!
        due to orthogonality of Hermite polynomials.
        """
        factorials = np.array(
            [factorial(k) for k in range(self.degree + 1)],
            dtype=np.float64,
        )
        scale = float(np.max(np.abs(self._coeffs), initial=0.0))
        if scale <= 0.0:
            raise ValueError("coefficients must define a finite positive density")
        scaled_coefficients = self._coeffs / scale
        norm_sq = float(np.sum(scaled_coefficients**2 * factorials))
        if not np.isfinite(norm_sq) or norm_sq <= 0.0:
            raise ValueError("coefficients must define a finite positive density")
        self._coeffs = scaled_coefficients / np.sqrt(norm_sq)

    def _moment_matrices(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return the matrices ``M_r`` with ``E[x^r g(x)^2] = c @ M_r @ c``.

        Expectations are under the standard normal, so the moments of the
        curve are ratios to ``c @ M_0 @ c``. A Gauss-Hermite rule with
        ``degree + 2`` nodes integrates these polynomials exactly.
        """
        nodes, weights = roots_hermitenorm(self.degree + 2)
        weights = weights / weights.sum()
        basis = self._hermite_polynomials(nodes)
        weighted = basis * weights[:, None]
        return (
            weighted.T @ basis,
            weighted.T @ (nodes[:, None] * basis),
            weighted.T @ (nodes[:, None] ** 2 * basis),
        )

    def _hermite_polynomials(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute probabilist's Hermite polynomials H_k(x).

        Returns shape (n_points, degree+1)
        """
        x = np.atleast_1d(x).ravel()
        n = len(x)
        H = np.zeros((n, self.degree + 1))

        H[:, 0] = 1.0
        if self.degree >= 1:
            H[:, 1] = x
        for k in range(2, self.degree + 1):
            H[:, k] = x * H[:, k - 1] - (k - 1) * H[:, k - 2]

        return H

    def _polynomial_value(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute polynomial g(theta) = sum_k c_k * H_k(theta)."""
        H = self._hermite_polynomials(theta)
        return H @ self._coeffs

    def log_density(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        # scipy.stats is slow to import, so only the densities that use it do.
        from scipy import stats

        theta = _as_univariate_points(theta)

        log_phi = stats.norm.logpdf(theta)

        g = self._polynomial_value(theta)
        g_squared = g**2

        g_squared = np.clip(g_squared, 1e-300, None)

        return log_phi + np.log(g_squared)

    @property
    def mean(self) -> float:
        """Return the analytic curve mean."""
        norm, first, _ = self._moments
        coefficients = self._coeffs
        return float(coefficients @ first @ coefficients) / float(
            coefficients @ norm @ coefficients
        )

    @property
    def variance(self) -> float:
        """Return the analytic curve variance."""
        norm, _, second = self._moments
        coefficients = self._coeffs
        second_moment = float(coefficients @ second @ coefficients) / float(
            coefficients @ norm @ coefficients
        )
        return max(second_moment - self.mean**2, 0.0)

    def update(
        self,
        theta_points: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        """Maximize the expected log-likelihood of the posterior counts."""
        theta = _as_univariate_points(theta_points, name="theta_points")
        counts = _normalize_weights(weights, n_points=len(theta))
        if self.degree == 0:
            return

        # Optimize over the coordinates of the orthonormal polynomials
        # H_k / sqrt(k!), which keeps high degrees well conditioned.
        scale = 1.0 / np.sqrt(
            np.array([factorial(k) for k in range(self.degree + 1)], dtype=np.float64)
        )
        basis = self._hermite_polynomials(theta) * scale
        norm, first, second = (
            scale[:, None] * matrix * scale for matrix in self._moments
        )
        # Quadratic constraints c @ M @ c = target: unit normalization and,
        # when standardizing, zero mean and unit second moment.
        matrices = [norm]
        targets = [1.0]
        if self.standardize:
            matrices += [first, second - norm]
            targets += [0.0, 0.0]

        def fit(coefficients: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            values = basis @ coefficients
            safe = np.where(np.abs(values) < 1e-150, 1e-150, values)
            total = float(coefficients @ norm @ coefficients)
            value = float(counts @ np.log(safe**2)) - np.log(total)
            gradient = (
                2.0 * (basis.T @ (counts / safe)) - 2.0 * (norm @ coefficients) / total
            )
            return value, gradient

        def negative_fit(
            coefficients: NDArray[np.float64],
        ) -> tuple[float, NDArray[np.float64]]:
            value, gradient = fit(coefficients)
            return -value, -gradient

        constraints = [
            {
                "type": "eq",
                "fun": lambda c, m=matrix, t=target: float(c @ m @ c) - t,
                "jac": lambda c, m=matrix: 2.0 * (m @ c),
            }
            for matrix, target in zip(matrices, targets, strict=True)
        ]
        current = self._coeffs / scale
        with np.errstate(all="ignore"):
            result = minimize(
                negative_fit,
                current,
                jac=True,
                method="SLSQP",
                constraints=constraints,
                options={"maxiter": 200, "ftol": 1e-12},
            )
        candidate = _solve_quadratic_constraints(result.x, matrices, targets)
        if candidate is None:
            return
        feasible = _solve_quadratic_constraints(
            current, matrices, targets, max_iter=0, tol=1e-10
        )
        if feasible is not None and fit(candidate)[0] < fit(current)[0]:
            return
        if candidate[0] < 0:
            candidate = -candidate
        self._coeffs = candidate * scale

    @property
    def coefficients(self) -> NDArray[np.float64]:
        """Return an owned copy of the normalized curve coefficients."""
        return self._coeffs.copy()

    @property
    def n_parameters(self) -> int:
        """Free coefficients: the degree, less two moments when standardized."""
        return max(self.degree - 2, 0) if self.standardize else self.degree


class MixtureDensity(LatentDensity):
    """Mixture of Gaussians latent density.

    Each update takes one EM step of the mixture fit to the posterior counts
    at the quadrature nodes. With ``standardize=True`` the step is followed by
    the affine rescaling of all component means and variances that gives the
    mixture mean 0 and variance 1, so the latent location and scale cannot
    drift into the item parameters. From a standardized mixture, the step is
    halved until the rescaled mixture fits the counts at least as well as the
    current one.

    Parameters
    ----------
    n_components : int
        Number of mixture components
    means : ndarray, optional
        Component means. Defaults to equally spaced means on ``[-1.5, 1.5]``.
    variances : ndarray, optional
        Component variances. Defaults to 0.5 for every component.
    weights : ndarray, optional
        Component weights, normalized to sum to one. Defaults to equal weights.
    standardize : bool, default=True
        Hold the mixture at mean 0 and variance 1 during EM. The default
        means and variances are then rescaled to that standard; supplied
        values are kept until the first update. Disable it only when fixed
        item parameters identify the latent location and scale.

    Notes
    -----
    Standardization constrains two of the ``3 * n_components - 1`` mixture
    parameters, which leaves ``3 * n_components - 3`` for
    :attr:`n_parameters`.
    """

    def __init__(
        self,
        n_components: int = 2,
        means: NDArray[np.float64] | None = None,
        variances: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        standardize: bool = True,
    ) -> None:
        self.n_components = _validate_count(n_components, name="n_components")
        if not isinstance(standardize, (bool, np.bool_)):
            raise TypeError("standardize must be a boolean")
        self.standardize = bool(standardize)

        if means is not None:
            self.means = np.asarray(means, dtype=np.float64)
            if self.means.shape != (self.n_components,):
                raise ValueError(f"means must have shape ({self.n_components},)")
            if not np.all(np.isfinite(self.means)):
                raise ValueError("means must contain only finite values")
            self.means = self.means.copy()
        else:
            self.means = np.linspace(-1.5, 1.5, self.n_components)

        if variances is not None:
            self.variances = np.asarray(variances, dtype=np.float64)
            if self.variances.shape != (self.n_components,):
                raise ValueError(f"variances must have shape ({self.n_components},)")
            if not np.all(np.isfinite(self.variances)) or np.any(self.variances <= 0.0):
                raise ValueError("variances must contain finite positive values")
            self.variances = self.variances.copy()
        else:
            self.variances = np.ones(self.n_components) * 0.5

        if weights is not None:
            self.weights = _normalize_probabilities(
                weights,
                name="weights",
                expected_length=self.n_components,
            )
        else:
            self.weights = np.ones(self.n_components) / self.n_components

        if self.standardize and means is None and variances is None:
            self.means, self.variances = self._standardized(
                self.means, self.variances, self.weights
            )

    @staticmethod
    def _standardized(
        means: NDArray[np.float64],
        variances: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Rescale components so the mixture has mean 0 and variance 1."""
        mean = float(weights @ means)
        variance = float(weights @ (variances + (means - mean) ** 2))
        return (means - mean) / np.sqrt(variance), variances / variance

    def _component_log_density(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return log weighted component densities for each point."""
        centered = theta[:, None] - self.means[None, :]
        log_normal = -0.5 * (
            np.log(2.0 * np.pi * self.variances)[None, :]
            + centered**2 / self.variances[None, :]
        )
        with np.errstate(divide="ignore"):
            log_weights = np.log(self.weights)
        return log_normal + log_weights[None, :]

    def log_density(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        points = _as_univariate_points(theta)
        return logsumexp(self._component_log_density(points), axis=1)

    def _component_probabilities(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return normalized component probabilities for validated points."""
        log_components = self._component_log_density(theta)
        log_marginal = logsumexp(log_components, axis=1, keepdims=True)
        return np.exp(log_components - log_marginal)

    def component_probabilities(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return conditional component probabilities at each latent value.

        The result has shape ``(n_points, n_components)`` and each row sums
        to one. Evaluation remains stable in remote tails and for components
        with zero prior weight.
        """
        points = _as_univariate_points(theta)
        return self._component_probabilities(points)

    def classify(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.int_]:
        """Return the most probable component index for each latent value."""
        return np.argmax(self.component_probabilities(theta), axis=1).astype(
            np.int_, copy=False
        )

    def cdf(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the cumulative mixture distribution at latent values."""
        from scipy import stats

        points = _as_univariate_points(theta)
        standardized = (points[:, None] - self.means[None, :]) / np.sqrt(
            self.variances[None, :]
        )
        return stats.norm.cdf(standardized) @ self.weights

    @property
    def mean(self) -> float:
        """Return the analytic mixture mean."""
        return float(self.weights @ self.means)

    @property
    def variance(self) -> float:
        """Return the analytic mixture variance."""
        second_moment = float(self.weights @ (self.variances + self.means**2))
        return max(second_moment - self.mean**2, 0.0)

    @property
    def standard_deviation(self) -> float:
        """Return the analytic mixture standard deviation."""
        return float(np.sqrt(self.variance))

    def sample(
        self,
        n_samples: int,
        random_state: int | np.random.Generator | None = None,
        *,
        return_components: bool = False,
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.int_]]:
        """Draw reproducible latent values from the fitted mixture.

        Parameters
        ----------
        n_samples : int
            Number of draws. Zero returns empty arrays.
        random_state : int, Generator, or None
            Seed or existing NumPy generator.
        return_components : bool, default=False
            Also return the generating component index for each draw.
        """
        sample_count = _validate_count(
            n_samples,
            name="n_samples",
            allow_zero=True,
        )
        if not isinstance(return_components, (bool, np.bool_)):
            raise TypeError("return_components must be a boolean")
        if isinstance(random_state, np.random.Generator):
            rng = random_state
        elif random_state is None or (
            isinstance(random_state, Integral)
            and not isinstance(random_state, (bool, np.bool_))
        ):
            rng = np.random.default_rng(
                None if random_state is None else int(random_state)
            )
        else:
            raise TypeError("random_state must be an integer, Generator, or None")

        components = rng.choice(
            self.n_components,
            size=sample_count,
            p=self.weights,
        ).astype(np.int_, copy=False)
        samples = rng.normal(
            loc=self.means[components],
            scale=np.sqrt(self.variances[components]),
        ).astype(np.float64, copy=False)
        if return_components:
            return samples, components
        return samples

    def update(
        self,
        theta_points: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        """Update mixture parameters via EM."""
        theta = _as_univariate_points(theta_points, name="theta_points")
        normalized_weights = _normalize_weights(weights, n_points=len(theta))

        responsibilities = self._component_probabilities(theta)
        weighted_responsibilities = responsibilities * normalized_weights[:, None]
        component_mass = weighted_responsibilities.sum(axis=0)
        active = component_mass > PROB_EPSILON

        updated_means = self.means.copy()
        updated_variances = self.variances.copy()
        if np.any(active):
            updated_means[active] = (
                weighted_responsibilities[:, active].T @ theta
            ) / component_mass[active]

            centered = theta[:, None] - updated_means[None, :]
            updated_variances[active] = (
                weighted_responsibilities[:, active] * centered[:, active] ** 2
            ).sum(axis=0) / component_mass[active]
            updated_variances = np.maximum(updated_variances, 0.01)

        updated_weights = np.maximum(component_mass, PROB_EPSILON)
        updated_weights = updated_weights / updated_weights.sum()

        if not self.standardize:
            self.means = updated_means
            self.variances = updated_variances
            self.weights = updated_weights
            return

        start = (self.means, self.variances, self.weights)
        current = (
            float(normalized_weights @ self.log_density(theta))
            if _is_standard(self.mean, self.variance)
            else None
        )
        step = 1.0
        for _ in range(30):
            self.weights = start[2] + step * (updated_weights - start[2])
            self.means, self.variances = self._standardized(
                start[0] + step * (updated_means - start[0]),
                start[1] + step * (updated_variances - start[1]),
                self.weights,
            )
            if (
                current is None
                or float(normalized_weights @ self.log_density(theta)) >= current
            ):
                return
            step *= 0.5
        self.means, self.variances, self.weights = start

    @property
    def n_parameters(self) -> int:
        """Free parameters, less the mean and variance when standardized."""
        n_free = 3 * self.n_components - 1
        return n_free - 2 if self.standardize else n_free


class CustomDensity(LatentDensity):
    """User-defined custom latent density.

    Parameters
    ----------
    log_density_func : callable
        Function that takes theta array and returns log density values.
    update_func : callable, optional
        Function to update density parameters. Takes (theta_points, weights).
    n_params : int
        Number of parameters in the custom density.
    """

    def __init__(
        self,
        log_density_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        update_func: Callable[[NDArray[np.float64], NDArray[np.float64]], None]
        | None = None,
        n_params: int = 0,
    ) -> None:
        if not callable(log_density_func):
            raise TypeError("log_density_func must be callable")
        if update_func is not None and not callable(update_func):
            raise TypeError("update_func must be callable or None")
        self._log_density_func = log_density_func
        self._update_func = update_func
        self._n_params = _validate_count(n_params, name="n_params", allow_zero=True)

    def log_density(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        n_points = _point_count(theta)
        values = np.asarray(self._log_density_func(theta), dtype=np.float64)
        if values.size != n_points:
            raise ValueError("log_density_func must return one value per theta point")
        values = values.reshape(n_points)
        if np.any(np.isnan(values)):
            raise ValueError("log_density_func returned NaN values")
        return values

    def update(
        self,
        theta_points: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        if self._update_func is not None:
            self._update_func(theta_points, weights)

    @property
    def n_parameters(self) -> int:
        return self._n_params


def create_density(
    density_type: str = "gaussian",
    **kwargs: Any,
) -> LatentDensity:
    """Factory function to create latent density objects.

    Parameters
    ----------
    density_type : str
        Type of density: 'gaussian', 'empirical', 'davidian',
        'mixture', or 'custom'
    **kwargs
        Additional arguments passed to density constructor

    Returns
    -------
    LatentDensity
        The density object
    """
    density_types = {
        "gaussian": GaussianDensity,
        "normal": GaussianDensity,
        "empirical": EmpiricalHistogram,
        "histogram": EmpiricalHistogram,
        "eh": EmpiricalHistogram,
        "ehw": EmpiricalHistogramWoods,
        "empiricalhist_woods": EmpiricalHistogramWoods,
        "davidian": DavidianCurve,
        "mixture": MixtureDensity,
        "custom": CustomDensity,
    }

    if not isinstance(density_type, str):
        raise TypeError("density_type must be a string")
    normalized_type = density_type.strip().lower()
    if normalized_type not in density_types:
        raise ValueError(
            f"Unknown density type: {density_type}. "
            f"Choose from: {list(density_types.keys())}"
        )

    return density_types[normalized_type](**kwargs)
