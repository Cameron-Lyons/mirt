"""Gaussian kernel equating.

Implements the kernel method of test equating of von Davier, Holland and
Thayer (2004) for the equivalent-groups design, and its IRT observed-score
variant (Andersson and Wiberg, 2017), which continuizes model-implied score
distributions from Lord-Wingersky recursion.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, ndtr, ndtri

from mirt.equating.linking import LinkingResult
from mirt.equating.score_equating import (
    ScoreEquatingResult,
    _bracketed_root,
    _population_score_distributions,
    _validate_distribution,
)

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

BandwidthSpec = Literal["penalty", "linear"] | float | tuple[float, float]
_T = TypeVar("_T")

_PENALTY_OFFSET = 0.25
_PENALTY_GRID_SIZE = 121
_MIN_BANDWIDTH = 0.05
_LINEAR_BANDWIDTH_FACTOR = 1000.0
_LOGLINEAR_MAX_ITERATIONS = 100
# Design columns are orthonormal under the observed scores, so these
# tolerances are on a fixed scale.
_LOGLINEAR_GRADIENT_TOLERANCE = 1e-13
_LOGLINEAR_DECREMENT_TOLERANCE = 1e-24
_LOGLINEAR_FAILURE = (
    "log-linear presmoothing did not converge; lower the presmoothing degree"
)
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


@dataclass
class KernelEquatingResult(ScoreEquatingResult):
    """Result of Gaussian kernel equating.

    Extends :class:`ScoreEquatingResult`. ``theta`` holds the population
    grid of the IRT variant and is empty when equating score distributions
    directly. ``standard_errors`` holds standard errors of equating when
    sample sizes are supplied.

    Attributes
    ----------
    bandwidth_old : float
        Continuization bandwidth of the old form.
    bandwidth_new : float
        Continuization bandwidth of the new form.
    score_dist_old : NDArray[np.float64]
        Old-form score probabilities that were continuized, after any
        presmoothing.
    score_dist_new : NDArray[np.float64]
        New-form score probabilities that were continuized, after any
        presmoothing.
    """

    bandwidth_old: float
    bandwidth_new: float
    score_dist_old: NDArray[np.float64]
    score_dist_new: NDArray[np.float64]


@dataclass(frozen=True)
class _Continuization:
    """Mean- and variance-preserving Gaussian kernel continuization."""

    probabilities: NDArray[np.float64]
    bandwidth: float
    mean: float
    variance: float
    shrinkage: float

    @classmethod
    def fit(
        cls, probabilities: NDArray[np.float64], bandwidth: float
    ) -> "_Continuization":
        mean, variance = _score_moments(probabilities)
        shrinkage = float(np.sqrt(variance / (variance + bandwidth**2)))
        return cls(probabilities, bandwidth, mean, variance, shrinkage)

    @property
    def scale(self) -> float:
        return self.shrinkage * self.bandwidth

    def standardized(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``R_j(x)`` for every point and score."""
        scores = np.arange(len(self.probabilities), dtype=np.float64)
        centers = self.shrinkage * scores + (1.0 - self.shrinkage) * self.mean
        return (points[:, None] - centers) / self.scale

    def cdf(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(ndtr(self.standardized(points)) @ self.probabilities)

    def survival(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(ndtr(-self.standardized(points)) @ self.probabilities)

    def density(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        kernel = np.exp(-0.5 * self.standardized(points) ** 2) * _INV_SQRT_2PI
        return np.asarray(kernel @ self.probabilities / self.scale)

    def density_slope(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        standardized = self.standardized(points)
        kernel = np.exp(-0.5 * standardized**2) * _INV_SQRT_2PI
        return np.asarray(-(standardized * kernel) @ self.probabilities / self.scale**2)

    def quantile(
        self, lower_tail: NDArray[np.float64], upper_tail: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Invert the continuized distribution at complementary tail masses.

        Masses below one half are inverted through the lower tail and the
        rest through the upper tail, keeping relative precision in both.
        Roots are found on the probit scale, where Gaussian tails are nearly
        linear.
        """
        scores = np.arange(len(self.probabilities), dtype=np.float64)
        centers = self.shrinkage * scores + (1.0 - self.shrinkage) * self.mean
        lowest, highest = float(np.min(centers)), float(np.max(centers))
        tiny = np.finfo(np.float64).tiny
        use_lower = lower_tail <= upper_tail
        result = np.empty(len(lower_tail), dtype=np.float64)
        for lower_branch in (True, False):
            mask = use_lower == lower_branch
            if not np.any(mask):
                continue
            tail = lower_tail if lower_branch else upper_tail
            probits = ndtri(np.clip(tail[mask], tiny, 0.5))
            # Mixture bounds: G(lowest + s z) <= Phi(z) <= G(highest + s z).
            if lower_branch:
                start = lowest + self.scale * (probits - 1.0)
                stop = highest + self.scale * (probits + 1.0)
                function = self._lower_probit
                targets = probits
            else:
                start = lowest - self.scale * (probits + 1.0)
                stop = highest - self.scale * (probits - 1.0)
                function = self._upper_probit
                targets = -probits
            # Kernel centers split each mixture bound into narrow brackets.
            knots = np.unique(np.concatenate((start, centers, stop)))
            values = np.maximum.accumulate(function(knots))
            upper = np.searchsorted(values, targets, side="left")
            result[mask] = _bracketed_root(
                function,
                targets,
                knots[upper - 1],
                knots[upper],
                values[upper - 1],
                values[upper],
            )
        return result

    def _lower_probit(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(ndtri(self.cdf(points)))

    def _upper_probit(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(-ndtri(self.survival(points)))

    def mass_jacobian(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``dF(x)/dr_j`` including the dependence of the moments on r.

        This is ``Phi(R_j(x)) - M_j(x) f(x)`` of von Davier et al. (2004).
        """
        scores = np.arange(len(self.probabilities), dtype=np.float64)
        standardized_scores = (scores - self.mean) ** 2 / self.variance
        moment_term = (
            0.5
            * (points[:, None] - self.mean)
            * (1.0 - self.shrinkage**2)
            * standardized_scores
            + (1.0 - self.shrinkage) * scores
        )
        return np.asarray(
            ndtr(self.standardized(points))
            - moment_term * self.density(points)[:, None]
        )


def kernel_equating(
    score_dist_old: NDArray[np.float64],
    score_dist_new: NDArray[np.float64],
    *,
    bandwidth: BandwidthSpec = "penalty",
    kappa: float = 0.0,
    presmoothing: int | tuple[int | None, int | None] | None = None,
    n_old: int | None = None,
    n_new: int | None = None,
) -> KernelEquatingResult:
    """Perform Gaussian kernel equating for an equivalent-groups design.

    Each discrete score distribution is continuized with a Gaussian kernel
    that preserves its mean and variance. Each old score ``x`` is mapped to
    ``G_h^{-1}(F_h(x))``, where ``F_h`` and ``G_h`` are the continuized
    old and new distribution functions (von Davier, Holland and Thayer,
    2004).

    Parameters
    ----------
    score_dist_old : NDArray
        Old-form score frequencies or probabilities for scores ``0..K_X``.
    score_dist_new : NDArray
        New-form score frequencies or probabilities for scores ``0..K_Y``.
    bandwidth : {"penalty", "linear"} or float or tuple[float, float]
        Continuization bandwidths. ``"penalty"`` selects each bandwidth by
        minimizing ``PEN1 + kappa * PEN2``. ``"linear"`` uses ``1000`` times
        each score standard deviation, which reproduces linear equating. A
        positive number is used for both forms; a pair sets old and new
        bandwidths separately.
    kappa : float
        Weight of the second penalty, which counts score points around which
        the continuized density is U-shaped. The default ``0`` minimizes only
        the squared density misfit at the score points, as R's kequate does.
    presmoothing : int or tuple[int | None, int | None] or None
        Polynomial degree of a log-linear model fitted by maximum likelihood
        to each score distribution before continuization. It preserves the
        first ``degree`` moments and must be less than the number of scores
        with positive probability. A pair sets the old and new degrees
        separately. None continuizes a distribution as given.
    n_old, n_new : int or None
        Sample sizes behind the two distributions. When both are given, the
        standard errors of equating are computed by the delta method from
        the multinomial (or log-linear) sampling covariance, holding the
        bandwidths fixed.

    Returns
    -------
    KernelEquatingResult
        Equated scores, bandwidths, and optional standard errors.

    Notes
    -----
    ``observed_score_equating(..., smoothing="kernel")`` smooths discrete
    distributions before percentile-rank matching; it is not kernel
    equating.
    """
    old = _validate_distribution(score_dist_old, "score_dist_old")
    new = _validate_distribution(score_dist_new, "score_dist_new")
    _require_spread(old, "score_dist_old")
    _require_spread(new, "score_dist_new")
    degrees = _pair(presmoothing, "presmoothing")
    sample_sizes = _sample_sizes(n_old, n_new)
    designs: list[NDArray[np.float64] | None] = [None, None]
    distributions = [old, new]
    for index, (name, degree) in enumerate(zip(("old", "new"), degrees, strict=True)):
        if degree is None:
            continue
        degree = _validate_degree(degree, distributions[index], name)
        designs[index] = _polynomial_design(distributions[index], degree)
        distributions[index] = _fit_loglinear(distributions[index], designs[index])
    old, new = distributions

    fitted = _kernel_equate(old, new, bandwidth, kappa)
    if sample_sizes is not None:
        continuized_old, continuized_new, equated = fitted
        jacobian_old, jacobian_new = _equating_jacobians(
            continuized_old, continuized_new, equated
        )
        variance = _delta_variance(
            jacobian_old, old, sample_sizes[0], designs[0]
        ) + _delta_variance(jacobian_new, new, sample_sizes[1], designs[1])
        standard_errors: NDArray[np.float64] | None = np.sqrt(np.maximum(variance, 0.0))
    else:
        standard_errors = None
    return _result(fitted, standard_errors, np.empty(0), "kernel")


def irt_kernel_equating(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    theta_distribution: NDArray[np.float64] | None = None,
    theta_grid: NDArray[np.float64] | None = None,
    n_theta: int = 61,
    items_old: list[int] | None = None,
    items_new: list[int] | None = None,
    linking_result: LinkingResult | None = None,
    *,
    bandwidth: BandwidthSpec = "penalty",
    kappa: float = 0.0,
    batch_size: int | None = None,
) -> KernelEquatingResult:
    """Perform IRT observed-score kernel equating.

    Score distributions implied by each model in one reference population
    are computed with Lord-Wingersky recursion, as in
    :func:`observed_score_equating`, and then equated with Gaussian kernel
    continuization (Andersson and Wiberg, 2017).

    Parameters
    ----------
    model_old : BaseItemModel
        Reference form model.
    model_new : BaseItemModel
        New form model.
    theta_distribution : NDArray | None
        Probability masses at each point on the old/reference theta scale.
        Default: weights proportional to standard normal density.
    theta_grid : NDArray | None
        Grid of theta values for integration.
    n_theta : int
        Number of theta points if grid not provided.
    items_old : list[int] | None
        Subset of items for old form.
    items_new : list[int] | None
        Subset of items for new form.
    linking_result : LinkingResult | None
        Constants mapping new abilities onto the old/reference scale as
        ``theta_old = A * theta_new + B``.
    bandwidth : {"penalty", "linear"} or float or tuple[float, float]
        Continuization bandwidths; see :func:`kernel_equating`.
    kappa : float
        Weight of the second bandwidth penalty; see :func:`kernel_equating`.
    batch_size : int | None
        Maximum theta points evaluated together during recursion.

    Returns
    -------
    KernelEquatingResult
        Equated scores and bandwidths. Standard errors are not computed
        because they depend on the item parameter covariance.
    """
    theta_grid, old, new = _population_score_distributions(
        model_old,
        model_new,
        theta_distribution,
        theta_grid,
        n_theta,
        items_old,
        items_new,
        linking_result,
        batch_size,
    )
    fitted = _kernel_equate(old, new, bandwidth, kappa)
    return _result(fitted, None, theta_grid, "irt_kernel")


def _kernel_equate(
    old: NDArray[np.float64],
    new: NDArray[np.float64],
    bandwidth: BandwidthSpec,
    kappa: float,
) -> tuple[_Continuization, _Continuization, NDArray[np.float64]]:
    """Continuize both distributions and map every old score."""
    kappa = float(kappa)
    if not np.isfinite(kappa) or kappa < 0.0:
        raise ValueError("kappa must be finite and non-negative")
    _require_spread(old, "score_dist_old")
    _require_spread(new, "score_dist_new")
    if isinstance(bandwidth, str):
        if bandwidth == "penalty":
            bandwidths = (
                _penalty_bandwidth(old, kappa),
                _penalty_bandwidth(new, kappa),
            )
        elif bandwidth == "linear":
            bandwidths = (_linear_bandwidth(old), _linear_bandwidth(new))
        else:
            raise ValueError("bandwidth must be 'penalty', 'linear', or positive")
    else:
        pair = _pair(bandwidth, "bandwidth")
        bandwidths = (_positive(pair[0]), _positive(pair[1]))

    continuized_old = _Continuization.fit(old, bandwidths[0])
    continuized_new = _Continuization.fit(new, bandwidths[1])
    scores = np.arange(len(old), dtype=np.float64)
    equated = continuized_new.quantile(
        continuized_old.cdf(scores), continuized_old.survival(scores)
    )
    return continuized_old, continuized_new, equated


def _result(
    fitted: tuple[_Continuization, _Continuization, NDArray[np.float64]],
    standard_errors: NDArray[np.float64] | None,
    theta: NDArray[np.float64],
    method: str,
) -> KernelEquatingResult:
    continuized_old, continuized_new, equated = fitted
    return KernelEquatingResult(
        old_scores=np.arange(len(continuized_old.probabilities), dtype=np.float64),
        new_scores=equated,
        theta=theta,
        standard_errors=standard_errors,
        method=method,
        bandwidth_old=continuized_old.bandwidth,
        bandwidth_new=continuized_new.bandwidth,
        score_dist_old=continuized_old.probabilities,
        score_dist_new=continuized_new.probabilities,
    )


def _bandwidth_penalty(
    probabilities: NDArray[np.float64], bandwidth: float, kappa: float
) -> float:
    """Return ``PEN1 + kappa * PEN2`` of von Davier et al. (2004, sec. 4.5)."""
    continuized = _Continuization.fit(probabilities, bandwidth)
    scores = np.arange(len(probabilities), dtype=np.float64)
    misfit = float(np.sum((probabilities - continuized.density(scores)) ** 2))
    if kappa == 0.0:
        return misfit
    falling_left = continuized.density_slope(scores - _PENALTY_OFFSET) < 0.0
    falling_right = continuized.density_slope(scores + _PENALTY_OFFSET) < 0.0
    return misfit + kappa * float(np.sum(falling_left & ~falling_right))


def _penalty_bandwidth(probabilities: NDArray[np.float64], kappa: float) -> float:
    """Minimize the bandwidth penalty on a log grid, then refine locally."""
    deviation = np.sqrt(_score_moments(probabilities)[1])
    log_grid = np.linspace(
        np.log(_MIN_BANDWIDTH),
        np.log(10.0 * max(1.0, deviation)),
        _PENALTY_GRID_SIZE,
    )
    values = np.array(
        [_bandwidth_penalty(probabilities, np.exp(h), kappa) for h in log_grid]
    )
    best = int(np.argmin(values))
    lower = log_grid[max(best - 1, 0)]
    upper = log_grid[min(best + 1, len(log_grid) - 1)]
    refined = minimize_scalar(
        lambda log_h: _bandwidth_penalty(probabilities, float(np.exp(log_h)), kappa),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if refined.success and float(refined.fun) <= values[best]:
        return float(np.exp(refined.x))
    return float(np.exp(log_grid[best]))


def _linear_bandwidth(probabilities: NDArray[np.float64]) -> float:
    return float(_LINEAR_BANDWIDTH_FACTOR * np.sqrt(_score_moments(probabilities)[1]))


def _score_moments(probabilities: NDArray[np.float64]) -> tuple[float, float]:
    """Return the mean and variance of a score distribution on 0..K."""
    scores = np.arange(len(probabilities), dtype=np.float64)
    mean = float(probabilities @ scores)
    return mean, float(probabilities @ (scores - mean) ** 2)


def _equating_jacobians(
    continuized_old: _Continuization,
    continuized_new: _Continuization,
    equated: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return derivatives of each equated score with respect to r and s."""
    scores = np.arange(len(continuized_old.probabilities), dtype=np.float64)
    density = continuized_new.density(equated)[:, None]
    jacobian_old = continuized_old.mass_jacobian(scores) / density
    jacobian_new = -continuized_new.mass_jacobian(equated) / density
    return jacobian_old, jacobian_new


def _delta_variance(
    jacobian: NDArray[np.float64],
    probabilities: NDArray[np.float64],
    sample_size: int,
    design: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Return ``diag(J Cov(r) J^T)`` for multinomial or log-linear estimates.

    The log-linear covariance ``S B (B^T S B)^-1 B^T S``, with ``S`` the
    multinomial covariance, equals ``D^1/2 P D^1/2`` for the projection ``P``
    onto the whitened design, which stays accurate for tiny fitted masses.
    """
    if design is None:
        centered = jacobian - (jacobian @ probabilities)[:, None]
        variance = centered**2 @ probabilities
    else:
        root = np.sqrt(probabilities)
        projection = _whitened_basis(_whitened_design(design, probabilities))
        variance = np.sum(((jacobian * root) @ projection) ** 2, axis=1)
    return np.asarray(variance / sample_size, dtype=np.float64)


def _polynomial_design(
    probabilities: NDArray[np.float64], degree: int
) -> NDArray[np.float64]:
    """Return polynomials of degree 1..degree in the standardized score.

    The columns are orthonormal under the observed distribution, which keeps
    Newton steps well conditioned near the fit even when the observed
    scores occupy a small part of the score range. They span the same
    log-linear model as raw score powers.
    """
    mean, variance = _score_moments(probabilities)
    standardized = (np.arange(len(probabilities)) - mean) / np.sqrt(variance)
    vandermonde = np.vander(standardized, degree + 1, increasing=True)
    _, triangular = np.linalg.qr(np.sqrt(probabilities)[:, None] * vandermonde)
    basis = solve_triangular(triangular, vandermonde.T, trans="T").T
    return np.asarray(basis[:, 1:], dtype=np.float64)


def _fit_loglinear(
    probabilities: NDArray[np.float64], design: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Fit a polynomial log-linear model by damped Newton maximum likelihood.

    Iterations start at the discrete normal distribution with the observed
    mean and variance. Each Newton step is the least-squares solution of
    ``D^1/2 C step = D^-1/2 (r - p)``, where ``p`` is the current fit,
    ``D = diag(p)`` and ``C`` is the design centered at ``p``. This avoids
    the squared condition number of the normal equations. The fit has
    converged when its design moments match the observed ones.
    """
    mean, variance = _score_moments(probabilities)
    half_square = 0.5 * (np.arange(len(probabilities)) - mean) ** 2 / variance
    centered = design - np.mean(design, axis=0)
    coefficients = np.linalg.lstsq(
        centered, np.mean(half_square) - half_square, rcond=None
    )[0]
    log_likelihood = _loglinear_log_likelihood(probabilities, design, coefficients)
    for _ in range(_LOGLINEAR_MAX_ITERATIONS):
        fitted = _loglinear_probabilities(design, coefficients)
        if np.max(np.abs(design.T @ (probabilities - fitted))) <= (
            _LOGLINEAR_GRADIENT_TOLERANCE
        ):
            return fitted
        whitened = _whitened_design(design, fitted)
        root = np.sqrt(fitted)
        residual = np.divide(
            probabilities - fitted, root, out=np.zeros_like(root), where=root > 0.0
        )
        step = np.linalg.lstsq(whitened, residual, rcond=None)[0]
        # Squared Newton decrement: twice the predicted likelihood gain.
        decrement = float(np.sum((whitened @ step) ** 2))
        if decrement <= _LOGLINEAR_DECREMENT_TOLERANCE:
            return fitted
        rounding = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(log_likelihood))
        scale = 1.0
        while True:
            candidate = _loglinear_log_likelihood(
                probabilities, design, coefficients + scale * step
            )
            if candidate - log_likelihood >= 0.25 * scale * decrement - rounding:
                break
            scale *= 0.5
            if scale < 1e-12:
                raise ValueError(_LOGLINEAR_FAILURE)
        coefficients = coefficients + scale * step
        log_likelihood = candidate
    raise ValueError(_LOGLINEAR_FAILURE)


def _whitened_design(
    design: NDArray[np.float64], probabilities: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return ``D^1/2 (B - 1 p^T B)``, whose Gram matrix is the information."""
    centered = design - probabilities @ design
    return np.asarray(np.sqrt(probabilities)[:, None] * centered, dtype=np.float64)


def _whitened_basis(whitened: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return an orthonormal basis of the numerically nonzero column space."""
    left, singular, _ = np.linalg.svd(whitened, full_matrices=False)
    cutoff = singular[0] * max(whitened.shape) * np.finfo(np.float64).eps
    return np.asarray(left[:, singular > cutoff], dtype=np.float64)


def _loglinear_probabilities(
    design: NDArray[np.float64], coefficients: NDArray[np.float64]
) -> NDArray[np.float64]:
    logits = design @ coefficients
    weights = np.exp(logits - np.max(logits))
    return np.asarray(weights / np.sum(weights), dtype=np.float64)


def _loglinear_log_likelihood(
    probabilities: NDArray[np.float64],
    design: NDArray[np.float64],
    coefficients: NDArray[np.float64],
) -> float:
    logits = design @ coefficients
    return float(probabilities @ logits - logsumexp(logits))


def _require_spread(distribution: NDArray[np.float64], name: str) -> None:
    if np.count_nonzero(distribution) < 2:
        raise ValueError(f"{name} must have positive probability at two scores")


def _validate_degree(degree: int, probabilities: NDArray[np.float64], name: str) -> int:
    """Require fewer parameters than observed scores, below saturation.

    With more observed scores than the degree, the observed moments lie
    inside the moment polytope, so the maximum likelihood fit exists.
    """
    if isinstance(degree, (bool, np.bool_)) or not isinstance(
        degree, (int, np.integer)
    ):
        raise ValueError("presmoothing degrees must be integers")
    maximum = min(len(probabilities) - 2, np.count_nonzero(probabilities) - 1)
    if not 1 <= int(degree) <= maximum:
        raise ValueError(
            f"presmoothing degree for the {name} form must be between 1 and "
            f"{maximum}, below its number of scores with positive probability"
        )
    return int(degree)


def _sample_sizes(n_old: int | None, n_new: int | None) -> tuple[int, int] | None:
    if n_old is None and n_new is None:
        return None
    if n_old is None or n_new is None:
        raise ValueError("n_old and n_new must be given together")
    sizes = []
    for name, value in (("n_old", n_old), ("n_new", n_new)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
        sizes.append(int(value))
    return sizes[0], sizes[1]


def _pair(value: _T | tuple[_T, _T], name: str) -> tuple[_T, _T]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"{name} pair must have two entries")
        return value[0], value[1]
    return value, value


def _positive(value: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("bandwidth must be 'penalty', 'linear', or positive")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("bandwidth must be 'penalty', 'linear', or positive") from None
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("bandwidth must be finite and positive")
    return result
