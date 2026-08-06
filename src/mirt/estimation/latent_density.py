"""Latent density specifications for IRT models.

This module provides various latent trait distribution options:
- Gaussian (standard normal, default)
- Empirical histogram (nonparametric)
- Davidian curves (semi-parametric)
- Custom density functions
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from math import factorial
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.special import logsumexp

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
        diff = points - self.mean
        mahal = np.sum(diff @ self._precision * diff, axis=1)
        return self._log_norm - 0.5 * mahal

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
            self.mean = np.sum(normalized_weights[:, None] * points, axis=0)

        if self.estimate_cov:
            diff = points - self.mean
            self.cov = np.sum(
                normalized_weights[:, None, None]
                * (diff[:, :, None] * diff[:, None, :]),
                axis=0,
            )
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
    """Empirical histogram density with Woods extrapolation.

    Extends the standard empirical histogram approach using the method
    of Woods (2007) which extrapolates probabilities at the tails of
    the distribution to improve estimation in sparse regions.

    Parameters
    ----------
    n_bins : int
        Number of histogram bins (equals n_quadpts typically)
    extrapolation_factor : float
        Factor controlling extrapolation strength (0.5-2.0 typical)

    References
    ----------
    Woods, C. M. (2007). Empirical histograms in item response theory
        with ordinal data. Educational and Psychological Measurement,
        67, 73-87.
    """

    def __init__(
        self,
        n_bins: int | None = None,
        extrapolation_factor: float = 1.0,
        initial_probs: NDArray[np.float64] | None = None,
    ) -> None:
        self.n_bins = None if n_bins is None else _validate_count(n_bins, name="n_bins")
        if isinstance(extrapolation_factor, (bool, np.bool_)) or not isinstance(
            extrapolation_factor, Real
        ):
            raise TypeError("extrapolation_factor must be a real number")
        extrapolation_factor = float(extrapolation_factor)
        if not np.isfinite(extrapolation_factor) or extrapolation_factor < 0.0:
            raise ValueError("extrapolation_factor must be finite and non-negative")
        self.extrapolation_factor: float = extrapolation_factor
        self._probs: NDArray[np.float64] | None = None
        self._theta_points: NDArray[np.float64] | None = None

        if initial_probs is not None:
            self._probs = _normalize_probabilities(
                initial_probs,
                name="initial_probs",
                expected_length=self.n_bins,
            )
            self.n_bins = len(self._probs)

    def _initialize(
        self, n_points: int, theta_points: NDArray[np.float64] | None = None
    ) -> None:
        """Initialize with uniform distribution."""
        if self.n_bins is not None and self.n_bins != n_points:
            raise ValueError(
                f"density has {self.n_bins} bins but received {n_points} points"
            )
        if self._probs is None:
            self.n_bins = n_points
            self._probs = np.ones(n_points) / n_points
        if theta_points is not None:
            self._theta_points = np.asarray(theta_points, dtype=np.float64).copy()

    def _extrapolate_tails(self, probs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply Woods extrapolation to tail probabilities.

        The method fits a polynomial to the log-probabilities in the
        interior region and extrapolates to the tails.
        """
        n = len(probs)
        if n < 5:
            return probs

        log_probs = np.log(np.clip(probs, PROB_EPSILON, None))

        n_interior = max(3, n // 3)
        lower_idx = n_interior
        upper_idx = n - n_interior

        interior_mask = np.zeros(n, dtype=bool)
        interior_mask[lower_idx:upper_idx] = True

        if interior_mask.sum() < 3:
            return probs

        x = np.arange(n)
        try:
            poly = np.polyfit(x[interior_mask], log_probs[interior_mask], deg=2)

            lower_extrap = np.exp(np.polyval(poly, x[:lower_idx]))
            upper_extrap = np.exp(np.polyval(poly, x[upper_idx:]))

            extrap_probs = probs.copy()
            alpha = self.extrapolation_factor

            extrap_probs[:lower_idx] = (1 - alpha) * probs[
                :lower_idx
            ] + alpha * lower_extrap
            extrap_probs[upper_idx:] = (1 - alpha) * probs[
                upper_idx:
            ] + alpha * upper_extrap

            extrap_probs = np.clip(extrap_probs, PROB_EPSILON, None)
            extrap_probs = extrap_probs / extrap_probs.sum()

            return extrap_probs
        except (np.linalg.LinAlgError, ValueError):
            return probs

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
        self._initialize(n_points, np.asarray(theta))
        return normalize_log_mass(self.log_density(theta))

    def update(
        self,
        theta_points: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        n_points = _point_count(theta_points, name="theta_points")
        normalized_weights = _normalize_weights(weights, n_points=n_points)
        self._initialize(n_points, theta_points)

        raw_probs = np.clip(normalized_weights, PROB_EPSILON, None)

        self._probs = self._extrapolate_tails(raw_probs)

        self._probs = np.clip(self._probs, PROB_EPSILON, None)
        self._probs = self._probs / self._probs.sum()

    @property
    def probabilities(self) -> NDArray[np.float64] | None:
        """Return an owned copy of the extrapolated probability masses."""
        return None if self._probs is None else self._probs.copy()

    @property
    def n_parameters(self) -> int:
        if self._probs is None:
            return 0
        return len(self._probs) - 1


class DavidianCurve(LatentDensity):
    """Davidian curve semi-parametric density.

    Uses a polynomial transformation of the standard normal to create
    flexible density shapes while maintaining smoothness.

    The density is: f(theta) = phi(theta) * [sum_k c_k * H_k(theta)]^2

    where phi is standard normal and H_k are Hermite polynomials.

    Parameters
    ----------
    degree : int
        Degree of the polynomial (number of c coefficients).
        Higher values allow more flexible shapes.
    coefficients : ndarray, optional
        Initial coefficients. Default starts at standard normal.

    References
    ----------
    Davidian, M., & Gallant, A. R. (1993). The nonlinear mixed effects
    model with a smooth random effects density. Biometrika, 80(3), 475-488.
    """

    def __init__(
        self,
        degree: int = 4,
        coefficients: NDArray[np.float64] | None = None,
    ) -> None:
        self.degree = _validate_count(degree, name="degree", allow_zero=True)

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
        theta = _as_univariate_points(theta)

        log_phi = stats.norm.logpdf(theta)

        g = self._polynomial_value(theta)
        g_squared = g**2

        g_squared = np.clip(g_squared, 1e-300, None)

        return log_phi + np.log(g_squared)

    def update(
        self,
        theta_points: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        """Update Davidian curve coefficients via weighted least squares."""
        theta = _as_univariate_points(theta_points, name="theta_points")
        normalized_weights = _normalize_weights(weights, n_points=len(theta))

        H = self._hermite_polynomials(theta)

        phi = stats.norm.pdf(theta)

        target = np.sqrt(normalized_weights / (phi + 1e-300))

        try:
            weighted_h = H * phi[:, None]
            HtWH = H.T @ weighted_h
            HtWy = H.T @ (phi * target)

            reg = REGULARIZATION_EPSILON * np.eye(self.degree + 1)
            coefficients = np.linalg.solve(HtWH + reg, HtWy)

            if coefficients[0] < 0:
                coefficients = -coefficients
            if not np.all(np.isfinite(coefficients)):
                return
            self._coeffs = coefficients
            self._normalize_coefficients()

        except np.linalg.LinAlgError:
            return

    @property
    def coefficients(self) -> NDArray[np.float64]:
        """Return an owned copy of the normalized curve coefficients."""
        return self._coeffs.copy()

    @property
    def n_parameters(self) -> int:
        return self.degree


class MixtureDensity(LatentDensity):
    """Mixture of Gaussians latent density.

    Parameters
    ----------
    n_components : int
        Number of mixture components
    """

    def __init__(
        self,
        n_components: int = 2,
        means: NDArray[np.float64] | None = None,
        variances: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
    ) -> None:
        self.n_components = _validate_count(n_components, name="n_components")

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

    def update(
        self,
        theta_points: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        """Update mixture parameters via EM."""
        theta = _as_univariate_points(theta_points, name="theta_points")
        normalized_weights = _normalize_weights(weights, n_points=len(theta))

        log_components = self._component_log_density(theta)
        log_responsibilities = log_components - logsumexp(
            log_components,
            axis=1,
            keepdims=True,
        )
        weighted_responsibilities = (
            np.exp(log_responsibilities) * normalized_weights[:, None]
        )
        component_mass = weighted_responsibilities.sum(axis=0)
        active = component_mass > PROB_EPSILON

        if np.any(active):
            updated_means = self.means.copy()
            updated_means[active] = (
                weighted_responsibilities[:, active].T @ theta
            ) / component_mass[active]

            centered = theta[:, None] - updated_means[None, :]
            updated_variances = self.variances.copy()
            updated_variances[active] = (
                weighted_responsibilities[:, active] * centered[:, active] ** 2
            ).sum(axis=0) / component_mass[active]

            self.means = updated_means
            self.variances = np.maximum(updated_variances, 0.01)

        self.weights = np.maximum(component_mass, PROB_EPSILON)
        self.weights = self.weights / self.weights.sum()

    @property
    def n_parameters(self) -> int:
        return 3 * self.n_components - 1


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
