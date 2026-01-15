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
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from mirt.constants import PROB_EPSILON, REGULARIZATION_EPSILON

if TYPE_CHECKING:
    pass


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
        n_dimensions: int = 1,
    ) -> None:
        self.n_dimensions = n_dimensions

        if mean is None:
            self.mean = np.zeros(n_dimensions)
        else:
            self.mean = np.asarray(mean)
            self.n_dimensions = len(self.mean)

        if cov is None:
            self.cov = np.eye(self.n_dimensions)
        else:
            self.cov = np.asarray(cov)

        self.estimate_mean = estimate_mean
        self.estimate_cov = estimate_cov

        self._update_precision()

    def _update_precision(self) -> None:
        """Update precision matrix and normalizing constant."""
        self._precision = np.linalg.inv(self.cov)
        self._log_det = np.linalg.slogdet(self.cov)[1]
        self._log_norm = -0.5 * (self.n_dimensions * np.log(2 * np.pi) + self._log_det)

    def log_density(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        theta = np.atleast_2d(theta)
        diff = theta - self.mean
        mahal = np.sum(diff @ self._precision * diff, axis=1)
        return self._log_norm - 0.5 * mahal

    def update(
        self,
        theta_points: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        if not self.estimate_mean and not self.estimate_cov:
            return

        weights = weights / weights.sum()

        if self.estimate_mean:
            self.mean = np.sum(weights[:, None] * theta_points, axis=0)

        if self.estimate_cov:
            diff = theta_points - self.mean
            self.cov = np.sum(
                weights[:, None, None] * (diff[:, :, None] * diff[:, None, :]),
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
        self.n_bins = n_bins
        if initial_probs is not None:
            self._probs = np.asarray(initial_probs)
            self._probs = self._probs / self._probs.sum()
            self.n_bins = len(self._probs)
        else:
            self._probs = None

    def _initialize(self, n_points: int) -> None:
        """Initialize with uniform distribution."""
        if self._probs is None or len(self._probs) != n_points:
            self.n_bins = n_points
            self._probs = np.ones(n_points) / n_points

    def log_density(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._probs is None:
            raise ValueError("Histogram not initialized. Call update() first.")
        return np.log(np.clip(self._probs, 1e-300, None))

    def update(
        self,
        theta_points: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        self._initialize(len(weights))
        self._probs = weights / weights.sum()
        self._probs = np.clip(self._probs, PROB_EPSILON, None)
        self._probs = self._probs / self._probs.sum()

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
        self.n_bins = n_bins
        self.extrapolation_factor = extrapolation_factor
        self._probs: NDArray[np.float64] | None = None
        self._theta_points: NDArray[np.float64] | None = None

        if initial_probs is not None:
            self._probs = np.asarray(initial_probs)
            self._probs = self._probs / self._probs.sum()
            self.n_bins = len(self._probs)

    def _initialize(
        self, n_points: int, theta_points: NDArray[np.float64] | None = None
    ) -> None:
        """Initialize with uniform distribution."""
        if self._probs is None or len(self._probs) != n_points:
            self.n_bins = n_points
            self._probs = np.ones(n_points) / n_points
        if theta_points is not None:
            self._theta_points = theta_points

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
        return np.log(np.clip(self._probs, 1e-300, None))

    def update(
        self,
        theta_points: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        self._initialize(len(weights), theta_points)

        raw_probs = weights / weights.sum()
        raw_probs = np.clip(raw_probs, PROB_EPSILON, None)

        self._probs = self._extrapolate_tails(raw_probs)

        self._probs = np.clip(self._probs, PROB_EPSILON, None)
        self._probs = self._probs / self._probs.sum()

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
        self.degree = degree

        if coefficients is not None:
            self._coeffs = np.asarray(coefficients)
        else:
            self._coeffs = np.zeros(degree + 1)
            self._coeffs[0] = 1.0

        self._normalize_coefficients()

    def _normalize_coefficients(self) -> None:
        """Normalize coefficients so density integrates to 1.

        For Hermite polynomial expansion f(x) = phi(x) * g(x)^2 where
        g(x) = sum_k c_k * H_k(x), the integral is sum_k c_k^2 * k!
        due to orthogonality of Hermite polynomials.
        """
        factorials = np.array([np.math.factorial(k) for k in range(self.degree + 1)])
        norm_sq = np.sum(self._coeffs**2 * factorials)
        if norm_sq > 0:
            self._coeffs = self._coeffs / np.sqrt(norm_sq)

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
        theta = np.atleast_1d(theta)
        if theta.ndim == 2:
            theta = theta[:, 0]

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
        theta = theta_points
        if theta.ndim == 2:
            theta = theta[:, 0]

        weights = weights / weights.sum()

        H = self._hermite_polynomials(theta)

        phi = stats.norm.pdf(theta)

        target = np.sqrt(weights / (phi + 1e-300))

        try:
            W = np.diag(phi)
            HtWH = H.T @ W @ H
            HtWy = H.T @ W @ target

            reg = REGULARIZATION_EPSILON * np.eye(self.degree + 1)
            self._coeffs = np.linalg.solve(HtWH + reg, HtWy)

            if self._coeffs[0] < 0:
                self._coeffs = -self._coeffs

        except np.linalg.LinAlgError:
            pass

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
        self.n_components = n_components

        if means is not None:
            self.means = np.asarray(means)
        else:
            self.means = np.linspace(-1.5, 1.5, n_components)

        if variances is not None:
            self.variances = np.asarray(variances)
        else:
            self.variances = np.ones(n_components) * 0.5

        if weights is not None:
            self.weights = np.asarray(weights)
            self.weights = self.weights / self.weights.sum()
        else:
            self.weights = np.ones(n_components) / n_components

    def log_density(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        theta = np.atleast_1d(theta)
        if theta.ndim == 2:
            theta = theta[:, 0]

        log_components = np.zeros((len(theta), self.n_components))

        for k in range(self.n_components):
            log_components[:, k] = np.log(self.weights[k]) + stats.norm.logpdf(
                theta, self.means[k], np.sqrt(self.variances[k])
            )

        from mirt.utils.numeric import logsumexp

        return logsumexp(log_components, axis=1)

    def update(
        self,
        theta_points: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        """Update mixture parameters via EM."""
        theta = theta_points
        if theta.ndim == 2:
            theta = theta[:, 0]

        weights = weights / weights.sum()

        resp = np.zeros((len(theta), self.n_components))
        for k in range(self.n_components):
            resp[:, k] = self.weights[k] * stats.norm.pdf(
                theta, self.means[k], np.sqrt(self.variances[k])
            )
        resp = resp / (resp.sum(axis=1, keepdims=True) + 1e-300)

        weighted_resp = resp * weights[:, None]

        for k in range(self.n_components):
            nk = weighted_resp[:, k].sum()
            if nk > PROB_EPSILON:
                self.means[k] = np.sum(weighted_resp[:, k] * theta) / nk
                self.variances[k] = (
                    np.sum(weighted_resp[:, k] * (theta - self.means[k]) ** 2) / nk
                )
                self.variances[k] = max(self.variances[k], 0.01)
                self.weights[k] = nk

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
        self._log_density_func = log_density_func
        self._update_func = update_func
        self._n_params = n_params

    def log_density(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._log_density_func(theta)

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

    if density_type.lower() not in density_types:
        raise ValueError(
            f"Unknown density type: {density_type}. "
            f"Choose from: {list(density_types.keys())}"
        )

    return density_types[density_type.lower()](**kwargs)
