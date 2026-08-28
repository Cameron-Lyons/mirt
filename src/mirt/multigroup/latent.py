from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from mirt._prior_mass import gaussian_log_quadrature_mass
from mirt.constants import PROB_EPSILON, REGULARIZATION_EPSILON


def _positive_integer(value: int, name: str, minimum: int = 1) -> int:
    """Return a validated positive integer configuration value."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        qualifier = f"at least {minimum}" if minimum > 1 else "positive"
        raise ValueError(f"{name} must be an integer that is {qualifier}")
    return int(value)


def _boolean(value: bool, name: str) -> bool:
    """Return a validated Boolean flag."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _group_index(group_idx: int, n_groups: int) -> int:
    """Return a validated group index without accepting Boolean aliases."""
    if isinstance(group_idx, (bool, np.bool_)) or not isinstance(
        group_idx, (int, np.integer)
    ):
        raise TypeError("group_idx must be an integer")
    index = int(group_idx)
    if index < 0 or index >= n_groups:
        raise IndexError(f"group_idx {index} out of range [0, {n_groups})")
    return index


def _validated_gaussian_parameters(
    mean: NDArray[np.float64],
    cov: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Validate and copy a finite, nondegenerate Gaussian specification."""
    mean_array = np.asarray(mean, dtype=np.float64)
    if mean_array.ndim != 1 or mean_array.size == 0:
        raise ValueError("mean must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(mean_array)):
        raise ValueError("mean must contain only finite values")

    covariance = np.asarray(cov, dtype=np.float64)
    expected_shape = (mean_array.size, mean_array.size)
    if covariance.shape != expected_shape:
        raise ValueError(f"cov must have shape {expected_shape}")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("cov must contain only finite values")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        raise ValueError("cov must be symmetric")

    covariance = (covariance + covariance.T) * 0.5
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("cov must be positive definite") from exc
    return mean_array.copy(), covariance.copy(), cholesky


def _theta_points(
    theta: NDArray[np.float64],
    n_factors: int,
) -> NDArray[np.float64]:
    """Normalize scalar, vector, and matrix theta input to point rows."""
    points = np.asarray(theta, dtype=np.float64)
    if points.ndim == 0:
        if n_factors != 1:
            raise ValueError(f"theta must have {n_factors} columns")
        points = points.reshape(1, 1)
    elif points.ndim == 1:
        if n_factors == 1:
            points = points[:, None]
        elif points.shape == (n_factors,):
            points = points[None, :]
        else:
            raise ValueError(f"theta must have shape (n_points, {n_factors})")
    elif points.ndim != 2 or points.shape[1] != n_factors:
        raise ValueError(f"theta must have shape (n_points, {n_factors})")

    if not np.all(np.isfinite(points)):
        raise ValueError("theta must contain only finite values")
    return points


def _posterior_weights(
    weights: NDArray[np.float64],
    n_points: int,
) -> NDArray[np.float64]:
    """Validate posterior counts and return stable normalized weights."""
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (n_points,):
        raise ValueError(f"weights must have shape ({n_points},)")
    if not np.all(np.isfinite(values)):
        raise ValueError("weights must contain only finite values")
    if np.any(values < 0.0):
        raise ValueError("weights must be non-negative")

    with np.errstate(over="ignore"):
        total = float(np.sum(values))
    if total < PROB_EPSILON:
        return np.empty(0, dtype=np.float64)
    if np.isfinite(total):
        return values / total

    maximum = float(np.max(values))
    scaled = values / maximum
    return scaled / np.sum(scaled)


@dataclass
class GroupLatentDistribution:
    """Latent distribution specification for one group.

    Attributes
    ----------
    mean : ndarray
        Mean vector, shape (n_factors,).
    cov : ndarray
        Covariance matrix, shape (n_factors, n_factors).
    is_reference : bool
        Whether this is the reference group with fixed parameters.
    estimate_mean : bool
        Whether to estimate the mean during EM.
    estimate_cov : bool
        Whether to estimate the covariance during EM.
    """

    mean: NDArray[np.float64]
    cov: NDArray[np.float64]
    is_reference: bool = False
    estimate_mean: bool = True
    estimate_cov: bool = True
    _precision: NDArray[np.float64] = field(init=False, repr=False)
    _cov_snapshot: NDArray[np.float64] = field(init=False, repr=False)
    _log_det: float = field(init=False, repr=False)
    _log_norm: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.is_reference = _boolean(self.is_reference, "is_reference")
        self.estimate_mean = _boolean(self.estimate_mean, "estimate_mean")
        self.estimate_cov = _boolean(self.estimate_cov, "estimate_cov")
        if self.is_reference:
            self.estimate_mean = False
            self.estimate_cov = False
        self._update_precision()

    def _set_parameters(
        self,
        mean: NDArray[np.float64],
        cov: NDArray[np.float64],
    ) -> None:
        """Validate and atomically install Gaussian parameters and caches."""
        mean, covariance, cholesky = _validated_gaussian_parameters(mean, cov)
        identity = np.eye(mean.size, dtype=np.float64)
        precision = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, identity))
        log_det = float(2.0 * np.sum(np.log(np.diag(cholesky))))
        log_norm = -0.5 * (mean.size * np.log(2 * np.pi) + log_det)

        self.mean = mean
        self.cov = covariance
        self._precision = precision
        self._cov_snapshot = covariance.copy()
        self._log_det = log_det
        self._log_norm = log_norm

    def _update_precision(self) -> None:
        """Validate public fields and atomically refresh matrix caches."""
        self._set_parameters(self.mean, self.cov)

    def _validated_state(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Validate public fields and refresh a cache after direct mutation."""
        mean = np.asarray(self.mean, dtype=np.float64)
        covariance = np.asarray(self.cov, dtype=np.float64)
        expected_shape = (mean.size, mean.size) if mean.ndim == 1 else (-1, -1)
        if (
            mean.ndim != 1
            or mean.size == 0
            or not np.all(np.isfinite(mean))
            or covariance.shape != expected_shape
            or not np.all(np.isfinite(covariance))
        ):
            _validated_gaussian_parameters(mean, covariance)
        if not np.array_equal(covariance, self._cov_snapshot):
            self._update_precision()
            return self.mean, self.cov
        return mean, covariance

    def log_density(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute log density at theta points.

        Parameters
        ----------
        theta : ndarray
            Theta values, shape (n_points, n_factors).

        Returns
        -------
        ndarray
            Log density values, shape (n_points,).
        """
        mean, _ = self._validated_state()
        points = _theta_points(theta, mean.size)
        diff = points - mean
        mahal = np.sum(diff @ self._precision * diff, axis=1)
        mahal = np.maximum(mahal, 0.0)
        return self._log_norm - 0.5 * mahal

    def log_quadrature_mass(
        self,
        theta: NDArray[np.float64],
        quadrature_weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return normalized group-prior masses on standard-normal GH nodes."""
        mean, covariance = self._validated_state()
        points = _theta_points(theta, mean.size)
        return gaussian_log_quadrature_mass(
            points, quadrature_weights, mean, covariance
        )

    @property
    def n_factors(self) -> int:
        """Number of latent factors."""
        mean, _ = self._validated_state()
        return mean.size

    @property
    def n_free_parameters(self) -> int:
        """Count free parameters in this distribution."""
        n = 0
        if self.estimate_mean:
            n += self.n_factors
        if self.estimate_cov:
            n += self.n_factors * (self.n_factors + 1) // 2
        return n

    def copy(self) -> GroupLatentDistribution:
        """Create a deep copy."""
        return GroupLatentDistribution(
            mean=self.mean.copy(),
            cov=self.cov.copy(),
            is_reference=self.is_reference,
            estimate_mean=self.estimate_mean,
            estimate_cov=self.estimate_cov,
        )


class MultigroupLatentDensity:
    """Collection of group-specific latent distributions.

    This class manages latent distributions for multiple groups, with
    one reference group having fixed mean=0 and cov=I for identification.

    Parameters
    ----------
    n_groups : int
        Number of groups.
    n_factors : int
        Number of latent factors (dimensions).
    reference_group : int
        Index of the reference group (0-indexed).
    """

    def __init__(
        self,
        n_groups: int,
        n_factors: int = 1,
        reference_group: int = 0,
    ) -> None:
        n_groups = _positive_integer(n_groups, "n_groups", minimum=2)
        n_factors = _positive_integer(n_factors, "n_factors")
        if isinstance(reference_group, (bool, np.bool_)) or not isinstance(
            reference_group, (int, np.integer)
        ):
            raise TypeError("reference_group must be an integer")
        reference_group = int(reference_group)
        if reference_group < 0 or reference_group >= n_groups:
            raise ValueError(
                f"reference_group must be in [0, {n_groups}), got {reference_group}"
            )

        self.n_groups = n_groups
        self.n_factors = n_factors
        self.reference_group = reference_group

        self.distributions: list[GroupLatentDistribution] = []
        for g in range(n_groups):
            is_ref = g == reference_group
            dist = GroupLatentDistribution(
                mean=np.zeros(n_factors),
                cov=np.eye(n_factors),
                is_reference=is_ref,
                estimate_mean=not is_ref,
                estimate_cov=not is_ref,
            )
            self.distributions.append(dist)

    def _distribution(self, group_idx: int) -> GroupLatentDistribution:
        """Return a distribution after consistently validating its index."""
        return self.distributions[_group_index(group_idx, self.n_groups)]

    def log_density(
        self,
        theta: NDArray[np.float64],
        group_idx: int,
    ) -> NDArray[np.float64]:
        """Compute log density for a specific group.

        Parameters
        ----------
        theta : ndarray
            Theta values, shape (n_points, n_factors).
        group_idx : int
            Group index.

        Returns
        -------
        ndarray
            Log density values, shape (n_points,).
        """
        return self._distribution(group_idx).log_density(theta)

    def log_density_all(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log densities for every group in one aligned matrix.

        Returns an array with shape ``(n_points, n_groups)``. A vector is
        interpreted as multiple points for a unidimensional density and as
        one point for a multidimensional density.
        """
        points = _theta_points(theta, self.n_factors)
        return np.column_stack(
            [distribution.log_density(points) for distribution in self.distributions]
        )

    def log_quadrature_mass(
        self,
        theta: NDArray[np.float64],
        quadrature_weights: NDArray[np.float64],
        group_idx: int,
    ) -> NDArray[np.float64]:
        """Return normalized prior masses for one group."""
        return self._distribution(group_idx).log_quadrature_mass(
            theta, quadrature_weights
        )

    def update(
        self,
        theta_points: NDArray[np.float64],
        weights: NDArray[np.float64],
        group_idx: int,
    ) -> None:
        """Update group distribution from posterior weights.

        Parameters
        ----------
        theta_points : ndarray
            Quadrature points, shape (n_quad, n_factors).
        weights : ndarray
            Posterior weights summed across persons, shape (n_quad,).
        group_idx : int
            Group index to update.
        """
        dist = self._distribution(group_idx)
        theta_array = _theta_points(theta_points, self.n_factors)
        weights_norm = _posterior_weights(weights, len(theta_array))
        if dist.is_reference or weights_norm.size == 0:
            return

        mean = dist.mean
        covariance = dist.cov

        if dist.estimate_mean:
            mean = weights_norm @ theta_array

        if dist.estimate_cov:
            diff = theta_array - mean
            covariance = diff.T @ (weights_norm[:, None] * diff)
            covariance = (covariance + covariance.T) * 0.5
            covariance += REGULARIZATION_EPSILON * np.eye(self.n_factors)

        dist._set_parameters(mean, covariance)

    def get_group_mean(self, group_idx: int) -> NDArray[np.float64]:
        """Get mean for a specific group."""
        mean, _ = self._distribution(group_idx)._validated_state()
        return mean.copy()

    def get_group_cov(self, group_idx: int) -> NDArray[np.float64]:
        """Get covariance for a specific group."""
        _, covariance = self._distribution(group_idx)._validated_state()
        return covariance.copy()

    def set_group_distribution(
        self,
        group_idx: int,
        mean: NDArray[np.float64] | None = None,
        cov: NDArray[np.float64] | None = None,
    ) -> None:
        """Set distribution parameters for a specific group.

        Parameters
        ----------
        group_idx : int
            Group index.
        mean : ndarray, optional
            New mean vector.
        cov : ndarray, optional
            New covariance matrix.
        """
        dist = self._distribution(group_idx)
        if dist.is_reference:
            raise ValueError("Cannot modify reference group distribution")

        candidate_mean = dist.mean if mean is None else mean
        candidate_cov = dist.cov if cov is None else cov
        dist._set_parameters(candidate_mean, candidate_cov)

    @property
    def n_parameters(self) -> int:
        """Total number of free parameters across all distributions."""
        return sum(d.n_free_parameters for d in self.distributions)

    def get_latent_parameters(self) -> dict[int, dict[str, NDArray[np.float64]]]:
        """Get all latent distribution parameters.

        Returns
        -------
        dict
            Dictionary mapping group index to dict with 'mean' and 'cov'.
        """
        result = {}
        for g, dist in enumerate(self.distributions):
            mean, covariance = dist._validated_state()
            result[g] = {
                "mean": mean.copy(),
                "cov": covariance.copy(),
                "is_reference": dist.is_reference,
            }
        return result

    def copy(self) -> MultigroupLatentDensity:
        """Create a deep copy."""
        new_density = MultigroupLatentDensity(
            n_groups=self.n_groups,
            n_factors=self.n_factors,
            reference_group=self.reference_group,
        )
        new_density.distributions = [d.copy() for d in self.distributions]
        return new_density
