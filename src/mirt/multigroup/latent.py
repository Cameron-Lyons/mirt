from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON, REGULARIZATION_EPSILON

if TYPE_CHECKING:
    pass


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
    _log_det: float = field(init=False, repr=False)
    _log_norm: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float64)
        self.cov = np.asarray(self.cov, dtype=np.float64)
        if self.is_reference:
            self.estimate_mean = False
            self.estimate_cov = False
        self._update_precision()

    def _update_precision(self) -> None:
        """Update precision matrix and normalizing constant."""
        n_factors = len(self.mean)
        try:
            self._precision = np.linalg.inv(self.cov)
        except np.linalg.LinAlgError:
            self._precision = np.linalg.pinv(self.cov)
        sign, log_det = np.linalg.slogdet(self.cov)
        self._log_det = log_det if sign > 0 else -np.inf
        self._log_norm = -0.5 * (n_factors * np.log(2 * np.pi) + self._log_det)

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
        theta = np.atleast_2d(theta)
        diff = theta - self.mean
        mahal = np.sum(diff @ self._precision * diff, axis=1)
        return self._log_norm - 0.5 * mahal

    @property
    def n_factors(self) -> int:
        """Number of latent factors."""
        return len(self.mean)

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
        if n_groups < 2:
            raise ValueError("n_groups must be at least 2")
        if n_factors < 1:
            raise ValueError("n_factors must be at least 1")
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
        if group_idx < 0 or group_idx >= self.n_groups:
            raise IndexError(f"group_idx {group_idx} out of range [0, {self.n_groups})")
        return self.distributions[group_idx].log_density(theta)

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
        if group_idx < 0 or group_idx >= self.n_groups:
            raise IndexError(f"group_idx {group_idx} out of range [0, {self.n_groups})")

        dist = self.distributions[group_idx]
        if dist.is_reference:
            return

        weights_sum = weights.sum()
        if weights_sum < PROB_EPSILON:
            return

        weights_norm = weights / weights_sum

        if dist.estimate_mean:
            dist.mean = np.sum(weights_norm[:, None] * theta_points, axis=0)

        if dist.estimate_cov:
            diff = theta_points - dist.mean
            dist.cov = np.sum(
                weights_norm[:, None, None] * (diff[:, :, None] * diff[:, None, :]),
                axis=0,
            )
            dist.cov = (dist.cov + dist.cov.T) / 2
            dist.cov += REGULARIZATION_EPSILON * np.eye(self.n_factors)

        dist._update_precision()

    def get_group_mean(self, group_idx: int) -> NDArray[np.float64]:
        """Get mean for a specific group."""
        return self.distributions[group_idx].mean.copy()

    def get_group_cov(self, group_idx: int) -> NDArray[np.float64]:
        """Get covariance for a specific group."""
        return self.distributions[group_idx].cov.copy()

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
        if group_idx < 0 or group_idx >= self.n_groups:
            raise IndexError(f"group_idx {group_idx} out of range [0, {self.n_groups})")

        dist = self.distributions[group_idx]
        if dist.is_reference:
            raise ValueError("Cannot modify reference group distribution")

        if mean is not None:
            dist.mean = np.asarray(mean, dtype=np.float64)
        if cov is not None:
            dist.cov = np.asarray(cov, dtype=np.float64)
        dist._update_precision()

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
            result[g] = {
                "mean": dist.mean.copy(),
                "cov": dist.cov.copy(),
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
