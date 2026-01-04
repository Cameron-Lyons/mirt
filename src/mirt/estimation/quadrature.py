"""Gauss-Hermite quadrature for numerical integration."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.special import roots_hermite


class GaussHermiteQuadrature:
    """Gauss-Hermite quadrature for integrating over normal distributions.

    This class provides nodes (quadrature points) and weights for
    numerically approximating integrals of the form:

        ∫ f(x) × φ(x) dx

    where φ(x) is the standard normal density. The approximation is:

        ∫ f(x) × φ(x) dx ≈ Σ w_i × f(x_i)

    For multidimensional integration, tensor products of 1D rules are used.

    Parameters
    ----------
    n_points : int, default=21
        Number of quadrature points per dimension.
    n_dimensions : int, default=1
        Number of dimensions (factors) for integration.
    mean : ndarray, optional
        Mean of the multivariate normal. Default is zeros.
    cov : ndarray, optional
        Covariance matrix. Default is identity.

    Attributes
    ----------
    nodes : ndarray of shape (n_total_points, n_dimensions)
        Quadrature nodes (points).
    weights : ndarray of shape (n_total_points,)
        Quadrature weights.
    n_points : int
        Number of points per dimension.
    n_dimensions : int
        Number of dimensions.

    Examples
    --------
    >>> quad = GaussHermiteQuadrature(n_points=21, n_dimensions=1)
    >>> print(quad.nodes.shape)
    (21, 1)
    >>> print(quad.weights.shape)
    (21,)

    >>> # Approximate E[X] where X ~ N(0,1) - should be 0
    >>> np.sum(quad.weights * quad.nodes.ravel())
    0.0

    >>> # Approximate E[X²] where X ~ N(0,1) - should be 1
    >>> np.sum(quad.weights * quad.nodes.ravel()**2)
    1.0
    """

    def __init__(
        self,
        n_points: int = 21,
        n_dimensions: int = 1,
        mean: Optional[NDArray[np.float64]] = None,
        cov: Optional[NDArray[np.float64]] = None,
    ) -> None:
        if n_points < 1:
            raise ValueError("n_points must be at least 1")
        if n_dimensions < 1:
            raise ValueError("n_dimensions must be at least 1")

        self.n_points = n_points
        self.n_dimensions = n_dimensions

        # Set default mean and covariance
        if mean is None:
            self._mean = np.zeros(n_dimensions)
        else:
            self._mean = np.asarray(mean)
            if self._mean.shape != (n_dimensions,):
                raise ValueError(f"mean must have shape ({n_dimensions},)")

        if cov is None:
            self._cov = np.eye(n_dimensions)
        else:
            self._cov = np.asarray(cov)
            if self._cov.shape != (n_dimensions, n_dimensions):
                raise ValueError(
                    f"cov must have shape ({n_dimensions}, {n_dimensions})"
                )

        # Compute quadrature nodes and weights
        self._nodes, self._weights = self._compute_quadrature()

    def _compute_quadrature(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute quadrature nodes and weights."""
        # Get 1D Gauss-Hermite nodes and weights
        # scipy's roots_hermite returns physicist's Hermite polynomials
        nodes_1d, weights_1d = roots_hermite(self.n_points)

        # Transform to probabilist's convention (standard normal)
        # Physicist: ∫ f(x) exp(-x²) dx
        # Probabilist: ∫ f(x) (1/√(2π)) exp(-x²/2) dx
        nodes_1d = nodes_1d * np.sqrt(2)
        weights_1d = weights_1d / np.sqrt(np.pi)

        if self.n_dimensions == 1:
            nodes = nodes_1d.reshape(-1, 1)
            weights = weights_1d

            # Transform for non-standard mean/variance
            if not np.allclose(self._mean, 0) or not np.allclose(self._cov, 1):
                std = np.sqrt(self._cov[0, 0])
                nodes = nodes * std + self._mean[0]
                # Weights don't change for location-scale transform

            return nodes, weights

        # Multidimensional: tensor product
        n_total = self.n_points ** self.n_dimensions

        # Create tensor product grid
        grids = [nodes_1d] * self.n_dimensions
        mesh = np.meshgrid(*grids, indexing='ij')
        nodes = np.column_stack([g.ravel() for g in mesh])

        # Tensor product of weights
        weight_grids = [weights_1d] * self.n_dimensions
        weight_mesh = np.meshgrid(*weight_grids, indexing='ij')
        weights = np.ones(n_total)
        for wg in weight_mesh:
            weights *= wg.ravel()

        # Transform for non-standard mean/covariance
        if not np.allclose(self._mean, 0) or not np.allclose(self._cov, np.eye(self.n_dimensions)):
            # Cholesky decomposition for covariance transformation
            L = np.linalg.cholesky(self._cov)
            nodes = nodes @ L.T + self._mean

        return nodes, weights

    @property
    def nodes(self) -> NDArray[np.float64]:
        """Quadrature nodes (points)."""
        return self._nodes.copy()

    @property
    def weights(self) -> NDArray[np.float64]:
        """Quadrature weights."""
        return self._weights.copy()

    @property
    def n_total_points(self) -> int:
        """Total number of quadrature points."""
        return len(self._weights)

    def update_distribution(
        self,
        mean: Optional[NDArray[np.float64]] = None,
        cov: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Update the mean and/or covariance and recompute quadrature.

        Parameters
        ----------
        mean : ndarray, optional
            New mean vector.
        cov : ndarray, optional
            New covariance matrix.
        """
        if mean is not None:
            self._mean = np.asarray(mean)
        if cov is not None:
            self._cov = np.asarray(cov)

        self._nodes, self._weights = self._compute_quadrature()

    def integrate(
        self,
        func: callable,
    ) -> float:
        """Integrate a function against the normal distribution.

        Approximates ∫ f(x) × φ(x; μ, Σ) dx

        Parameters
        ----------
        func : callable
            Function to integrate. Should accept array of shape
            (n_points, n_dimensions) and return array of shape (n_points,).

        Returns
        -------
        float
            Approximate integral value.
        """
        values = func(self._nodes)
        return float(np.sum(self._weights * values))

    def __repr__(self) -> str:
        return (
            f"GaussHermiteQuadrature(n_points={self.n_points}, "
            f"n_dimensions={self.n_dimensions}, "
            f"n_total_points={self.n_total_points})"
        )


def create_quadrature_grid(
    n_points: int = 21,
    n_dimensions: int = 1,
    theta_range: tuple[float, float] = (-6.0, 6.0),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create a simple rectangular quadrature grid.

    This is an alternative to Gauss-Hermite when uniform spacing is desired.

    Parameters
    ----------
    n_points : int
        Number of points per dimension.
    n_dimensions : int
        Number of dimensions.
    theta_range : tuple of float
        Range (min, max) for theta values.

    Returns
    -------
    nodes : ndarray of shape (n_total, n_dimensions)
        Grid points.
    weights : ndarray of shape (n_total,)
        Uniform weights (summing to 1).
    """
    # Create 1D grid
    points_1d = np.linspace(theta_range[0], theta_range[1], n_points)

    if n_dimensions == 1:
        nodes = points_1d.reshape(-1, 1)
        # Approximate weights from standard normal density
        from scipy.stats import norm
        weights = norm.pdf(points_1d)
        weights = weights / weights.sum()
        return nodes, weights

    # Multidimensional grid
    grids = [points_1d] * n_dimensions
    mesh = np.meshgrid(*grids, indexing='ij')
    nodes = np.column_stack([g.ravel() for g in mesh])

    # Weights from multivariate normal density
    from scipy.stats import multivariate_normal
    rv = multivariate_normal(mean=np.zeros(n_dimensions))
    weights = rv.pdf(nodes)
    weights = weights / weights.sum()

    return nodes, weights
