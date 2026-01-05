from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.special import roots_hermite


class GaussHermiteQuadrature:
    def __init__(
        self,
        n_points: int = 21,
        n_dimensions: int = 1,
        mean: NDArray[np.float64] | None = None,
        cov: NDArray[np.float64] | None = None,
    ) -> None:
        if n_points < 1:
            raise ValueError("n_points must be at least 1")
        if n_dimensions < 1:
            raise ValueError("n_dimensions must be at least 1")

        self.n_points = n_points
        self.n_dimensions = n_dimensions

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

        self._nodes, self._weights = self._compute_quadrature()

    def _compute_quadrature(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        nodes_1d, weights_1d = roots_hermite(self.n_points)

        nodes_1d = nodes_1d * np.sqrt(2)
        weights_1d = weights_1d / np.sqrt(np.pi)

        if self.n_dimensions == 1:
            nodes = nodes_1d.reshape(-1, 1)
            weights = weights_1d

            if not np.allclose(self._mean, 0) or not np.allclose(self._cov, 1):
                std = np.sqrt(self._cov[0, 0])
                nodes = nodes * std + self._mean[0]

            return nodes, weights

        n_total = self.n_points**self.n_dimensions

        grids = [nodes_1d] * self.n_dimensions
        mesh = np.meshgrid(*grids, indexing="ij")
        nodes = np.column_stack([g.ravel() for g in mesh])

        weight_grids = [weights_1d] * self.n_dimensions
        weight_mesh = np.meshgrid(*weight_grids, indexing="ij")
        weights = np.ones(n_total)
        for wg in weight_mesh:
            weights *= wg.ravel()

        if not np.allclose(self._mean, 0) or not np.allclose(
            self._cov, np.eye(self.n_dimensions)
        ):
            L = np.linalg.cholesky(self._cov)
            nodes = nodes @ L.T + self._mean

        return nodes, weights

    @property
    def nodes(self) -> NDArray[np.float64]:
        return self._nodes.copy()

    @property
    def weights(self) -> NDArray[np.float64]:
        return self._weights.copy()

    @property
    def n_total_points(self) -> int:
        return len(self._weights)

    def update_distribution(
        self,
        mean: NDArray[np.float64] | None = None,
        cov: NDArray[np.float64] | None = None,
    ) -> None:
        if mean is not None:
            self._mean = np.asarray(mean)
        if cov is not None:
            self._cov = np.asarray(cov)

        self._nodes, self._weights = self._compute_quadrature()

    def integrate(
        self,
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ) -> float:
        values = func(self._nodes)
        return float(np.sum(self._weights * values))

    def __repr__(self) -> str:
        return (
            f"GaussHermiteQuadrature(n_points={self.n_points}, "
            f"n_dimensions={self.n_dimensions}, "
            f"n_total_points={self.n_total_points})"
        )
