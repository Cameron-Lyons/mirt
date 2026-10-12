"""Quadrature grids for marginal maximum likelihood estimation.

EM and Bock-Lieberman estimation integrate the item likelihoods over the
latent trait on a product grid of ``n_points ** n_dimensions`` nodes. Two
rules are available:

- :class:`RectangularQuadrature`, the equally spaced grid on ``[-6, 6]`` with
  normal-density weights that R mirt uses. Person likelihoods of long or
  highly discriminating tests are sharply peaked, and an evenly fine grid
  resolves them with far fewer nodes than Gauss-Hermite rules, which spend
  most of their nodes in the tails.
- :class:`GaussHermiteQuadrature`, which integrates smooth functions against
  the normal density more accurately when only a few points per dimension
  are affordable.

:func:`default_quadpts` gives R mirt's dimension-dependent point counts and
:func:`build_quadrature` selects the rule.
"""

from collections.abc import Callable
from numbers import Integral
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.special import roots_hermite

from mirt.exceptions import MirtValidationError

QuadratureRule: TypeAlias = Literal["auto", "rectangular", "gauss-hermite"]
QUADRATURE_RULES: tuple[str, ...] = ("auto", "rectangular", "gauss-hermite")

# Default EM points per latent dimension by number of factors, as R mirt's
# ``select_quadpts``. Six factors keep the smallest count.
_DEFAULT_QUADPTS_BY_FACTORS = {1: 61, 2: 31, 3: 15, 4: 9, 5: 7}
_MIN_DEFAULT_QUADPTS = 5
# Default grids may have at most this many nodes (R mirt's ``MAXQUAD``).
MAX_DEFAULT_GRID_NODES = 20_000
# Under ``"auto"``, grids with at least this many points per dimension are
# rectangular. Gauss-Hermite nodes are spaced about pi / sqrt(n) apart near
# the mode, against 12 / (n - 1) for the rectangular grid, so the
# rectangular grid is the finer one from about 15 points on.
_RECTANGULAR_MIN_POINTS = 15
# Bounds of the rectangular grid in every dimension (R mirt's ``theta_lim``).
RECTANGULAR_LIMITS = (-6.0, 6.0)


def default_quadpts(n_factors: int) -> int:
    """Return the default EM quadrature points per latent dimension.

    The counts are those of R mirt: 61 points for one factor, 31 for two, 15
    for three, 9 for four, 7 for five and 5 for six, so that every default
    grid has at most :data:`MAX_DEFAULT_GRID_NODES` nodes.

    Parameters
    ----------
    n_factors : int
        Number of latent dimensions.

    Returns
    -------
    int
        Quadrature points per dimension.

    Raises
    ------
    MirtValidationError
        If ``n_factors`` exceeds six, where even five points per dimension
        exceed the node limit. Fixed-grid quadrature is then impractical;
        Metropolis-Hastings Robbins-Monro (``fit_mirt(estimation="MHRM")``)
        or quasi-Monte Carlo EM (:class:`~mirt.estimation.mcem.QMCEMEstimator`)
        integrate such models instead.
    """
    n_factors = int(n_factors)
    n_points = _DEFAULT_QUADPTS_BY_FACTORS.get(n_factors, _MIN_DEFAULT_QUADPTS)
    if n_points**n_factors > MAX_DEFAULT_GRID_NODES:
        raise MirtValidationError(
            f"a quadrature grid for {n_factors} factors needs "
            f"{n_points**n_factors:,} nodes even with {n_points} points per "
            "dimension; fit the model with estimation='MHRM' or QMCEMEstimator, "
            "or pass n_quadpts explicitly",
            parameter="n_quadpts",
            value=None,
            expected=f"an explicit grid size, or at most {len(_DEFAULT_QUADPTS_BY_FACTORS) + 1} factors",
        )
    return n_points


def resolve_quadpts(n_quadpts: int | None, n_factors: int) -> int:
    """Validate an explicit grid size or return :func:`default_quadpts`.

    Parameters
    ----------
    n_quadpts : int or None
        Requested points per dimension, or ``None`` for the default.
    n_factors : int
        Number of latent dimensions.

    Returns
    -------
    int
        Quadrature points per dimension.

    Raises
    ------
    MirtValidationError
        If ``n_quadpts`` is not an integer of at least 5, or the default grid
        is too large (see :func:`default_quadpts`).
    """
    if n_quadpts is None:
        return default_quadpts(n_factors)
    return validate_quadpts(n_quadpts)


def validate_quadpts(n_quadpts: object) -> int:
    """Return an explicit grid size as an ``int`` of at least 5.

    Raises
    ------
    MirtValidationError
        If ``n_quadpts`` is not an integer, or is below 5.
    """
    if isinstance(n_quadpts, (bool, np.bool_)) or not isinstance(n_quadpts, Integral):
        raise MirtValidationError(
            "n_quadpts must be an integer or None",
            parameter="n_quadpts",
            value=n_quadpts,
            expected="integer >= 5, or None for the default",
        )
    if n_quadpts < _MIN_DEFAULT_QUADPTS:
        raise MirtValidationError(
            f"n_quadpts should be at least {_MIN_DEFAULT_QUADPTS}",
            parameter="n_quadpts",
            value=n_quadpts,
            expected=f">= {_MIN_DEFAULT_QUADPTS}",
        )
    return int(n_quadpts)


def validate_quadrature_rule(rule: object) -> QuadratureRule:
    """Return a validated quadrature rule name.

    Raises
    ------
    MirtValidationError
        If ``rule`` is not ``"auto"``, ``"rectangular"`` or
        ``"gauss-hermite"``.
    """
    if not isinstance(rule, str) or rule not in QUADRATURE_RULES:
        raise MirtValidationError(
            "quadrature must be 'auto', 'rectangular' or 'gauss-hermite'",
            parameter="quadrature",
            value=rule,
            expected=", ".join(repr(name) for name in QUADRATURE_RULES),
        )
    if rule == "rectangular":
        return "rectangular"
    return "gauss-hermite" if rule == "gauss-hermite" else "auto"


def resolve_quadrature_rule(
    rule: QuadratureRule, n_points: int
) -> Literal["rectangular", "gauss-hermite"]:
    """Return the rule that ``"auto"`` selects for ``n_points`` per dimension.

    ``"auto"`` is rectangular for grids of 15 or more points per dimension,
    which includes the defaults for up to three factors, and Gauss-Hermite
    for coarser grids. The other rules are returned unchanged.
    """
    rule = validate_quadrature_rule(rule)
    if rule != "auto":
        return rule
    return "rectangular" if n_points >= _RECTANGULAR_MIN_POINTS else "gauss-hermite"


def build_quadrature(
    n_points: int,
    n_dimensions: int,
    rule: QuadratureRule = "auto",
) -> "GaussHermiteQuadrature | RectangularQuadrature":
    """Build the standard-normal quadrature grid of an estimator.

    Parameters
    ----------
    n_points : int
        Points per dimension.
    n_dimensions : int
        Number of latent dimensions.
    rule : {"auto", "rectangular", "gauss-hermite"}, default="auto"
        Quadrature rule (see :func:`resolve_quadrature_rule`).

    Returns
    -------
    GaussHermiteQuadrature or RectangularQuadrature
        Grid whose weights integrate against the standard normal density.
    """
    if resolve_quadrature_rule(rule, n_points) == "rectangular":
        return RectangularQuadrature(n_points=n_points, n_dimensions=n_dimensions)
    return GaussHermiteQuadrature(n_points=n_points, n_dimensions=n_dimensions)


def _validate_grid_shape(
    n_points: int,
    n_dimensions: int,
    mean: NDArray[np.float64] | None,
    cov: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate grid sizes and return the mean and covariance arrays."""
    if n_points < 1:
        raise MirtValidationError(
            "n_points must be at least 1",
            parameter="n_points",
            value=n_points,
            expected=">= 1",
        )
    if n_dimensions < 1:
        raise MirtValidationError(
            "n_dimensions must be at least 1",
            parameter="n_dimensions",
            value=n_dimensions,
            expected=">= 1",
        )
    if mean is None:
        mean_array = np.zeros(n_dimensions)
    else:
        mean_array = np.asarray(mean)
        if mean_array.shape != (n_dimensions,):
            raise MirtValidationError(
                f"mean must have shape ({n_dimensions},)",
                parameter="mean",
                value=mean_array.shape,
                expected=f"({n_dimensions},)",
            )
    if cov is None:
        cov_array = np.eye(n_dimensions)
    else:
        cov_array = np.asarray(cov)
        if cov_array.shape != (n_dimensions, n_dimensions):
            raise MirtValidationError(
                f"cov must have shape ({n_dimensions}, {n_dimensions})",
                parameter="cov",
                value=cov_array.shape,
                expected=f"({n_dimensions}, {n_dimensions})",
            )
    return mean_array, cov_array


class GaussHermiteQuadrature:
    def __init__(
        self,
        n_points: int = 21,
        n_dimensions: int = 1,
        mean: NDArray[np.float64] | None = None,
        cov: NDArray[np.float64] | None = None,
    ) -> None:
        self._mean, self._cov = _validate_grid_shape(n_points, n_dimensions, mean, cov)
        self.n_points = n_points
        self.n_dimensions = n_dimensions

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


class RectangularQuadrature:
    """Equally spaced product grid with normal-density weights.

    Every dimension has ``n_points`` nodes spaced evenly on ``limits``, and
    each node of the product grid is weighted by the normal density with
    ``mean`` and ``cov`` there, normalized to sum to one. This is the grid of
    R mirt (``quadpts`` points on ``theta_lim = c(-6, 6)``). Latent densities
    reweight it by their density ratio to the standard normal, so an
    estimated density is evaluated at the fixed nodes, as in R.

    Parameters
    ----------
    n_points : int, default=61
        Points per dimension.
    n_dimensions : int, default=1
        Number of latent dimensions.
    mean : ndarray of shape (n_dimensions,), optional
        Mean of the weighting density; zero by default.
    cov : ndarray of shape (n_dimensions, n_dimensions), optional
        Covariance of the weighting density; the identity by default.
    limits : tuple of float, default=(-6.0, 6.0)
        Lower and upper node in every dimension.
    """

    def __init__(
        self,
        n_points: int = 61,
        n_dimensions: int = 1,
        mean: NDArray[np.float64] | None = None,
        cov: NDArray[np.float64] | None = None,
        limits: tuple[float, float] = RECTANGULAR_LIMITS,
    ) -> None:
        self._mean, self._cov = _validate_grid_shape(n_points, n_dimensions, mean, cov)
        low, high = (float(value) for value in limits)
        if not (np.isfinite(low) and np.isfinite(high) and low < high):
            raise MirtValidationError(
                "limits must be finite with lower < upper",
                parameter="limits",
                value=limits,
                expected="(lower, upper) with lower < upper",
            )
        self.n_points = n_points
        self.n_dimensions = n_dimensions
        self.limits = (low, high)
        self._nodes, self._weights = self._compute_quadrature()

    def _compute_quadrature(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        axis = np.linspace(*self.limits, self.n_points)
        mesh = np.meshgrid(*([axis] * self.n_dimensions), indexing="ij")
        nodes = np.column_stack([grid.ravel() for grid in mesh])
        difference = nodes - self._mean
        solved = np.linalg.solve(self._cov, difference.T).T
        log_density = -0.5 * np.sum(difference * solved, axis=1)
        weights = np.exp(log_density - log_density.max())
        return nodes, weights / weights.sum()

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
        """Reweight the fixed nodes by a normal density with new moments."""
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
            f"RectangularQuadrature(n_points={self.n_points}, "
            f"n_dimensions={self.n_dimensions}, "
            f"n_total_points={self.n_total_points})"
        )
