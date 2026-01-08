"""Non-parametric and semi-parametric IRT models.

These models relax the parametric assumptions of standard IRT models,
allowing more flexible item response functions. This includes:
- Monotonic spline models
- Monotonic polynomial models
- Kernel-smoothed IRFs

References:
    Ramsay, J. O. (1991). Kernel smoothing approaches to nonparametric
        item characteristic curve estimation. Psychometrika, 56(4), 611-630.

    Woods, C. M. (2006). Ramsay-curve item response theory (RC-IRT) to detect
        and correct for nonnormal latent variables. Psychological Methods.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt.models.base import DichotomousItemModel


class MonotonicSplineModel(DichotomousItemModel):
    """Monotonic Spline Item Response Model.

    Uses I-splines (integrated B-splines) to create monotonically
    increasing item response functions without assuming a specific
    parametric form.

    Parameters
    ----------
    n_items : int
        Number of items
    n_knots : int
        Number of interior knots for spline
    degree : int
        Degree of spline (default 3 for cubic)
    item_names : list of str, optional
        Names for items

    Notes
    -----
    The IRF is defined as:
        P(X=1|θ) = c + (d - c) * sum_k w_k * I_k(θ)

    where I_k are I-spline basis functions and w_k >= 0 ensures monotonicity.
    """

    model_name = "MonotonicSpline"
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_knots: int = 4,
        degree: int = 3,
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("Spline model only supports unidimensional analysis")
        super().__init__(n_items, n_factors=1, item_names=item_names)

        self.n_knots = n_knots
        self.degree = degree
        self._n_basis = n_knots + degree + 1

    def _initialize_parameters(self) -> None:
        self._parameters["log_weights"] = np.zeros((self.n_items, self._n_basis))

        self._parameters["lower"] = np.zeros(self.n_items)
        self._parameters["upper"] = np.ones(self.n_items)

        knots = np.linspace(-3, 3, self.n_knots)
        self._knots = knots

    @property
    def weights(self) -> NDArray[np.float64]:
        """Non-negative spline weights."""
        return np.exp(self._parameters["log_weights"])

    @property
    def lower(self) -> NDArray[np.float64]:
        return self._parameters["lower"]

    @property
    def upper(self) -> NDArray[np.float64]:
        return self._parameters["upper"]

    def _ispline_basis(
        self,
        theta: NDArray[np.float64],
        knot_idx: int,
    ) -> NDArray[np.float64]:
        """Compute I-spline basis function at given theta values.

        I-splines are integrated B-splines and are monotonically increasing.
        """
        theta = np.asarray(theta).ravel()

        all_knots = np.concatenate(
            [
                np.full(self.degree + 1, -4),
                self._knots,
                np.full(self.degree + 1, 4),
            ]
        )

        from scipy.interpolate import BSpline

        c = np.zeros(len(all_knots) - self.degree - 1)
        if knot_idx < len(c):
            c[knot_idx] = 1

        bspline = BSpline(all_knots, c, self.degree, extrapolate=True)

        ispline_vals = np.zeros_like(theta)

        for i, t in enumerate(theta):
            grid = np.linspace(-4, t, 50)
            b_vals = bspline(grid)
            ispline_vals[i] = np.trapz(b_vals, grid)

        max_val = np.trapz(bspline(np.linspace(-4, 4, 100)), np.linspace(-4, 4, 100))
        if max_val > 0:
            ispline_vals = ispline_vals / max_val

        return np.clip(ispline_vals, 0, 1)

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()
        n_persons = len(theta_1d)

        c = self._parameters["lower"]
        d = self._parameters["upper"]
        w = self.weights

        if item_idx is not None:
            p_star = np.zeros(n_persons)
            for k in range(self._n_basis):
                basis = self._ispline_basis(theta_1d, k)
                p_star += w[item_idx, k] * basis

            p_star = p_star / (np.sum(w[item_idx]) + 1e-10)
            return c[item_idx] + (d[item_idx] - c[item_idx]) * p_star

        probs = np.zeros((n_persons, self.n_items))
        for j in range(self.n_items):
            p_star = np.zeros(n_persons)
            for k in range(self._n_basis):
                basis = self._ispline_basis(theta_1d, k)
                p_star += w[j, k] * basis

            p_star = p_star / (np.sum(w[j]) + 1e-10)
            probs[:, j] = c[j] + (d[j] - c[j]) * p_star

        return probs

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        p = self.probability(theta, item_idx)

        h = 1e-5
        p_plus = self.probability(theta + h, item_idx)
        p_minus = self.probability(theta - h, item_idx)

        dp = (p_plus - p_minus) / (2 * h)

        return (dp**2) / (p * (1 - p) + 1e-10)


class MonotonicPolynomialModel(DichotomousItemModel):
    """Monotonic Polynomial Item Response Model.

    Uses Bernstein polynomials to create monotonically increasing IRFs.
    Bernstein polynomials with non-negative coefficients guarantee monotonicity.

    Parameters
    ----------
    n_items : int
        Number of items
    degree : int
        Polynomial degree (higher = more flexible)
    item_names : list of str, optional
        Names for items

    Notes
    -----
    The IRF uses Bernstein polynomial basis:
        P(X=1|θ) = c + (d - c) * sum_k w_k * B_k,n(g(θ))

    where B_k,n are Bernstein basis polynomials, w_k >= 0 for monotonicity,
    and g(θ) maps theta to [0, 1].

    References
    ----------
    Liang, L., & Browne, M. W. (2015). A quasi-parametric method for
        fitting flexible item response functions. Journal of Educational
        and Behavioral Statistics.
    """

    model_name = "MonotonicPolynomial"
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        degree: int = 5,
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("Polynomial model only supports unidimensional")
        super().__init__(n_items, n_factors=1, item_names=item_names)

        self.degree = degree

    def _initialize_parameters(self) -> None:
        self._parameters["log_coefficients"] = np.zeros((self.n_items, self.degree + 1))

        self._parameters["location"] = np.zeros(self.n_items)
        self._parameters["scale"] = np.ones(self.n_items)

        self._parameters["lower"] = np.zeros(self.n_items)
        self._parameters["upper"] = np.ones(self.n_items)

    @property
    def coefficients(self) -> NDArray[np.float64]:
        """Non-negative polynomial coefficients."""
        return np.exp(self._parameters["log_coefficients"])

    def _bernstein_basis(
        self,
        t: NDArray[np.float64],
        k: int,
        n: int,
    ) -> NDArray[np.float64]:
        """Compute Bernstein basis polynomial B_{k,n}(t)."""
        from scipy.special import comb

        t = np.clip(t, 0, 1)
        return comb(n, k) * (t**k) * ((1 - t) ** (n - k))

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()
        n_persons = len(theta_1d)

        loc = self._parameters["location"]
        scale = self._parameters["scale"]
        c = self._parameters["lower"]
        d = self._parameters["upper"]
        w = self.coefficients

        if item_idx is not None:
            z = scale[item_idx] * (theta_1d - loc[item_idx])
            t = 1.0 / (1.0 + np.exp(-z))

            p_star = np.zeros(n_persons)
            for k in range(self.degree + 1):
                basis = self._bernstein_basis(t, k, self.degree)
                p_star += w[item_idx, k] * basis

            p_star = p_star / (np.sum(w[item_idx]) + 1e-10)

            return c[item_idx] + (d[item_idx] - c[item_idx]) * p_star

        probs = np.zeros((n_persons, self.n_items))
        for j in range(self.n_items):
            z = scale[j] * (theta_1d - loc[j])
            t = 1.0 / (1.0 + np.exp(-z))

            p_star = np.zeros(n_persons)
            for k in range(self.degree + 1):
                basis = self._bernstein_basis(t, k, self.degree)
                p_star += w[j, k] * basis

            p_star = p_star / (np.sum(w[j]) + 1e-10)
            probs[:, j] = c[j] + (d[j] - c[j]) * p_star

        return probs

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        p = self.probability(theta, item_idx)

        h = 1e-5
        p_plus = self.probability(theta + h, item_idx)
        p_minus = self.probability(theta - h, item_idx)

        dp = (p_plus - p_minus) / (2 * h)

        return (dp**2) / (p * (1 - p) + 1e-10)


class KernelSmoothingModel(DichotomousItemModel):
    """Kernel-Smoothed Item Response Model.

    Non-parametric IRF estimation using kernel smoothing.
    The IRF is estimated by smoothing observed proportions correct
    across the theta continuum.

    Parameters
    ----------
    n_items : int
        Number of items
    bandwidth : float
        Kernel bandwidth (larger = smoother)
    item_names : list of str, optional
        Names for items

    Notes
    -----
    This model requires theta estimates from another model and observed
    responses for calibration. The fitted IRF is then used for new
    examinees.
    """

    model_name = "KernelSmoothing"
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        bandwidth: float = 0.5,
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("Kernel smoothing only supports unidimensional")
        super().__init__(n_items, n_factors=1, item_names=item_names)

        self.bandwidth = bandwidth
        self._theta_grid: NDArray[np.float64] | None = None
        self._irf_values: NDArray[np.float64] | None = None

    def _initialize_parameters(self) -> None:
        self._theta_grid = np.linspace(-4, 4, 81)
        self._irf_values = np.full((self.n_items, len(self._theta_grid)), 0.5)

    def calibrate(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> None:
        """Calibrate IRFs using kernel smoothing.

        Parameters
        ----------
        responses : ndarray
            Response matrix (n_persons, n_items)
        theta : ndarray
            Ability estimates for each person
        """
        responses = np.asarray(responses)
        theta = np.asarray(theta).ravel()

        grid = self._theta_grid

        for j in range(self.n_items):
            valid = responses[:, j] >= 0
            resp_j = responses[valid, j]
            theta_j = theta[valid]

            for g, t in enumerate(grid):
                weights = np.exp(-0.5 * ((theta_j - t) / self.bandwidth) ** 2)
                weights_sum = np.sum(weights) + 1e-10

                self._irf_values[j, g] = np.sum(weights * resp_j) / weights_sum

        self._is_fitted = True

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        if self._theta_grid is None or self._irf_values is None:
            raise ValueError("Model must be calibrated before computing probabilities")

        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()
        n_persons = len(theta_1d)

        if item_idx is not None:
            return np.interp(theta_1d, self._theta_grid, self._irf_values[item_idx])

        probs = np.zeros((n_persons, self.n_items))
        for j in range(self.n_items):
            probs[:, j] = np.interp(theta_1d, self._theta_grid, self._irf_values[j])

        return probs

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        p = self.probability(theta, item_idx)

        h = 0.01
        p_plus = self.probability(theta + h, item_idx)
        p_minus = self.probability(theta - h, item_idx)

        dp = (p_plus - p_minus) / (2 * h)

        return (dp**2) / (p * (1 - p) + 1e-10)
