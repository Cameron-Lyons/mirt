"""Weighted Likelihood Estimation (WLE) scorer.

WLE (Warm's Weighted Likelihood Estimation) reduces the bias in ML estimates,
particularly at extreme ability levels. It adds a correction term based on
the first derivative of the test information function.

Reference:
    Warm, T. A. (1989). Weighted likelihood estimation of ability in item
    response theory. Psychometrika, 54(3), 427-450.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, minimize_scalar

from mirt.results.score_result import ScoreResult

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


class WLEScorer:
    """Weighted Likelihood Estimation scorer.

    WLE provides bias-reduced ability estimates by incorporating a correction
    term based on the ratio of the first derivative of test information to
    twice the test information. This correction pulls extreme estimates toward
    the center of the ability distribution.

    Parameters
    ----------
    bounds : tuple of float, optional
        Lower and upper bounds for theta search. Default is (-6.0, 6.0).
    tol : float, optional
        Tolerance for convergence. Default is 1e-6.

    Attributes
    ----------
    bounds : tuple of float
        The theta search bounds.
    tol : float
        Convergence tolerance.

    Notes
    -----
    WLE is recommended over ML when:
    - Sample sizes are small
    - Many examinees have extreme response patterns
    - Bias reduction is important for the application

    The WLE estimate maximizes the weighted likelihood:
        WL(theta) = L(theta) * sqrt(I(theta))

    where L(theta) is the likelihood and I(theta) is the test information.
    """

    def __init__(
        self,
        bounds: tuple[float, float] = (-6.0, 6.0),
        tol: float = 1e-6,
    ) -> None:
        self.bounds = bounds
        self.tol = tol

    def score(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> ScoreResult:
        """Compute WLE ability estimates.

        Parameters
        ----------
        model : BaseItemModel
            A fitted IRT model.
        responses : ndarray of shape (n_persons, n_items)
            Response matrix with integer responses. Missing values should be
            coded as negative integers.

        Returns
        -------
        ScoreResult
            Object containing theta estimates and standard errors.

        Raises
        ------
        ValueError
            If the model is not fitted.
        """
        if not model.is_fitted:
            raise ValueError("Model must be fitted before scoring")

        responses = np.asarray(responses)
        n_persons = responses.shape[0]
        n_factors = model.n_factors

        if n_factors > 1:
            return self._score_multidimensional(model, responses)

        theta_wle = np.zeros(n_persons)
        theta_se = np.zeros(n_persons)

        for i in range(n_persons):
            resp_i = responses[i]
            theta_wle[i], theta_se[i] = self._estimate_person(model, resp_i)

        return ScoreResult(
            theta=theta_wle,
            standard_error=theta_se,
            method="WLE",
        )

    def _estimate_person(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> tuple[float, float]:
        """Estimate theta for a single person using WLE."""

        valid_mask = responses >= 0
        if not valid_mask.any():
            return 0.0, np.inf

        def neg_weighted_log_likelihood(theta: float) -> float:
            theta_arr = np.array([[theta]])

            ll = model.log_likelihood(responses[None, :], theta_arr)[0]

            info = self._test_information(model, theta_arr, valid_mask)[0]

            if info > 1e-10:
                wl = ll + 0.5 * np.log(info)
            else:
                wl = ll

            return -wl

        result = minimize_scalar(
            neg_weighted_log_likelihood,
            bounds=self.bounds,
            method="bounded",
            options={"xatol": self.tol},
        )

        theta_hat = result.x

        theta_arr = np.array([[theta_hat]])
        info = self._test_information(model, theta_arr, valid_mask)[0]

        if info > 1e-10:
            se = 1.0 / np.sqrt(info)
        else:
            se = np.inf

        return theta_hat, se

    def _test_information(
        self,
        model: BaseItemModel,
        theta: NDArray[np.float64],
        valid_mask: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Compute test information at given theta values."""
        info = np.zeros(theta.shape[0])

        for j in range(model.n_items):
            if valid_mask[j]:
                info += model.information(theta, j)

        return info

    def _score_multidimensional(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> ScoreResult:
        """Estimate multidimensional WLE scores."""
        n_persons = responses.shape[0]
        n_factors = model.n_factors

        theta_wle = np.zeros((n_persons, n_factors))
        theta_se = np.zeros((n_persons, n_factors))

        for i in range(n_persons):
            resp_i = responses[i]
            valid_mask = resp_i >= 0

            if not valid_mask.any():
                theta_wle[i] = 0.0
                theta_se[i] = np.inf
                continue

            def neg_weighted_log_likelihood(theta_vec: NDArray[np.float64]) -> float:
                theta_arr = theta_vec.reshape(1, -1)
                ll = model.log_likelihood(resp_i[None, :], theta_arr)[0]
                info = self._test_information(model, theta_arr, valid_mask)[0]

                if info > 1e-10:
                    wl = ll + 0.5 * np.log(info)
                else:
                    wl = ll

                return -wl

            x0 = np.zeros(n_factors)

            result = minimize(
                neg_weighted_log_likelihood,
                x0,
                method="L-BFGS-B",
                bounds=[(self.bounds[0], self.bounds[1])] * n_factors,
                options={"ftol": self.tol},
            )

            theta_wle[i] = result.x

            theta_arr = result.x.reshape(1, -1)
            info = self._test_information(model, theta_arr, valid_mask)[0]

            if info > 1e-10:
                theta_se[i] = 1.0 / np.sqrt(info / n_factors)
            else:
                theta_se[i] = np.inf

        return ScoreResult(
            theta=theta_wle,
            standard_error=theta_se,
            method="WLE",
        )

    def __repr__(self) -> str:
        return f"WLEScorer(bounds={self.bounds}, tol={self.tol})"
