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

from mirt.constants import PROB_EPSILON
from mirt.results.score_result import ScoreResult
from mirt.scoring._common import (
    observed_test_information,
    resolve_n_jobs,
    score_responses_parallel,
    unique_response_patterns,
    validate_scoring_responses,
)
from mirt.utils.numeric import compute_hessian_se

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
    n_jobs : int, optional
        Number of response patterns to optimize in parallel. ``-1`` uses all
        available CPU cores. Default is 1.

    Attributes
    ----------
    bounds : tuple of float
        The theta search bounds.
    tol : float
        Convergence tolerance.
    n_jobs : int
        Number of parallel workers.

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
        n_jobs: int = 1,
    ) -> None:
        try:
            lower, upper = (float(value) for value in bounds)
        except (TypeError, ValueError) as exc:
            raise ValueError("bounds must contain exactly two finite values") from exc
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("bounds must contain finite values with lower < upper")
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tol must be finite and positive")
        resolve_n_jobs(n_jobs)

        self.bounds = (lower, upper)
        self.tol = float(tol)
        self.n_jobs = int(n_jobs)

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

        responses = validate_scoring_responses(model, responses)
        patterns, inverse = unique_response_patterns(responses)
        n_factors = model.n_factors

        def score_person(
            index: int,
        ) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64]]:
            person_responses = patterns[index]
            if n_factors == 1:
                return self._estimate_person(model, person_responses)
            return self._estimate_multidimensional_person(model, person_responses)

        theta_wle, theta_se = score_responses_parallel(
            model=model,
            responses=patterns,
            n_jobs=self.n_jobs,
            score_person=score_person,
        )

        return ScoreResult(
            theta=theta_wle[inverse],
            standard_error=theta_se[inverse],
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

            if info > PROB_EPSILON:
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

        if info > PROB_EPSILON:
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
        return observed_test_information(model, theta, valid_mask)

    def _estimate_multidimensional_person(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Estimate one multidimensional WLE score and curvature-based SE."""
        n_factors = model.n_factors
        valid_mask = responses >= 0

        if not valid_mask.any():
            return np.zeros(n_factors), np.full(n_factors, np.inf)

        def neg_weighted_log_likelihood(theta_vec: NDArray[np.float64]) -> float:
            theta_arr = theta_vec.reshape(1, -1)
            ll = model.log_likelihood(responses[None, :], theta_arr)[0]
            info = self._test_information(model, theta_arr, valid_mask)[0]

            if info > PROB_EPSILON:
                ll += 0.5 * np.log(info)
            return float(-ll)

        result = minimize(
            neg_weighted_log_likelihood,
            np.zeros(n_factors),
            method="L-BFGS-B",
            bounds=[self.bounds] * n_factors,
            options={"ftol": self.tol},
        )
        standard_error = compute_hessian_se(neg_weighted_log_likelihood, result.x)
        return result.x, standard_error

    def __repr__(self) -> str:
        return f"WLEScorer(bounds={self.bounds}, tol={self.tol}, n_jobs={self.n_jobs})"
