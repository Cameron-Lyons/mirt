"""Weighted Likelihood Estimation (WLE) scorer.

WLE (Warm's Weighted Likelihood Estimation) reduces the bias in ML estimates,
particularly at extreme ability levels. It adds a correction term based on
the first derivative of the test information function.

References
----------
Warm, T. A. (1989). Weighted likelihood estimation of ability in item
    response theory. Psychometrika, 54(3), 427-450.
Wang, C. (2015). On latent trait estimation in multidimensional compensatory
    item response models. Psychometrika, 80(2), 428-449.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, minimize_scalar

from mirt._model_defaults import uses_builtin_model_hooks
from mirt.constants import PROB_EPSILON
from mirt.results.score_result import ScoreResult
from mirt.scoring._common import (
    measured_factors,
    observed_information_matrix,
    observed_test_information,
    score_pattern_chunks,
    score_responses_parallel,
    supports_row_batched_scoring,
    unique_response_patterns,
    validate_scoring_responses,
)
from mirt.scoring._optimization import (
    batched_newton_minimize,
    bounded_scalar_minimize,
    free_block_standard_errors,
    validate_theta_bounds,
)
from mirt.utils._parallel import resolve_n_jobs

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


def _half_log_determinant(
    information: NDArray[np.float64],
    measured: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return ``0.5 * log det`` of each measured-factor information block.

    Unmeasured factors carry no information, so their rows and columns are
    dropped. A block whose determinant is not positive yields ``-inf``.
    """
    pair = measured[:, :, None] & measured[:, None, :]
    block = np.where(pair, information, np.eye(information.shape[-1]))
    sign, log_determinant = np.linalg.slogdet(block)
    finite = (sign > 0.0) & np.isfinite(log_determinant)
    return np.where(finite, 0.5 * np.where(finite, log_determinant, 0.0), -np.inf)


def _information_standard_errors(
    information: NDArray[np.float64],
    measured: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return ``sqrt(diag(I^-1))`` for measured factors and inf otherwise."""
    standard_error = free_block_standard_errors(information, measured)
    standard_error[~measured] = np.inf
    return standard_error


# Spacing of the grid that brackets the global maximum of the weighted
# likelihood, and the number of its highest grid maxima refined per pattern.
_WLE_GRID_SPACING = 0.25
_WLE_MAX_PEAKS = 3


def _wle_grid(theta_min: float, theta_max: float) -> NDArray[np.float64]:
    """Return equally spaced points at most 0.25 apart spanning the bounds."""
    n_points = int(np.ceil((theta_max - theta_min) / _WLE_GRID_SPACING)) + 1
    return np.linspace(theta_min, theta_max, max(n_points, 2))


def _wle_peak_brackets(
    values: NDArray[np.float64],
    grid: NDArray[np.float64],
    estimate: NDArray[np.float64],
    estimate_value: NDArray[np.float64],
) -> Iterator[tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.float64]]]:
    """Yield search brackets around grid maxima that local estimates miss.

    ``values`` holds each row's weighted log-likelihood at the ``grid``
    points and ``estimate_value`` its value at the row's local ``estimate``.
    A point at least as high as its neighbours is a grid maximum, bracketed
    by those neighbours. For each of the ``_WLE_MAX_PEAKS`` highest grid
    maxima of every row, this yields ``(rows, lower, upper)`` for the rows
    whose estimate lies outside that bracket or below that grid maximum, one
    rank at a time. A row whose estimate tops its only grid maximum yields
    nothing.
    """
    n_rows, n_grid = values.shape
    padded = np.full((n_rows, n_grid + 2), -np.inf)
    padded[:, 1:-1] = values
    peaks = (values >= padded[:, :-2]) & (values >= padded[:, 2:]) & ~np.isnan(values)
    order = np.argsort(np.where(peaks, -values, np.inf), axis=1, kind="stable")
    all_rows = np.arange(n_rows)
    for rank in range(min(_WLE_MAX_PEAKS, n_grid)):
        point = order[:, rank]
        lower = grid[np.maximum(point - 1, 0)]
        upper = grid[np.minimum(point + 1, n_grid - 1)]
        missed = peaks[all_rows, point] & (
            (estimate < lower)
            | (estimate > upper)
            | (estimate_value < values[all_rows, point])
        )
        rows = np.flatnonzero(missed)
        if rows.size:
            yield rows, lower[rows], upper[rows]


def _item_information_grid(
    model: BaseItemModel,
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return every item's information at the theta rows, ``(n_theta, n_items)``.

    Item totals are formed as :func:`observed_test_information` forms them.
    """
    n_theta = theta.shape[0]
    if not model.is_polytomous:
        information = np.asarray(model.information(theta), dtype=np.float64)
        if information.shape == (n_theta, model.n_items):
            return information
    return np.column_stack(
        [
            np.asarray(model.information(theta, item_idx), dtype=np.float64)
            .reshape(n_theta, -1)
            .sum(axis=1)
            for item_idx in range(model.n_items)
        ]
    )


def _weighted_log_likelihood_grid(
    model: BaseItemModel,
    patterns: NDArray[np.int_],
    observed: NDArray[np.bool_],
    grid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return every pattern's ``log L + 0.5 log I`` at every grid point."""
    theta = grid[:, None]
    log_likelihood = np.asarray(
        model.log_likelihood_batch(patterns, theta), dtype=np.float64
    )
    information = observed.astype(np.float64) @ _item_information_grid(model, theta).T
    weighted = information > PROB_EPSILON
    log_info = np.log(information, out=np.zeros_like(information), where=weighted)
    return np.where(weighted, log_likelihood + 0.5 * log_info, log_likelihood)


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
        available CPU cores. Default is 1. Built-in unidimensional and
        multidimensional models are scored with one row-batched search that
        does not use a thread pool.

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

    where L(theta) is the likelihood and I(theta) is the test information of
    the observed items. With one factor the maximum is global over
    ``bounds``: ``WL`` can have several local maxima when weakly and highly
    discriminating items are mixed, so after a bounded local search the
    criterion is scanned on a grid at most 0.25 apart, and the three
    highest grid maxima away from or above the estimate are refined; the
    estimate moves to a higher maximum when one is found. R's
    ``fscores(method="WLE")`` searches locally and can stop on a lower
    maximum. For multidimensional models ``I(theta)`` is the determinant of
    the Fisher information matrix (Wang, 2015), as in R's
    mirt, and standard errors are the square roots of the diagonal of its
    inverse at the estimate. Built-in multidimensional models are scored for
    every distinct pattern with one row-batched projected Newton search from
    the origin, whose unconverged rows fall back to L-BFGS-B. Factors that no
    observed item measures are left out of the determinant and reported as
    zero with infinite standard errors. A pattern whose observed items give a
    singular information matrix for the factors they measure has no WLE and
    receives zero with NaN standard errors for those factors.
    """

    def __init__(
        self,
        bounds: tuple[float, float] = (-6.0, 6.0),
        tol: float = 1e-6,
        n_jobs: int = 1,
    ) -> None:
        lower, upper = validate_theta_bounds(bounds)
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

        default_search = type(self) is WLEScorer and not (
            {"_estimate_person", "_test_information"} & vars(self).keys()
        )
        if n_factors == 1 and default_search and supports_row_batched_scoring(model):
            theta_wle, theta_se = score_pattern_chunks(
                patterns,
                lambda chunk: self._estimate_unidimensional_batch(model, chunk),
            )
            return ScoreResult(
                theta=theta_wle[inverse],
                standard_error=theta_se[inverse],
                method="WLE",
            )

        if (
            n_factors > 1
            and type(self) is WLEScorer
            and not {"_estimate_multidimensional_person", "_test_information_matrix"}
            & vars(self).keys()
            and supports_row_batched_scoring(model)
            and uses_builtin_model_hooks(model, likelihood=True)
        ):
            theta_wle, theta_se = score_pattern_chunks(
                patterns,
                lambda chunk: self._estimate_multidimensional_batch(model, chunk),
            )
            return ScoreResult(
                theta=theta_wle[inverse],
                standard_error=theta_se[inverse],
                method="WLE",
            )

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
        theta_hat, best = float(result.x), -float(result.fun)

        # The weighted likelihood can have several local maxima; search the
        # grid maxima that the bounded search did not reach. The grid is
        # evaluated in one call, as EAP evaluates its quadrature points.
        grid = _wle_grid(*self.bounds)
        grid_theta = grid[:, None]
        grid_ll = np.asarray(
            model.log_likelihood_batch(responses[None, :], grid_theta), dtype=np.float64
        ).reshape(-1)
        grid_info = np.asarray(
            self._test_information(model, grid_theta, valid_mask), dtype=np.float64
        ).reshape(-1)
        weighted = grid_info > PROB_EPSILON
        log_info = np.log(grid_info, out=np.zeros_like(grid_info), where=weighted)
        values = np.where(weighted, grid_ll + 0.5 * log_info, grid_ll)
        for _, lower, upper in _wle_peak_brackets(
            values[None, :], grid, np.array([theta_hat]), np.array([best])
        ):
            candidate = minimize_scalar(
                neg_weighted_log_likelihood,
                bounds=(float(lower[0]), float(upper[0])),
                method="bounded",
                options={"xatol": self.tol},
            )
            if -float(candidate.fun) > best:
                theta_hat, best = float(candidate.x), -float(candidate.fun)

        theta_arr = np.array([[theta_hat]])
        info = self._test_information(model, theta_arr, valid_mask)[0]

        if info > PROB_EPSILON:
            se = 1.0 / np.sqrt(info)
        else:
            se = np.inf

        return theta_hat, se

    def _estimate_unidimensional_batch(
        self,
        model: BaseItemModel,
        patterns: NDArray[np.int_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Estimate every pattern as :meth:`_estimate_person` would.

        All patterns share one bounded Brent search, so each optimizer step
        needs a single stacked likelihood and information evaluation. A grid
        scan of every pattern's criterion then locates higher local maxima,
        which a second shared search refines.
        """
        observed = patterns >= 0
        theta = np.zeros(patterns.shape[0], dtype=np.float64)
        standard_error = np.full(patterns.shape[0], np.inf, dtype=np.float64)
        rows = np.flatnonzero(observed.any(axis=1))
        if rows.size == 0:
            return theta, standard_error
        patterns = patterns[rows]
        observed = observed[rows]

        def neg_weighted_log_likelihood(
            subset: NDArray[np.intp],
            values: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            theta_arr = values[:, None]
            ll = np.asarray(
                model.log_likelihood(patterns[subset], theta_arr), dtype=np.float64
            )
            info = observed_test_information(model, theta_arr, observed[subset])
            weighted = info > PROB_EPSILON
            log_info = np.log(info, out=np.zeros_like(info), where=weighted)
            return -np.where(weighted, ll + 0.5 * log_info, ll)

        estimate, fun = bounded_scalar_minimize(
            neg_weighted_log_likelihood,
            rows.size,
            *self.bounds,
            xatol=self.tol,
        )

        def restricted(
            subset: NDArray[np.intp],
            selected: NDArray[np.intp],
            values: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            return neg_weighted_log_likelihood(subset[selected], values)

        # Move estimates at a local maximum to a higher grid maximum's peak.
        grid = _wle_grid(*self.bounds)
        values = _weighted_log_likelihood_grid(model, patterns, observed, grid)
        for subset, lower, upper in _wle_peak_brackets(
            values, grid, estimate.copy(), -fun
        ):
            candidate, candidate_fun = bounded_scalar_minimize(
                partial(restricted, subset),
                subset.size,
                lower,
                upper,
                xatol=self.tol,
            )
            better = candidate_fun < fun[subset]
            estimate[subset[better]] = candidate[better]
            fun[subset[better]] = candidate_fun[better]
        info = observed_test_information(model, estimate[:, None], observed)
        estimate_se = np.full(rows.size, np.inf, dtype=np.float64)
        informative = info > PROB_EPSILON
        estimate_se[informative] = 1.0 / np.sqrt(info[informative])
        theta[rows] = estimate
        standard_error[rows] = estimate_se
        return theta, standard_error

    def _test_information(
        self,
        model: BaseItemModel,
        theta: NDArray[np.float64],
        valid_mask: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Compute test information at given theta values."""
        return observed_test_information(model, theta, valid_mask)

    def _test_information_matrix(
        self,
        model: BaseItemModel,
        theta: NDArray[np.float64],
        valid_mask: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Compute observed-item Fisher information matrices at theta rows."""
        return observed_information_matrix(model, theta, valid_mask)

    def _estimate_multidimensional_person(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Estimate one multidimensional WLE score with L-BFGS-B.

        The weighted likelihood is ``L(theta) * sqrt(det I(theta))`` over the
        factors measured by the observed items.
        """
        n_factors = model.n_factors
        valid_mask = responses >= 0

        if not valid_mask.any():
            return np.zeros(n_factors), np.full(n_factors, np.inf)
        measured = measured_factors(model, valid_mask[None, :])

        def neg_weighted_log_likelihood(theta_vec: NDArray[np.float64]) -> float:
            theta_arr = theta_vec.reshape(1, -1)
            ll = model.log_likelihood(responses[None, :], theta_arr)[0]
            information = self._test_information_matrix(model, theta_arr, valid_mask)
            return float(-(ll + _half_log_determinant(information, measured)[0]))

        start = np.zeros(n_factors)
        if not np.isfinite(neg_weighted_log_likelihood(start)):
            return start, np.where(measured[0], np.nan, np.inf)
        result = minimize(
            neg_weighted_log_likelihood,
            start,
            method="L-BFGS-B",
            bounds=[self.bounds] * n_factors,
            options={"ftol": self.tol},
        )
        theta_hat = np.where(measured[0], result.x, 0.0)
        information = self._test_information_matrix(
            model, theta_hat[None, :], valid_mask
        )
        return theta_hat, _information_standard_errors(information, measured)[0]

    def _estimate_multidimensional_batch(
        self,
        model: BaseItemModel,
        patterns: NDArray[np.int_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Estimate every pattern's WLE with a shared Newton search.

        Rows for which the projected Newton search does not converge are
        re-solved by :meth:`_estimate_multidimensional_person`.
        """
        n_patterns, n_factors = patterns.shape[0], model.n_factors
        observed = patterns >= 0
        theta = np.zeros((n_patterns, n_factors), dtype=np.float64)
        standard_error = np.full((n_patterns, n_factors), np.inf, dtype=np.float64)
        rows = np.flatnonzero(observed.any(axis=1))
        if rows.size == 0:
            return theta, standard_error
        measured = measured_factors(model, observed[rows])

        def neg_weighted_log_likelihood(
            subset: NDArray[np.intp],
            values: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            selected = rows[subset]
            ll = np.asarray(
                model.log_likelihood(patterns[selected], values), dtype=np.float64
            )
            information = observed_information_matrix(model, values, observed[selected])
            return -(ll + _half_log_determinant(information, measured[subset]))

        estimate, fun, converged = batched_newton_minimize(
            neg_weighted_log_likelihood,
            np.zeros((rows.size, n_factors)),
            *self.bounds,
        )
        estimate = np.where(measured, estimate, 0.0)
        solved = np.flatnonzero(converged)
        information = observed_information_matrix(
            model, estimate[solved], observed[rows[solved]]
        )
        theta[rows[solved]] = estimate[solved]
        standard_error[rows[solved]] = _information_standard_errors(
            information, measured[solved]
        )
        # Rows without a finite weighted likelihood at the origin have a
        # singular information matrix everywhere, hence no WLE.
        singular = np.flatnonzero(~converged & ~np.isfinite(fun))
        standard_error[rows[singular]] = np.where(measured[singular], np.nan, np.inf)
        for row in rows[~converged & np.isfinite(fun)]:
            theta[row], standard_error[row] = self._estimate_multidimensional_person(
                model, patterns[row]
            )
        return theta, standard_error

    def __repr__(self) -> str:
        return f"WLEScorer(bounds={self.bounds}, tol={self.tol}, n_jobs={self.n_jobs})"
