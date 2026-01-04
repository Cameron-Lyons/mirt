"""Base class for parameter estimation algorithms."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


class BaseEstimator(ABC):
    """Abstract base class for IRT parameter estimation algorithms.

    This class defines the interface for all estimation methods including
    EM algorithm, MHRM, and others.

    Parameters
    ----------
    max_iter : int, default=500
        Maximum number of iterations.
    tol : float, default=1e-4
        Convergence tolerance (change in log-likelihood).
    verbose : bool, default=False
        Whether to print progress information.

    Attributes
    ----------
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Verbosity flag.
    convergence_history : list of float
        Log-likelihood values at each iteration.
    """

    def __init__(
        self,
        max_iter: int = 500,
        tol: float = 1e-4,
        verbose: bool = False,
    ) -> None:
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if tol <= 0:
            raise ValueError("tol must be positive")

        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self._convergence_history: list[float] = []

    @abstractmethod
    def fit(
        self,
        model: "BaseItemModel",
        responses: NDArray[np.int_],
        **kwargs,
    ) -> "FitResult":
        """Estimate model parameters from response data.

        Parameters
        ----------
        model : BaseItemModel
            The IRT model to fit. Parameters will be updated in place.
        responses : ndarray of shape (n_persons, n_items)
            Response matrix. Missing values should be coded as -1.
        **kwargs
            Additional arguments specific to the estimation method.

        Returns
        -------
        FitResult
            Object containing fitted parameters, standard errors, and
            fit statistics.
        """
        ...

    @property
    def convergence_history(self) -> list[float]:
        """Return log-likelihood history across iterations."""
        return self._convergence_history.copy()

    def _check_convergence(
        self,
        old_ll: float,
        new_ll: float,
    ) -> bool:
        """Check if algorithm has converged.

        Parameters
        ----------
        old_ll : float
            Previous log-likelihood.
        new_ll : float
            Current log-likelihood.

        Returns
        -------
        bool
            True if converged.
        """
        return abs(new_ll - old_ll) < self.tol

    def _validate_responses(
        self,
        responses: NDArray[np.int_],
        n_items: int,
    ) -> NDArray[np.int_]:
        """Validate and clean response matrix.

        Parameters
        ----------
        responses : ndarray
            Raw response matrix.
        n_items : int
            Expected number of items.

        Returns
        -------
        ndarray
            Validated response matrix.
        """
        responses = np.asarray(responses)

        if responses.ndim != 2:
            raise ValueError(f"responses must be 2D, got {responses.ndim}D")

        if responses.shape[1] != n_items:
            raise ValueError(
                f"responses has {responses.shape[1]} items, expected {n_items}"
            )

        return responses

    def _log_iteration(
        self,
        iteration: int,
        log_likelihood: float,
        **kwargs,
    ) -> None:
        """Log iteration progress if verbose mode is on.

        Parameters
        ----------
        iteration : int
            Current iteration number.
        log_likelihood : float
            Current log-likelihood value.
        **kwargs
            Additional values to log.
        """
        if self.verbose:
            extras = ", ".join(f"{k}={v:.4f}" for k, v in kwargs.items())
            msg = f"Iteration {iteration:4d}: LL = {log_likelihood:.4f}"
            if extras:
                msg += f", {extras}"
            print(msg)

    def _compute_aic(
        self,
        log_likelihood: float,
        n_parameters: int,
    ) -> float:
        """Compute Akaike Information Criterion.

        AIC = -2 × LL + 2 × k

        Parameters
        ----------
        log_likelihood : float
            Final log-likelihood.
        n_parameters : int
            Number of free parameters.

        Returns
        -------
        float
            AIC value.
        """
        return -2 * log_likelihood + 2 * n_parameters

    def _compute_bic(
        self,
        log_likelihood: float,
        n_parameters: int,
        n_observations: int,
    ) -> float:
        """Compute Bayesian Information Criterion.

        BIC = -2 × LL + k × log(n)

        Parameters
        ----------
        log_likelihood : float
            Final log-likelihood.
        n_parameters : int
            Number of free parameters.
        n_observations : int
            Number of observations (persons).

        Returns
        -------
        float
            BIC value.
        """
        return -2 * log_likelihood + n_parameters * np.log(n_observations)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"max_iter={self.max_iter}, "
            f"tol={self.tol})"
        )
