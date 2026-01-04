from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


class BaseEstimator(ABC):
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
    ) -> "FitResult": ...

    @property
    def convergence_history(self) -> list[float]:
        return self._convergence_history.copy()

    def _check_convergence(
        self,
        old_ll: float,
        new_ll: float,
    ) -> bool:
        return abs(new_ll - old_ll) < self.tol

    def _validate_responses(
        self,
        responses: NDArray[np.int_],
        n_items: int,
    ) -> NDArray[np.int_]:
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
        return -2 * log_likelihood + 2 * n_parameters

    def _compute_bic(
        self,
        log_likelihood: float,
        n_parameters: int,
        n_observations: int,
    ) -> float:
        return -2 * log_likelihood + n_parameters * np.log(n_observations)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_iter={self.max_iter}, tol={self.tol})"
