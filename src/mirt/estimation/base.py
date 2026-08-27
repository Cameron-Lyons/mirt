from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mirt.exceptions import MirtValidationError
from mirt.utils.data import validate_responses

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
            raise MirtValidationError(
                "max_iter must be at least 1",
                parameter="max_iter",
                value=max_iter,
                expected=">= 1",
            )
        if tol <= 0:
            raise MirtValidationError(
                "tol must be positive",
                parameter="tol",
                value=tol,
                expected="> 0",
            )

        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self._convergence_history: list[float] = []

    @abstractmethod
    def fit(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        **kwargs: Any,
    ) -> FitResult: ...

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
        return validate_responses(responses, n_items=n_items)

    def _log_iteration(
        self,
        iteration: int,
        log_likelihood: float,
        **kwargs: float,
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

    def _get_item_params_and_bounds(
        self,
        model: BaseItemModel,
        item_idx: int,
    ) -> tuple[NDArray[np.float64], list[tuple[float, float]]]:
        """Get current item parameters and their bounds for optimization."""
        params_list: list[float] = []
        bounds: list[tuple[float, float]] = []
        params = model.parameters
        free_masks = model.free_parameter_masks
        bounds_map = {
            "discrimination": (0.1, 5.0),
            "difficulty": (-6.0, 6.0),
            "intercepts": (-6.0, 6.0),
            "thresholds": (-6.0, 6.0),
            "steps": (-6.0, 6.0),
            "guessing": (0.0, 0.5),
            "upper": (0.5, 1.0),
            "asymmetry": (0.1, 5.0),
        }

        for name, values in params.items():
            if values.ndim == 0 or values.shape[0] != model.n_items:
                continue

            canonical = model._canonical_parameter_values(name, values)
            item_values = np.asarray(canonical[item_idx]).reshape(-1)
            item_mask = np.asarray(free_masks[name][item_idx], dtype=np.bool_).reshape(
                -1
            )
            params_list.extend(item_values[item_mask].tolist())
            if name == "slopes" and model.model_name == "NRM":
                bound = (-5.0, 5.0)
            elif "discrimination" in name or "slope" in name:
                bound = (0.1, 5.0)
            else:
                bound = bounds_map.get(name, (-6.0, 6.0))
            bounds.extend([bound] * int(np.count_nonzero(item_mask)))

        return np.asarray(params_list, dtype=np.float64), bounds

    def _set_item_params(
        self,
        model: BaseItemModel,
        item_idx: int,
        params: NDArray[np.float64],
    ) -> None:
        """Set item parameters from flat array."""
        idx = 0
        free_masks = model.free_parameter_masks

        for name, values in model.parameters.items():
            if values.ndim == 0 or values.shape[0] != model.n_items:
                continue

            canonical = model._canonical_parameter_values(name, values)
            item_values = np.asarray(canonical[item_idx]).copy()
            item_flat = item_values.reshape(-1)
            item_mask = np.asarray(free_masks[name][item_idx], dtype=np.bool_).reshape(
                -1
            )
            n_free = int(np.count_nonzero(item_mask))
            item_flat[item_mask] = params[idx : idx + n_free]

            value: float | NDArray[np.float64]
            if item_values.ndim == 0:
                value = float(item_flat[0])
            else:
                value = item_flat.reshape(item_values.shape)
            model.set_item_parameter(item_idx, name, value)
            idx += n_free

        if idx != params.size:
            raise ValueError(f"Expected {idx} item parameters, got {params.size}")
