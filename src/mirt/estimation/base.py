from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

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
        model: "BaseItemModel",
        item_idx: int,
    ) -> tuple[NDArray[np.float64], list[tuple[float, float]]]:
        """Get current item parameters and their bounds for optimization."""
        params_list: list[float] = []
        bounds: list[tuple[float, float]] = []

        model_name = model.model_name
        params = model.parameters

        if model_name in ("1PL", "2PL", "3PL", "4PL"):
            if model_name != "1PL":
                a = params["discrimination"]
                if a.ndim == 1:
                    params_list.append(float(a[item_idx]))
                    bounds.append((0.1, 5.0))
                else:
                    params_list.extend(a[item_idx].tolist())
                    bounds.extend([(0.1, 5.0)] * model.n_factors)

            b = params["difficulty"][item_idx]
            params_list.append(float(b))
            bounds.append((-6.0, 6.0))

            if model_name in ("3PL", "4PL"):
                c = params["guessing"][item_idx]
                params_list.append(float(c))
                bounds.append((0.0, 0.5))

            if model_name == "4PL":
                d = params["upper"][item_idx]
                params_list.append(float(d))
                bounds.append((0.5, 1.0))

        else:
            for name, values in params.items():
                if values.ndim == 1 and len(values) == model.n_items:
                    params_list.append(float(values[item_idx]))
                    if "discrimination" in name or "slope" in name:
                        bounds.append((0.1, 5.0))
                    else:
                        bounds.append((-6.0, 6.0))
                elif values.ndim == 2 and values.shape[0] == model.n_items:
                    params_list.extend(values[item_idx].tolist())
                    if "discrimination" in name or "slope" in name:
                        bounds.extend([(0.1, 5.0)] * values.shape[1])
                    else:
                        bounds.extend([(-6.0, 6.0)] * values.shape[1])

        return np.array(params_list), bounds

    def _set_item_params(
        self,
        model: "BaseItemModel",
        item_idx: int,
        params: NDArray[np.float64],
    ) -> None:
        """Set item parameters from flat array."""
        model_name = model.model_name
        idx = 0

        if model_name in ("1PL", "2PL", "3PL", "4PL"):
            if model_name != "1PL":
                a = model.parameters["discrimination"]
                if a.ndim == 1:
                    model.set_item_parameter(item_idx, "discrimination", params[idx])
                    idx += 1
                else:
                    n_factors = a.shape[1]
                    model.set_item_parameter(
                        item_idx, "discrimination", params[idx : idx + n_factors]
                    )
                    idx += n_factors

            model.set_item_parameter(item_idx, "difficulty", params[idx])
            idx += 1

            if model_name in ("3PL", "4PL"):
                model.set_item_parameter(item_idx, "guessing", params[idx])
                idx += 1

            if model_name == "4PL":
                model.set_item_parameter(item_idx, "upper", params[idx])

        else:
            for name, values in model.parameters.items():
                if values.ndim == 1 and len(values) == model.n_items:
                    model.set_item_parameter(item_idx, name, params[idx])
                    idx += 1
                elif values.ndim == 2 and values.shape[0] == model.n_items:
                    n_vals = values.shape[1]
                    model.set_item_parameter(item_idx, name, params[idx : idx + n_vals])
                    idx += n_vals
