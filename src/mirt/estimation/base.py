from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt.exceptions import MirtModelError, MirtValidationError
from mirt.utils.data import validate_responses

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult

StartValues: TypeAlias = Literal["default", "model"] | Mapping[str, ArrayLike]


_ITEM_PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "discrimination": (0.1, 5.0),
    "difficulty": (-6.0, 6.0),
    "intercepts": (-6.0, 6.0),
    "thresholds": (-6.0, 6.0),
    "steps": (-6.0, 6.0),
    "guessing": (0.0, 0.5),
    "upper": (0.5, 1.0),
    "asymmetry": (0.1, 5.0),
}


def _parameter_bounds(model: BaseItemModel, name: str) -> tuple[float, float]:
    """Return the box that EM item optimizers use for a stored parameter.

    Parameters
    ----------
    model : BaseItemModel
        Model owning the parameter. Nominal-response slopes may be negative.
    name : str
        Stored parameter name.

    Returns
    -------
    tuple of float
        Lower and upper bound shared by every free coordinate of ``name``.
        Qualified parameters of a mixed-format model use their component's.
    """
    from mirt.models.mixed_format import MixedItemModel

    if isinstance(model, MixedItemModel):
        component, local = model.parameter_component(name)
        return _parameter_bounds(model.component_models[component], local)
    if name == "slopes" and model.model_name == "NRM":
        return (-5.0, 5.0)
    if "discrimination" in name or "slope" in name:
        return _ITEM_PARAMETER_BOUNDS["discrimination"]
    return _ITEM_PARAMETER_BOUNDS.get(name, (-6.0, 6.0))


def _stagger_factor_slopes(
    model: BaseItemModel,
    name: str,
    values: NDArray[np.float64],
    free: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Stagger default slopes across factors with identical free coordinates.

    Factors that are free on the same slope coordinates, as in an exploratory
    model, and start with equal slopes are exchangeable under a standard
    normal prior. Exact M-steps then keep their slope columns equal, and EM
    stops at the saddle point where the factors act as one. Item ``j``
    instead starts at its default slope on factor ``j mod n_factors`` and at
    half of it on the other factors. Other parameters and confirmatory slope
    patterns are returned unchanged.
    """
    from mirt.models.mixed_format import MixedItemModel

    owner, local = model, name
    if isinstance(model, MixedItemModel):
        component, local = model.parameter_component(name)
        owner = model.component_models[component]
    n_items, n_factors = owner.n_items, owner.n_factors
    if (
        n_factors < 2
        or local not in ("discrimination", "slopes")
        or values.ndim < 2
        or values.shape[0] != n_items
        or values.shape[-1] != n_factors
    ):
        return values
    columns = np.asarray(free, dtype=np.bool_).reshape(-1, n_factors).T
    used = columns[columns.any(axis=1)]
    if len(np.unique(used, axis=0)) == len(used):
        return values
    weights = np.full((n_items, n_factors), 0.5)
    weights[np.arange(n_items), np.arange(n_items) % n_factors] = 1.0
    return values * weights.reshape((n_items,) + (1,) * (values.ndim - 2) + (-1,))


def _initialize_free_parameters(model: BaseItemModel) -> None:
    """Reset starting values while preserving fixed independent coordinates.

    Slopes of factors with identical free coordinates start staggered (see
    ``_stagger_factor_slopes``) so that exploratory multidimensional fits do
    not stall at the equal-slope saddle point.
    """
    original = {
        name: model._canonical_parameter_values(name, values)
        for name, values in model.parameters.items()
    }
    masks = model.free_parameter_masks
    model._initialize_parameters()
    updates = {}
    for name, values in model.parameters.items():
        initial = values.copy()
        values = _stagger_factor_slopes(model, name, values, masks[name])
        np.copyto(values, original[name], where=~masks[name])
        values = model._canonical_parameter_values(name, values)
        if not np.array_equal(values, initial, equal_nan=True):
            updates[name] = values
    if updates:
        model.set_parameters(**updates)


def _validate_start(
    start: StartValues,
) -> Literal["default", "model"] | dict[str, NDArray[np.float64]]:
    """Return a validated ``start`` option, converting mappings to float arrays."""
    if isinstance(start, str):
        if start not in ("default", "model"):
            raise MirtValidationError(
                "start must be 'default', 'model', or a mapping of parameter values",
                parameter="start",
                value=start,
                expected="'default', 'model', or mapping",
            )
        return "default" if start == "default" else "model"
    if not isinstance(start, Mapping):
        raise MirtValidationError(
            "start must be 'default', 'model', or a mapping of parameter values",
            parameter="start",
            value=type(start).__name__,
            expected="'default', 'model', or mapping",
        )
    values: dict[str, NDArray[np.float64]] = {}
    for name, value in start.items():
        try:
            array = np.array(value, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                f"starting values for {name!r} must be numeric",
                parameter="start",
                value=name,
            ) from exc
        if not np.all(np.isfinite(array)):
            raise MirtValidationError(
                f"starting values for {name!r} must be finite",
                parameter="start",
                value=name,
            )
        values[str(name)] = array
    return values


def _apply_starting_values(model: BaseItemModel, start: StartValues) -> None:
    """Install the starting values that an estimator's ``start`` option requests.

    ``"default"`` resets the free coordinates of an unfitted model to the
    family defaults and keeps the values of a fitted model as a warm start.
    ``"model"`` keeps the current values. A mapping is applied with
    ``set_parameters`` after the ``"default"`` initialization, so it can also
    set the values at which fixed coordinates are held.
    """
    start = _validate_start(start)
    if start == "model":
        return
    values: dict[str, NDArray[np.float64]] = {}
    if not isinstance(start, str):
        stored = model.parameters
        for name, array in start.items():
            if name not in stored:
                raise MirtValidationError(
                    f"Unknown parameter in start: {name}",
                    parameter="start",
                    value=name,
                    expected=", ".join(stored),
                )
            if array.shape != stored[name].shape:
                raise MirtValidationError(
                    f"starting values for {name!r} must have shape "
                    f"{stored[name].shape}",
                    parameter="start",
                    value=array.shape,
                    expected=str(stored[name].shape),
                )
        values = start
    snapshot = model.parameters
    try:
        if not model._is_fitted:
            _initialize_free_parameters(model)
        if values:
            model.set_parameters(**values)
    except Exception:
        model._parameters.update(snapshot)
        raise


def _free_masks_from_fixed(
    model: BaseItemModel, fixed: Mapping[str, ArrayLike]
) -> dict[str, NDArray[np.bool_]]:
    """Convert ``{parameter: fixed mask}`` into free-parameter masks.

    ``True`` marks a coordinate held at its starting value. A scalar fixes or
    frees a whole parameter. Coordinates that the model family fixes stay
    fixed.
    """
    if not isinstance(fixed, Mapping):
        raise MirtValidationError(
            "fixed must map parameter names to Boolean masks",
            parameter="fixed",
            value=type(fixed).__name__,
            expected="Mapping[str, bool array]",
        )
    masks = model.free_parameter_masks
    free: dict[str, NDArray[np.bool_]] = {}
    for name, value in fixed.items():
        if name not in masks:
            raise MirtValidationError(
                f"Unknown parameter in fixed: {name}",
                parameter="fixed",
                value=name,
                expected=", ".join(masks),
            )
        mask = np.asarray(value)
        if mask.dtype != np.bool_:
            raise MirtValidationError(
                f"fixed mask for {name!r} must be Boolean",
                parameter="fixed",
                value=name,
            )
        try:
            mask = np.broadcast_to(mask, masks[name].shape)
        except ValueError as exc:
            raise MirtValidationError(
                f"fixed mask for {name!r} must be a scalar or have shape "
                f"{masks[name].shape}",
                parameter="fixed",
                value=mask.shape,
                expected=str(masks[name].shape),
            ) from exc
        free[name] = masks[name] & ~mask
    return free


def _free_shared_parameters(model: BaseItemModel) -> tuple[str, ...]:
    """Return the parameters shared by all items that have a free coordinate."""
    shared = model._shared_parameters
    if not shared:
        return ()
    masks = model.free_parameter_masks
    return tuple(
        name for name in model._parameters if name in shared and np.any(masks[name])
    )


def _reject_parameter_restrictions(model: BaseItemModel, estimator: str) -> None:
    """Raise when user masks fix parameters that ``estimator`` would move."""
    if getattr(model, "_free_parameter_restrictions", None):
        raise MirtValidationError(
            f"{estimator} cannot hold parameters fixed by "
            "set_free_parameter_masks; use EMEstimator or MCEMEstimator, or "
            "clear the masks with set_free_parameter_masks(None)",
            parameter="model",
            expected="no free-parameter restrictions",
        )


class BaseEstimator(ABC):
    # Whether the M-step also updates parameters shared by all items, such as
    # rating-scale thresholds. Itemwise estimators without that step refuse
    # models whose shared parameters are free rather than leave them unchanged.
    _estimates_shared_parameters: bool = False
    # Whether the estimator honors coordinates fixed by set_free_parameter_masks.
    _holds_fixed_parameters: bool = True

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

    def _check_shared_parameters(self, model: BaseItemModel) -> None:
        """Refuse free shared parameters that this estimator cannot update.

        Raises
        ------
        MirtModelError
            If ``model`` is a mixed-format model, or has free parameters
            shared by all items and the estimator's M-step updates item
            parameters only.
        """
        from mirt.models.mixed_format import MixedItemModel

        if isinstance(model, MixedItemModel):
            # Its parameters are not item-indexed, so itemwise optimization
            # would silently leave them unchanged.
            raise MirtModelError(
                f"{type(self).__name__} cannot optimize a MixedItemModel; fit "
                "it with MixedFormatEMEstimator or fit_mirt(model=[...])",
                model_type=model.model_name,
            )
        if self._estimates_shared_parameters:
            return
        shared = _free_shared_parameters(model)
        if shared:
            advice = "use EMEstimator or BLEstimator"
            if self._holds_fixed_parameters:
                advice += ", or hold them fixed with set_free_parameter_masks"
            raise MirtModelError(
                f"{type(self).__name__} cannot estimate {', '.join(shared)}, "
                f"which {model.model_name} shares across items; {advice}",
                model_type=model.model_name,
            )

    def _get_item_params_and_bounds(
        self,
        model: BaseItemModel,
        item_idx: int,
    ) -> tuple[NDArray[np.float64], list[tuple[float, float]]]:
        """Get current item parameters and their bounds for optimization.

        Parameters shared by all items are left out. Estimators without a
        shared-parameter step raise ``MirtModelError`` when one is free, and
        every itemwise estimator for a mixed-format model.
        """
        self._check_shared_parameters(model)
        params_list: list[float] = []
        bounds: list[tuple[float, float]] = []
        params = model.parameters
        free_masks = model.free_parameter_masks

        for name, values in params.items():
            if not model._item_indexed(name):
                continue

            canonical = model._canonical_parameter_values(name, values)
            item_values = np.asarray(canonical[item_idx]).reshape(-1)
            item_mask = np.asarray(free_masks[name][item_idx], dtype=np.bool_).reshape(
                -1
            )
            params_list.extend(item_values[item_mask].tolist())
            bound = _parameter_bounds(model, name)
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
            if not model._item_indexed(name):
                continue

            item_values = np.asarray(values[item_idx]).copy()
            item_flat = item_values.reshape(-1)
            item_mask = np.asarray(free_masks[name][item_idx], dtype=np.bool_).reshape(
                -1
            )
            n_free = int(np.count_nonzero(item_mask))
            item_flat[item_mask] = params[idx : idx + n_free]

            values[item_idx] = item_flat.reshape(item_values.shape)
            canonical = model._canonical_parameter_values(name, values)
            row = np.asarray(canonical[item_idx])
            value: float | NDArray[np.float64] = float(row) if row.ndim == 0 else row
            model.set_item_parameter(item_idx, name, value)
            idx += n_free

        if idx != params.size:
            raise ValueError(f"Expected {idx} item parameters, got {params.size}")
