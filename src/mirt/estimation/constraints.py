"""Parameter constraints for IRT model estimation.

This module provides a system for specifying and enforcing constraints
on IRT model parameters during estimation:
- Equality constraints (parameter1 = parameter2)
- Fixed value constraints (parameter = constant)
- Linear constraints (sum of parameters = constant)
- Inequality constraints (parameter > bound)
- Bound constraints (lower < parameter < upper)

References
----------
- Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation
  of item parameters: Application of an EM algorithm. Psychometrika, 46, 443-459.
- Thissen, D., & Wainer, H. (1982). Some standard errors in item response theory.
  Psychometrika, 47, 397-412.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


@dataclass
class ParameterConstraint:
    """Base class for parameter constraints."""

    param_name: str
    item_indices: list[int] | None = None

    def apply(self, model: BaseItemModel) -> None:
        """Apply the constraint to the model."""
        raise NotImplementedError

    def is_satisfied(self, model: BaseItemModel, tol: float = 1e-6) -> bool:
        """Check if the constraint is satisfied."""
        raise NotImplementedError

    def penalty(self, model: BaseItemModel) -> float:
        """Return penalty for constraint violation (for optimization)."""
        raise NotImplementedError


@dataclass
class FixedConstraint(ParameterConstraint):
    """Fix a parameter to a specific value.

    Examples
    --------
    >>> # Fix discrimination of item 0 to 1.0
    >>> FixedConstraint("discrimination", [0], value=1.0)
    >>> # Fix all guessing parameters to 0.2
    >>> FixedConstraint("guessing", value=0.2)
    """

    value: float = 0.0

    def apply(self, model: BaseItemModel) -> None:
        params = model.parameters
        if self.param_name not in params:
            return

        values = params[self.param_name].copy()

        if self.item_indices is None:
            if values.ndim == 1:
                values[:] = self.value
            else:
                values[:, :] = self.value
        else:
            for idx in self.item_indices:
                if values.ndim == 1:
                    values[idx] = self.value
                else:
                    values[idx, :] = self.value

        model.set_parameters(**{self.param_name: values})

    def is_satisfied(self, model: BaseItemModel, tol: float = 1e-6) -> bool:
        params = model.parameters
        if self.param_name not in params:
            return True

        values = params[self.param_name]

        if self.item_indices is None:
            return np.allclose(values, self.value, atol=tol)
        else:
            for idx in self.item_indices:
                if values.ndim == 1:
                    if not np.isclose(values[idx], self.value, atol=tol):
                        return False
                else:
                    if not np.allclose(values[idx], self.value, atol=tol):
                        return False
            return True

    def penalty(self, model: BaseItemModel) -> float:
        params = model.parameters
        if self.param_name not in params:
            return 0.0

        values = params[self.param_name]

        if self.item_indices is None:
            return float(np.sum((values - self.value) ** 2))
        else:
            pen = 0.0
            for idx in self.item_indices:
                if values.ndim == 1:
                    pen += (values[idx] - self.value) ** 2
                else:
                    pen += np.sum((values[idx] - self.value) ** 2)
            return float(pen)


@dataclass
class EqualityConstraint(ParameterConstraint):
    """Constrain parameters to be equal across items.

    Examples
    --------
    >>> # Constrain discrimination to be equal for items 0, 1, 2
    >>> EqualityConstraint("discrimination", [0, 1, 2])
    >>> # Constrain all difficulties to be equal
    >>> EqualityConstraint("difficulty")
    """

    def apply(self, model: BaseItemModel) -> None:
        params = model.parameters
        if self.param_name not in params:
            return

        values = params[self.param_name].copy()
        indices = self.item_indices if self.item_indices else list(range(len(values)))

        if values.ndim == 1:
            mean_val = np.mean([values[i] for i in indices])
            for idx in indices:
                values[idx] = mean_val
        else:
            mean_val = np.mean([values[i] for i in indices], axis=0)
            for idx in indices:
                values[idx] = mean_val

        model.set_parameters(**{self.param_name: values})

    def is_satisfied(self, model: BaseItemModel, tol: float = 1e-6) -> bool:
        params = model.parameters
        if self.param_name not in params:
            return True

        values = params[self.param_name]
        indices = self.item_indices if self.item_indices else list(range(len(values)))

        if len(indices) < 2:
            return True

        ref_val = values[indices[0]]
        for idx in indices[1:]:
            if values.ndim == 1:
                if not np.isclose(values[idx], ref_val, atol=tol):
                    return False
            else:
                if not np.allclose(values[idx], ref_val, atol=tol):
                    return False
        return True

    def penalty(self, model: BaseItemModel) -> float:
        params = model.parameters
        if self.param_name not in params:
            return 0.0

        values = params[self.param_name]
        indices = self.item_indices if self.item_indices else list(range(len(values)))

        if len(indices) < 2:
            return 0.0

        if values.ndim == 1:
            mean_val = np.mean([values[i] for i in indices])
            return float(sum((values[i] - mean_val) ** 2 for i in indices))
        else:
            mean_val = np.mean([values[i] for i in indices], axis=0)
            return float(sum(np.sum((values[i] - mean_val) ** 2) for i in indices))


@dataclass
class BoundConstraint(ParameterConstraint):
    """Constrain parameters within bounds.

    Examples
    --------
    >>> # Constrain discrimination to be between 0.25 and 4.0
    >>> BoundConstraint("discrimination", lower=0.25, upper=4.0)
    >>> # Constrain guessing for items 0-4 to be at most 0.35
    >>> BoundConstraint("guessing", [0,1,2,3,4], upper=0.35)
    """

    lower: float | None = None
    upper: float | None = None

    def apply(self, model: BaseItemModel) -> None:
        params = model.parameters
        if self.param_name not in params:
            return

        values = params[self.param_name].copy()

        if self.item_indices is None:
            if self.lower is not None:
                values = np.maximum(values, self.lower)
            if self.upper is not None:
                values = np.minimum(values, self.upper)
        else:
            for idx in self.item_indices:
                if self.lower is not None:
                    if values.ndim == 1:
                        values[idx] = max(values[idx], self.lower)
                    else:
                        values[idx] = np.maximum(values[idx], self.lower)
                if self.upper is not None:
                    if values.ndim == 1:
                        values[idx] = min(values[idx], self.upper)
                    else:
                        values[idx] = np.minimum(values[idx], self.upper)

        model.set_parameters(**{self.param_name: values})

    def is_satisfied(self, model: BaseItemModel, tol: float = 1e-6) -> bool:
        params = model.parameters
        if self.param_name not in params:
            return True

        values = params[self.param_name]

        if self.item_indices is None:
            check_values = values
        else:
            if values.ndim == 1:
                check_values = values[self.item_indices]
            else:
                check_values = values[self.item_indices]

        if self.lower is not None:
            if np.any(check_values < self.lower - tol):
                return False
        if self.upper is not None:
            if np.any(check_values > self.upper + tol):
                return False

        return True

    def penalty(self, model: BaseItemModel) -> float:
        params = model.parameters
        if self.param_name not in params:
            return 0.0

        values = params[self.param_name]

        if self.item_indices is None:
            check_values = values
        else:
            if values.ndim == 1:
                check_values = values[self.item_indices]
            else:
                check_values = values[self.item_indices]

        pen = 0.0
        if self.lower is not None:
            violations = np.maximum(0, self.lower - check_values)
            pen += float(np.sum(violations**2))
        if self.upper is not None:
            violations = np.maximum(0, check_values - self.upper)
            pen += float(np.sum(violations**2))

        return pen


@dataclass
class LinearConstraint(ParameterConstraint):
    """Linear constraint on parameters.

    Constrains: sum(coefficients * parameters) = target

    Examples
    --------
    >>> # Mean difficulty = 0 (for identification)
    >>> LinearConstraint("difficulty", target=0.0, type="mean")
    >>> # Sum of loadings for factor 1 = n_items
    >>> LinearConstraint("loadings", target=10, type="sum", factor=0)
    """

    target: float = 0.0
    constraint_type: Literal["sum", "mean"] = "mean"
    coefficients: NDArray[np.float64] | None = None
    factor: int | None = None

    def apply(self, model: BaseItemModel) -> None:
        params = model.parameters
        if self.param_name not in params:
            return

        values = params[self.param_name].copy()
        indices = self.item_indices if self.item_indices else list(range(len(values)))

        if values.ndim == 1:
            current_vals = values[indices]
        else:
            f = self.factor if self.factor is not None else 0
            current_vals = values[indices, f]

        if self.coefficients is not None:
            current = np.dot(self.coefficients, current_vals)
            n = np.sum(self.coefficients)
        elif self.constraint_type == "mean":
            current = np.mean(current_vals)
            n = 1
        else:
            current = np.sum(current_vals)
            n = len(current_vals)

        adjustment = (self.target - current * n) / len(current_vals)

        if values.ndim == 1:
            values[indices] += adjustment
        else:
            f = self.factor if self.factor is not None else 0
            values[indices, f] += adjustment

        model.set_parameters(**{self.param_name: values})

    def is_satisfied(self, model: BaseItemModel, tol: float = 1e-6) -> bool:
        params = model.parameters
        if self.param_name not in params:
            return True

        values = params[self.param_name]
        indices = self.item_indices if self.item_indices else list(range(len(values)))

        if values.ndim == 1:
            current_vals = values[indices]
        else:
            f = self.factor if self.factor is not None else 0
            current_vals = values[indices, f]

        if self.coefficients is not None:
            current = np.dot(self.coefficients, current_vals)
        elif self.constraint_type == "mean":
            current = np.mean(current_vals)
        else:
            current = np.sum(current_vals)

        return np.isclose(current, self.target, atol=tol)

    def penalty(self, model: BaseItemModel) -> float:
        params = model.parameters
        if self.param_name not in params:
            return 0.0

        values = params[self.param_name]
        indices = self.item_indices if self.item_indices else list(range(len(values)))

        if values.ndim == 1:
            current_vals = values[indices]
        else:
            f = self.factor if self.factor is not None else 0
            current_vals = values[indices, f]

        if self.coefficients is not None:
            current = np.dot(self.coefficients, current_vals)
        elif self.constraint_type == "mean":
            current = np.mean(current_vals)
        else:
            current = np.sum(current_vals)

        return (current - self.target) ** 2


@dataclass
class CustomConstraint(ParameterConstraint):
    """User-defined constraint function.

    Examples
    --------
    >>> def my_constraint(model):
    ...     # Custom constraint logic
    ...     params = model.parameters
    ...     params["difficulty"][0] = params["difficulty"][1]
    ...     model.set_parameters(**params)
    >>> CustomConstraint("custom", apply_func=my_constraint)
    """

    apply_func: Callable[[BaseItemModel], None] | None = None
    check_func: Callable[[BaseItemModel], bool] | None = None
    penalty_func: Callable[[BaseItemModel], float] | None = None

    def apply(self, model: BaseItemModel) -> None:
        if self.apply_func is not None:
            self.apply_func(model)

    def is_satisfied(self, model: BaseItemModel, tol: float = 1e-6) -> bool:
        if self.check_func is not None:
            return self.check_func(model)
        return True

    def penalty(self, model: BaseItemModel) -> float:
        if self.penalty_func is not None:
            return self.penalty_func(model)
        return 0.0


@dataclass
class ConstraintSet:
    """Collection of constraints to apply together.

    Examples
    --------
    >>> constraints = ConstraintSet()
    >>> constraints.add(FixedConstraint("discrimination", [0], value=1.0))
    >>> constraints.add(EqualityConstraint("discrimination", [1, 2, 3]))
    >>> constraints.add(LinearConstraint("difficulty", target=0.0))
    """

    constraints: list[ParameterConstraint] = field(default_factory=list)

    def add(self, constraint: ParameterConstraint) -> ConstraintSet:
        """Add a constraint to the set."""
        self.constraints.append(constraint)
        return self

    def apply_all(self, model: BaseItemModel) -> None:
        """Apply all constraints to the model."""
        for constraint in self.constraints:
            constraint.apply(model)

    def all_satisfied(self, model: BaseItemModel, tol: float = 1e-6) -> bool:
        """Check if all constraints are satisfied."""
        return all(c.is_satisfied(model, tol) for c in self.constraints)

    def total_penalty(self, model: BaseItemModel) -> float:
        """Get total penalty for constraint violations."""
        return sum(c.penalty(model) for c in self.constraints)

    def summary(self, model: BaseItemModel) -> list[dict[str, Any]]:
        """Get summary of constraint satisfaction."""
        return [
            {
                "constraint": type(c).__name__,
                "param": c.param_name,
                "items": c.item_indices,
                "satisfied": c.is_satisfied(model),
                "penalty": c.penalty(model),
            }
            for c in self.constraints
        ]

    def __len__(self) -> int:
        return len(self.constraints)

    def __iter__(self):
        return iter(self.constraints)


def fix_parameter(
    param_name: str,
    value: float,
    items: list[int] | None = None,
) -> FixedConstraint:
    """Create a fixed value constraint."""
    return FixedConstraint(param_name, items, value)


def equal_parameters(
    param_name: str,
    items: list[int] | None = None,
) -> EqualityConstraint:
    """Create an equality constraint."""
    return EqualityConstraint(param_name, items)


def bound_parameter(
    param_name: str,
    lower: float | None = None,
    upper: float | None = None,
    items: list[int] | None = None,
) -> BoundConstraint:
    """Create a bound constraint."""
    return BoundConstraint(param_name, items, lower, upper)


def mean_constraint(
    param_name: str,
    target: float = 0.0,
    items: list[int] | None = None,
) -> LinearConstraint:
    """Create a mean constraint (for identification)."""
    return LinearConstraint(param_name, items, target, "mean")


def create_1pl_constraints() -> ConstraintSet:
    """Create standard 1PL model constraints.

    - All discriminations equal
    - Mean difficulty = 0 (identification)
    """
    return ConstraintSet(
        [
            EqualityConstraint("discrimination"),
            LinearConstraint("difficulty", target=0.0, constraint_type="mean"),
        ]
    )


def create_rasch_constraints() -> ConstraintSet:
    """Create Rasch model constraints.

    - All discriminations fixed to 1.0
    - Mean difficulty = 0 (identification)
    """
    return ConstraintSet(
        [
            FixedConstraint("discrimination", value=1.0),
            LinearConstraint("difficulty", target=0.0, constraint_type="mean"),
        ]
    )
