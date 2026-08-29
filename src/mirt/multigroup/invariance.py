from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from mirt.multigroup.model import MultigroupModel
    from mirt.multigroup.results import MultigroupFitResult


DISCRIMINATION_PARAMS = {
    "discrimination",
    "slopes",
    "loadings",
    "general_loadings",
    "specific_loadings",
}
INTERCEPT_PARAMS = {"difficulty", "intercepts", "thresholds", "steps", "location"}
INVARIANCE_LEVELS = ("configural", "metric", "scalar", "strict")


def _validate_item_indices(
    item_indices: list[int] | None,
    parameter: str,
) -> list[int] | None:
    """Validate a partial-invariance item list without changing its order."""
    if item_indices is None:
        return None
    if not isinstance(item_indices, list):
        raise ValueError(f"{parameter} must be a list of item indices or None")
    if any(
        isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer))
        for item in item_indices
    ):
        raise ValueError(f"{parameter} must contain only integer item indices")

    normalized = [int(item) for item in item_indices]
    if any(item < 0 for item in normalized):
        raise ValueError(f"{parameter} must contain only non-negative item indices")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{parameter} must not contain duplicate item indices")
    return normalized


def _merge_item_indices(current: list[int], additional: list[int]) -> None:
    """Append new indices while retaining first-appearance order."""
    seen = set(current)
    for item in additional:
        if item not in seen:
            current.append(item)
            seen.add(item)


def _parameter_count(result: MultigroupFitResult, label: str) -> int:
    """Return a validated non-negative model parameter count."""
    value = result.n_parameters
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
    ):
        raise ValueError(
            f"{label} model parameter count must be a non-negative integer"
        )
    return int(value)


@dataclass
class InvarianceSpec:
    """Specification for measurement invariance level.

    Parameters
    ----------
    level : str
        Invariance level: 'configural', 'metric', 'scalar', or 'strict'.
    free_discrimination : list[int], optional
        Items with discrimination freed for partial invariance.
    free_intercepts : list[int], optional
        Items with intercepts freed for partial invariance.
    """

    level: Literal["configural", "metric", "scalar", "strict"]
    free_discrimination: list[int] | None = None
    free_intercepts: list[int] | None = None
    _discrimination_params: set[str] = field(
        default_factory=lambda: DISCRIMINATION_PARAMS.copy(), repr=False
    )
    _intercept_params: set[str] = field(
        default_factory=lambda: INTERCEPT_PARAMS.copy(), repr=False
    )

    def __post_init__(self) -> None:
        """Reject invalid or ineffectual partial-invariance specifications."""
        if self.level not in INVARIANCE_LEVELS:
            valid = ", ".join(INVARIANCE_LEVELS)
            raise ValueError(
                f"Unknown invariance level: {self.level}. Choose from: {valid}"
            )

        self.free_discrimination = _validate_item_indices(
            self.free_discrimination,
            "free_discrimination",
        )
        self.free_intercepts = _validate_item_indices(
            self.free_intercepts,
            "free_intercepts",
        )

        if self.level == "configural" and (
            self.free_discrimination is not None or self.free_intercepts is not None
        ):
            raise ValueError(
                "partial-invariance items are not applicable to configural invariance"
            )
        if self.level == "metric" and self.free_intercepts is not None:
            raise ValueError("free_intercepts requires scalar or strict invariance")

    def get_shared_parameters(self, model: MultigroupModel) -> list[str]:
        """Get list of parameters that should be shared based on invariance level.

        Parameters
        ----------
        model : MultigroupModel
            The multigroup model.

        Returns
        -------
        list[str]
            Parameter names that should be shared.
        """
        if self.level == "configural":
            return []

        parameter_names = model.parameter_names
        if self.level == "strict":
            return list(parameter_names)

        shared_families = self._discrimination_params
        if self.level == "scalar":
            shared_families = shared_families | self._intercept_params
        return [name for name in parameter_names if name in shared_families]

    def get_free_items(self, param_name: str) -> list[int] | None:
        """Get items that should be freed for partial invariance.

        Parameters
        ----------
        param_name : str
            Parameter name.

        Returns
        -------
        list[int] | None
            List of item indices to free, or None if all shared.
        """
        if param_name in self._discrimination_params:
            return self.free_discrimination
        elif param_name in self._intercept_params:
            return self.free_intercepts
        return None

    def apply_to_model(self, model: MultigroupModel) -> None:
        """Apply invariance constraints to a multigroup model.

        Parameters
        ----------
        model : MultigroupModel
            The model to configure.
        """
        shared_params = self.get_shared_parameters(model)
        for parameter, indices in (
            ("free_discrimination", self.free_discrimination),
            ("free_intercepts", self.free_intercepts),
        ):
            if indices is not None and any(index >= model.n_items for index in indices):
                raise ValueError(
                    f"{parameter} must contain indices below {model.n_items}"
                )

        for param_name in model.parameter_names:
            model.set_group_specific_parameter(param_name)

        for param_name in shared_params:
            free_items = self.get_free_items(param_name)

            if free_items is None:
                model.set_shared_parameter(param_name)
            else:
                model.set_shared_parameter(param_name)
                model.set_group_specific_parameter(param_name, item_indices=free_items)

    def __repr__(self) -> str:
        parts = [f"level={self.level}"]
        if self.free_discrimination:
            parts.append(f"free_disc={self.free_discrimination}")
        if self.free_intercepts:
            parts.append(f"free_int={self.free_intercepts}")
        return f"InvarianceSpec({', '.join(parts)})"


def parse_invariance(
    invariance: str | InvarianceSpec,
    free_items: dict[str, list[int]] | None = None,
) -> InvarianceSpec:
    """Parse invariance specification.

    Parameters
    ----------
    invariance : str or InvarianceSpec
        Either an invariance level string or InvarianceSpec object.
    free_items : dict, optional
        For partial invariance: {param_name: [item_indices]} to free.

    Returns
    -------
    InvarianceSpec
        Parsed invariance specification.
    """
    if isinstance(invariance, InvarianceSpec):
        if free_items is not None:
            raise ValueError(
                "free_items cannot be combined with an InvarianceSpec; "
                "set partial-invariance items on the specification"
            )
        return invariance

    if invariance not in INVARIANCE_LEVELS:
        raise ValueError(
            f"Unknown invariance level: {invariance}. "
            f"Choose from: {', '.join(INVARIANCE_LEVELS)}"
        )

    free_disc: list[int] = []
    free_int: list[int] = []

    if free_items is not None:
        for param, items in free_items.items():
            validated = _validate_item_indices(items, f"free_items[{param!r}]")
            if validated is None:
                raise ValueError(f"free_items[{param!r}] must be a list")
            if param in DISCRIMINATION_PARAMS:
                _merge_item_indices(free_disc, validated)
            elif param in INTERCEPT_PARAMS:
                _merge_item_indices(free_int, validated)
            else:
                supported = sorted(DISCRIMINATION_PARAMS | INTERCEPT_PARAMS)
                raise ValueError(
                    f"Unknown free_items parameter: {param}. "
                    f"Choose from: {', '.join(supported)}"
                )

    return InvarianceSpec(
        level=invariance,
        free_discrimination=free_disc or None,
        free_intercepts=free_int or None,
    )


def invariance_lrt(
    constrained: MultigroupFitResult,
    free: MultigroupFitResult,
) -> dict[str, float]:
    """Likelihood ratio test for nested invariance models.

    Parameters
    ----------
    constrained : MultigroupFitResult
        More constrained model (e.g., metric invariance).
    free : MultigroupFitResult
        Less constrained model (e.g., configural invariance).

    Returns
    -------
    dict
        Dictionary with 'chi2', 'df', 'p_value'.

    Raises
    ------
    ValueError
        If the models are not nested (constrained should have higher -2LL).
    """
    from scipy import stats

    ll_free = float(free.log_likelihood)
    ll_constrained = float(constrained.log_likelihood)
    if not np.isfinite(ll_free) or not np.isfinite(ll_constrained):
        raise ValueError("log-likelihoods must be finite")

    scale = max(1.0, abs(ll_free), abs(ll_constrained))
    tolerance = max(1e-3, 64 * np.finfo(np.float64).eps * scale)

    if ll_constrained > ll_free + tolerance:
        raise ValueError(
            f"Models may not be nested: constrained LL ({ll_constrained:.4f}) > "
            f"free LL ({ll_free:.4f})"
        )
    chi2 = max(2.0 * (ll_free - ll_constrained), 0.0)

    free_parameters = _parameter_count(free, "free")
    constrained_parameters = _parameter_count(constrained, "constrained")
    df = free_parameters - constrained_parameters
    if df <= 0:
        raise ValueError(
            f"Constrained model must have fewer parameters: "
            f"constrained={constrained_parameters}, free={free_parameters}"
        )

    p_value = stats.chi2.sf(chi2, int(df))

    return {
        "chi2": float(chi2),
        "df": int(df),
        "p_value": float(p_value),
    }


def compute_delta_fit(
    constrained: MultigroupFitResult,
    free: MultigroupFitResult,
) -> dict[str, float]:
    """Compute change in fit indices between models.

    Parameters
    ----------
    constrained : MultigroupFitResult
        More constrained model.
    free : MultigroupFitResult
        Less constrained model.

    Returns
    -------
    dict
        Dictionary with delta values for various fit indices.
    """
    fit_values = np.asarray(
        [
            constrained.aic,
            free.aic,
            constrained.bic,
            free.bic,
            constrained.log_likelihood,
            free.log_likelihood,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(fit_values)):
        raise ValueError("fit statistics must be finite")

    delta_aic = float(constrained.aic - free.aic)
    delta_bic = float(constrained.bic - free.bic)
    delta_ll = float(constrained.log_likelihood - free.log_likelihood)

    return {
        "delta_LL": delta_ll,
        "delta_AIC": delta_aic,
        "delta_BIC": delta_bic,
    }


@dataclass
class InvarianceTestResult:
    """Result from invariance hierarchy testing."""

    comparison: str
    chi2: float
    df: int
    p_value: float
    delta_aic: float
    delta_bic: float
    significant: bool


def test_invariance_step(
    constrained: MultigroupFitResult,
    free: MultigroupFitResult,
    comparison_name: str,
    alpha: float = 0.05,
) -> InvarianceTestResult:
    """Test a single step in the invariance hierarchy.

    Parameters
    ----------
    constrained : MultigroupFitResult
        More constrained model.
    free : MultigroupFitResult
        Less constrained model.
    comparison_name : str
        Name for this comparison (e.g., "configural vs metric").
    alpha : float
        Significance level.

    Returns
    -------
    InvarianceTestResult
        Test results.
    """
    if not isinstance(comparison_name, str) or not comparison_name.strip():
        raise ValueError("comparison_name must be a non-empty string")
    if (
        isinstance(alpha, (bool, np.bool_))
        or not isinstance(alpha, (int, float, np.integer, np.floating))
        or not np.isfinite(alpha)
        or not 0.0 < alpha < 1.0
    ):
        raise ValueError("alpha must be a finite value between 0 and 1")

    lrt = invariance_lrt(constrained, free)
    delta = compute_delta_fit(constrained, free)

    return InvarianceTestResult(
        comparison=comparison_name,
        chi2=lrt["chi2"],
        df=lrt["df"],
        p_value=lrt["p_value"],
        delta_aic=delta["delta_AIC"],
        delta_bic=delta["delta_BIC"],
        significant=lrt["p_value"] < alpha,
    )


INVARIANCE_HIERARCHY = list(INVARIANCE_LEVELS)


def get_invariance_hierarchy_pairs() -> list[tuple[str, str]]:
    """Get pairs of invariance levels for sequential testing.

    Returns
    -------
    list of tuple
        Pairs of (free_level, constrained_level).
    """
    return list(zip(INVARIANCE_HIERARCHY, INVARIANCE_HIERARCHY[1:]))
