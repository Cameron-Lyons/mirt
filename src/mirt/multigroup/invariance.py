from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from scipy import stats

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

        shared = []
        param_names = set(model.parameter_names)

        if self.level in ("metric", "scalar", "strict"):
            for p in self._discrimination_params:
                if p in param_names:
                    shared.append(p)

        if self.level in ("scalar", "strict"):
            for p in self._intercept_params:
                if p in param_names:
                    shared.append(p)

        if self.level == "strict":
            for p in param_names:
                if p not in shared:
                    shared.append(p)

        return shared

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
        return invariance

    if invariance not in ("configural", "metric", "scalar", "strict"):
        raise ValueError(
            f"Unknown invariance level: {invariance}. "
            "Choose from: configural, metric, scalar, strict"
        )

    free_disc = None
    free_int = None

    if free_items is not None:
        for param, items in free_items.items():
            if param in DISCRIMINATION_PARAMS or param == "discrimination":
                free_disc = items
            elif param in INTERCEPT_PARAMS or param in ("difficulty", "intercepts"):
                free_int = items

    return InvarianceSpec(
        level=invariance,
        free_discrimination=free_disc,
        free_intercepts=free_int,
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
    ll_free = free.log_likelihood
    ll_constrained = constrained.log_likelihood

    chi2 = -2 * (ll_constrained - ll_free)

    if chi2 < -0.001:
        raise ValueError(
            f"Models may not be nested: constrained LL ({ll_constrained:.4f}) > "
            f"free LL ({ll_free:.4f})"
        )
    chi2 = max(chi2, 0.0)

    df = free.n_parameters - constrained.n_parameters
    if df <= 0:
        raise ValueError(
            f"Constrained model must have fewer parameters: "
            f"constrained={constrained.n_parameters}, free={free.n_parameters}"
        )

    p_value = 1 - stats.chi2.cdf(chi2, df)

    return {
        "chi2": chi2,
        "df": df,
        "p_value": p_value,
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
    delta_aic = constrained.aic - free.aic
    delta_bic = constrained.bic - free.bic
    delta_ll = constrained.log_likelihood - free.log_likelihood

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


INVARIANCE_HIERARCHY = ["configural", "metric", "scalar", "strict"]


def get_invariance_hierarchy_pairs() -> list[tuple[str, str]]:
    """Get pairs of invariance levels for sequential testing.

    Returns
    -------
    list of tuple
        Pairs of (free_level, constrained_level).
    """
    pairs = []
    for i in range(len(INVARIANCE_HIERARCHY) - 1):
        pairs.append((INVARIANCE_HIERARCHY[i], INVARIANCE_HIERARCHY[i + 1]))
    return pairs
