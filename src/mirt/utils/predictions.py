"""Prediction functions for mixed-effects IRT models.

Provides functions for extracting random and fixed effect
predictions from mixed-effects IRT models.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.estimation.mixed import MixedEffectsFitResult


@dataclass
class RandomEffects:
    """Container for random effect predictions.

    Attributes
    ----------
    theta : NDArray[np.float64]
        Person ability estimates (random effects).
    theta_se : NDArray[np.float64]
        Standard errors of ability estimates.
    group_effects : dict | None
        Group-level random effects if applicable.
    """

    theta: NDArray[np.float64]
    theta_se: NDArray[np.float64]
    group_effects: dict | None = None


@dataclass
class FixedEffects:
    """Container for fixed effect predictions.

    Attributes
    ----------
    item_parameters : dict
        Fixed item parameters (discrimination, difficulty, etc.).
    covariate_effects : dict | None
        Effects of person/item covariates if applicable.
    covariate_standard_errors : dict | None
        Standard errors keyed identically to ``covariate_effects``.
    person_intercept, item_intercept : float | None
        Intercepts from fitted person and item covariate regressions.
    """

    item_parameters: dict
    covariate_effects: dict | None = None
    covariate_standard_errors: dict | None = None
    person_intercept: float | None = None
    item_intercept: float | None = None


def randef(
    result: "MixedEffectsFitResult",
    level: str = "person",
) -> RandomEffects:
    """Extract random effect predictions from mixed-effects IRT model.

    Parameters
    ----------
    result : MixedEffectsFitResult
        Result from fitting a mixed-effects IRT model.
    level : str
        Level of random effects to extract:
        - "person": Person ability estimates (default)
        - "group": Group-level random effects

    Returns
    -------
    RandomEffects
        Container with random effect estimates and standard errors.

    Examples
    --------
    >>> result = MixedEffectsIRT(...).fit(responses, person_data)
    >>> re = randef(result, level="person")
    >>> print(f"Mean ability: {np.mean(re.theta):.3f}")
    """
    if level not in {"person", "group"}:
        raise ValueError(f"Unknown level: {level}. Use 'person' or 'group'.")

    theta_values = getattr(result, "theta", None)
    if theta_values is None:
        raise ValueError("result does not contain estimated person abilities")
    theta = np.asarray(theta_values, dtype=np.float64).copy()

    theta_se_values = getattr(result, "theta_se", None)
    theta_se = (
        np.full(theta.shape, np.nan, dtype=np.float64)
        if theta_se_values is None
        else np.asarray(theta_se_values, dtype=np.float64).copy()
    )
    if theta_se.shape != theta.shape:
        raise ValueError(
            "result theta and theta standard errors must have equal shapes"
        )

    group_effects = None
    if level == "group":
        group_effects = getattr(result, "group_effects", None)
        if group_effects is None:
            group_effects = getattr(result, "random_effects", None)
        if group_effects is None:
            raise ValueError("result does not contain group-level random effects")

    return RandomEffects(
        theta=theta,
        theta_se=theta_se,
        group_effects=group_effects,
    )


def fixef(
    result: "MixedEffectsFitResult",
) -> FixedEffects:
    """Extract fixed effect estimates from mixed-effects IRT model.

    Parameters
    ----------
    result : MixedEffectsFitResult
        Result from fitting a mixed-effects IRT model.

    Returns
    -------
    FixedEffects
        Container with fixed effect estimates.

    Examples
    --------
    >>> result = MixedEffectsIRT(...).fit(responses, person_data)
    >>> fe = fixef(result)
    >>> print(f"Item difficulties: {fe.item_parameters['difficulty']}")
    """
    if not hasattr(result, "model"):
        raise ValueError("result does not contain a fitted model")
    item_params = result.model.parameters

    covariate_effects: dict[str, float] = {}
    covariate_standard_errors: dict[str, float] = {}
    for prefix in ("person", "item"):
        effects = getattr(result, f"{prefix}_effects", None)
        if effects is None:
            continue
        effects = np.asarray(effects, dtype=np.float64)
        names = getattr(result, f"{prefix}_covariate_names", ()) or tuple(
            f"{prefix}_{idx}" for idx in range(len(effects))
        )
        if len(names) != len(effects):
            raise ValueError(f"result {prefix} effect names have the wrong length")
        standard_errors = getattr(result, f"{prefix}_effect_se", None)
        if standard_errors is None:
            standard_errors = np.full(effects.shape, np.nan)
        else:
            standard_errors = np.asarray(standard_errors, dtype=np.float64)
            if standard_errors.shape != effects.shape:
                raise ValueError(
                    f"result {prefix} effects and standard errors must have equal shapes"
                )
        for idx, name in enumerate(names):
            key = f"{prefix}:{name}"
            covariate_effects[key] = float(effects[idx])
            covariate_standard_errors[key] = float(standard_errors[idx])

    return FixedEffects(
        item_parameters=item_params,
        covariate_effects=covariate_effects or None,
        covariate_standard_errors=covariate_standard_errors or None,
        person_intercept=(
            float(result.person_intercept)
            if result.person_effects is not None
            else None
        ),
        item_intercept=(
            float(result.item_intercept) if result.item_effects is not None else None
        ),
    )


def predict_mixed(
    result: "MixedEffectsFitResult",
    new_theta: NDArray[np.float64] | None = None,
    new_covariates: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Predict response probabilities from mixed-effects model.

    Parameters
    ----------
    result : MixedEffectsFitResult
        Fitted mixed-effects IRT result.
    new_theta : NDArray[np.float64], optional
        New ability values. If None, uses estimated theta.
    new_covariates : NDArray[np.float64], optional
        New person covariate values. Abilities are computed from the fitted
        person intercept and effects before response probabilities are evaluated.

    Returns
    -------
    NDArray[np.float64]
        Predicted probabilities. Shape: (n_persons, n_items).

    Examples
    --------
    >>> result = MixedEffectsIRT(...).fit(responses, person_data)
    >>> # Predict at specific ability levels
    >>> new_theta = np.array([[-1], [0], [1]])
    >>> probs = predict_mixed(result, new_theta)
    """
    model = result.model
    if new_theta is not None and new_covariates is not None:
        raise ValueError("provide either new_theta or new_covariates, not both")

    if new_covariates is not None:
        effects = result.person_effects
        if effects is None:
            raise ValueError("result does not contain person covariate effects")
        effects = np.asarray(effects, dtype=np.float64)
        covariates = np.asarray(new_covariates, dtype=np.float64)
        if covariates.ndim == 1:
            covariates = (
                covariates.reshape(-1, 1)
                if len(effects) == 1
                else covariates.reshape(1, -1)
            )
        if covariates.ndim != 2 or covariates.shape[1] != len(effects):
            raise ValueError(f"new_covariates must have {len(effects)} columns")
        if not np.all(np.isfinite(covariates)):
            raise ValueError("new_covariates must contain only finite values")
        theta = result.person_intercept + covariates @ effects
    else:
        theta_values = new_theta if new_theta is not None else result.theta
        if theta_values is None:
            raise ValueError("result does not contain abilities for prediction")
        theta = np.asarray(theta_values, dtype=np.float64)

    if theta.ndim == 0:
        theta = theta.reshape(1, 1)
    elif theta.ndim == 1:
        theta = theta.reshape(-1, 1) if model.n_factors == 1 else theta.reshape(1, -1)
    if theta.ndim != 2 or theta.shape[1] != model.n_factors:
        raise ValueError(f"ability values must have {model.n_factors} columns")
    if not np.all(np.isfinite(theta)):
        raise ValueError("ability values must contain only finite values")

    return model.probability(theta)


def conditional_effects(
    result: "MixedEffectsFitResult",
    covariate_name: str,
    values: NDArray[np.float64] | list[float],
) -> dict[str, NDArray[np.float64]]:
    """Compute conditional effects at specific covariate values.

    Parameters
    ----------
    result : MixedEffectsFitResult
        Fitted mixed-effects IRT result.
    covariate_name : str
        Name of the covariate.
    values : array-like
        Covariate values at which to compute effects.

    Returns
    -------
    dict
        Dictionary with "values", "effects", and "se" arrays.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a finite one-dimensional array")

    requested_prefix = None
    requested_name = covariate_name
    if ":" in covariate_name:
        requested_prefix, requested_name = covariate_name.split(":", 1)
        if requested_prefix not in {"person", "item"}:
            raise ValueError("covariate prefix must be 'person' or 'item'")

    matches: list[tuple[str, int]] = []
    for prefix in ("person", "item"):
        if requested_prefix is not None and prefix != requested_prefix:
            continue
        effects = getattr(result, f"{prefix}_effects", None)
        if effects is None:
            continue
        names = getattr(result, f"{prefix}_covariate_names", ()) or tuple(
            f"{prefix}_{idx}" for idx in range(len(effects))
        )
        if len(names) != len(effects):
            raise ValueError(f"result {prefix} effect names have the wrong length")
        matches.extend(
            (prefix, idx) for idx, name in enumerate(names) if name == requested_name
        )

    if not matches:
        raise KeyError(f"unknown covariate: {covariate_name}")
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous covariate name {covariate_name!r}; use a person: or item: prefix"
        )

    prefix, index = matches[0]
    coefficient = float(getattr(result, f"{prefix}_effects")[index])
    standard_errors = getattr(result, f"{prefix}_effect_se", None)
    if standard_errors is None:
        coefficient_se = np.nan
    else:
        standard_errors = np.asarray(standard_errors, dtype=np.float64)
        effects = np.asarray(getattr(result, f"{prefix}_effects"), dtype=np.float64)
        if standard_errors.shape != effects.shape:
            raise ValueError(
                f"result {prefix} effects and standard errors must have equal shapes"
            )
        coefficient_se = float(standard_errors[index])
    return {
        "values": values,
        "effects": coefficient * values,
        "se": abs(coefficient_se) * np.abs(values),
    }


def shrinkage_estimates(
    result: "MixedEffectsFitResult",
) -> dict[str, float | None]:
    """Compute shrinkage statistics for random effects.

    Shrinkage measures how much random effects are pulled toward
    the population mean.

    Parameters
    ----------
    result : MixedEffectsFitResult
        Fitted mixed-effects IRT result.

    Returns
    -------
    dict
        Dictionary with shrinkage statistics:
        - "reliability": Reliability of random effects
        - "shrinkage": Proportion of shrinkage (1 - reliability)
        - "icc": Intraclass correlation if applicable
    """
    if result.theta is None or result.theta_se is None:
        raise ValueError("result must contain abilities and their standard errors")
    theta = np.asarray(result.theta, dtype=np.float64).ravel()
    theta_se = np.asarray(result.theta_se, dtype=np.float64).ravel()
    if theta.shape != theta_se.shape or theta.size == 0:
        raise ValueError(
            "abilities and standard errors must have equal non-empty shapes"
        )
    if (
        not np.all(np.isfinite(theta))
        or not np.all(np.isfinite(theta_se))
        or np.any(theta_se < 0.0)
    ):
        raise ValueError("abilities and standard errors must be finite and valid")

    obs_var = np.var(theta)
    mean_error_var = np.mean(theta_se**2)

    if obs_var > 0:
        true_var = max(obs_var - mean_error_var, 0)
        reliability = true_var / obs_var
    else:
        reliability = 0.0

    shrinkage = 1 - reliability

    icc = None
    if hasattr(result, "variance_components"):
        vc = result.variance_components
        if "between_group" in vc and "within_group" in vc:
            total_var = vc["between_group"] + vc["within_group"]
            if total_var > 0:
                icc = vc["between_group"] / total_var

    return {
        "reliability": float(reliability),
        "shrinkage": float(shrinkage),
        "icc": icc,
    }
