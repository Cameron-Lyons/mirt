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
    """

    item_parameters: dict
    covariate_effects: dict | None = None


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
    if level == "person":
        theta = result.theta if hasattr(result, "theta") else np.zeros(0)
        theta = np.atleast_1d(theta)

        if hasattr(result, "theta_se"):
            theta_se = result.theta_se
        else:
            theta_se = np.full_like(theta, np.nan)

        return RandomEffects(
            theta=theta,
            theta_se=theta_se,
            group_effects=None,
        )

    elif level == "group":
        group_effects = {}

        if hasattr(result, "group_effects"):
            group_effects = result.group_effects
        elif hasattr(result, "random_effects"):
            group_effects = result.random_effects

        theta = result.theta if hasattr(result, "theta") else np.zeros(0)
        theta_se = (
            result.theta_se
            if hasattr(result, "theta_se")
            else np.full_like(theta, np.nan)
        )

        return RandomEffects(
            theta=theta,
            theta_se=theta_se,
            group_effects=group_effects,
        )

    else:
        raise ValueError(f"Unknown level: {level}. Use 'person' or 'group'.")


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
    item_params = {}

    if hasattr(result, "model"):
        model = result.model
        if hasattr(model, "discrimination"):
            item_params["discrimination"] = np.asarray(model.discrimination)
        if hasattr(model, "difficulty"):
            item_params["difficulty"] = np.asarray(model.difficulty)
        if hasattr(model, "guessing"):
            item_params["guessing"] = np.asarray(model.guessing)

    covariate_effects = None
    if hasattr(result, "covariate_effects"):
        covariate_effects = result.covariate_effects
    elif hasattr(result, "fixed_effects"):
        covariate_effects = result.fixed_effects

    return FixedEffects(
        item_parameters=item_params,
        covariate_effects=covariate_effects,
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
        New covariate values for prediction.

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
    if new_theta is None:
        new_theta = result.theta if hasattr(result, "theta") else np.zeros((1, 1))

    new_theta = np.atleast_2d(new_theta)
    if new_theta.shape[1] == 1 and new_theta.shape[0] > 1:
        pass
    elif new_theta.ndim == 1:
        new_theta = new_theta.reshape(-1, 1)

    model = result.model
    probs = model.probability(new_theta)

    return probs


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
    values = np.asarray(values)

    if hasattr(result, "covariate_effects") and result.covariate_effects is not None:
        if covariate_name in result.covariate_effects:
            coef = result.covariate_effects[covariate_name]
            effects = coef * values

            se = np.full_like(effects, np.nan)

            return {
                "values": values,
                "effects": effects,
                "se": se,
            }

    return {
        "values": values,
        "effects": np.zeros_like(values),
        "se": np.full_like(values, np.nan),
    }


def shrinkage_estimates(
    result: "MixedEffectsFitResult",
) -> dict[str, float]:
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
    theta = result.theta if hasattr(result, "theta") else np.zeros(1)
    theta_se = (
        result.theta_se if hasattr(result, "theta_se") else np.full_like(theta, 0.1)
    )

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
