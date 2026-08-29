"""Prediction functions for mixed-effects IRT models.

Provides functions for extracting random and fixed effect
predictions from mixed-effects IRT models.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from mirt.estimation.mixed import MixedEffectsFitResult


_MIXED_PREDICTION_MAX_PROBABILITY_VALUES = 1_000_000


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
    person_intercept_standard_error, item_intercept_standard_error : float | None
        Standard errors for the corresponding intercepts.
    """

    item_parameters: dict
    covariate_effects: dict | None = None
    covariate_standard_errors: dict | None = None
    person_intercept: float | None = None
    item_intercept: float | None = None
    person_intercept_standard_error: float | None = None
    item_intercept_standard_error: float | None = None


def _effect_metadata(
    result: "MixedEffectsFitResult",
    prefix: str,
) -> tuple[NDArray[np.float64], tuple[str, ...], NDArray[np.float64]] | None:
    """Return validated coefficients, names, and standard errors for one level."""
    values = getattr(result, f"{prefix}_effects", None)
    if values is None:
        return None

    effects = np.asarray(values, dtype=np.float64)
    if effects.ndim != 1 or effects.size == 0:
        raise ValueError(f"result {prefix} effects must be a non-empty vector")
    if not np.all(np.isfinite(effects)):
        raise ValueError(f"result {prefix} effects must contain only finite values")

    raw_names = getattr(result, f"{prefix}_covariate_names", ())
    names = tuple(raw_names) if raw_names is not None else ()
    if not names:
        names = tuple(f"{prefix}_{idx}" for idx in range(effects.size))
    if len(names) != effects.size:
        raise ValueError(f"result {prefix} effect names have the wrong length")
    if any(not isinstance(name, str) or not name or ":" in name for name in names):
        raise ValueError(
            f"result {prefix} effect names must be non-empty strings without ':'"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"result {prefix} effect names must be unique")

    standard_error_values = getattr(result, f"{prefix}_effect_se", None)
    if standard_error_values is None:
        standard_errors = np.full(effects.shape, np.nan)
    else:
        standard_errors = np.asarray(standard_error_values, dtype=np.float64)
        if standard_errors.shape != effects.shape:
            raise ValueError(
                f"result {prefix} effects and standard errors must have equal shapes"
            )
        if np.any(standard_errors < 0.0) or np.any(
            ~(np.isfinite(standard_errors) | np.isnan(standard_errors))
        ):
            raise ValueError(
                f"result {prefix} standard errors must be non-negative or NaN"
            )
    return effects, names, standard_errors


def _intercept_metadata(
    result: "MixedEffectsFitResult",
    prefix: str,
) -> tuple[float, float]:
    """Return a validated intercept and its optional standard error."""
    try:
        intercept = float(getattr(result, f"{prefix}_intercept"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"result does not contain a valid {prefix} intercept") from exc
    if not np.isfinite(intercept):
        raise ValueError(f"result {prefix} intercept must be finite")

    try:
        standard_error = float(getattr(result, f"{prefix}_intercept_se", np.nan))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"result does not contain a valid {prefix} intercept standard error"
        ) from exc
    if standard_error < 0.0 or not (
        np.isfinite(standard_error) or np.isnan(standard_error)
    ):
        raise ValueError(
            f"result {prefix} intercept standard error must be non-negative or NaN"
        )
    return intercept, standard_error


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
    if theta.size == 0 or not np.all(np.isfinite(theta)):
        raise ValueError("result abilities must be non-empty and finite")

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
    if theta_se_values is not None and (
        not np.all(np.isfinite(theta_se)) or np.any(theta_se < 0.0)
    ):
        raise ValueError(
            "result ability standard errors must be finite and non-negative"
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
        group_effects=deepcopy(group_effects),
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
    if not hasattr(result, "model") or result.model is None:
        raise ValueError("result does not contain a fitted model")
    try:
        item_params = deepcopy(result.model.parameters)
    except AttributeError as exc:
        raise ValueError("result model does not expose fitted parameters") from exc

    covariate_effects: dict[str, float] = {}
    covariate_standard_errors: dict[str, float] = {}
    intercepts: dict[str, float | None] = {"person": None, "item": None}
    intercept_standard_errors: dict[str, float | None] = {
        "person": None,
        "item": None,
    }
    for prefix in ("person", "item"):
        metadata = _effect_metadata(result, prefix)
        if metadata is None:
            continue
        effects, names, standard_errors = metadata
        intercepts[prefix], intercept_standard_errors[prefix] = _intercept_metadata(
            result, prefix
        )
        for idx, name in enumerate(names):
            key = f"{prefix}:{name}"
            covariate_effects[key] = float(effects[idx])
            covariate_standard_errors[key] = float(standard_errors[idx])

    return FixedEffects(
        item_parameters=item_params,
        covariate_effects=covariate_effects or None,
        covariate_standard_errors=covariate_standard_errors or None,
        person_intercept=intercepts["person"],
        item_intercept=intercepts["item"],
        person_intercept_standard_error=intercept_standard_errors["person"],
        item_intercept_standard_error=intercept_standard_errors["item"],
    )


def predict_mixed(
    result: "MixedEffectsFitResult",
    new_theta: NDArray[np.float64] | None = None,
    new_covariates: NDArray[np.float64] | None = None,
    item_idx: int | None = None,
    *,
    integrate_uncertainty: bool = False,
    standard_errors: ArrayLike | None = None,
    n_quadpts: int = 21,
    chunk_size: int | None = None,
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
    item_idx : int, optional
        Zero-based item index. Supplying an index avoids evaluating every item.
    integrate_uncertainty : bool, optional
        Average probabilities over Gaussian ability uncertainty instead of using
        plug-in ability estimates. This is available for unidimensional models.
    standard_errors : array-like, optional
        Ability standard errors. Scalars are broadcast across people. Stored
        predictions use ``result.theta_se`` by default; covariate predictions use
        the square root of ``result.residual_variance``. Explicit abilities require
        explicit standard errors.
    n_quadpts : int, optional
        Number of Gauss-Hermite integration points, at least 2.
    chunk_size : int, optional
        Maximum people evaluated per integration chunk. By default, a bounded
        chunk size is chosen from the requested item and category dimensions.

    Returns
    -------
    NDArray[np.float64]
        Predicted probabilities. Dichotomous models return ``(n_persons,
        n_items)`` for all items or ``(n_persons,)`` for one item. Polytomous
        models add a final category dimension.

    Examples
    --------
    >>> result = MixedEffectsIRT(...).fit(responses, person_data)
    >>> # Predict at specific ability levels
    >>> new_theta = np.array([[-1], [0], [1]])
    >>> probs = predict_mixed(result, new_theta)
    >>> averaged = predict_mixed(result, integrate_uncertainty=True)
    """
    if not isinstance(integrate_uncertainty, (bool, np.bool_)):
        raise TypeError("integrate_uncertainty must be a boolean")
    if standard_errors is not None and not integrate_uncertainty:
        raise ValueError("standard_errors requires integrate_uncertainty=True")
    if integrate_uncertainty:
        if (
            isinstance(n_quadpts, (bool, np.bool_))
            or not isinstance(n_quadpts, (int, np.integer))
            or n_quadpts < 2
        ):
            raise ValueError("n_quadpts must be an integer of at least 2")
        n_quadpts = int(n_quadpts)
        if chunk_size is not None:
            if (
                isinstance(chunk_size, (bool, np.bool_))
                or not isinstance(chunk_size, (int, np.integer))
                or chunk_size < 1
            ):
                raise ValueError("chunk_size must be a positive integer")
            chunk_size = int(chunk_size)

    model = getattr(result, "model", None)
    if model is None or not callable(getattr(model, "probability", None)):
        raise ValueError("result does not contain a fitted probability model")
    n_factors = getattr(model, "n_factors", None)
    if (
        isinstance(n_factors, bool)
        or not isinstance(n_factors, (int, np.integer))
        or n_factors < 1
    ):
        raise ValueError("result model does not expose a valid factor count")
    if item_idx is not None:
        n_items = getattr(model, "n_items", None)
        if (
            isinstance(item_idx, bool)
            or not isinstance(item_idx, (int, np.integer))
            or isinstance(n_items, bool)
            or not isinstance(n_items, (int, np.integer))
            or item_idx < 0
            or item_idx >= n_items
        ):
            raise IndexError(f"item_idx must be in [0, {n_items})")
        item_idx = int(item_idx)
    if new_theta is not None and new_covariates is not None:
        raise ValueError("provide either new_theta or new_covariates, not both")

    if new_covariates is not None:
        metadata = _effect_metadata(result, "person")
        if metadata is None:
            raise ValueError("result does not contain person covariate effects")
        effects = metadata[0]
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
        intercept, _ = _intercept_metadata(result, "person")
        theta = intercept + covariates @ effects
    else:
        theta_values = (
            new_theta if new_theta is not None else getattr(result, "theta", None)
        )
        if theta_values is None:
            raise ValueError("result does not contain abilities for prediction")
        theta = np.asarray(theta_values, dtype=np.float64)

    if theta.ndim == 0:
        theta = theta.reshape(1, 1)
    elif theta.ndim == 1:
        theta = theta.reshape(-1, 1) if n_factors == 1 else theta.reshape(1, -1)
    if theta.ndim != 2 or theta.shape[1] != n_factors:
        raise ValueError(f"ability values must have {n_factors} columns")
    if not np.all(np.isfinite(theta)):
        raise ValueError("ability values must contain only finite values")

    if integrate_uncertainty:
        if n_factors != 1:
            raise ValueError(
                "uncertainty integration is available only for unidimensional models"
            )
        if standard_errors is None:
            if new_theta is not None:
                raise ValueError(
                    "standard_errors are required with explicit ability values"
                )
            if new_covariates is not None:
                try:
                    residual_variance = float(result.residual_variance)
                except (AttributeError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "result does not contain a valid residual variance"
                    ) from exc
                if residual_variance < 0.0 or not np.isfinite(residual_variance):
                    raise ValueError(
                        "result residual variance must be finite and non-negative"
                    )
                uncertainty_values: ArrayLike = np.sqrt(residual_variance)
            else:
                uncertainty_values = getattr(result, "theta_se", None)
                if uncertainty_values is None:
                    raise ValueError(
                        "result does not contain ability standard errors for integration"
                    )
        else:
            uncertainty_values = standard_errors

        uncertainty = np.asarray(uncertainty_values, dtype=np.float64)
        if uncertainty.ndim == 0:
            uncertainty = np.full(theta.shape[0], float(uncertainty))
        elif uncertainty.ndim == 2 and uncertainty.shape[1] == 1:
            uncertainty = uncertainty[:, 0]
        if uncertainty.ndim != 1 or uncertainty.shape[0] != theta.shape[0]:
            raise ValueError(
                f"standard_errors must be scalar or contain {theta.shape[0]} values"
            )
        if not np.all(np.isfinite(uncertainty)) or np.any(uncertainty < 0.0):
            raise ValueError("standard_errors must be finite and non-negative")
        return _integrated_mixed_probabilities(
            model,
            theta,
            uncertainty,
            item_idx,
            n_quadpts,
            chunk_size,
        )

    return np.asarray(model.probability(theta, item_idx=item_idx), dtype=np.float64)


def _integrated_mixed_probabilities(
    model: object,
    theta: NDArray[np.float64],
    standard_errors: NDArray[np.float64],
    item_idx: int | None,
    n_quadpts: int,
    chunk_size: int | None,
) -> NDArray[np.float64]:
    """Average response probabilities over Gaussian ability uncertainty."""
    probability = getattr(model, "probability")
    if theta.shape[0] == 0:
        return np.asarray(probability(theta, item_idx=item_idx), dtype=np.float64)

    nodes, weights = np.polynomial.hermite.hermgauss(n_quadpts)
    nodes *= np.sqrt(2.0)
    weights /= np.sqrt(np.pi)

    if chunk_size is None:
        n_items = 1 if item_idx is not None else int(getattr(model, "n_items", 1))
        category_factor = 1
        categories = getattr(model, "n_categories", None)
        if categories is not None:
            category_values = np.asarray(categories)
            category_factor = int(
                category_values[item_idx]
                if item_idx is not None
                else np.max(category_values)
            )
        values_per_person = max(1, n_quadpts * n_items * category_factor)
        chunk_size = max(
            1,
            _MIXED_PREDICTION_MAX_PROBABILITY_VALUES // values_per_person,
        )

    integrated: NDArray[np.float64] | None = None
    for start in range(0, theta.shape[0], chunk_size):
        stop = min(start + chunk_size, theta.shape[0])
        expanded = (
            theta[start:stop, 0, None]
            + standard_errors[start:stop, None] * nodes[None, :]
        )
        probabilities = np.asarray(
            probability(expanded.reshape(-1, 1), item_idx=item_idx),
            dtype=np.float64,
        )
        probabilities = probabilities.reshape(
            stop - start,
            n_quadpts,
            *probabilities.shape[1:],
        )
        averaged = np.tensordot(probabilities, weights, axes=([1], [0]))
        if integrated is None:
            integrated = np.empty(
                (theta.shape[0], *averaged.shape[1:]),
                dtype=np.float64,
            )
        integrated[start:stop] = averaged

    assert integrated is not None
    return integrated


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
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a non-empty finite one-dimensional array")
    if not isinstance(covariate_name, str) or not covariate_name:
        raise ValueError("covariate_name must be a non-empty string")

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
        metadata = _effect_metadata(result, prefix)
        if metadata is None:
            continue
        _, names, _ = metadata
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
    metadata = _effect_metadata(result, prefix)
    assert metadata is not None
    effects, _, standard_errors = metadata
    coefficient = float(effects[index])
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
            between = float(vc["between_group"])
            within = float(vc["within_group"])
            if not np.isfinite(between) or not np.isfinite(within):
                raise ValueError("variance components must be finite")
            if between < 0.0 or within < 0.0:
                raise ValueError("variance components must be non-negative")
            total_var = between + within
            if total_var > 0:
                icc = between / total_var

    return {
        "reliability": float(reliability),
        "shrinkage": float(shrinkage),
        "icc": icc,
    }
