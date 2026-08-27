"""Model extraction utilities for IRT models.

Provides functions for extracting and converting model parameters
to different formats.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.constants import PROB_CLIP_MAX, PROB_CLIP_MIN, PROB_EPSILON
from mirt.exceptions import MirtDataError, MirtModelError, MirtValidationError
from mirt.utils.data import validate_responses

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


@dataclass
class ItemParameters:
    """Container for extracted item parameters.

    Attributes
    ----------
    item_idx : int
        Item index.
    model_type : str
        Model type (e.g., "2PL", "GRM").
    discrimination : NDArray[np.float64]
        Discrimination parameter(s). Shape depends on dimensionality.
    difficulty : float | NDArray[np.float64]
        Difficulty parameter(s). Scalar for dichotomous, array for polytomous.
    guessing : float | None
        Lower asymptote (for 3PL/4PL).
    slipping : float | None
        Upper asymptote (for 4PL).
    parameters : dict
        Exact parameter values for this item, keyed by the model's native
        parameter names.
    """

    item_idx: int
    model_type: str
    discrimination: NDArray[np.float64]
    difficulty: float | NDArray[np.float64]
    guessing: float | None = None
    slipping: float | None = None
    parameters: dict[str, float | NDArray[np.float64]] = field(default_factory=dict)

    @property
    def upper(self) -> float | None:
        """Upper asymptote using the fitted model's terminology."""
        return self.slipping


@dataclass
class ModelValues:
    """Container for all model parameter values.

    Attributes
    ----------
    model_type : str
        Model type (e.g., "2PL", "GRM").
    n_items : int
        Number of items.
    n_dimensions : int
        Number of latent dimensions.
    discrimination : NDArray[np.float64]
        Discrimination matrix. Shape: (n_items, n_dims).
    difficulty : NDArray[np.float64]
        Difficulty parameters. Shape depends on model.
    guessing : NDArray[np.float64] | None
        Lower asymptotes if applicable.
    slipping : NDArray[np.float64] | None
        Upper asymptotes if applicable.
    parameters : dict
        Exact model parameter arrays, keyed by their native names.
    """

    model_type: str
    n_items: int
    n_dimensions: int
    discrimination: NDArray[np.float64]
    difficulty: NDArray[np.float64]
    guessing: NDArray[np.float64] | None = None
    slipping: NDArray[np.float64] | None = None
    parameters: dict[str, NDArray[np.float64]] = field(default_factory=dict)

    @property
    def upper(self) -> NDArray[np.float64] | None:
        """Upper asymptotes using the fitted model's terminology."""
        return self.slipping


def mod2values(model: "BaseItemModel") -> ModelValues:
    """Extract all parameter values from model.

    Converts model parameters to a standardized format for
    inspection and modification.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.

    Returns
    -------
    ModelValues
        Container with all model parameters.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL")
    >>> values = mod2values(result.model)
    >>> print(f"Discriminations shape: {values.discrimination.shape}")
    >>> print(f"Mean difficulty: {np.mean(values.difficulty):.2f}")
    """
    n_items = model.n_items
    n_dims = model.n_factors
    parameters = {
        name: np.asarray(values, dtype=np.float64).copy()
        for name, values in model.parameters.items()
    }

    model_type = str(getattr(model, "model_name", model.__class__.__name__))

    disc = parameters.get("discrimination", parameters.get("slopes"))
    if disc is None:
        discrimination = np.ones((n_items, n_dims), dtype=np.float64)
    elif disc.ndim == 0 or disc.size == 1:
        discrimination = np.full((n_items, n_dims), float(disc.ravel()[0]))
    elif disc.ndim == 1:
        discrimination = disc.reshape(-1, 1)
    else:
        discrimination = disc.copy()

    difficulty = np.zeros(n_items, dtype=np.float64)
    for name in ("difficulty", "thresholds", "steps", "intercepts"):
        if name in parameters:
            difficulty = parameters[name].copy()
            break

    guessing = parameters.get("guessing")
    slipping = parameters.get("upper", parameters.get("slipping"))

    return ModelValues(
        model_type=model_type,
        n_items=n_items,
        n_dimensions=n_dims,
        discrimination=discrimination,
        difficulty=difficulty,
        guessing=None if guessing is None else guessing.copy(),
        slipping=None if slipping is None else slipping.copy(),
        parameters=parameters,
    )


def extract_item(
    model: "BaseItemModel",
    item_idx: int,
) -> ItemParameters:
    """Extract parameters for a single item.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    item_idx : int
        Index of the item to extract.

    Returns
    -------
    ItemParameters
        Container with item parameters.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL")
    >>> item = extract_item(result.model, item_idx=0)
    >>> print(f"Item 0 discrimination: {item.discrimination}")
    >>> print(f"Item 0 difficulty: {item.difficulty:.2f}")
    """
    values = mod2values(model)

    if (
        not isinstance(item_idx, (int, np.integer))
        or isinstance(item_idx, (bool, np.bool_))
        or item_idx < 0
        or item_idx >= values.n_items
    ):
        raise MirtValidationError(
            f"item_idx {item_idx} out of range [0, {values.n_items})",
            parameter="item_idx",
            value=item_idx,
            expected=f"0 <= item_idx < {values.n_items}",
        )

    discrimination = values.discrimination[item_idx]

    if values.difficulty.ndim == 1:
        difficulty = float(values.difficulty[item_idx])
    else:
        difficulty = values.difficulty[item_idx]

    guessing = None
    if values.guessing is not None:
        guessing = float(values.guessing[item_idx])

    slipping = None
    if values.slipping is not None:
        slipping = float(values.slipping[item_idx])

    item_parameters: dict[str, float | NDArray[np.float64]] = {}
    for name, parameter in values.parameters.items():
        is_shared_rating_parameter = model.model_name in {"RSM", "GRSM"} and name in {
            "discrimination",
            "thresholds",
        }
        if (
            not is_shared_rating_parameter
            and parameter.ndim > 0
            and parameter.shape[0] == values.n_items
        ):
            selected = parameter[item_idx]
        else:
            selected = parameter
        if np.ndim(selected) == 0:
            item_parameters[name] = float(selected)
        else:
            item_parameters[name] = np.asarray(selected, dtype=np.float64).copy()

    return ItemParameters(
        item_idx=item_idx,
        model_type=values.model_type,
        discrimination=discrimination,
        difficulty=difficulty,
        guessing=guessing,
        slipping=slipping,
        parameters=item_parameters,
    )


def coef(
    model: "BaseItemModel",
    irt_pars: bool = True,
) -> dict[str, NDArray[np.float64]]:
    """Extract model coefficients in dictionary format.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    irt_pars : bool
        If True, return copies of the model's native parameter arrays.
        If False, convert compatible logistic models to slope-intercept form.
        Default True.

    Returns
    -------
    dict
        Dictionary with parameter arrays.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL")
    >>> params = coef(result.model)
    >>> print(params["discrimination"])
    >>> print(params["difficulty"])
    """
    parameters = {
        name: np.asarray(values, dtype=np.float64).copy()
        for name, values in model.parameters.items()
    }

    if irt_pars:
        return parameters

    if "discrimination" not in parameters or "difficulty" not in parameters:
        raise MirtModelError(
            "Slope-intercept conversion requires discrimination and difficulty parameters",
            model_type=model.model_name,
        )

    discrimination = parameters["discrimination"]
    difficulty = parameters["difficulty"]
    if difficulty.shape != (model.n_items,):
        raise MirtModelError(
            "Slope-intercept conversion requires one difficulty per item",
            model_type=model.model_name,
            value=difficulty.shape,
            expected=f"({model.n_items},)",
        )
    if discrimination.ndim == 1:
        slopes = discrimination.reshape(-1, 1)
    elif discrimination.ndim == 2:
        slopes = discrimination
    else:
        raise MirtModelError(
            "Slope-intercept conversion requires a vector or matrix of discriminations",
            model_type=model.model_name,
            value=discrimination.shape,
        )
    if slopes.shape[0] != model.n_items:
        raise MirtModelError(
            "Discrimination parameters must have one row per item",
            model_type=model.model_name,
            value=slopes.shape,
            expected=f"({model.n_items}, n_factors)",
        )

    result = {
        "slope": slopes.copy(),
        "intercept": -slopes.sum(axis=1) * difficulty,
    }
    for name, values in parameters.items():
        if name not in {"discrimination", "difficulty"}:
            result[name] = values.copy()
    return result


def itemplot_data(
    model: "BaseItemModel",
    item_idx: int,
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_points: int = 101,
    reference_direction: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Get data for item characteristic curve plot.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    item_idx : int
        Index of item.
    theta_range : tuple
        Range of theta values.
    n_points : int
        Number of points.
    reference_direction : NDArray, optional
        Direction through the latent space for multidimensional models.
        Defaults to the first factor.

    Returns
    -------
    dict
        Dictionary with "theta", "probability", and "information" arrays.
    """
    if (
        not isinstance(item_idx, (int, np.integer))
        or isinstance(item_idx, (bool, np.bool_))
        or item_idx < 0
        or item_idx >= model.n_items
    ):
        raise MirtValidationError(
            "item_idx is out of range",
            parameter="item_idx",
            value=item_idx,
            expected=f"0 <= item_idx < {model.n_items}",
        )
    if (
        not isinstance(n_points, (int, np.integer))
        or isinstance(n_points, (bool, np.bool_))
        or n_points < 2
    ):
        raise MirtValidationError(
            "n_points must be an integer of at least 2",
            parameter="n_points",
            value=n_points,
        )
    try:
        theta_min, theta_max = (float(value) for value in theta_range)
    except (TypeError, ValueError) as exc:
        raise MirtValidationError(
            "theta_range must contain exactly two numeric values",
            parameter="theta_range",
            value=theta_range,
        ) from exc
    if not np.isfinite([theta_min, theta_max]).all() or theta_min >= theta_max:
        raise MirtValidationError(
            "theta_range must contain finite increasing bounds",
            parameter="theta_range",
            value=theta_range,
        )

    if reference_direction is None:
        direction = np.zeros(model.n_factors, dtype=np.float64)
        direction[0] = 1.0
    else:
        direction = np.asarray(reference_direction, dtype=np.float64)
        if direction.shape != (model.n_factors,) or not np.all(np.isfinite(direction)):
            raise MirtValidationError(
                "reference_direction must be a finite vector with one value per factor",
                parameter="reference_direction",
                value=direction.shape,
                expected=f"({model.n_factors},)",
            )
        norm = np.linalg.norm(direction)
        if norm <= PROB_EPSILON:
            raise MirtValidationError(
                "reference_direction must have nonzero length",
                parameter="reference_direction",
            )
        direction = direction / norm

    theta = np.linspace(theta_min, theta_max, n_points)
    theta_2d = theta[:, None] * direction[None, :]

    probs = model.probability(theta_2d, item_idx=item_idx)
    if probs.ndim > 1:
        probs = probs[:, 0] if probs.shape[1] == 1 else probs

    item_info = np.asarray(
        model.information(theta_2d, item_idx=item_idx), dtype=np.float64
    )

    return {
        "theta": theta,
        "probability": probs,
        "information": item_info,
    }


def estfun(
    model: "BaseItemModel",
    responses: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Extract empirical estimating functions from a fitted model.

    Computes the score function (gradient of log-likelihood) for each
    person. Used for sandwich estimators of standard errors and for
    detecting influential observations.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    responses : NDArray[np.float64]
        Response matrix. Shape: (n_persons, n_items).
    theta : NDArray[np.float64]
        Ability estimates. Shape: (n_persons,) or (n_persons, n_factors).

    Returns
    -------
    NDArray[np.float64]
        Estimating functions (scores) for each person-parameter combination.
        Shape: (n_persons, n_parameters). Columns follow ``model.parameters``
        insertion order and include each array's free entries in row-major order.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL")
    >>> scores = fscores(result, responses)
    >>> ef = estfun(result.model, responses, scores.theta)
    >>> # Sum should be close to zero at MLE
    >>> print(f"Sum of estimating functions: {ef.sum(axis=0)}")

    Notes
    -----
    The estimating functions are the first derivatives of the log-likelihood
    with respect to the item parameters. At the MLE, these should sum to
    approximately zero across all persons.

    Standard 1PL through 4PL scores use vectorized analytic derivatives.
    Other model families use vectorized person likelihoods with central
    finite differences, allowing the same interface to support polytomous
    and less common response models.

    These can be used to compute:
    - Robust (sandwich) standard errors
    - Influence functions for individual observations
    - Model-based residuals
    """
    raw_responses = np.asarray(responses)
    if raw_responses.dtype.kind == "f" and np.isnan(raw_responses).any():
        raw_responses = raw_responses.copy()
        raw_responses[np.isnan(raw_responses)] = -1
    validated_responses = validate_responses(
        raw_responses, n_items=model.n_items, allow_missing=True
    )

    theta_2d = np.asarray(theta, dtype=np.float64)
    if theta_2d.ndim == 1:
        if model.n_factors != 1:
            raise MirtValidationError(
                "theta must be a matrix for a multidimensional model",
                parameter="theta",
                value=theta_2d.shape,
                expected=f"(n_persons, {model.n_factors})",
            )
        theta_2d = theta_2d.reshape(-1, 1)
    if theta_2d.ndim != 2 or theta_2d.shape[1] != model.n_factors:
        raise MirtValidationError(
            "theta has an incompatible shape",
            parameter="theta",
            value=theta_2d.shape,
            expected=f"(n_persons, {model.n_factors})",
        )
    if theta_2d.shape[0] != validated_responses.shape[0]:
        raise MirtDataError(
            "theta and responses must contain the same number of persons",
            n_persons=validated_responses.shape[0],
            n_items=validated_responses.shape[1],
        )
    if not np.all(np.isfinite(theta_2d)):
        raise MirtValidationError(
            "theta values must be finite",
            parameter="theta",
        )

    valid = validated_responses >= 0
    if model.is_polytomous:
        category_counts = model.n_categories
        for item_idx, n_categories in enumerate(category_counts):
            if np.any(
                validated_responses[valid[:, item_idx], item_idx] >= n_categories
            ):
                raise MirtDataError(
                    f"responses for item {item_idx} exceed its category range",
                    n_persons=validated_responses.shape[0],
                    n_items=model.n_items,
                )
    elif np.any(validated_responses[valid] > 1):
        raise MirtDataError(
            "Dichotomous responses must be 0, 1, or missing",
            n_persons=validated_responses.shape[0],
            n_items=model.n_items,
        )

    parameters = model.parameters
    if model.model_name in {"1PL", "2PL", "3PL", "4PL"}:
        return _logistic_estfun(
            model,
            validated_responses,
            theta_2d,
            parameters,
        )
    return _numerical_estfun(model, validated_responses, theta_2d, parameters)


def _logistic_estfun(
    model: "BaseItemModel",
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    parameters: dict[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Compute vectorized conditional scores for standard logistic models."""
    discrimination = np.asarray(parameters["discrimination"], dtype=np.float64)
    if discrimination.ndim == 1:
        discrimination_2d = discrimination.reshape(-1, 1)
    else:
        discrimination_2d = discrimination
    difficulty = np.asarray(parameters["difficulty"], dtype=np.float64)

    guessing = np.asarray(
        parameters.get("guessing", np.zeros(model.n_items)), dtype=np.float64
    )
    upper = np.asarray(
        parameters.get("upper", np.ones(model.n_items)), dtype=np.float64
    )

    logits = (
        theta @ discrimination_2d.T
        - discrimination_2d.sum(axis=1)[None, :] * difficulty[None, :]
    )
    logistic = sigmoid(logits)
    probabilities = guessing[None, :] + (upper - guessing)[None, :] * logistic
    probabilities = np.clip(probabilities, PROB_CLIP_MIN, PROB_CLIP_MAX)

    valid = responses >= 0
    residual = np.where(valid, responses - probabilities, 0.0)
    likelihood_scale = residual / (probabilities * (1 - probabilities) + PROB_EPSILON)
    curve_derivative = (upper - guessing)[None, :] * logistic * (1 - logistic)

    chunks: list[NDArray[np.float64]] = []
    free_masks = model.free_parameter_masks
    for name in parameters:
        if name == "discrimination":
            derivative = curve_derivative[:, :, None] * (
                theta[:, None, :] - difficulty[None, :, None]
            )
            chunk = (likelihood_scale[:, :, None] * derivative).reshape(
                responses.shape[0], -1
            )
        elif name == "difficulty":
            derivative = -curve_derivative * discrimination_2d.sum(axis=1)[None, :]
            chunk = likelihood_scale * derivative
        elif name == "guessing":
            chunk = likelihood_scale * (1 - logistic)
        elif name == "upper":
            chunk = likelihood_scale * logistic
        else:
            return _numerical_estfun(model, responses, theta, parameters)

        free_mask = free_masks[name].ravel()
        if np.any(free_mask):
            chunks.append(chunk[:, free_mask])

    return (
        np.concatenate(chunks, axis=1)
        if chunks
        else np.empty((responses.shape[0], 0), dtype=np.float64)
    )


def _numerical_estfun(
    model: "BaseItemModel",
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    parameters: dict[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Compute conditional scores for arbitrary models with central differences."""
    n_persons = responses.shape[0]
    n_parameters = model.n_parameters
    scores = np.empty((n_persons, n_parameters), dtype=np.float64)
    work_model = model.copy()
    for name, values in parameters.items():
        work_model._parameters[name] = model._canonical_parameter_values(name, values)

    free_masks = model.free_parameter_masks
    relative_step = np.cbrt(np.finfo(np.float64).eps)

    column = 0
    for name, values in parameters.items():
        work_values = work_model._parameters[name]
        for flat_index in np.flatnonzero(free_masks[name].ravel()):
            index = np.unravel_index(flat_index, values.shape)
            original = float(work_values[index])
            step = relative_step * max(1.0, abs(original))

            work_values[index] = original + step
            ll_plus = np.asarray(
                work_model.log_likelihood(responses, theta), dtype=np.float64
            )
            work_values[index] = original - step
            ll_minus = np.asarray(
                work_model.log_likelihood(responses, theta), dtype=np.float64
            )
            work_values[index] = original

            scores[:, column] = (ll_plus - ll_minus) / (2 * step)
            column += 1

    return scores


def estfun_summary(
    model: "BaseItemModel",
    responses: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Compute summary statistics for estimating functions.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    responses : NDArray[np.float64]
        Response matrix.
    theta : NDArray[np.float64]
        Ability estimates.

    Returns
    -------
    dict
        Dictionary containing:
        - sum: Sum of estimating functions (should be ~0)
        - mean: Mean estimating function
        - var: Variance of estimating functions
        - meat: "Meat" matrix for sandwich estimator (sum of outer products)
    """
    ef = estfun(model, responses, theta)

    n_persons = ef.shape[0]

    meat = ef.T @ ef / n_persons

    return {
        "sum": ef.sum(axis=0),
        "mean": ef.mean(axis=0),
        "var": ef.var(axis=0),
        "meat": meat,
    }
