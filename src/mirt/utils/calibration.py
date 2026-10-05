"""Fixed-item calibration and test equating functions.

Provides functions for calibrating new items to an existing scale
and equating test forms. :func:`fixed_calib` uses the Rust backend for
performance when available; :func:`fixed_item_calibration` calibrates any
item model family by marginal maximum likelihood.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtDataError, MirtValidationError

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult

try:
    from mirt.backends.rust._helpers import RUST_AVAILABLE
    from mirt.backends.rust.calibration import fixed_calib_em
except ImportError:
    RUST_AVAILABLE = False


@dataclass
class CalibrationResult:
    """Result of fixed-item calibration.

    Attributes
    ----------
    new_discrimination : NDArray[np.float64]
        Calibrated discrimination parameters for new items.
    new_difficulty : NDArray[np.float64]
        Calibrated difficulty parameters for new items.
    theta : NDArray[np.float64]
        Estimated abilities on the anchor scale.
    log_likelihood : float
        Final log-likelihood.
    n_iterations : int
        Number of iterations.
    converged : bool
        Whether estimation converged.
    """

    new_discrimination: NDArray[np.float64]
    new_difficulty: NDArray[np.float64]
    theta: NDArray[np.float64]
    log_likelihood: float
    n_iterations: int
    converged: bool


@dataclass
class EquatingResult:
    """Result of test equating.

    Attributes
    ----------
    A : float
        Slope of linear transformation.
    B : float
        Intercept of linear transformation.
    method : str
        Equating method used.
    anchor_items : list[int]
        Indices of anchor items.
    rmse : float
        Root mean square error of equating.
    """

    A: float
    B: float
    method: str
    anchor_items: list[int]
    rmse: float


def _normalize_binary_responses(
    responses: NDArray[np.float64],
) -> NDArray[np.int32]:
    """Validate binary responses and normalize missing values to -1."""
    responses_array = np.asarray(responses)

    if responses_array.ndim != 2:
        raise MirtDataError(f"responses must be 2D array, got {responses_array.ndim}D")

    n_persons, n_items = responses_array.shape
    if n_persons == 0:
        raise MirtDataError("responses must contain at least one person")
    if n_items == 0:
        raise MirtDataError("responses must contain at least one item")
    if responses_array.dtype.kind not in "biuf":
        raise MirtDataError("response data must be numeric")

    nan_mask = (
        np.isnan(responses_array)
        if responses_array.dtype.kind == "f"
        else np.zeros(responses_array.shape, dtype=bool)
    )
    if responses_array.dtype.kind == "f" and np.any(
        ~np.isfinite(responses_array) & ~nan_mask
    ):
        raise MirtDataError("responses must not contain infinite values")

    missing = nan_mask | (responses_array < 0)
    observed = responses_array[~missing]
    if np.any((observed != 0) & (observed != 1)):
        raise MirtDataError(
            "fixed-item calibration requires binary responses coded 0 or 1"
        )

    normalized = np.full(responses_array.shape, -1, dtype=np.int32)
    normalized[responses_array == 0] = 0
    normalized[responses_array == 1] = 1
    return normalized


def _validate_item_indices(
    indices: list[int],
    *,
    name: str,
    n_items: int,
) -> list[int]:
    """Return validated, unique response-column indices."""
    try:
        values = list(indices)
    except TypeError as exc:
        raise MirtValidationError(
            f"{name} must be a sequence of item indices", parameter=name
        ) from exc

    if not values:
        raise MirtValidationError(
            f"{name} must contain at least one item", parameter=name
        )
    if any(
        isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
        for index in values
    ):
        raise MirtValidationError(
            f"{name} must contain integer item indices", parameter=name
        )

    validated = [int(index) for index in values]
    if len(set(validated)) != len(validated):
        raise MirtValidationError(
            f"{name} must not contain duplicate indices", parameter=name
        )
    if any(index < 0 or index >= n_items for index in validated):
        raise MirtValidationError(
            f"{name} contains an out-of-bounds item index",
            parameter=name,
            value=validated,
            expected=f"indices between 0 and {n_items - 1}",
        )
    return validated


def _validate_positive_integer(
    value: int,
    *,
    name: str,
    minimum: int = 1,
) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise MirtValidationError(
            f"{name} must be an integer greater than or equal to {minimum}",
            parameter=name,
            value=value,
        )
    return int(value)


def _validate_bounds(
    bounds: tuple[float, float],
    *,
    name: str,
) -> tuple[float, float]:
    try:
        values = np.asarray(bounds, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MirtValidationError(
            f"{name} must contain two finite numbers", parameter=name
        ) from exc

    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise MirtValidationError(
            f"{name} must contain two finite numbers", parameter=name
        )
    lower, upper = float(values[0]), float(values[1])
    if lower >= upper:
        raise MirtValidationError(
            f"{name} lower bound must be less than its upper bound",
            parameter=name,
            value=(lower, upper),
        )
    return lower, upper


def _validate_finite_number(value: float, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise MirtValidationError(
            f"{name} must be a finite number", parameter=name, value=value
        )
    try:
        validated = float(value)
    except (TypeError, ValueError) as exc:
        raise MirtValidationError(
            f"{name} must be a finite number", parameter=name, value=value
        ) from exc
    if not np.isfinite(validated):
        raise MirtValidationError(
            f"{name} must be a finite number", parameter=name, value=value
        )
    return validated


def _item_log_likelihood(
    responses: NDArray[np.int32],
    theta_grid: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute stable person-by-quadrature item log likelihoods."""
    logits = discrimination[None, :] * (theta_grid[:, None] - difficulty[None, :])
    log_probability = -np.logaddexp(0.0, -logits)
    log_complement = -np.logaddexp(0.0, logits)

    correct = (responses == 1).astype(np.float64)
    incorrect = (responses == 0).astype(np.float64)
    return correct @ log_probability.T + incorrect @ log_complement.T


def _fixed_calib_e_step(
    anchor_log_likelihood: NDArray[np.float64],
    new_responses: NDArray[np.int32],
    theta_grid: NDArray[np.float64],
    log_weights: NDArray[np.float64],
    new_discrimination: NDArray[np.float64],
    new_difficulty: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    new_log_likelihood = _item_log_likelihood(
        new_responses,
        theta_grid,
        new_discrimination,
        new_difficulty,
    )
    log_joint = anchor_log_likelihood + new_log_likelihood + log_weights[None, :]
    row_maximum = np.max(log_joint, axis=1, keepdims=True)
    scaled_joint = np.exp(log_joint - row_maximum)
    scaled_marginal = np.sum(scaled_joint, axis=1, keepdims=True)
    posterior = scaled_joint / scaled_marginal
    log_marginal = row_maximum[:, 0] + np.log(scaled_marginal[:, 0])
    return posterior, float(np.sum(log_marginal))


def _fixed_calib_m_step(
    responses: NDArray[np.int32],
    posterior: NDArray[np.float64],
    theta_grid: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    disc_bounds: tuple[float, float],
    diff_bounds: tuple[float, float],
    prob_clamp: tuple[float, float],
    min_count: float,
    min_valid_points: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Update new-item parameters from posterior expected counts."""
    valid = (responses >= 0).astype(np.float64)
    correct = (responses == 1).astype(np.float64)
    counts = valid.T @ posterior
    correct_counts = correct.T @ posterior

    updated_disc = discrimination.copy()
    updated_diff = difficulty.copy()
    valid_points = counts > min_count
    eligible = np.count_nonzero(valid_points, axis=1) >= min_valid_points
    if not np.any(eligible):
        return updated_disc, updated_diff

    proportions = correct_counts / np.maximum(counts, PROB_EPSILON)
    proportions = np.clip(proportions, *prob_clamp)
    logits = np.log(proportions / (1.0 - proportions))
    regression_weights = np.where(valid_points, counts, 0.0)
    weight_sums = np.sum(regression_weights, axis=1)
    safe_weight_sums = np.where(weight_sums > 0.0, weight_sums, 1.0)

    mean_theta = (
        np.sum(regression_weights * theta_grid[None, :], axis=1) / safe_weight_sums
    )
    mean_logit = np.sum(regression_weights * logits, axis=1) / safe_weight_sums
    centered_theta = theta_grid[None, :] - mean_theta[:, None]
    variance = (
        np.sum(
            regression_weights * centered_theta * centered_theta,
            axis=1,
        )
        / safe_weight_sums
    )
    covariance = (
        np.sum(
            regression_weights * centered_theta * (logits - mean_logit[:, None]),
            axis=1,
        )
        / safe_weight_sums
    )

    eligible &= variance > PROB_EPSILON
    fitted_disc = np.clip(
        np.divide(
            covariance,
            variance,
            out=np.zeros_like(covariance),
            where=eligible,
        ),
        *disc_bounds,
    )
    fitted_diff = np.clip(mean_theta - mean_logit / fitted_disc, *diff_bounds)
    updated_disc[eligible] = fitted_disc[eligible]
    updated_diff[eligible] = fitted_diff[eligible]

    return updated_disc, updated_diff


def fixed_calib(
    responses: NDArray[np.float64],
    anchor_model: "BaseItemModel",
    anchor_items: list[int],
    new_items: list[int] | None = None,
    model_type: str = "2PL",
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    use_rust: bool = True,
    disc_bounds: tuple[float, float] = (0.2, 5.0),
    diff_bounds: tuple[float, float] = (-5.0, 5.0),
    prob_clamp: tuple[float, float] = (0.01, 0.99),
    init_disc: float = 1.0,
    init_diff: float = 0.0,
    min_count: float = 1.0,
    min_valid_points: int = 3,
) -> CalibrationResult:
    """Calibrate new items while holding anchor items fixed.

    This function estimates parameters for new items on a scale
    defined by anchor items with known (fixed) parameters.

    Parameters
    ----------
    responses : NDArray[np.float64]
        Binary response matrix containing both anchor and new items, with shape
        ``(n_persons, n_total_items)``. NaN and negative values are treated as
        missing.
    anchor_model : BaseItemModel
        Model with fixed parameters for anchor items.
    anchor_items : list of int
        Column indices of anchor items in responses.
    new_items : list of int, optional
        Column indices of new items. If None, all non-anchor items.
    model_type : str
        Model type for new items. Default "2PL".
    n_quadpts : int
        Number of quadrature points. Default 21.
    max_iter : int
        Maximum iterations. Default 500.
    tol : float
        Convergence tolerance. Default 1e-4.
    use_rust : bool
        Use Rust backend for performance. Default True.
    disc_bounds : tuple[float, float]
        Bounds for discrimination parameters (min, max). Default (0.2, 5.0).
    diff_bounds : tuple[float, float]
        Bounds for difficulty parameters (min, max). Default (-5.0, 5.0).
    prob_clamp : tuple[float, float]
        Bounds for probability clipping (min, max). Default (0.01, 0.99).
    init_disc : float
        Initial discrimination value. Default 1.0.
    init_diff : float
        Initial difficulty value. Default 0.0.
    min_count : float
        Minimum count threshold for valid quadrature points. Default 1.0.
    min_valid_points : int
        Minimum number of valid points for regression. Default 3.

    Returns
    -------
    CalibrationResult
        Calibrated parameters for new items.

    Examples
    --------
    >>> result = fixed_calib(
    ...     responses=all_responses,
    ...     anchor_model=existing_model,
    ...     anchor_items=[0, 1, 2, 3, 4],
    ... )
    >>> print(f"New item difficulties: {result.new_difficulty}")
    """
    responses_int = _normalize_binary_responses(responses)
    n_total_items = responses_int.shape[1]

    if not isinstance(model_type, str) or model_type.casefold() != "2pl":
        raise MirtValidationError(
            "fixed-item calibration currently supports only the 2PL model",
            parameter="model_type",
            value=model_type,
            expected="2PL",
        )

    n_quadpts = _validate_positive_integer(n_quadpts, name="n_quadpts", minimum=2)
    max_iter = _validate_positive_integer(max_iter, name="max_iter")
    min_valid_points = _validate_positive_integer(
        min_valid_points, name="min_valid_points", minimum=2
    )
    if min_valid_points > n_quadpts:
        raise MirtValidationError(
            "min_valid_points must not exceed n_quadpts",
            parameter="min_valid_points",
            value=min_valid_points,
        )

    tol = _validate_finite_number(tol, name="tol")
    min_count = _validate_finite_number(min_count, name="min_count")
    init_disc = _validate_finite_number(init_disc, name="init_disc")
    init_diff = _validate_finite_number(init_diff, name="init_diff")
    if tol <= 0:
        raise MirtValidationError(
            "tol must be a positive finite number", parameter="tol", value=tol
        )
    if min_count < 0:
        raise MirtValidationError(
            "min_count must be a non-negative finite number",
            parameter="min_count",
            value=min_count,
        )

    disc_bounds = _validate_bounds(disc_bounds, name="disc_bounds")
    diff_bounds = _validate_bounds(diff_bounds, name="diff_bounds")
    prob_clamp = _validate_bounds(prob_clamp, name="prob_clamp")
    if disc_bounds[0] <= 0:
        raise MirtValidationError(
            "disc_bounds must be strictly positive", parameter="disc_bounds"
        )
    if not 0 < prob_clamp[0] < prob_clamp[1] < 1:
        raise MirtValidationError(
            "prob_clamp values must lie strictly between 0 and 1",
            parameter="prob_clamp",
        )
    if not disc_bounds[0] <= init_disc <= disc_bounds[1]:
        raise MirtValidationError(
            "init_disc must be finite and within disc_bounds",
            parameter="init_disc",
            value=init_disc,
        )
    if not diff_bounds[0] <= init_diff <= diff_bounds[1]:
        raise MirtValidationError(
            "init_diff must be finite and within diff_bounds",
            parameter="init_diff",
            value=init_diff,
        )

    anchor_items = _validate_item_indices(
        anchor_items, name="anchor_items", n_items=n_total_items
    )

    if new_items is None:
        anchor_set = set(anchor_items)
        new_items = [i for i in range(n_total_items) if i not in anchor_set]
    new_items = _validate_item_indices(
        new_items, name="new_items", n_items=n_total_items
    )
    if set(anchor_items) & set(new_items):
        raise MirtValidationError(
            "anchor_items and new_items must be disjoint",
            parameter="new_items",
        )

    if getattr(anchor_model, "n_factors", 1) != 1:
        raise MirtValidationError(
            "fixed-item calibration requires a unidimensional anchor model",
            parameter="anchor_model",
        )
    if getattr(anchor_model, "n_items", len(anchor_items)) != len(anchor_items):
        raise MirtValidationError(
            "anchor_model must contain one parameter set per anchor item",
            parameter="anchor_model",
            expected=f"{len(anchor_items)} items",
        )

    try:
        anchor_disc = np.asarray(anchor_model.discrimination, dtype=np.float64).reshape(
            -1
        )
        anchor_diff = np.asarray(anchor_model.difficulty, dtype=np.float64).reshape(-1)
    except (AttributeError, TypeError, ValueError) as exc:
        raise MirtValidationError(
            "anchor_model must provide numeric discrimination and difficulty parameters",
            parameter="anchor_model",
        ) from exc

    if anchor_disc.size != len(anchor_items) or anchor_diff.size != len(anchor_items):
        raise MirtValidationError(
            "anchor_model parameter counts must match anchor_items",
            parameter="anchor_model",
        )
    if (
        not np.all(np.isfinite(anchor_disc))
        or not np.all(anchor_disc > 0)
        or not np.all(np.isfinite(anchor_diff))
    ):
        raise MirtValidationError(
            "anchor_model parameters must be finite with positive discriminations",
            parameter="anchor_model",
        )

    can_use_rust = use_rust and RUST_AVAILABLE

    theta_grid = np.linspace(-4, 4, n_quadpts)
    weights = np.exp(-0.5 * theta_grid**2)
    weights = weights / np.sum(weights)

    if can_use_rust:
        new_disc, new_diff, theta_est, log_likelihood, n_iterations, converged = (
            fixed_calib_em(
                responses_int,
                anchor_items,
                new_items,
                anchor_disc.astype(np.float64),
                anchor_diff.astype(np.float64),
                theta_grid.astype(np.float64),
                weights.astype(np.float64),
                max_iter,
                tol,
                disc_bounds,
                diff_bounds,
                prob_clamp,
                init_disc,
                init_diff,
                min_count,
                min_valid_points,
            )
        )

        return CalibrationResult(
            new_discrimination=np.asarray(new_disc),
            new_difficulty=np.asarray(new_diff),
            theta=np.asarray(theta_est),
            log_likelihood=float(log_likelihood),
            n_iterations=int(n_iterations),
            converged=bool(converged),
        )

    n_new = len(new_items)
    new_disc = np.full(n_new, init_disc)
    new_diff = np.full(n_new, init_diff)
    anchor_responses = responses_int[:, anchor_items]
    new_responses = responses_int[:, new_items]
    anchor_ll = _item_log_likelihood(
        anchor_responses, theta_grid, anchor_disc, anchor_diff
    )
    log_weights = np.log(weights)

    converged = False
    previous_log_likelihood = -np.inf

    for iteration in range(max_iter):
        posterior, log_likelihood = _fixed_calib_e_step(
            anchor_ll,
            new_responses,
            theta_grid,
            log_weights,
            new_disc,
            new_diff,
        )
        if abs(log_likelihood - previous_log_likelihood) < tol:
            converged = True
            break
        previous_log_likelihood = log_likelihood

        old_disc = new_disc.copy()
        old_diff = new_diff.copy()
        new_disc, new_diff = _fixed_calib_m_step(
            new_responses,
            posterior,
            theta_grid,
            new_disc,
            new_diff,
            disc_bounds,
            diff_bounds,
            prob_clamp,
            min_count,
            min_valid_points,
        )

        param_change = np.max(np.abs(new_disc - old_disc)) + np.max(
            np.abs(new_diff - old_diff)
        )
        if param_change < tol:
            converged = True
            break

    posterior, log_likelihood = _fixed_calib_e_step(
        anchor_ll,
        new_responses,
        theta_grid,
        log_weights,
        new_disc,
        new_diff,
    )
    theta_est = posterior @ theta_grid

    return CalibrationResult(
        new_discrimination=new_disc,
        new_difficulty=new_diff,
        theta=theta_est,
        log_likelihood=log_likelihood,
        n_iterations=iteration + 1,
        converged=converged,
    )


@dataclass
class FixedItemCalibrationResult:
    """Result of marginal maximum likelihood fixed-item calibration.

    Attributes
    ----------
    fit_result : FitResult
        EM fit of the calibration model. Anchor parameters keep their supplied
        values and report zero standard errors.
    latent_mean : ndarray of shape (n_factors,)
        Estimated mean of the calibration population on the anchor scale.
    latent_cov : ndarray of shape (n_factors, n_factors)
        Estimated covariance of the calibration population.
    anchor_items : list of int
        Response columns whose parameters were held fixed.
    new_items : list of int
        Response columns whose parameters were estimated.
    """

    fit_result: "FitResult"
    latent_mean: NDArray[np.float64]
    latent_cov: NDArray[np.float64]
    anchor_items: list[int]
    new_items: list[int]

    @property
    def model(self) -> "BaseItemModel":
        """Calibrated model containing the anchor and new items."""
        return self.fit_result.model

    @property
    def new_item_parameters(self) -> dict[str, NDArray[np.float64]]:
        """Item-indexed parameters of the new items only."""
        model = self.fit_result.model
        return {
            name: values[self.new_items]
            for name, values in model.parameters.items()
            if values.ndim >= 1 and values.shape[0] == model.n_items
        }


def _anchor_parameter_values(
    model: "BaseItemModel",
    anchor_items: list[int],
    anchor_parameters: "BaseItemModel | FitResult | Mapping[str, ArrayLike]",
) -> dict[str, NDArray[np.float64]]:
    """Return complete parameter arrays with the anchor rows replaced."""
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult

    if isinstance(anchor_parameters, FitResult):
        anchor_parameters = anchor_parameters.model
    if isinstance(anchor_parameters, BaseItemModel):
        if anchor_parameters.n_items != len(anchor_items):
            raise MirtValidationError(
                "the anchor model must contain one item per anchor item",
                parameter="anchor_parameters",
                value=anchor_parameters.n_items,
                expected=str(len(anchor_items)),
            )
        anchor_counts = getattr(anchor_parameters, "n_categories", None)
        model_counts = getattr(model, "n_categories", None)
        if (anchor_counts is None) != (model_counts is None) or (
            model_counts is not None
            and not np.array_equal(
                np.broadcast_to(anchor_counts, len(anchor_items)),
                np.broadcast_to(model_counts, model.n_items)[anchor_items],
            )
        ):
            raise MirtValidationError(
                "anchor items must have the same category counts in both models",
                parameter="anchor_parameters",
            )
        supplied: Mapping[str, ArrayLike] = anchor_parameters.parameters
    elif isinstance(anchor_parameters, Mapping):
        supplied = anchor_parameters
    else:
        raise MirtValidationError(
            "anchor_parameters must be a model, a fit result, or a mapping",
            parameter="anchor_parameters",
        )

    current = model.parameters
    unknown = sorted(set(supplied) - set(current))
    if unknown:
        raise MirtValidationError(
            f"Unknown anchor parameters: {', '.join(unknown)}",
            parameter="anchor_parameters",
            expected=", ".join(current),
        )
    masks = model.free_parameter_masks
    updated: dict[str, NDArray[np.float64]] = {}
    for name, values in current.items():
        item_indexed = values.ndim >= 1 and values.shape[0] == model.n_items
        if name not in supplied:
            # Omission is safe only where no anchor coordinate is free.
            if np.any(masks[name][anchor_items] if item_indexed else masks[name]):
                raise MirtValidationError(
                    f"anchor_parameters must provide {name}",
                    parameter="anchor_parameters",
                )
            continue
        try:
            replacement = np.asarray(supplied[name], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                f"anchor parameter {name} must be numeric",
                parameter="anchor_parameters",
            ) from exc
        expected = (
            (len(anchor_items), *values.shape[1:]) if item_indexed else values.shape
        )
        if replacement.shape != expected:
            raise MirtValidationError(
                f"anchor parameter {name} must have shape {expected}",
                parameter="anchor_parameters",
                value=replacement.shape,
            )
        if not np.all(np.isfinite(replacement)):
            raise MirtValidationError(
                f"anchor parameter {name} must be finite",
                parameter="anchor_parameters",
            )
        if item_indexed:
            values[anchor_items] = replacement
        else:
            values = replacement.copy()
        updated[name] = values
    return updated


def fixed_item_calibration(
    responses: ArrayLike,
    model: "BaseItemModel",
    anchor_items: list[int],
    anchor_parameters: "BaseItemModel | FitResult | Mapping[str, ArrayLike] | None" = None,
    *,
    estimate_mean: bool = True,
    estimate_cov: bool = True,
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-5,
    compute_standard_errors: bool = True,
) -> FixedItemCalibrationResult:
    """Calibrate new items by MML while anchor items stay fixed (FIPC).

    Fixed-item parameter calibration places new items on the scale of
    previously calibrated anchor items: anchor parameters are held at their
    known values, while EM estimates the new items' parameters together with
    the mean and covariance of the calibration population. Any dichotomous or
    polytomous item model family is supported, unlike :func:`fixed_calib`,
    which fits only 2PL items under a fixed standard normal population.

    Parameters
    ----------
    responses : array-like of shape (n_persons, n_items)
        Responses to the anchor and new items. Negative codes and ``NaN`` are
        missing, so new items given to a subsample may be calibrated as well.
    model : BaseItemModel
        Item model covering every response column, for example
        ``GradedResponseModel(n_items, n_categories=5)``. It is copied, not
        modified. Its existing free-parameter masks are respected, and the
        current values of new-item parameters are their starting values.
    anchor_items : list of int
        Response columns of the anchor items.
    anchor_parameters : BaseItemModel, FitResult or mapping, optional
        Known anchor parameters. A model or fit result must contain exactly
        the anchor items, in ``anchor_items`` order, with parameter arrays
        shaped like ``model``'s. A mapping gives each parameter's anchor rows
        by name; parameters whose anchor coordinates are fixed by the model
        family, such as 1PL discriminations, may be omitted. By default the
        anchor rows already stored in ``model`` are used.
    estimate_mean : bool, default=True
        Whether to estimate the calibration population's latent mean.
    estimate_cov : bool, default=True
        Whether to estimate the calibration population's latent covariance.
    n_quadpts : int, default=21
        Quadrature points per latent dimension.
    max_iter : int, default=500
        Maximum EM iterations.
    tol : float, default=1e-5
        Convergence tolerance for the change in marginal log-likelihood.
    compute_standard_errors : bool, default=True
        Whether to compute standard errors of the new-item parameters.

    Returns
    -------
    FixedItemCalibrationResult
        Fit result, estimated latent mean and covariance, and the anchor and
        new item indices.

    Notes
    -----
    Parameters shared by all items, such as rating-scale thresholds, belong to
    the anchored scale and remain fixed. Standard errors are conditional on
    the estimated latent mean and covariance. To score people on the anchor
    scale, pass the estimated population as the scoring prior, for example
    ``fscores(result.model, responses, prior_mean=result.latent_mean,
    prior_cov=result.latent_cov)``.

    Examples
    --------
    >>> import numpy as np
    >>> from mirt import TwoParameterLogistic, simdata
    >>> from mirt.utils.calibration import fixed_item_calibration
    >>> a = np.linspace(0.8, 1.6, 10)
    >>> b = np.linspace(-1.5, 1.5, 10)
    >>> theta = np.random.default_rng(1).normal(0.5, 1.2, 800)
    >>> data = simdata(theta=theta, discrimination=a, difficulty=b, seed=1)
    >>> anchors = TwoParameterLogistic(6).set_parameters(
    ...     discrimination=a[:6], difficulty=b[:6]
    ... )
    >>> result = fixed_item_calibration(
    ...     data, TwoParameterLogistic(10), list(range(6)), anchors
    ... )
    >>> bool(np.allclose(result.model.parameters["difficulty"][:6], b[:6]))
    True
    """
    from mirt.estimation.em import EMEstimator
    from mirt.estimation.latent_density import GaussianDensity
    from mirt.models.base import BaseItemModel
    from mirt.utils.data import validate_responses

    if not isinstance(model, BaseItemModel):
        raise MirtValidationError(
            "model must be an item model covering every response column",
            parameter="model",
        )
    for name, flag in (
        ("estimate_mean", estimate_mean),
        ("estimate_cov", estimate_cov),
        ("compute_standard_errors", compute_standard_errors),
    ):
        if not isinstance(flag, (bool, np.bool_)):
            raise MirtValidationError(
                f"{name} must be a boolean", parameter=name, value=flag
            )
    validated = validate_responses(responses, n_items=model.n_items)
    anchor_items = _validate_item_indices(
        anchor_items, name="anchor_items", n_items=model.n_items
    )
    anchor_set = set(anchor_items)
    new_items = [item for item in range(model.n_items) if item not in anchor_set]
    if not new_items:
        raise MirtValidationError(
            "at least one response column must be a new item",
            parameter="anchor_items",
        )

    calibration_model = model.copy()
    if anchor_parameters is not None:
        calibration_model.set_parameters(
            **_anchor_parameter_values(
                calibration_model, anchor_items, anchor_parameters
            )
        )
    masks = calibration_model.free_parameter_masks
    for name, values in calibration_model.parameters.items():
        if values.ndim >= 1 and values.shape[0] == calibration_model.n_items:
            masks[name][anchor_items] = False
        else:
            masks[name][...] = False
    if not any(np.any(mask) for mask in masks.values()):
        raise MirtValidationError(
            "the new items have no free parameters to calibrate",
            parameter="model",
        )
    calibration_model.set_free_parameter_masks(masks)
    # Fitted models keep their values as EM starting values, so the anchor
    # coordinates are never reinitialized.
    calibration_model._is_fitted = True

    density = GaussianDensity(
        n_dimensions=calibration_model.n_factors,
        estimate_mean=bool(estimate_mean),
        estimate_cov=bool(estimate_cov),
    )
    estimator = EMEstimator(
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        latent_density=density,
        compute_standard_errors=bool(compute_standard_errors),
    )
    fit_result = estimator.fit(calibration_model, validated)
    return FixedItemCalibrationResult(
        fit_result=fit_result,
        latent_mean=np.array(density.mean, dtype=np.float64),
        latent_cov=np.array(density.cov, dtype=np.float64),
        anchor_items=anchor_items,
        new_items=new_items,
    )


def equate(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    anchor_items_old: list[int],
    anchor_items_new: list[int],
    method: Literal[
        "mean_sigma", "mean_mean", "stocking_lord", "haebara"
    ] = "stocking_lord",
) -> EquatingResult:
    """Equate two test forms using anchor items.

    .. deprecated::
        This function is deprecated. Use :func:`mirt.equating.link` instead,
        which provides more linking methods, diagnostics, and polytomous support.

    Finds transformation constants A and B such that:
        theta_new = A * theta_old + B

    The estimator is :func:`mirt.equating.link`, whose constants describe the
    opposite direction (``theta_old = A_link * theta_new + B_link``), so
    ``A = 1 / A_link`` and ``B = -B_link / A_link``.

    Parameters
    ----------
    model_old : BaseItemModel
        Model for old/reference form.
    model_new : BaseItemModel
        Model for new form.
    anchor_items_old : list of int
        Indices of anchor items in old model.
    anchor_items_new : list of int
        Indices of anchor items in new model.
    method : str
        Equating method:
        - "mean_sigma": Mean/sigma method
        - "mean_mean": Mean/mean method
        - "stocking_lord": Stocking-Lord method (characteristic curve)
        - "haebara": Haebara method

    Returns
    -------
    EquatingResult
        Transformation constants. ``rmse`` combines the discrimination and
        difficulty RMSEs of the anchors after placing the new form on the old
        scale; it is zero for an exact linear relation.

    See Also
    --------
    mirt.equating.link : Recommended replacement with more features.

    Examples
    --------
    >>> eq = equate(old_model, new_model, [0,1,2], [0,1,2])
    >>> theta_on_old_scale = transform_theta(theta_new, eq)
    """
    import warnings

    from mirt.equating.linking import link

    warnings.warn(
        "equate() is deprecated. Use mirt.equating.link() instead for more "
        "linking methods, diagnostics, and polytomous/multidimensional support.",
        DeprecationWarning,
        stacklevel=2,
    )
    if method not in ("mean_sigma", "mean_mean", "stocking_lord", "haebara"):
        raise ValueError(f"Unknown equating method: {method}")

    linked = link(
        model_old,
        model_new,
        anchor_items_old,
        anchor_items_new,
        method=method,
        compute_diagnostics=True,
    )
    assert linked.fit_statistics is not None
    A_link, B_link = linked.constants.A, linked.constants.B

    return EquatingResult(
        A=float(1.0 / A_link),
        B=float(-B_link / A_link),
        method=method,
        anchor_items=linked.anchor_items,
        rmse=float(linked.fit_statistics.weighted_rmse),
    )


def transform_theta(
    theta: NDArray[np.float64],
    equating_result: EquatingResult,
) -> NDArray[np.float64]:
    """Transform theta estimates using equating constants.

    :func:`equate` returns constants for ``theta_new = A * theta_old + B``,
    so new-form estimates map onto the old scale as ``(theta - B) / A``.

    Parameters
    ----------
    theta : NDArray[np.float64]
        Theta estimates from new form.
    equating_result : EquatingResult
        Result from equate() function.

    Returns
    -------
    NDArray[np.float64]
        Transformed theta on old/reference scale.
    """
    scale = _validate_finite_number(equating_result.A, name="A")
    shift = _validate_finite_number(equating_result.B, name="B")
    if scale <= 0.0:
        raise MirtValidationError("A must be positive", parameter="A", value=scale)
    return (np.asarray(theta, dtype=np.float64) - shift) / scale
