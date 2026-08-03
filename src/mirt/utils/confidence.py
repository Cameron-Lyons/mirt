"""Likelihood and delta-method confidence intervals for IRT models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt.exceptions import MirtDataError, MirtEstimationError, MirtValidationError
from mirt.utils.numeric import logsumexp

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

ParameterName = Literal["discrimination", "difficulty"]
ScoreCIMethod = Literal["wald", "likelihood"]

_PARAMETER_NAMES = ("discrimination", "difficulty")
_SCORE_CI_METHODS = ("wald", "likelihood")
_MIN_DISCRIMINATION = 1e-8
_MAX_SEARCH_DISTANCE = 100.0


@dataclass
class PLCIResult:
    """Result of a likelihood-drop interval for an item parameter."""

    param_name: str
    param_idx: int
    estimate: float
    lower: float
    upper: float
    alpha: float
    converged: bool
    log_likelihood: float = np.nan
    critical_drop: float = np.nan
    n_evaluations: int = 0
    used_profile_callback: bool = False


def PLCI(
    model: BaseItemModel,
    responses: ArrayLike,
    param_idx: int,
    param_name: ParameterName = "discrimination",
    alpha: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-3,
    n_quadpts: int = 21,
    profile_log_likelihood: Callable[[float], float] | None = None,
) -> PLCIResult:
    """Compute a likelihood-drop interval for an item parameter.

    By default, nuisance parameters remain fixed at their fitted values and
    the marginal likelihood is evaluated with Gauss-Hermite quadrature. A
    caller that can refit a constrained model may provide
    ``profile_log_likelihood`` to obtain a full profile-likelihood interval.

    This utility currently supports scalar discrimination and difficulty
    parameters from unidimensional dichotomous models. Missing responses must
    use the package convention of a negative response code.
    """
    from scipy import stats

    _validate_interval_model(model)
    parameter = _validate_parameter_name(param_name)
    item_index = _validate_item_index(param_idx, model.n_items)
    alpha_value = _validate_alpha(alpha)
    iterations = _validate_positive_integer(max_iter, "max_iter")
    tolerance = _validate_positive_scalar(tol, "tol")
    quadrature_points = _validate_positive_integer(n_quadpts, "n_quadpts")
    data = _validate_response_matrix(responses, model.n_items)
    if not np.any(data[:, item_index] >= 0):
        raise MirtDataError(
            "the selected item has no observed responses",
            n_persons=data.shape[0],
            n_items=data.shape[1],
            item_idx=item_index,
        )

    parameters = model.parameters
    if parameter not in parameters:
        raise MirtValidationError(
            f"the model has no {parameter} parameter",
            parameter="param_name",
            value=parameter,
        )
    parameter_values = np.asarray(parameters[parameter], dtype=np.float64)
    if parameter_values.shape != (model.n_items,):
        raise MirtValidationError(
            "PLCI requires one scalar parameter per item",
            parameter="param_name",
            value=parameter_values.shape,
            expected=f"({model.n_items},)",
        )
    if parameter == "discrimination" and getattr(model, "model_name", "") == "1PL":
        raise MirtValidationError(
            "discrimination is fixed in a 1PL model",
            parameter="param_name",
            value=parameter,
        )

    estimate = float(parameter_values[item_index])
    if not np.isfinite(estimate):
        raise MirtValidationError(
            "the selected parameter estimate must be finite",
            parameter="param_idx",
            value=estimate,
        )
    if parameter == "discrimination" and estimate <= 0.0:
        raise MirtValidationError(
            "the discrimination estimate must be positive",
            parameter="param_idx",
            value=estimate,
        )

    if profile_log_likelihood is not None and not callable(profile_log_likelihood):
        raise MirtValidationError(
            "profile_log_likelihood must be callable",
            parameter="profile_log_likelihood",
        )

    if profile_log_likelihood is None:
        from mirt.estimation.quadrature import GaussHermiteQuadrature

        quadrature = GaussHermiteQuadrature(
            n_points=quadrature_points,
            n_dimensions=1,
        )
        nodes = quadrature.nodes
        weights = quadrature.weights
    else:
        nodes = np.empty((0, 1), dtype=np.float64)
        weights = np.empty(0, dtype=np.float64)

    cache: dict[float, float] = {}

    def evaluate(candidate_value: float) -> float:
        key = float(candidate_value)
        if key in cache:
            return cache[key]
        if parameter == "discrimination" and key <= 0.0:
            return -np.inf

        if profile_log_likelihood is None:
            candidate_model = model.copy()
            candidate_model.set_item_parameter(item_index, parameter, key)
            value = _marginal_log_likelihood(
                candidate_model,
                data,
                nodes,
                weights,
            )
        else:
            try:
                value = float(profile_log_likelihood(key))
            except (TypeError, ValueError, ArithmeticError) as exc:
                raise MirtEstimationError(
                    "profile_log_likelihood failed",
                    parameter_value=key,
                ) from exc

        if not np.isfinite(value):
            raise MirtEstimationError(
                "profile log-likelihood must be finite",
                parameter_value=key,
                log_likelihood=value,
            )
        cache[key] = value
        return value

    maximum_ll = evaluate(estimate)
    critical_drop = float(stats.chi2.ppf(1.0 - alpha_value, df=1) / 2.0)
    target_ll = maximum_ll - critical_drop

    def objective(candidate_value: float) -> float:
        return evaluate(candidate_value) - target_ll

    search_limit = max(_MAX_SEARCH_DISTANCE, abs(estimate) + _MAX_SEARCH_DISTANCE)
    lower_limit = (
        _MIN_DISCRIMINATION if parameter == "discrimination" else -search_limit
    )
    upper_limit = search_limit
    lower, lower_converged = _find_likelihood_bound(
        objective,
        estimate,
        direction=-1,
        limit=lower_limit,
        max_iter=iterations,
        tol=tolerance,
    )
    upper, upper_converged = _find_likelihood_bound(
        objective,
        estimate,
        direction=1,
        limit=upper_limit,
        max_iter=iterations,
        tol=tolerance,
    )
    if not lower_converged:
        lower = 0.0 if parameter == "discrimination" else -np.inf
    if not upper_converged:
        upper = np.inf

    return PLCIResult(
        param_name=parameter,
        param_idx=item_index,
        estimate=estimate,
        lower=lower,
        upper=upper,
        alpha=alpha_value,
        converged=lower_converged and upper_converged,
        log_likelihood=maximum_ll,
        critical_drop=critical_drop,
        n_evaluations=len(cache),
        used_profile_callback=profile_log_likelihood is not None,
    )


def score_CI(
    model: BaseItemModel,
    theta: float,
    responses: ArrayLike | None = None,
    alpha: float = 0.05,
    method: ScoreCIMethod = "wald",
    max_iter: int = 100,
    tol: float = 1e-4,
) -> tuple[float, float]:
    """Compute a Wald or likelihood interval for a scalar ability estimate.

    Likelihood intervals require one dichotomous response pattern. Missing
    responses use a negative code. When an extreme response pattern has an
    unbounded likelihood interval, the corresponding bound is infinite.
    """
    from scipy import stats

    _validate_interval_model(model)
    theta_value = _validate_finite_scalar(theta, "theta")
    alpha_value = _validate_alpha(alpha)
    ci_method = _validate_score_method(method)
    iterations = _validate_positive_integer(max_iter, "max_iter")
    tolerance = _validate_positive_scalar(tol, "tol")

    if ci_method == "wald":
        information = np.asarray(
            model.information(np.array([[theta_value]], dtype=np.float64)),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(information)) or np.any(information < 0.0):
            raise MirtEstimationError(
                "model information must be finite and nonnegative"
            )
        test_information = float(np.sum(information))
        if test_information <= 0.0:
            raise MirtEstimationError(
                "a Wald interval requires positive test information"
            )
        standard_error = 1.0 / np.sqrt(test_information)
        critical_value = float(stats.norm.ppf(1.0 - alpha_value / 2.0))
        margin = critical_value * standard_error
        return theta_value - margin, theta_value + margin

    if responses is None:
        raise MirtValidationError(
            "responses are required for a likelihood interval",
            parameter="responses",
        )
    response_pattern = _validate_response_pattern(responses, model.n_items)
    critical_drop = float(stats.chi2.ppf(1.0 - alpha_value, df=1) / 2.0)

    def log_likelihood(candidate_theta: float) -> float:
        value = float(
            model.log_likelihood(
                response_pattern,
                np.array([[candidate_theta]], dtype=np.float64),
            )[0]
        )
        if not np.isfinite(value):
            raise MirtEstimationError(
                "score log-likelihood must be finite",
                log_likelihood=value,
            )
        return value

    target_ll = log_likelihood(theta_value) - critical_drop

    def objective(candidate_theta: float) -> float:
        return log_likelihood(candidate_theta) - target_ll

    search_limit = max(
        _MAX_SEARCH_DISTANCE,
        abs(theta_value) + _MAX_SEARCH_DISTANCE,
    )
    lower, lower_converged = _find_likelihood_bound(
        objective,
        theta_value,
        direction=-1,
        limit=-search_limit,
        max_iter=iterations,
        tol=tolerance,
    )
    upper, upper_converged = _find_likelihood_bound(
        objective,
        theta_value,
        direction=1,
        limit=search_limit,
        max_iter=iterations,
        tol=tolerance,
    )
    if not lower_converged:
        lower = -np.inf
    if not upper_converged:
        upper = np.inf
    return lower, upper


def delta_method(
    estimates: ArrayLike,
    vcov: ArrayLike,
    transform_func: Callable[[NDArray[np.float64]], float],
    eps: float = 1e-6,
) -> tuple[float, float]:
    """Compute a transformed estimate and delta-method standard error."""
    estimate_values = _as_finite_vector(estimates, "estimates")
    covariance = _validate_covariance(vcov, estimate_values.size)
    if not callable(transform_func):
        raise MirtValidationError(
            "transform_func must be callable",
            parameter="transform_func",
        )
    step = _validate_positive_scalar(eps, "eps")
    transformed = _evaluate_transform(transform_func, estimate_values)

    gradient = np.empty(estimate_values.size, dtype=np.float64)
    for index in range(estimate_values.size):
        plus = estimate_values.copy()
        minus = estimate_values.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (
            _evaluate_transform(transform_func, plus)
            - _evaluate_transform(transform_func, minus)
        ) / (2.0 * step)

    variance = float(gradient @ covariance @ gradient)
    scale = max(1.0, float(np.max(np.abs(covariance))))
    if variance < -1e-10 * scale:
        raise MirtEstimationError(
            "delta-method variance is negative",
            variance=variance,
        )
    standard_error = np.sqrt(max(variance, 0.0))
    return transformed, float(standard_error)


def _marginal_log_likelihood(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    nodes: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> float:
    """Evaluate a stable marginal log-likelihood for all response patterns."""
    normalized_weights = np.asarray(weights, dtype=np.float64)
    if (
        normalized_weights.ndim != 1
        or normalized_weights.size != nodes.shape[0]
        or not np.all(np.isfinite(normalized_weights))
        or np.any(normalized_weights <= 0.0)
    ):
        raise MirtValidationError(
            "quadrature weights must be finite and positive",
            parameter="weights",
        )
    normalized_weights = normalized_weights / normalized_weights.sum()
    conditional_ll = np.asarray(
        model.log_likelihood_batch(responses, nodes),
        dtype=np.float64,
    )
    expected_shape = (responses.shape[0], nodes.shape[0])
    if conditional_ll.shape != expected_shape or not np.all(
        np.isfinite(conditional_ll)
    ):
        raise MirtEstimationError(
            "model returned invalid batched log-likelihoods",
            expected_shape=expected_shape,
            actual_shape=conditional_ll.shape,
        )
    marginal_ll = logsumexp(
        conditional_ll + np.log(normalized_weights)[None, :],
        axis=1,
    )
    if not np.all(np.isfinite(marginal_ll)):
        raise MirtEstimationError("marginal log-likelihood is non-finite")
    return float(np.sum(marginal_ll))


def _find_likelihood_bound(
    objective: Callable[[float], float],
    center: float,
    *,
    direction: Literal[-1, 1],
    limit: float,
    max_iter: int,
    tol: float,
) -> tuple[float, bool]:
    """Expand from a center point and solve the first likelihood crossing."""
    from scipy.optimize import brentq

    step = max(0.25, abs(center) * 0.25)
    candidate = center
    for _ in range(max_iter):
        proposed = center + direction * step
        candidate = max(limit, proposed) if direction < 0 else min(limit, proposed)
        value = float(objective(candidate))
        if not np.isfinite(value):
            raise MirtEstimationError(
                "likelihood-bound objective must be finite",
                parameter_value=candidate,
            )
        if value <= 0.0:
            bracket = (candidate, center) if direction < 0 else (center, candidate)
            try:
                root = brentq(
                    objective,
                    *bracket,
                    xtol=tol,
                    maxiter=max_iter,
                )
            except (ValueError, RuntimeError):
                return candidate, False
            return float(root), True
        if candidate == limit:
            return candidate, False
        step *= 2.0
    return candidate, False


def _validate_interval_model(model: BaseItemModel) -> None:
    if getattr(model, "n_factors", None) != 1:
        raise MirtValidationError(
            "confidence intervals currently require a unidimensional model",
            parameter="model",
        )
    if getattr(model, "is_polytomous", False):
        raise MirtValidationError(
            "confidence intervals currently require a dichotomous model",
            parameter="model",
        )


def _validate_response_matrix(
    responses: ArrayLike,
    n_items: int,
) -> NDArray[np.int_]:
    from mirt.utils.data import validate_responses

    validated = validate_responses(responses, n_items=n_items)
    observed = validated[validated >= 0]
    if np.any(observed > 1):
        raise MirtDataError(
            "dichotomous responses must be coded as 0 or 1",
            n_persons=validated.shape[0],
            n_items=validated.shape[1],
        )
    return validated


def _validate_response_pattern(
    responses: ArrayLike,
    n_items: int,
) -> NDArray[np.int_]:
    raw = np.asarray(responses)
    if raw.ndim != 1:
        raise MirtDataError("responses must be a one-dimensional response pattern")
    validated = _validate_response_matrix(raw.reshape(1, -1), n_items)
    if not np.any(validated >= 0):
        raise MirtDataError(
            "responses must contain at least one observed response",
            n_persons=1,
            n_items=n_items,
        )
    return validated


def _validate_parameter_name(param_name: str) -> str:
    if param_name not in _PARAMETER_NAMES:
        raise MirtValidationError(
            "Unknown parameter",
            parameter="param_name",
            value=param_name,
            expected=", ".join(_PARAMETER_NAMES),
        )
    return param_name


def _validate_score_method(method: str) -> str:
    if method not in _SCORE_CI_METHODS:
        raise MirtValidationError(
            "Unknown score confidence-interval method",
            parameter="method",
            value=method,
            expected=", ".join(_SCORE_CI_METHODS),
        )
    return method


def _validate_item_index(param_idx: int, n_items: int) -> int:
    if (
        not isinstance(param_idx, (int, np.integer))
        or isinstance(param_idx, (bool, np.bool_))
        or not 0 <= int(param_idx) < n_items
    ):
        raise MirtValidationError(
            "param_idx is outside the item range",
            parameter="param_idx",
            value=param_idx,
            expected=f"0 <= param_idx < {n_items}",
        )
    return int(param_idx)


def _validate_alpha(alpha: float) -> float:
    value = _validate_finite_scalar(alpha, "alpha")
    if not 0.0 < value < 1.0:
        raise MirtValidationError(
            "alpha must be between 0 and 1",
            parameter="alpha",
            value=alpha,
            expected="0 < alpha < 1",
        )
    return value


def _validate_positive_integer(value: int, parameter: str) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or int(value) < 1
    ):
        raise MirtValidationError(
            f"{parameter} must be a positive integer",
            parameter=parameter,
            value=value,
            expected=">= 1",
        )
    return int(value)


def _validate_positive_scalar(value: float, parameter: str) -> float:
    result = _validate_finite_scalar(value, parameter)
    if result <= 0.0:
        raise MirtValidationError(
            f"{parameter} must be positive",
            parameter=parameter,
            value=value,
            expected="> 0",
        )
    return result


def _validate_finite_scalar(value: float, parameter: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise MirtValidationError(
            f"{parameter} must be a finite number",
            parameter=parameter,
            value=value,
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MirtValidationError(
            f"{parameter} must be a finite number",
            parameter=parameter,
            value=value,
        ) from exc
    if not np.isfinite(result):
        raise MirtValidationError(
            f"{parameter} must be a finite number",
            parameter=parameter,
            value=value,
        )
    return result


def _as_finite_vector(values: ArrayLike, parameter: str) -> NDArray[np.float64]:
    try:
        result = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise MirtValidationError(
            f"{parameter} must contain numeric values",
            parameter=parameter,
        ) from exc
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise MirtValidationError(
            f"{parameter} must be nonempty and finite",
            parameter=parameter,
        )
    return result


def _validate_covariance(vcov: ArrayLike, size: int) -> NDArray[np.float64]:
    try:
        covariance = np.asarray(vcov, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MirtValidationError(
            "vcov must contain numeric values",
            parameter="vcov",
        ) from exc
    if covariance.shape != (size, size):
        raise MirtValidationError(
            "vcov has an incompatible shape",
            parameter="vcov",
            value=covariance.shape,
            expected=f"({size}, {size})",
        )
    if not np.all(np.isfinite(covariance)):
        raise MirtValidationError(
            "vcov must contain only finite values",
            parameter="vcov",
        )
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        raise MirtValidationError(
            "vcov must be symmetric",
            parameter="vcov",
        )
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(1.0, float(np.max(np.abs(covariance))))
    if float(eigenvalues.min()) < -1e-10 * scale:
        raise MirtValidationError(
            "vcov must be positive semidefinite",
            parameter="vcov",
            value=float(eigenvalues.min()),
        )
    return covariance


def _evaluate_transform(
    transform_func: Callable[[NDArray[np.float64]], float],
    estimates: NDArray[np.float64],
) -> float:
    value = transform_func(estimates)
    result = np.asarray(value)
    if result.ndim != 0:
        raise MirtValidationError(
            "transform_func must return one scalar",
            parameter="transform_func",
            value=result.shape,
        )
    scalar = float(result)
    if not np.isfinite(scalar):
        raise MirtValidationError(
            "transform_func must return a finite value",
            parameter="transform_func",
            value=scalar,
        )
    return scalar
