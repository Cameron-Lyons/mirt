"""Statistical tests for IRT models.

Provides Wald and Lagrange (score) tests for parameter constraints
and model comparison.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from mirt._core import sigmoid
from mirt.constants import PROB_EPSILON

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


@dataclass
class WaldTestResult:
    """Result of a Wald test.

    Attributes
    ----------
    statistic : float
        Wald chi-square statistic.
    df : int
        Degrees of freedom.
    p_value : float
        P-value for the test.
    parameter_estimates : NDArray[np.float64]
        Parameter estimates or linear combinations being tested.
    standard_errors : NDArray[np.float64]
        Standard errors of estimates.
    constraint_values : NDArray[np.float64]
        Values under null hypothesis.
    """

    statistic: float
    df: int
    p_value: float
    parameter_estimates: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    constraint_values: NDArray[np.float64]


@dataclass
class LagrangeTestResult:
    """Result of a Lagrange (score) test.

    Attributes
    ----------
    statistic : float
        Score chi-square statistic.
    df : int
        Degrees of freedom.
    p_value : float
        P-value for the test.
    scores : NDArray[np.float64]
        Score vector (gradient) for constrained parameters.
    """

    statistic: float
    df: int
    p_value: float
    scores: NDArray[np.float64]


ParameterLayout = list[tuple[str, tuple[int, ...], slice]]


def _model_parameter_layout(
    model: "BaseItemModel",
) -> tuple[NDArray[np.float64], ParameterLayout]:
    """Flatten parameters in the model's canonical insertion order."""
    parameters = model.parameters
    if not parameters:
        raise ValueError("model does not expose any testable parameters")

    flattened: list[NDArray[np.float64]] = []
    layout: ParameterLayout = []
    offset = 0
    for name, values in parameters.items():
        array = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"model parameter {name!r} contains non-finite values")
        size = array.size
        flattened.append(array.ravel())
        layout.append((name, array.shape, slice(offset, offset + size)))
        offset += size

    return np.concatenate(flattened), layout


def _flatten_model_parameters(model: "BaseItemModel") -> NDArray[np.float64]:
    """Flatten model parameters in covariance-matrix order."""
    return _model_parameter_layout(model)[0]


def _validate_parameter_indices(
    param_indices: list[int] | NDArray[np.intp],
    n_parameters: int,
) -> NDArray[np.intp]:
    """Validate and normalize a non-empty set of unique parameter indices."""
    raw_indices = np.asarray(param_indices)
    if raw_indices.ndim != 1 or raw_indices.size == 0:
        raise ValueError("param_indices must be a non-empty one-dimensional array")
    if raw_indices.dtype.kind not in "iu" or raw_indices.dtype.kind == "b":
        raise ValueError("param_indices must contain integers")

    indices = raw_indices.astype(np.intp, copy=False)
    if np.any(indices < 0) or np.any(indices >= n_parameters):
        raise ValueError(f"param_indices must be between 0 and {n_parameters - 1}")
    if np.unique(indices).size != indices.size:
        raise ValueError("param_indices must not contain duplicates")
    return indices


def _validate_symmetric_matrix(
    matrix: NDArray[np.float64],
    expected_shape: tuple[int, int],
    name: str,
) -> NDArray[np.float64]:
    """Validate a finite symmetric matrix with the requested shape."""
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(array, array.T, atol=1e-10, rtol=1e-8):
        raise ValueError(f"{name} must be symmetric")
    return (array + array.T) / 2


def _validate_positive_definite(
    matrix: NDArray[np.float64],
    name: str,
) -> None:
    """Reject covariance submatrices that do not identify the hypothesis."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = np.finfo(np.float64).eps * matrix.shape[0] * scale
    if float(np.min(eigenvalues)) <= tolerance:
        raise ValueError(f"{name} must be positive definite")


def _validate_positive_semidefinite(
    matrix: NDArray[np.float64],
    name: str,
) -> None:
    """Reject a symmetric matrix with materially negative eigenvalues."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = np.sqrt(np.finfo(np.float64).eps) * matrix.shape[0] * scale
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")


def _resolve_vcov(
    model: "BaseItemModel",
    vcov: NDArray[np.float64] | None,
    n_parameters: int,
) -> NDArray[np.float64]:
    """Resolve a real parameter covariance matrix without fabricated defaults."""
    covariance = vcov
    if covariance is None:
        model_vcov = getattr(model, "vcov", None)
        if model_vcov is not None:
            covariance = model_vcov
        else:
            information_function = getattr(model, "information_matrix", None)
            if callable(information_function):
                information = _validate_symmetric_matrix(
                    information_function(),
                    (n_parameters, n_parameters),
                    "information matrix",
                )
                _validate_positive_definite(information, "information matrix")
                covariance = np.linalg.solve(
                    information, np.eye(n_parameters, dtype=np.float64)
                )
            else:
                raise ValueError(
                    "vcov is required unless the model exposes a covariance or "
                    "information matrix"
                )

    result = _validate_symmetric_matrix(
        covariance,
        (n_parameters, n_parameters),
        "vcov",
    )
    _validate_positive_semidefinite(result, "vcov")
    return result


def _validate_constraint_values(
    constraint_values: NDArray[np.float64] | list[float] | None,
    n_constraints: int,
) -> NDArray[np.float64]:
    """Normalize the right-hand side of a linear hypothesis."""
    if constraint_values is None:
        return np.zeros(n_constraints, dtype=np.float64)

    constraints = np.asarray(constraint_values, dtype=np.float64).reshape(-1)
    if constraints.shape != (n_constraints,):
        raise ValueError(
            f"constraint_values must have length {n_constraints}, "
            f"got {constraints.size}"
        )
    if not np.all(np.isfinite(constraints)):
        raise ValueError("constraint_values must contain only finite values")
    return constraints


def _quadratic_form(
    values: NDArray[np.float64],
    covariance: NDArray[np.float64],
) -> float:
    """Compute x' V^-1 x without explicitly forming an inverse."""
    statistic = float(values @ np.linalg.solve(covariance, values))
    return max(statistic, 0.0)


def wald(
    model: "BaseItemModel",
    param_indices: list[int] | NDArray[np.intp] | None = None,
    constraint_values: NDArray[np.float64] | list[float] | None = None,
    vcov: NDArray[np.float64] | None = None,
    *,
    contrast_matrix: NDArray[np.float64] | None = None,
) -> WaldTestResult:
    """Perform Wald test on model parameters.

    Tests H0: theta = constraint_values using the Wald statistic:
        W = (theta - c)' V^{-1} (theta - c)

    where V is the variance-covariance matrix of the parameter estimates.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    param_indices : array-like of int, optional
        Indices of parameters to test. Parameters follow the insertion order
        returned by ``model.parameters``. Mutually exclusive with
        ``contrast_matrix``.
    constraint_values : array-like of float, optional
        Values under null hypothesis. Default is zeros.
    vcov : NDArray[np.float64], optional
        Full parameter variance-covariance matrix in ``model.parameters``
        order. Required unless the model exposes ``vcov`` or an
        ``information_matrix()`` method.
    contrast_matrix : ndarray, optional
        Linear hypothesis matrix R for testing ``R @ parameters = values``.
        A one-dimensional contrast is accepted as a single-row matrix.

    Returns
    -------
    WaldTestResult
        Test results including statistic, df, and p-value.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL")
    >>> # Test if discrimination of item 0 equals 1.0
    >>> test = wald(
    ...     result.model,
    ...     param_indices=[0],
    ...     constraint_values=[1.0],
    ...     vcov=parameter_covariance,
    ... )
    >>> print(f"Wald chi-sq = {test.statistic:.3f}, p = {test.p_value:.4f}")
    """
    all_params = _flatten_model_parameters(model)
    n_total_parameters = all_params.size

    if contrast_matrix is None:
        if param_indices is None:
            raise ValueError("param_indices or contrast_matrix is required")
        indices = _validate_parameter_indices(param_indices, n_total_parameters)
        contrast = np.zeros((indices.size, n_total_parameters), dtype=np.float64)
        contrast[np.arange(indices.size), indices] = 1.0
    else:
        if param_indices is not None:
            raise ValueError("param_indices and contrast_matrix are mutually exclusive")
        contrast = np.asarray(contrast_matrix, dtype=np.float64)
        if contrast.ndim == 1:
            contrast = contrast.reshape(1, -1)
        if contrast.ndim != 2 or contrast.shape[0] == 0:
            raise ValueError("contrast_matrix must be a non-empty 2D matrix")
        if contrast.shape[1] != n_total_parameters:
            raise ValueError(
                "contrast_matrix must have one column per model parameter "
                f"({n_total_parameters}), got {contrast.shape[1]}"
            )
        if not np.all(np.isfinite(contrast)):
            raise ValueError("contrast_matrix must contain only finite values")
        if np.linalg.matrix_rank(contrast) != contrast.shape[0]:
            raise ValueError("contrast_matrix rows must be linearly independent")

    n_constraints = contrast.shape[0]
    constraints = _validate_constraint_values(constraint_values, n_constraints)
    estimates = contrast @ all_params
    full_vcov = _resolve_vcov(model, vcov, n_total_parameters)
    hypothesis_vcov = contrast @ full_vcov @ contrast.T
    hypothesis_vcov = (hypothesis_vcov + hypothesis_vcov.T) / 2
    _validate_positive_definite(hypothesis_vcov, "hypothesis covariance")

    difference = estimates - constraints
    statistic = _quadratic_form(difference, hypothesis_vcov)
    p_value = stats.chi2.sf(statistic, n_constraints)

    return WaldTestResult(
        statistic=statistic,
        df=n_constraints,
        p_value=float(p_value),
        parameter_estimates=estimates,
        standard_errors=np.sqrt(np.diag(hypothesis_vcov)),
        constraint_values=constraints,
    )


def _coerce_theta(
    theta: NDArray[np.float64],
    n_persons: int,
    n_factors: int,
) -> NDArray[np.float64]:
    """Normalize ability values without confusing persons and factors."""
    theta_array = np.asarray(theta, dtype=np.float64)
    if theta_array.ndim == 1:
        if n_factors == 1 and theta_array.size == n_persons:
            theta_array = theta_array.reshape(-1, 1)
        elif n_persons == 1 and theta_array.size == n_factors:
            theta_array = theta_array.reshape(1, -1)
        else:
            raise ValueError(
                f"theta must have shape ({n_persons}, {n_factors}), "
                f"got {theta_array.shape}"
            )
    if theta_array.shape != (n_persons, n_factors):
        raise ValueError(
            f"theta must have shape ({n_persons}, {n_factors}), got {theta_array.shape}"
        )
    if not np.all(np.isfinite(theta_array)):
        raise ValueError("theta must contain only finite values")
    return theta_array


def _set_model_parameter_vector(
    model: "BaseItemModel",
    values: NDArray[np.float64],
    layout: ParameterLayout,
) -> None:
    """Set the internal parameter vector while preserving names and shapes."""
    for name, shape, parameter_slice in layout:
        model._parameters[name] = values[parameter_slice].reshape(shape).copy()


def _conditional_log_likelihood(
    model: "BaseItemModel",
    responses: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> float:
    """Evaluate the model's native conditional log-likelihood."""
    values = np.asarray(model.log_likelihood(responses, theta), dtype=np.float64)
    result = float(np.sum(values))
    if not np.isfinite(result):
        raise ValueError("model log-likelihood returned a non-finite value")
    return result


def _analytic_score_vector(
    model: "BaseItemModel",
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """Return vectorized scores for the standard logistic parameterizations."""
    parameters = model.parameters
    parameter_names = tuple(parameters)
    if model.is_polytomous:
        return None

    probabilities = np.asarray(model.probability(theta), dtype=np.float64)
    if probabilities.shape != responses.shape:
        return None
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("model probability returned invalid values")

    valid = responses >= 0
    active = (
        valid & (probabilities > PROB_EPSILON) & (probabilities < 1.0 - PROB_EPSILON)
    )
    residuals = responses - probabilities
    residuals[~active] = 0.0

    if parameter_names == ("slopes", "intercepts"):
        slope_scores = residuals.T @ theta
        intercept_scores = np.sum(residuals, axis=0)
        return np.concatenate([slope_scores.ravel(), intercept_scores])

    if parameter_names == ("discrimination", "difficulty"):
        discrimination = parameters["discrimination"]
        difficulty = parameters["difficulty"]
        if discrimination.ndim == 1:
            discrimination_scores = np.sum(
                residuals * (theta[:, :1] - difficulty[None, :]), axis=0
            )
            difficulty_scores = np.sum(residuals * -discrimination[None, :], axis=0)
        else:
            discrimination_scores = np.sum(
                residuals[:, :, None] * (theta[:, None, :] - difficulty[None, :, None]),
                axis=0,
            )
            difficulty_scores = np.sum(
                residuals * -np.sum(discrimination, axis=1)[None, :], axis=0
            )
        return np.concatenate(
            [discrimination_scores.ravel(), difficulty_scores.ravel()]
        )

    if parameter_names not in (
        ("discrimination", "difficulty", "guessing"),
        ("discrimination", "difficulty", "guessing", "upper"),
    ):
        return None

    discrimination = parameters["discrimination"]
    difficulty = parameters["difficulty"]
    guessing = parameters["guessing"]
    upper = parameters.get("upper", np.ones_like(guessing))
    theta_column = theta[:, :1]
    logistic_probability = sigmoid(
        discrimination[None, :] * (theta_column - difficulty[None, :])
    )
    likelihood_derivative = np.divide(
        residuals,
        probabilities * (1.0 - probabilities),
        out=np.zeros_like(probabilities),
        where=active,
    )
    span = upper - guessing
    common_derivative = (
        likelihood_derivative
        * span[None, :]
        * logistic_probability
        * (1.0 - logistic_probability)
    )

    discrimination_scores = np.sum(
        common_derivative * (theta_column - difficulty[None, :]), axis=0
    )
    difficulty_scores = np.sum(common_derivative * -discrimination[None, :], axis=0)
    guessing_scores = np.sum(
        likelihood_derivative * (1.0 - logistic_probability), axis=0
    )
    scores = [discrimination_scores, difficulty_scores, guessing_scores]
    if "upper" in parameters:
        scores.append(np.sum(likelihood_derivative * logistic_probability, axis=0))
    return np.concatenate(scores)


def _validate_response_categories(
    model: "BaseItemModel",
    responses: NDArray[np.int_],
) -> None:
    """Validate observed categories while preserving negative missing codes."""
    if model.is_polytomous:
        for item_index, n_categories in enumerate(model.n_categories):
            observed = responses[:, item_index]
            invalid = (observed >= n_categories) & (observed >= 0)
            if np.any(invalid):
                raise ValueError(
                    f"responses for item {item_index} must be below {n_categories}"
                )
    elif np.any((responses >= 0) & (responses != 0) & (responses != 1)):
        raise ValueError(
            "dichotomous responses must be 0, 1, or a negative missing code"
        )


def _numerical_score_subset(
    model: "BaseItemModel",
    responses: NDArray[np.float64],
    theta: NDArray[np.float64],
    indices: NDArray[np.intp],
    step: float,
) -> NDArray[np.float64]:
    """Differentiate only the requested parameters using central differences."""
    parameters, layout = _model_parameter_layout(model)
    scores = np.empty(indices.size, dtype=np.float64)

    try:
        for output_index, parameter_index in enumerate(indices):
            delta = step * max(1.0, abs(float(parameters[parameter_index])))
            plus = parameters.copy()
            minus = parameters.copy()
            plus[parameter_index] += delta
            minus[parameter_index] -= delta

            _set_model_parameter_vector(model, plus, layout)
            likelihood_plus = _conditional_log_likelihood(model, responses, theta)
            _set_model_parameter_vector(model, minus, layout)
            likelihood_minus = _conditional_log_likelihood(model, responses, theta)
            scores[output_index] = (likelihood_plus - likelihood_minus) / (2 * delta)
    finally:
        _set_model_parameter_vector(model, parameters, layout)

    return scores


def lagrange(
    model: "BaseItemModel",
    responses: NDArray[np.float64],
    theta: NDArray[np.float64],
    param_indices: list[int] | NDArray[np.intp],
    vcov: NDArray[np.float64] | None = None,
    step: float = 1e-5,
) -> LagrangeTestResult:
    """Perform Lagrange (score) test for parameter constraints.

    Tests whether constrained parameters should be freed using
    the score statistic:
        LM = S' V S

    where S is the score (gradient) vector evaluated at the constrained
    estimates.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model (under constraints).
    responses : NDArray[np.float64]
        Response matrix. Shape: (n_persons, n_items).
    theta : NDArray[np.float64]
        Ability estimates. Shape: (n_persons, n_dims).
    param_indices : array-like of int
        Indices of constrained parameters to test, in the insertion order
        returned by ``model.parameters``.
    vcov : NDArray[np.float64], optional
        Inverse Fisher information (the full-model parameter covariance) in
        ``model.parameters`` order. Required unless the model exposes
        ``vcov`` or an ``information_matrix()`` method.
    step : float
        Relative central-difference step used when the model does not expose
        a compatible score function. Default 1e-5.

    Returns
    -------
    LagrangeTestResult
        Test results including statistic, df, and p-value.

    Examples
    --------
    >>> # Fit constrained model (e.g., Rasch with equal discriminations)
    >>> result = fit_mirt(responses, model="1PL")
    >>> # Test if discriminations should be freed
    >>> test = lagrange(
    ...     result.model,
    ...     responses,
    ...     result.theta,
    ...     param_indices=[0, 2, 4],
    ...     vcov=parameter_covariance,
    ... )
    >>> print(f"LM chi-sq = {test.statistic:.3f}, p = {test.p_value:.4f}")
    """
    responses = np.asarray(responses, dtype=np.float64)
    if responses.ndim != 2 or responses.shape[0] == 0:
        raise ValueError("responses must be a non-empty 2D matrix")
    if responses.shape[1] != model.n_items:
        raise ValueError(
            f"responses must contain {model.n_items} items, got {responses.shape[1]}"
        )
    if np.any(np.isinf(responses)):
        raise ValueError("responses must not contain infinite values")
    observed_responses = responses[~np.isnan(responses)]
    if np.any(observed_responses != np.floor(observed_responses)):
        raise ValueError("observed responses must be integer-valued")
    response_matrix = np.where(np.isnan(responses), -1, responses).astype(np.int_)
    _validate_response_categories(model, response_matrix)

    theta_array = _coerce_theta(
        theta,
        n_persons=responses.shape[0],
        n_factors=model.n_factors,
    )
    all_params = _flatten_model_parameters(model)
    n_total_parameters = all_params.size
    indices = _validate_parameter_indices(param_indices, n_total_parameters)

    if (
        isinstance(step, (bool, np.bool_))
        or not isinstance(step, (int, float, np.integer, np.floating))
        or not np.isfinite(step)
        or step <= 0
    ):
        raise ValueError("step must be a finite positive value")

    score_function = getattr(model, "score_function", None)
    if callable(score_function):
        raw_scores = np.asarray(
            score_function(response_matrix, theta_array), dtype=np.float64
        )
        if raw_scores.shape == (responses.shape[0], n_total_parameters):
            score_vector = np.sum(raw_scores, axis=0)
        elif raw_scores.shape == (n_total_parameters,):
            score_vector = raw_scores
        else:
            raise ValueError(
                "score_function must return shape "
                f"({n_total_parameters},) or "
                f"({responses.shape[0]}, {n_total_parameters})"
            )
        if not np.all(np.isfinite(score_vector)):
            raise ValueError("score_function returned non-finite values")
        score_subset = score_vector[indices]
    else:
        score_vector = _analytic_score_vector(model, response_matrix, theta_array)
        if score_vector is None:
            score_subset = _numerical_score_subset(
                model,
                response_matrix,
                theta_array,
                indices,
                float(step),
            )
        else:
            if score_vector.shape != (n_total_parameters,) or not np.all(
                np.isfinite(score_vector)
            ):
                raise ValueError("analytic score calculation returned invalid values")
            score_subset = score_vector[indices]

    full_vcov = _resolve_vcov(model, vcov, n_total_parameters)
    vcov_subset = full_vcov[np.ix_(indices, indices)]
    _validate_positive_definite(vcov_subset, "tested parameter covariance")

    statistic = max(float(score_subset @ vcov_subset @ score_subset), 0.0)
    p_value = stats.chi2.sf(statistic, indices.size)

    return LagrangeTestResult(
        statistic=statistic,
        df=indices.size,
        p_value=float(p_value),
        scores=score_subset,
    )


def likelihood_ratio(
    ll_full: float,
    ll_reduced: float,
    df_diff: int,
) -> tuple[float, float]:
    """Compute likelihood ratio test statistic.

    Parameters
    ----------
    ll_full : float
        Log-likelihood of full (less constrained) model.
    ll_reduced : float
        Log-likelihood of reduced (more constrained) model.
    df_diff : int
        Difference in degrees of freedom (number of constraints).

    Returns
    -------
    statistic : float
        Chi-square statistic (-2 * (ll_reduced - ll_full)).
    p_value : float
        P-value from chi-square distribution.
    """
    if not np.isfinite(ll_full) or not np.isfinite(ll_reduced):
        raise ValueError("log-likelihoods must be finite")
    if isinstance(df_diff, (bool, np.bool_)) or not isinstance(
        df_diff, (int, np.integer)
    ):
        raise ValueError("df_diff must be an integer")
    if df_diff <= 0:
        raise ValueError("df_diff must be positive")

    scale = max(1.0, abs(ll_full), abs(ll_reduced))
    tolerance = 64 * np.finfo(np.float64).eps * scale
    if ll_reduced > ll_full + tolerance:
        raise ValueError(
            "ll_full must be at least as large as ll_reduced for nested models"
        )

    statistic = max(2 * (ll_full - ll_reduced), 0.0)
    p_value = stats.chi2.sf(statistic, int(df_diff))
    return float(statistic), float(p_value)
