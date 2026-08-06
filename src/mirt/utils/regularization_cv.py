"""Regularization-path selection for multidimensional IRT models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.constants import PROB_EPSILON
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.estimation.regularized import RegularizedMIRTEstimator, RegularizedMIRTResult
from mirt.exceptions import MirtDataError, MirtEstimationError, MirtValidationError
from mirt.utils.data import validate_responses
from mirt.utils.numeric import logsumexp

Penalty = Literal["lasso", "ridge", "elastic_net"]
CVCriterion = Literal["log_likelihood", "bic", "ebic"]
PathCriterion = Literal["aic", "bic", "ebic"]

_PENALTIES = ("lasso", "ridge", "elastic_net")
_CV_CRITERIA = ("log_likelihood", "bic", "ebic")
_PATH_CRITERIA = ("aic", "bic", "ebic")
_MAX_SCORE_MATRIX_ELEMENTS = 1_000_000


@dataclass
class RegularizationCVResult:
    """Result from cross-validation for regularization parameter selection."""

    lambda_values: list[float]
    mean_scores: list[float]
    std_scores: list[float]
    best_lambda: float
    best_score: float
    best_result: RegularizedMIRTResult
    fold_scores: list[list[float]]
    criterion: str
    one_se_lambda: float | None = None
    mean_nonzero: list[float] = field(default_factory=list)

    @property
    def selected_lambda(self) -> float:
        """Penalty used for the final full-data fit."""
        if self.one_se_lambda is not None:
            return self.one_se_lambda
        return self.best_lambda

    def summary(self) -> str:
        """Format the complete regularization path as a compact table."""
        lines = []
        width = 70

        lines.append("=" * width)
        lines.append(f"{'Cross-Validation for Regularization':^{width}}")
        lines.append("=" * width)
        lines.append(f"Criterion:          {self.criterion}")
        lines.append(f"Best Lambda:        {self.best_lambda:.6f}")
        lines.append(f"Best Score:         {self.best_score:.4f}")
        if self.one_se_lambda is not None:
            lines.append(f"1-SE Lambda:        {self.one_se_lambda:.6f}")
        lines.append("-" * width)
        lines.append(
            f"\n{'Lambda':>12} {'Mean Score':>12} {'Std':>10} {'Non-zero':>10}"
        )
        lines.append("-" * width)

        for index, (lambda_value, mean, std) in enumerate(
            zip(self.lambda_values, self.mean_scores, self.std_scores, strict=True)
        ):
            nonzero = (
                f"{self.mean_nonzero[index]:.1f}"
                if index < len(self.mean_nonzero)
                else "--"
            )
            marker = "*" if np.isclose(lambda_value, self.selected_lambda) else " "
            lines.append(
                f"{marker}{lambda_value:>11.6f} {mean:>12.4f} "
                f"{std:>10.4f} {nonzero:>10}"
            )
        lines.append("-" * width)
        lines.append("* selected for the final fit")
        return "\n".join(lines)


def cv_select_lambda(
    responses: NDArray[np.int_],
    penalty: Penalty = "lasso",
    alpha: float = 1.0,
    n_factors: int = 2,
    lambda_values: list[float] | None = None,
    n_lambda: int = 20,
    n_folds: int = 5,
    criterion: CVCriterion = "log_likelihood",
    one_se_rule: bool = False,
    n_quadpts: int = 15,
    max_iter: int = 200,
    tol: float = 1e-3,
    verbose: bool = False,
    seed: int | None = None,
) -> RegularizationCVResult:
    """Select regularization strength with person-level cross-validation.

    Held-out scores integrate over the latent trait distribution rather than
    evaluating every person at a zero trait vector. All stored scores use a
    higher-is-better orientation; BIC and EBIC are negated and normalized by
    the held-out fold size.
    """
    responses = _validate_regularization_inputs(
        responses,
        penalty=penalty,
        alpha=alpha,
        n_factors=n_factors,
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
    )
    _validate_choice(criterion, _CV_CRITERIA, "criterion")
    _validate_positive_integer(n_folds, "n_folds", minimum=2)
    if not isinstance(one_se_rule, (bool, np.bool_)):
        raise MirtValidationError(
            "one_se_rule must be a boolean",
            parameter="one_se_rule",
            value=one_se_rule,
        )
    if n_folds > responses.shape[0]:
        raise MirtValidationError(
            "n_folds cannot exceed the number of persons",
            parameter="n_folds",
            value=n_folds,
            expected=f"<= {responses.shape[0]}",
        )

    estimator_options = {
        "penalty": penalty,
        "alpha": alpha,
        "n_factors": n_factors,
        "n_quadpts": n_quadpts,
        "max_iter": max_iter,
        "tol": tol,
        "verbose": False,
    }
    base_estimator = RegularizedMIRTEstimator(**estimator_options)
    lambda_values = _resolve_lambda_values(
        responses,
        base_estimator,
        lambda_values=lambda_values,
        n_lambda=n_lambda,
    )

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(responses.shape[0])
    fold_indices = np.empty(responses.shape[0], dtype=np.int_)
    for fold, indices in enumerate(np.array_split(shuffled, n_folds)):
        fold_indices[indices] = fold

    fold_scores: list[list[float]] = []
    mean_scores: list[float] = []
    std_scores: list[float] = []
    mean_nonzero: list[float] = []

    for lambda_index, lambda_value in enumerate(lambda_values):
        if verbose:
            print(f"Lambda {lambda_index + 1}/{len(lambda_values)}: {lambda_value:.6f}")

        scores_for_lambda: list[float] = []
        nonzero_for_lambda: list[float] = []
        for fold in range(n_folds):
            train_data = responses[fold_indices != fold]
            test_data = responses[fold_indices == fold]
            estimator = RegularizedMIRTEstimator(**estimator_options)

            try:
                result = estimator.fit(train_data, lambda_val=lambda_value)
                score = _score_regularized_result(
                    result,
                    test_data,
                    criterion=criterion,
                    n_quadpts=n_quadpts,
                )
            except (
                ValueError,
                RuntimeError,
                ArithmeticError,
                FloatingPointError,
                np.linalg.LinAlgError,
            ):
                score = np.nan
            scores_for_lambda.append(float(score))
            if np.isfinite(score):
                nonzero_for_lambda.append(float(result.n_nonzero))

        fold_scores.append(scores_for_lambda)
        finite_scores = np.asarray(scores_for_lambda, dtype=np.float64)
        finite_scores = finite_scores[np.isfinite(finite_scores)]
        if finite_scores.size == 0:
            mean_scores.append(np.nan)
            std_scores.append(np.nan)
            mean_nonzero.append(np.nan)
        else:
            mean_scores.append(float(finite_scores.mean()))
            std_scores.append(
                float(finite_scores.std(ddof=1)) if finite_scores.size > 1 else 0.0
            )
            mean_nonzero.append(float(np.mean(nonzero_for_lambda)))

    finite_means = np.isfinite(mean_scores)
    if not np.any(finite_means):
        raise MirtEstimationError(
            "Every regularization fit failed; no lambda could be selected"
        )

    comparable_means = np.where(finite_means, mean_scores, -np.inf)
    best_index = int(np.argmax(comparable_means))
    best_lambda = lambda_values[best_index]
    best_score = mean_scores[best_index]

    one_se_lambda = None
    if one_se_rule:
        best_fold_scores = np.asarray(fold_scores[best_index], dtype=np.float64)
        best_fold_scores = best_fold_scores[np.isfinite(best_fold_scores)]
        standard_error = (
            float(best_fold_scores.std(ddof=1) / np.sqrt(best_fold_scores.size))
            if best_fold_scores.size > 1
            else 0.0
        )
        threshold = best_score - standard_error
        eligible = [
            value
            for value, score in zip(lambda_values, mean_scores, strict=True)
            if np.isfinite(score) and score >= threshold
        ]
        one_se_lambda = max(eligible)

    final_lambda = one_se_lambda if one_se_lambda is not None else best_lambda
    if verbose:
        print(f"\nFitting final model with lambda = {final_lambda:.6f}")

    final_estimator = RegularizedMIRTEstimator(
        penalty=penalty,
        alpha=alpha,
        n_factors=n_factors,
        n_quadpts=n_quadpts,
        max_iter=max_iter * 2,
        tol=tol / 10,
        verbose=verbose,
    )
    best_result = final_estimator.fit(responses, lambda_val=final_lambda)

    return RegularizationCVResult(
        lambda_values=lambda_values,
        mean_scores=mean_scores,
        std_scores=std_scores,
        best_lambda=best_lambda,
        best_score=best_score,
        best_result=best_result,
        fold_scores=fold_scores,
        criterion=criterion,
        one_se_lambda=one_se_lambda,
        mean_nonzero=mean_nonzero,
    )


def _compute_test_ll(
    result: RegularizedMIRTResult,
    test_data: NDArray[np.int_],
    n_quadpts: int = 15,
) -> float:
    """Compute held-out marginal log-likelihood with vectorized quadrature."""
    test_data = validate_responses(test_data, n_items=result.loadings.shape[0])
    observed = test_data[test_data >= 0]
    if np.any(observed > 1):
        raise MirtDataError(
            "regularized MIRT requires dichotomous responses",
            n_persons=test_data.shape[0],
            n_items=test_data.shape[1],
        )
    _validate_positive_integer(n_quadpts, "n_quadpts", minimum=1)

    loadings = np.asarray(result.loadings, dtype=np.float64)
    intercepts = np.asarray(result.intercepts, dtype=np.float64)
    if loadings.ndim != 2 or intercepts.shape != (loadings.shape[0],):
        raise MirtValidationError(
            "regularized result parameters have incompatible shapes",
            parameter="result",
            value=(loadings.shape, intercepts.shape),
        )

    quadrature = GaussHermiteQuadrature(
        n_points=n_quadpts,
        n_dimensions=loadings.shape[1],
    )
    nodes = quadrature.nodes
    log_weights = np.log(np.maximum(quadrature.weights, PROB_EPSILON))
    probabilities = sigmoid(nodes @ loadings.T + intercepts[None, :])
    probabilities = np.clip(probabilities, PROB_EPSILON, 1 - PROB_EPSILON)
    log_probability = np.log(probabilities)
    log_complement = np.log1p(-probabilities)

    block_size = max(
        1,
        _MAX_SCORE_MATRIX_ELEMENTS // quadrature.n_total_points,
    )
    total = 0.0
    for start in range(0, test_data.shape[0], block_size):
        block = test_data[start : start + block_size]
        positive = block == 1
        negative = block == 0
        conditional_ll = positive @ log_probability.T + negative @ log_complement.T
        total += float(np.sum(logsumexp(conditional_ll + log_weights[None, :], axis=1)))
    return total


def _score_regularized_result(
    result: RegularizedMIRTResult,
    test_data: NDArray[np.int_],
    criterion: CVCriterion,
    n_quadpts: int,
) -> float:
    """Convert held-out likelihood or information criterion to a score."""
    test_ll = _compute_test_ll(result, test_data, n_quadpts=n_quadpts)
    n_persons, n_items = test_data.shape
    if criterion == "log_likelihood":
        return test_ll / n_persons

    information_criterion = -2 * test_ll + result.n_parameters * np.log(n_persons)
    if criterion == "ebic":
        n_factors = result.loadings.shape[1]
        information_criterion += n_items * n_factors * np.log(2)
    return -float(information_criterion) / n_persons


def information_criteria_path(
    responses: NDArray[np.int_],
    penalty: Penalty = "lasso",
    alpha: float = 1.0,
    n_factors: int = 2,
    lambda_values: list[float] | None = None,
    n_lambda: int = 20,
    criterion: PathCriterion = "bic",
    n_quadpts: int = 15,
    max_iter: int = 200,
    tol: float = 1e-3,
    verbose: bool = False,
) -> tuple[float, RegularizedMIRTResult, list[RegularizedMIRTResult]]:
    """Select regularization strength using a full-data information criterion."""
    responses = _validate_regularization_inputs(
        responses,
        penalty=penalty,
        alpha=alpha,
        n_factors=n_factors,
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
    )
    _validate_choice(criterion, _PATH_CRITERIA, "criterion")

    estimator = RegularizedMIRTEstimator(
        penalty=penalty,
        alpha=alpha,
        n_factors=n_factors,
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
    )
    resolved_lambdas = _resolve_lambda_values(
        responses,
        estimator,
        lambda_values=lambda_values,
        n_lambda=n_lambda,
    )
    results = estimator.fit_path(responses, lambda_values=resolved_lambdas)
    if not results:
        raise MirtEstimationError("The regularization path returned no fitted models")

    scores = [float(getattr(result, criterion)) for result in results]
    if not np.all(np.isfinite(scores)):
        raise MirtEstimationError(
            f"The regularization path produced non-finite {criterion.upper()} values"
        )
    best_index = int(np.argmin(scores))
    best_result = results[best_index]
    return best_result.lambda_val, best_result, results


def _validate_regularization_inputs(
    responses: NDArray[np.int_],
    *,
    penalty: str,
    alpha: float,
    n_factors: int,
    n_quadpts: int,
    max_iter: int,
    tol: float,
) -> NDArray[np.int_]:
    """Validate shared regularization-selection inputs."""
    _validate_choice(penalty, _PENALTIES, "penalty")
    _validate_positive_integer(n_factors, "n_factors", minimum=2)
    _validate_positive_integer(n_quadpts, "n_quadpts", minimum=1)
    _validate_positive_integer(max_iter, "max_iter", minimum=1)
    if (
        not isinstance(alpha, (int, float, np.integer, np.floating))
        or isinstance(alpha, (bool, np.bool_))
        or not np.isfinite(alpha)
        or not 0 <= float(alpha) <= 1
    ):
        raise MirtValidationError(
            "alpha must be a finite number between 0 and 1",
            parameter="alpha",
            value=alpha,
        )
    if (
        not isinstance(tol, (int, float, np.integer, np.floating))
        or isinstance(tol, (bool, np.bool_))
        or not np.isfinite(tol)
        or tol <= 0
    ):
        raise MirtValidationError(
            "tol must be a finite positive number",
            parameter="tol",
            value=tol,
        )

    validated = validate_responses(responses)
    observed = validated[validated >= 0]
    if np.any(observed > 1):
        raise MirtDataError(
            "regularized MIRT requires dichotomous responses",
            n_persons=validated.shape[0],
            n_items=validated.shape[1],
        )
    return validated


def _resolve_lambda_values(
    responses: NDArray[np.int_],
    estimator: RegularizedMIRTEstimator,
    *,
    lambda_values: list[float] | None,
    n_lambda: int,
) -> list[float]:
    """Validate or generate a numerically stable penalty path."""
    if lambda_values is not None:
        if not lambda_values:
            raise MirtValidationError(
                "lambda_values must contain at least one value",
                parameter="lambda_values",
            )
        resolved = []
        for value in lambda_values:
            if (
                not isinstance(value, (int, float, np.integer, np.floating))
                or isinstance(value, (bool, np.bool_))
                or not np.isfinite(value)
                or value < 0
            ):
                raise MirtValidationError(
                    "lambda values must be finite and nonnegative",
                    parameter="lambda_values",
                    value=value,
                )
            resolved.append(float(value))
        return resolved

    _validate_positive_integer(n_lambda, "n_lambda", minimum=1)
    lambda_max = float(estimator._compute_lambda_max(responses))
    if not np.isfinite(lambda_max) or lambda_max < 0:
        raise MirtEstimationError(
            "Unable to compute a finite nonnegative maximum lambda"
        )
    if lambda_max <= PROB_EPSILON:
        return [0.0]
    if n_lambda == 1:
        return [lambda_max]
    return np.geomspace(lambda_max, lambda_max * 0.01, n_lambda).tolist()


def _validate_choice(value: str, choices: tuple[str, ...], parameter: str) -> None:
    if value not in choices:
        raise MirtValidationError(
            f"Unknown {parameter}",
            parameter=parameter,
            value=value,
            expected=", ".join(choices),
        )


def _validate_positive_integer(value: int, parameter: str, minimum: int) -> None:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or value < minimum
    ):
        raise MirtValidationError(
            f"{parameter} must be an integer of at least {minimum}",
            parameter=parameter,
            value=value,
            expected=f">= {minimum}",
        )
