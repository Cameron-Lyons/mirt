from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.constants import PROB_EPSILON
from mirt.estimation.regularized import RegularizedMIRTEstimator, RegularizedMIRTResult


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

    def summary(self) -> str:
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

        return "\n".join(lines)


def cv_select_lambda(
    responses: NDArray[np.int_],
    penalty: Literal["lasso", "ridge", "elastic_net"] = "lasso",
    alpha: float = 1.0,
    n_factors: int = 2,
    lambda_values: list[float] | None = None,
    n_lambda: int = 20,
    n_folds: int = 5,
    criterion: Literal["log_likelihood", "bic", "ebic"] = "log_likelihood",
    one_se_rule: bool = False,
    n_quadpts: int = 15,
    max_iter: int = 200,
    tol: float = 1e-3,
    verbose: bool = False,
    seed: int | None = None,
) -> RegularizationCVResult:
    """Cross-validation for regularization parameter selection.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items). Missing values coded as -1.
    penalty : str
        Penalty type: "lasso", "ridge", or "elastic_net".
    alpha : float
        Elastic net mixing parameter (1 = LASSO, 0 = ridge).
    n_factors : int
        Number of latent factors.
    lambda_values : list, optional
        Explicit lambda values to evaluate. If None, generates sequence.
    n_lambda : int
        Number of lambda values if generating sequence.
    n_folds : int
        Number of cross-validation folds.
    criterion : str
        Selection criterion: "log_likelihood" (higher is better),
        "bic" or "ebic" (lower is better).
    one_se_rule : bool
        If True, return simplest model within one SE of best.
    n_quadpts : int
        Number of quadrature points.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Print progress.
    seed : int, optional
        Random seed for fold assignment.

    Returns
    -------
    RegularizationCVResult
        Cross-validation results with optimal lambda.
    """
    responses = np.asarray(responses, dtype=np.int32)
    n_persons = responses.shape[0]

    rng = np.random.default_rng(seed)
    fold_indices = rng.permutation(n_persons) % n_folds

    base_estimator = RegularizedMIRTEstimator(
        penalty=penalty,
        alpha=alpha,
        n_factors=n_factors,
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
    )

    if lambda_values is None:
        lambda_max = base_estimator._compute_lambda_max(responses)
        lambda_min = lambda_max * 0.01
        lambda_values = np.geomspace(lambda_max, lambda_min, n_lambda).tolist()

    fold_scores: list[list[float]] = [[] for _ in lambda_values]
    mean_scores: list[float] = []
    std_scores: list[float] = []

    for lam_idx, lam in enumerate(lambda_values):
        if verbose:
            print(f"Lambda {lam_idx + 1}/{len(lambda_values)}: {lam:.6f}")

        scores_for_lambda = []

        for fold in range(n_folds):
            train_mask = fold_indices != fold
            test_mask = fold_indices == fold

            train_data = responses[train_mask]
            test_data = responses[test_mask]

            estimator = RegularizedMIRTEstimator(
                penalty=penalty,
                alpha=alpha,
                n_factors=n_factors,
                n_quadpts=n_quadpts,
                max_iter=max_iter,
                tol=tol,
                verbose=False,
            )

            try:
                result = estimator.fit(train_data, lambda_val=lam)

                if criterion == "log_likelihood":
                    score = _compute_test_ll(result, test_data)
                elif criterion == "bic":
                    score = -result.bic
                else:
                    score = -result.ebic

                scores_for_lambda.append(score)
            except (
                ValueError,
                RuntimeError,
                ArithmeticError,
                FloatingPointError,
                np.linalg.LinAlgError,
            ):
                scores_for_lambda.append(np.nan)

        fold_scores[lam_idx] = scores_for_lambda
        mean_scores.append(np.nanmean(scores_for_lambda))
        std_scores.append(np.nanstd(scores_for_lambda))

    best_idx = int(np.nanargmax(mean_scores))
    best_lambda = lambda_values[best_idx]
    best_score = mean_scores[best_idx]

    one_se_lambda = None
    if one_se_rule:
        threshold = best_score - std_scores[best_idx]
        for i in range(len(lambda_values)):
            if mean_scores[i] >= threshold:
                one_se_lambda = lambda_values[i]
                break

    final_lambda = one_se_lambda if one_se_rule and one_se_lambda else best_lambda

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
    )


def _compute_test_ll(
    result: RegularizedMIRTResult,
    test_data: NDArray[np.int_],
) -> float:
    """Compute log-likelihood on test data."""
    loadings = result.loadings
    intercepts = result.intercepts
    n_persons = test_data.shape[0]
    n_items = test_data.shape[1]

    ll = 0.0
    for i in range(n_persons):
        theta = np.zeros(loadings.shape[1])

        for j in range(n_items):
            if test_data[i, j] < 0:
                continue

            z = np.dot(theta, loadings[j]) + intercepts[j]
            p = sigmoid(z)
            p = np.clip(p, PROB_EPSILON, 1 - PROB_EPSILON)

            if test_data[i, j] == 1:
                ll += np.log(p)
            else:
                ll += np.log(1 - p)

    return ll


def information_criteria_path(
    responses: NDArray[np.int_],
    penalty: Literal["lasso", "ridge", "elastic_net"] = "lasso",
    alpha: float = 1.0,
    n_factors: int = 2,
    lambda_values: list[float] | None = None,
    n_lambda: int = 20,
    criterion: Literal["aic", "bic", "ebic"] = "bic",
    n_quadpts: int = 15,
    max_iter: int = 200,
    tol: float = 1e-3,
    verbose: bool = False,
) -> tuple[float, RegularizedMIRTResult, list[RegularizedMIRTResult]]:
    """Select lambda using information criteria on full data.

    This is faster than CV but may overfit. Use for quick exploration.

    Parameters
    ----------
    responses : NDArray
        Response matrix.
    penalty : str
        Penalty type.
    alpha : float
        Elastic net mixing.
    n_factors : int
        Number of factors.
    lambda_values : list, optional
        Lambda values to evaluate.
    n_lambda : int
        Number of lambda values.
    criterion : str
        "aic", "bic", or "ebic".
    n_quadpts : int
        Number of quadrature points.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Print progress.

    Returns
    -------
    tuple
        (best_lambda, best_result, all_results)
    """
    estimator = RegularizedMIRTEstimator(
        penalty=penalty,
        alpha=alpha,
        n_factors=n_factors,
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
    )

    results = estimator.fit_path(
        responses, lambda_values=lambda_values, n_lambda=n_lambda
    )

    if criterion == "aic":
        scores = [r.aic for r in results]
    elif criterion == "bic":
        scores = [r.bic for r in results]
    else:
        scores = [r.ebic for r in results]

    best_idx = int(np.argmin(scores))
    best_result = results[best_idx]

    return best_result.lambda_val, best_result, results
