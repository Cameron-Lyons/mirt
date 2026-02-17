"""Batch model fitting utilities.

This module provides functions for fitting multiple IRT models
and comparing them efficiently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.results.fit_result import FitResult


ModelType = Literal["1PL", "2PL", "3PL", "4PL", "GRM", "GPCM", "PCM", "NRM"]


@dataclass
class BatchFitResult:
    """Result of batch model fitting.

    Attributes
    ----------
    results : dict[str, FitResult]
        Fitted results keyed by model type.
    comparison : Any
        DataFrame comparing all models.
    best_model : str
        Name of best model by BIC.

    Examples
    --------
    >>> batch_result = fit_models(["1PL", "2PL", "3PL"], responses)
    >>> print(batch_result.summary())
    >>> best = batch_result[batch_result.best_model]
    """

    results: dict[str, FitResult]
    comparison: Any
    best_model: str

    def __getitem__(self, model: str) -> FitResult:
        """Get result for a specific model.

        Parameters
        ----------
        model : str
            Model name (e.g., "2PL").

        Returns
        -------
        FitResult
            The fit result for that model.
        """
        return self.results[model]

    def summary(self) -> str:
        """Generate a text summary of batch fitting results.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines = ["Batch Model Fitting Results", "=" * 60]
        lines.append(f"Models fitted: {len(self.results)}")
        lines.append(f"Best model (BIC): {self.best_model}")
        lines.append("-" * 60)
        lines.append(
            f"{'Model':<10} {'LogLik':>12} {'AIC':>12} {'BIC':>12} {'Conv':>8}"
        )
        lines.append("-" * 60)
        for name, result in self.results.items():
            marker = " *" if name == self.best_model else ""
            lines.append(
                f"{name:<10} {result.log_likelihood:>12.2f} "
                f"{result.aic:>12.2f} {result.bic:>12.2f} "
                f"{str(result.converged):>8}{marker}"
            )
        lines.append("-" * 60)
        lines.append("* = best model by BIC")
        return "\n".join(lines)

    def get_best_result(self) -> FitResult:
        """Get the result for the best model.

        Returns
        -------
        FitResult
            The fit result for the best model by BIC.
        """
        return self.results[self.best_model]


def fit_models(
    models: list[ModelType],
    responses: NDArray[np.int_],
    n_categories: int | None = None,
    n_factors: int = 1,
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
) -> BatchFitResult:
    """Fit multiple IRT models to the same data.

    This function fits multiple model types to the same response data
    and returns a comparison of all models with information criteria.

    Parameters
    ----------
    models : list[ModelType]
        List of model types to fit (e.g., ["1PL", "2PL", "3PL"]).
    responses : NDArray
        Response matrix (n_persons, n_items).
    n_categories : int, optional
        Number of categories for polytomous models.
    n_factors : int, default=1
        Number of latent factors.
    n_quadpts : int, default=21
        Quadrature points for EM.
    max_iter : int, default=500
        Maximum EM iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    verbose : bool, default=False
        Print progress.

    Returns
    -------
    BatchFitResult
        Results containing all fitted models and comparison.

    Examples
    --------
    >>> from mirt import load_dataset
    >>> from mirt.utils.batch import fit_models
    >>> data = load_dataset("LSAT6")
    >>> batch_result = fit_models(
    ...     models=["1PL", "2PL", "3PL"],
    ...     responses=data["data"],
    ... )
    >>> print(batch_result.summary())
    >>> best_result = batch_result[batch_result.best_model]

    Notes
    -----
    The best model is determined by BIC (Bayesian Information Criterion),
    which balances model fit with model complexity.
    """
    from mirt import fit_mirt
    from mirt.diagnostics.comparison import compare_models

    responses = np.asarray(responses)

    results: dict[str, FitResult] = {}

    for model_type in models:
        if verbose:
            print(f"Fitting {model_type}...")

        result = fit_mirt(
            responses,
            model=model_type,
            n_categories=n_categories,
            n_factors=n_factors,
            n_quadpts=n_quadpts,
            max_iter=max_iter,
            tol=tol,
            verbose=False,
        )
        results[model_type] = result

        if verbose:
            print(f"  LL={result.log_likelihood:.2f}, converged={result.converged}")

    result_list = list(results.values())
    comparison = compare_models(result_list)

    bic_values = {name: r.bic for name, r in results.items()}
    best_model = min(bic_values, key=lambda k: bic_values[k])

    return BatchFitResult(
        results=results,
        comparison=comparison,
        best_model=best_model,
    )


def fit_model_grid(
    models: list[ModelType],
    responses: NDArray[np.int_],
    n_factors_range: list[int] | None = None,
    n_quadpts_range: list[int] | None = None,
    n_categories: int | None = None,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
) -> dict[str, FitResult]:
    """Fit models across a grid of hyperparameters.

    This function performs a grid search over model types and
    hyperparameters, returning all fitted models.

    Parameters
    ----------
    models : list[ModelType]
        Model types to fit.
    responses : NDArray
        Response matrix (n_persons, n_items).
    n_factors_range : list[int], optional
        Range of factor counts to try (for MIRT). Default: [1].
    n_quadpts_range : list[int], optional
        Range of quadrature points to try. Default: [21].
    n_categories : int, optional
        Number of categories for polytomous models.
    max_iter : int, default=500
        Maximum EM iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    verbose : bool, default=False
        Print progress.

    Returns
    -------
    dict[str, FitResult]
        Results keyed by "model_f{n_factors}_q{n_quadpts}".

    Examples
    --------
    >>> results = fit_model_grid(
    ...     models=["2PL"],
    ...     responses=data,
    ...     n_factors_range=[1, 2, 3],
    ...     n_quadpts_range=[11, 21, 31],
    ... )
    >>> for key, result in results.items():
    ...     print(f"{key}: BIC={result.bic:.2f}")
    """
    from mirt import fit_mirt

    responses = np.asarray(responses)

    if n_factors_range is None:
        n_factors_range = [1]
    if n_quadpts_range is None:
        n_quadpts_range = [21]

    results: dict[str, FitResult] = {}

    for model in models:
        for n_factors in n_factors_range:
            for n_quadpts in n_quadpts_range:
                key = f"{model}_f{n_factors}_q{n_quadpts}"

                if verbose:
                    print(f"Fitting {key}...")

                try:
                    result = fit_mirt(
                        responses,
                        model=model,
                        n_factors=n_factors,
                        n_categories=n_categories,
                        n_quadpts=n_quadpts,
                        max_iter=max_iter,
                        tol=tol,
                        verbose=False,
                    )
                    results[key] = result

                    if verbose:
                        print(f"  BIC={result.bic:.2f}, converged={result.converged}")
                except (
                    ValueError,
                    RuntimeError,
                    ArithmeticError,
                    FloatingPointError,
                    np.linalg.LinAlgError,
                ) as e:
                    if verbose:
                        print(f"  Failed: {e}")
                    continue

    return results


__all__ = [
    "fit_models",
    "fit_model_grid",
    "BatchFitResult",
]
