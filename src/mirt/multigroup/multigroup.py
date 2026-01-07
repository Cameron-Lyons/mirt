from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


class MultigroupResult:
    """Container for multigroup analysis results."""

    def __init__(
        self,
        group_results: list[FitResult],
        invariance: str,
        combined_ll: float,
        combined_aic: float,
        combined_bic: float,
        n_parameters: int,
        n_observations: int,
    ) -> None:
        self.group_results = group_results
        self.invariance = invariance
        self.combined_ll = combined_ll
        self.combined_aic = combined_aic
        self.combined_bic = combined_bic
        self.n_parameters = n_parameters
        self.n_observations = n_observations

    @property
    def model(self) -> BaseItemModel:
        """Return the model from the first group for compatibility."""
        return self.group_results[0].model

    @property
    def log_likelihood(self) -> float:
        """Combined log-likelihood."""
        return self.combined_ll

    @property
    def aic(self) -> float:
        return self.combined_aic

    @property
    def bic(self) -> float:
        return self.combined_bic

    def __repr__(self) -> str:
        return (
            f"MultigroupResult(invariance={self.invariance}, "
            f"n_groups={len(self.group_results)}, "
            f"LL={self.combined_ll:.2f})"
        )


def fit_multigroup(
    data: NDArray[np.int_],
    groups: NDArray,
    model: Literal["1PL", "2PL", "3PL", "GRM", "GPCM"] = "2PL",
    invariance: Literal["configural", "metric", "scalar", "strict"] = "configural",
    n_categories: int | None = None,
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
) -> FitResult:
    """Fit a multigroup IRT model with invariance constraints.

    Args:
        data: Response matrix (n_persons x n_items).
        groups: Group membership array (n_persons,).
        model: IRT model type.
        invariance: Level of measurement invariance:
            - 'configural': All parameters free across groups
            - 'metric': Discrimination/slopes constrained equal, intercepts free
            - 'scalar': Both discrimination and intercepts constrained equal
            - 'strict': All item parameters constrained equal
        n_categories: Number of categories for polytomous models.
        n_quadpts: Number of quadrature points for EM.
        max_iter: Maximum EM iterations.
        tol: Convergence tolerance.
        verbose: Print progress.

    Returns:
        FitResult containing the fitted model(s).
    """
    data = np.asarray(data)
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    if n_groups < 2:
        raise ValueError("At least 2 groups required for multiple group analysis")

    if verbose:
        print(f"Fitting {n_groups}-group {model} model with {invariance} invariance")

    if invariance == "configural":
        return _fit_configural(
            data,
            groups,
            unique_groups,
            model,
            n_categories,
            n_quadpts,
            max_iter,
            tol,
            verbose,
        )

    elif invariance == "metric":
        return _fit_metric(
            data,
            groups,
            unique_groups,
            model,
            n_categories,
            n_quadpts,
            max_iter,
            tol,
            verbose,
        )

    elif invariance == "scalar":
        return _fit_scalar(
            data,
            groups,
            unique_groups,
            model,
            n_categories,
            n_quadpts,
            max_iter,
            tol,
            verbose,
        )

    elif invariance == "strict":
        return _fit_strict(
            data,
            groups,
            unique_groups,
            model,
            n_categories,
            n_quadpts,
            max_iter,
            tol,
            verbose,
        )

    else:
        raise ValueError(f"Unknown invariance level: {invariance}")


def _fit_configural(
    data: NDArray[np.int_],
    groups: NDArray[np.int_] | NDArray[np.str_],
    unique_groups: NDArray[np.int_] | NDArray[np.str_],
    model: Literal["1PL", "2PL", "3PL", "GRM", "GPCM"],
    n_categories: int | None,
    n_quadpts: int,
    max_iter: int,
    tol: float,
    verbose: bool,
) -> FitResult:
    """Fit configural invariance: all parameters free across groups."""
    from mirt import fit_mirt

    group_results = []

    for g in unique_groups:
        group_mask = groups == g
        group_data = data[group_mask]

        if verbose:
            print(f"Fitting group {g} (n={group_mask.sum()})")

        result = fit_mirt(
            group_data,
            model=model,
            n_categories=n_categories,
            n_quadpts=n_quadpts,
            max_iter=max_iter,
            tol=tol,
            verbose=False,
        )
        group_results.append(result)

    combined_result = group_results[0]

    if verbose:
        total_ll = sum(r.log_likelihood for r in group_results)
        print(f"Combined log-likelihood: {total_ll:.4f}")

    return combined_result


def _fit_metric(
    data: NDArray[np.int_],
    groups: NDArray[np.int_] | NDArray[np.str_],
    unique_groups: NDArray[np.int_] | NDArray[np.str_],
    model: Literal["1PL", "2PL", "3PL", "GRM", "GPCM"],
    n_categories: int | None,
    n_quadpts: int,
    max_iter: int,
    tol: float,
    verbose: bool,
) -> FitResult:
    """Fit metric invariance: discrimination constrained equal, intercepts free.

    Strategy:
    1. Fit combined model to get shared discrimination parameters
    2. For each group, fit with discrimination fixed, only intercepts free
    """
    from mirt import fit_mirt
    from mirt.estimation.em import EMEstimator

    if verbose:
        print("Step 1: Fitting combined model to estimate shared discrimination")

    combined_result = fit_mirt(
        data,
        model=model,
        n_categories=n_categories,
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
    )

    shared_model = combined_result.model
    shared_params = shared_model.parameters

    if verbose:
        print("Step 2: Fitting group-specific intercepts with fixed discrimination")

    group_results = []
    total_ll = 0.0

    for g in unique_groups:
        group_mask = groups == g
        group_data = data[group_mask]

        if verbose:
            print(f"  Fitting group {g} (n={group_mask.sum()})")

        group_model = shared_model.copy()

        estimator = EMEstimator(
            n_quadpts=n_quadpts,
            max_iter=max_iter,
            tol=tol,
            verbose=False,
        )

        result = estimator.fit(group_model, group_data)

        if "discrimination" in shared_params:
            group_model.set_parameters(discrimination=shared_params["discrimination"])
        elif "slopes" in shared_params:
            group_model.set_parameters(slopes=shared_params["slopes"])
        elif "general_loadings" in shared_params:
            group_model.set_parameters(
                general_loadings=shared_params["general_loadings"]
            )

        group_results.append(result)
        total_ll += result.log_likelihood

    if verbose:
        print(f"Combined log-likelihood: {total_ll:.4f}")

    return group_results[0]


def _fit_scalar(
    data: NDArray[np.int_],
    groups: NDArray[np.int_] | NDArray[np.str_],
    unique_groups: NDArray[np.int_] | NDArray[np.str_],
    model: Literal["1PL", "2PL", "3PL", "GRM", "GPCM"],
    n_categories: int | None,
    n_quadpts: int,
    max_iter: int,
    tol: float,
    verbose: bool,
) -> FitResult:
    """Fit scalar invariance: both discrimination and intercepts constrained equal.

    This is equivalent to fitting a single model to the combined data.
    """
    from mirt import fit_mirt

    if verbose:
        print("Fitting single model to combined data (scalar invariance)")

    result = fit_mirt(
        data,
        model=model,
        n_categories=n_categories,
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
    )

    if verbose:
        print(f"Log-likelihood: {result.log_likelihood:.4f}")

    return result


def _fit_strict(
    data: NDArray[np.int_],
    groups: NDArray[np.int_] | NDArray[np.str_],
    unique_groups: NDArray[np.int_] | NDArray[np.str_],
    model: Literal["1PL", "2PL", "3PL", "GRM", "GPCM"],
    n_categories: int | None,
    n_quadpts: int,
    max_iter: int,
    tol: float,
    verbose: bool,
) -> FitResult:
    """Fit strict invariance: all item parameters constrained equal.

    For most IRT models, this is the same as scalar invariance.
    Strict invariance typically also constrains residual variances,
    but standard IRT models assume unit residual variance.
    """
    from mirt import fit_mirt

    if verbose:
        print("Fitting single model to combined data (strict invariance)")
        print("Note: For standard IRT models, strict = scalar invariance")

    result = fit_mirt(
        data,
        model=model,
        n_categories=n_categories,
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
    )

    if verbose:
        print(f"Log-likelihood: {result.log_likelihood:.4f}")

    return result


def compare_invariance(
    data: NDArray[np.int_],
    groups: NDArray,
    model: Literal["1PL", "2PL", "3PL", "GRM", "GPCM"] = "2PL",
    n_categories: int | None = None,
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
) -> dict[str, FitResult]:
    """Fit and compare different invariance levels.

    Args:
        data: Response matrix.
        groups: Group membership array.
        model: IRT model type.
        n_categories: Number of categories for polytomous models.
        n_quadpts: Number of quadrature points.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.
        verbose: Print progress.

    Returns:
        Dictionary mapping invariance level to FitResult.
    """
    results = {}

    invariance_levels: list[Literal["configural", "metric", "scalar", "strict"]] = [
        "configural",
        "metric",
        "scalar",
        "strict",
    ]
    for inv in invariance_levels:
        if verbose:
            print(f"\n{'=' * 50}")
            print(f"Fitting {inv} invariance")
            print("=" * 50)

        results[inv] = fit_multigroup(
            data,
            groups,
            model=model,
            invariance=inv,
            n_categories=n_categories,
            n_quadpts=n_quadpts,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
        )

    if verbose:
        print("\n" + "=" * 50)
        print("Model Comparison")
        print("=" * 50)
        print(f"{'Model':<12} {'LL':>12} {'AIC':>12} {'BIC':>12}")
        print("-" * 50)
        for inv, result in results.items():
            print(
                f"{inv:<12} {result.log_likelihood:>12.2f} "
                f"{result.aic:>12.2f} {result.bic:>12.2f}"
            )

    return results
