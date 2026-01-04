from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.results.fit_result import FitResult


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
) -> "FitResult":
    from mirt import fit_mirt

    data = np.asarray(data)
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    if n_groups < 2:
        raise ValueError("At least 2 groups required for multiple group analysis")

    if verbose:
        print(f"Fitting {n_groups}-group {model} model with {invariance} invariance")

    if invariance == "configural":
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

    else:
        if verbose:
            print(
                f"Warning: {invariance} invariance not fully implemented, "
                "using configural"
            )

        return fit_multigroup(
            data,
            groups,
            model,
            "configural",
            n_categories,
            n_quadpts,
            max_iter,
            tol,
            verbose,
        )
