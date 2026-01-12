"""Multiple group IRT analysis with measurement invariance testing.

This module provides simultaneous estimation of IRT models across multiple
groups with support for measurement invariance constraints at different
levels (configural, metric, scalar, strict).

Examples
--------
>>> from mirt import simdata
>>> from mirt.multigroup import fit_multigroup, compare_invariance
>>>
>>> # Generate data for two groups
>>> data1 = simdata(n_persons=500, n_items=20)
>>> data2 = simdata(n_persons=500, n_items=20)
>>> import numpy as np
>>> data = np.vstack([data1, data2])
>>> groups = np.array([0] * 500 + [1] * 500)
>>>
>>> # Fit with metric invariance
>>> result = fit_multigroup(data, groups, model="2PL", invariance="metric")
>>> print(result.summary())
>>>
>>> # Compare all invariance levels
>>> results = compare_invariance(data, groups, model="2PL")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from mirt.multigroup.estimator import MultigroupEMEstimator
from mirt.multigroup.invariance import (
    InvarianceSpec,
    InvarianceTestResult,
    get_invariance_hierarchy_pairs,
    invariance_lrt,
    parse_invariance,
    test_invariance_step,
)
from mirt.multigroup.latent import GroupLatentDistribution, MultigroupLatentDensity
from mirt.multigroup.model import MultigroupModel, ParameterLink
from mirt.multigroup.results import MultigroupFitResult

if TYPE_CHECKING:
    pass


def fit_multigroup(
    data: NDArray[np.int_],
    groups: NDArray,
    model: Literal["1PL", "2PL", "3PL", "4PL", "GRM", "GPCM", "PCM", "NRM"] = "2PL",
    invariance: Literal["configural", "metric", "scalar", "strict"]
    | InvarianceSpec = "configural",
    n_categories: int | None = None,
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
    reference_group: int | str = 0,
    free_items: dict[str, list[int]] | None = None,
) -> MultigroupFitResult:
    """Fit a multiple group IRT model with measurement invariance constraints.

    This function performs simultaneous estimation of IRT parameters across
    multiple groups, with options for testing measurement invariance at
    different levels.

    Parameters
    ----------
    data : ndarray of shape (n_persons, n_items)
        Combined response matrix for all groups.
    groups : ndarray of shape (n_persons,)
        Group membership indicator for each person.
    model : str
        IRT model type: "1PL", "2PL", "3PL", "4PL", "GRM", "GPCM", "PCM", "NRM".
    invariance : str or InvarianceSpec
        Level of measurement invariance:

        - 'configural': Same model structure, all parameters free
        - 'metric': Discrimination/slopes constrained equal
        - 'scalar': Discrimination and intercepts constrained equal
        - 'strict': All item parameters constrained equal

    n_categories : int, optional
        Number of response categories for polytomous models.
    n_quadpts : int
        Number of quadrature points for numerical integration.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Print iteration progress.
    reference_group : int or str
        Group to use as reference (mean=0, var=1). Can be index or label.
    free_items : dict, optional
        For partial invariance: {param_name: [item_indices]} to free.

    Returns
    -------
    MultigroupFitResult
        Fitted model with per-group parameters, latent distributions,
        and fit statistics.

    Examples
    --------
    >>> from mirt.multigroup import fit_multigroup
    >>> result = fit_multigroup(data, groups, model="2PL", invariance="metric")
    >>> print(result.summary())

    >>> # Partial invariance: free item 5's discrimination
    >>> result = fit_multigroup(
    ...     data, groups, model="2PL", invariance="metric",
    ...     free_items={"discrimination": [5]}
    ... )
    """
    from mirt.models.dichotomous import (
        FourParameterLogistic,
        OneParameterLogistic,
        ThreeParameterLogistic,
        TwoParameterLogistic,
    )
    from mirt.models.polytomous import (
        GeneralizedPartialCredit,
        GradedResponseModel,
        NominalResponseModel,
        PartialCreditModel,
    )

    data = np.asarray(data)
    groups = np.asarray(groups)

    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got {data.ndim}D")
    if groups.shape[0] != data.shape[0]:
        raise ValueError(
            f"groups length ({groups.shape[0]}) must match data rows ({data.shape[0]})"
        )

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    if n_groups < 2:
        raise ValueError("At least 2 groups required for multiple group analysis")

    group_labels = [str(g) for g in unique_groups]
    if isinstance(reference_group, str):
        if reference_group not in group_labels:
            raise ValueError(f"Unknown reference group: {reference_group}")
        ref_idx = group_labels.index(reference_group)
    else:
        ref_idx = reference_group

    n_items = data.shape[1]

    is_polytomous = model in ("GRM", "GPCM", "PCM", "NRM")
    if is_polytomous:
        if n_categories is None:
            n_categories = int(data[data >= 0].max()) + 1

    if model == "1PL":
        base_model = OneParameterLogistic(n_items=n_items)
    elif model == "2PL":
        base_model = TwoParameterLogistic(n_items=n_items)
    elif model == "3PL":
        base_model = ThreeParameterLogistic(n_items=n_items)
    elif model == "4PL":
        base_model = FourParameterLogistic(n_items=n_items)
    elif model == "GRM":
        base_model = GradedResponseModel(n_items=n_items, n_categories=n_categories)
    elif model == "GPCM":
        base_model = GeneralizedPartialCredit(
            n_items=n_items, n_categories=n_categories
        )
    elif model == "PCM":
        base_model = PartialCreditModel(n_items=n_items, n_categories=n_categories)
    elif model == "NRM":
        base_model = NominalResponseModel(n_items=n_items, n_categories=n_categories)
    else:
        raise ValueError(f"Unknown model: {model}")

    mg_model = MultigroupModel(
        base_model=base_model,
        n_groups=n_groups,
        group_labels=group_labels,
    )

    inv_spec = parse_invariance(invariance, free_items)

    group_responses = []
    for g_val in unique_groups:
        mask = groups == g_val
        group_responses.append(data[mask])

    estimator = MultigroupEMEstimator(
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
    )

    result = estimator.fit(
        model=mg_model,
        responses=group_responses,
        invariance=inv_spec,
        reference_group=ref_idx,
    )

    return result


def compare_invariance(
    data: NDArray[np.int_],
    groups: NDArray,
    model: Literal["1PL", "2PL", "3PL", "GRM", "GPCM", "PCM", "NRM"] = "2PL",
    n_categories: int | None = None,
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
    reference_group: int | str = 0,
) -> dict[str, MultigroupFitResult]:
    """Fit and compare different invariance levels.

    Parameters
    ----------
    data : ndarray
        Response matrix.
    groups : ndarray
        Group membership array.
    model : str
        IRT model type.
    n_categories : int, optional
        Number of categories for polytomous models.
    n_quadpts : int
        Number of quadrature points.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Print progress.
    reference_group : int or str
        Reference group for identification.

    Returns
    -------
    dict
        Dictionary mapping invariance level to MultigroupFitResult.
    """
    results = {}
    levels: list[Literal["configural", "metric", "scalar", "strict"]] = [
        "configural",
        "metric",
        "scalar",
        "strict",
    ]

    for level in levels:
        if verbose:
            print(f"\nFitting {level} invariance...")

        results[level] = fit_multigroup(
            data=data,
            groups=groups,
            model=model,
            invariance=level,
            n_categories=n_categories,
            n_quadpts=n_quadpts,
            max_iter=max_iter,
            tol=tol,
            verbose=False,
            reference_group=reference_group,
        )

        if verbose:
            r = results[level]
            print(f"  LL={r.log_likelihood:.4f}, AIC={r.aic:.4f}, BIC={r.bic:.4f}")

    return results


def test_invariance_hierarchy(
    data: NDArray[np.int_],
    groups: NDArray,
    model: Literal["1PL", "2PL", "3PL", "GRM", "GPCM", "PCM", "NRM"] = "2PL",
    n_categories: int | None = None,
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
    reference_group: int | str = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Test full invariance hierarchy with likelihood ratio tests.

    Parameters
    ----------
    data : ndarray
        Response matrix.
    groups : ndarray
        Group membership array.
    model : str
        IRT model type.
    n_categories : int, optional
        Number of categories for polytomous models.
    n_quadpts : int
        Number of quadrature points.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Print progress.
    reference_group : int or str
        Reference group for identification.
    alpha : float
        Significance level for LRT tests.

    Returns
    -------
    dict
        Dictionary with 'results' (fit results per level) and
        'comparisons' (LRT test results).
    """
    results = compare_invariance(
        data=data,
        groups=groups,
        model=model,
        n_categories=n_categories,
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        reference_group=reference_group,
    )

    comparisons = []
    pairs = get_invariance_hierarchy_pairs()

    for free_level, constrained_level in pairs:
        try:
            test_result = test_invariance_step(
                constrained=results[constrained_level],
                free=results[free_level],
                comparison_name=f"{free_level} vs {constrained_level}",
                alpha=alpha,
            )
            comparisons.append(test_result)
        except ValueError as e:
            if verbose:
                print(
                    f"Warning: Could not compare {free_level} vs {constrained_level}: {e}"
                )

    if verbose:
        print("\n" + "=" * 60)
        print("Invariance Hierarchy Test Results")
        print("=" * 60)
        print(f"{'Comparison':<25} {'Chi2':>10} {'df':>6} {'p':>10} {'Sig':>6}")
        print("-" * 60)
        for c in comparisons:
            sig = "*" if c.significant else ""
            print(
                f"{c.comparison:<25} {c.chi2:>10.4f} {c.df:>6} {c.p_value:>10.4f} {sig:>6}"
            )
        print("=" * 60)

    return {
        "results": results,
        "comparisons": comparisons,
    }


__all__ = [
    "fit_multigroup",
    "compare_invariance",
    "test_invariance_hierarchy",
    "MultigroupFitResult",
    "MultigroupModel",
    "MultigroupEMEstimator",
    "MultigroupLatentDensity",
    "GroupLatentDistribution",
    "InvarianceSpec",
    "InvarianceTestResult",
    "ParameterLink",
    "invariance_lrt",
    "parse_invariance",
]
