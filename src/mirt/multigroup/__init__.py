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
>>>
>>> # Likelihood-ratio DIF tests over selected anchor items
>>> from mirt.multigroup import multigroup_dif, select_dif_anchors
>>> anchors = select_dif_anchors(data, groups, model="2PL")
>>> table = multigroup_dif(data, groups, scheme="add", anchors=anchors)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

    from mirt.multigroup.invariance import InvarianceSpec
    from mirt.multigroup.model import MultigroupModel
    from mirt.multigroup.results import MultigroupFitResult

    _ItemFamily = Literal["1PL", "2PL", "3PL", "4PL", "GRM", "GPCM", "PCM", "NRM"]


_LAZY_IMPORTS = {
    "multigroup_dif": ("mirt.multigroup.dif", "multigroup_dif"),
    "select_dif_anchors": ("mirt.multigroup.dif", "select_dif_anchors"),
    "MultigroupFitResult": ("mirt.multigroup.results", "MultigroupFitResult"),
    "MultigroupModel": ("mirt.multigroup.model", "MultigroupModel"),
    "MultigroupEMEstimator": (
        "mirt.multigroup.estimator",
        "MultigroupEMEstimator",
    ),
    "MultigroupLatentDensity": (
        "mirt.multigroup.latent",
        "MultigroupLatentDensity",
    ),
    "GroupLatentDistribution": (
        "mirt.multigroup.latent",
        "GroupLatentDistribution",
    ),
    "InvarianceSpec": ("mirt.multigroup.invariance", "InvarianceSpec"),
    "InvarianceTestResult": (
        "mirt.multigroup.invariance",
        "InvarianceTestResult",
    ),
    "ParameterLink": ("mirt.multigroup.model", "ParameterLink"),
    "invariance_lrt": ("mirt.multigroup.invariance", "invariance_lrt"),
    "parse_invariance": ("mirt.multigroup.invariance", "parse_invariance"),
}


def fit_multigroup(
    data: NDArray[np.int_] | Any,
    groups: NDArray,
    model: _ItemFamily | Sequence[_ItemFamily] = "2PL",
    invariance: Literal["configural", "metric", "scalar", "strict"]
    | InvarianceSpec = "configural",
    n_categories: int | Sequence[int] | None = None,
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
    reference_group: int | str | None = None,
    free_items: dict[str, list[int]] | None = None,
    item_names: list[str] | None = None,
) -> MultigroupFitResult:
    """Fit a multiple group IRT model with measurement invariance constraints.

    This function performs simultaneous estimation of IRT parameters across
    multiple groups, with options for testing measurement invariance at
    different levels.

    Parameters
    ----------
    data : ndarray or DataFrame of shape (n_persons, n_items)
        Combined response matrix for all groups. Missing responses are coded
        as negative values or ``NaN``.
    groups : ndarray of shape (n_persons,)
        Group membership indicator for each person.
    model : str or sequence of str
        IRT model type: "1PL", "2PL", "3PL", "4PL", "GRM", "GPCM", "PCM", "NRM".
        As in ``fit_mirt``, a per-item sequence naming one family fits that
        family; mixed families raise ``MirtValidationError``.
    invariance : str or InvarianceSpec
        Level of measurement invariance:

        - 'configural': Same model structure, all parameters free
        - 'metric': Discrimination/slopes constrained equal
        - 'scalar': Discrimination and intercepts constrained equal
        - 'strict': All item parameters constrained equal

    n_categories : int or sequence of int, optional
        Category count for all polytomous items, or one count per item. If
        None, each item's count is inferred from its largest code observed in
        any group, with a minimum of two.
    n_quadpts : int
        Number of quadrature points for numerical integration.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Print iteration progress.
    reference_group : int or str, optional
        Group to use as reference (mean=0, var=1): an index into the sorted
        group labels, or a label matched against ``str(label)``. An integer
        that is also the label of another group raises ``ValueError``.
        Defaults to the first group in sorted order.
    free_items : dict, optional
        For partial invariance: {param_name: [item_indices]} to free.
    item_names : list of str, optional
        Names for each item. If None, unique DataFrame column names are used
        when available.

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
    from mirt.multigroup.estimator import MultigroupEMEstimator
    from mirt.multigroup.invariance import parse_invariance

    mg_model, group_responses, ref_idx = _prepare_multigroup(
        data,
        groups,
        model,
        n_categories=n_categories,
        reference_group=reference_group,
        item_names=item_names,
    )
    inv_spec = parse_invariance(invariance, free_items)

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


def _prepare_multigroup(
    data: NDArray[np.int_] | Any,
    groups: NDArray,
    model: str | Sequence[str],
    *,
    n_categories: int | Sequence[int] | None,
    reference_group: int | str | None,
    item_names: list[str] | None,
) -> tuple[MultigroupModel, list[NDArray[np.int_]], int]:
    """Build an unfitted multigroup model, per-group responses and reference index.

    Groups are ordered by their sorted unique labels. A string
    ``reference_group`` is matched against ``str(label)``; an integer is a
    group index and must not be the label of a different group; ``None`` is
    the first group.
    """
    import numpy as np

    from mirt.models._factory import build_item_model, single_item_family
    from mirt.multigroup.model import MultigroupModel
    from mirt.utils.data import response_column_names, validate_responses

    if item_names is None:
        item_names = response_column_names(data)
    data = validate_responses(data)
    # Like fit_mirt, a per-item sequence of one family fits that family.
    model = single_item_family(model, data.shape[1], operation="multigroup models")
    groups = np.asarray(groups)

    if groups.shape[0] != data.shape[0]:
        raise ValueError(
            f"groups length ({groups.shape[0]}) must match data rows ({data.shape[0]})"
        )

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    if n_groups < 2:
        raise ValueError("At least 2 groups required for multiple group analysis")

    group_labels = [str(g) for g in unique_groups]
    if reference_group is None:
        ref_idx = 0
    elif isinstance(reference_group, str):
        if reference_group not in group_labels:
            raise ValueError(f"Unknown reference group: {reference_group}")
        ref_idx = group_labels.index(reference_group)
    else:
        ref_idx = reference_group
        label = str(reference_group)
        if (
            not isinstance(reference_group, (bool, np.bool_))
            and isinstance(reference_group, (int, np.integer))
            and label in group_labels
            and group_labels.index(label) != reference_group
        ):
            advice = f"pass reference_group={label!r} to select that group by label"
            if 0 <= reference_group < n_groups:
                advice += (
                    f", or reference_group={group_labels[reference_group]!r} for "
                    f"group index {reference_group}"
                )
            raise ValueError(
                f"reference_group={reference_group} is a group index, but it is "
                f"also the label of group {group_labels.index(label)}; {advice}"
            )

    # Categories come from the pooled data so every group shares one structure.
    base_model = build_item_model(
        model,
        data.shape[1],
        n_categories=n_categories,
        item_names=item_names,
        responses=data,
    )

    mg_model = MultigroupModel(
        base_model=base_model,
        n_groups=n_groups,
        group_labels=group_labels,
    )
    group_responses = [data[groups == g_val] for g_val in unique_groups]
    return mg_model, group_responses, ref_idx


def compare_invariance(
    data: NDArray[np.int_],
    groups: NDArray,
    model: Literal["1PL", "2PL", "3PL", "GRM", "GPCM", "PCM", "NRM"] = "2PL",
    n_categories: int | Sequence[int] | None = None,
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
    reference_group: int | str | None = None,
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
    n_categories : int or sequence of int, optional
        Category count for all polytomous items, or one count per item.
    n_quadpts : int
        Number of quadrature points.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Print progress.
    reference_group : int or str, optional
        Reference group for identification, as in :func:`fit_multigroup`.

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
    n_categories: int | Sequence[int] | None = None,
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
    reference_group: int | str | None = None,
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
    n_categories : int or sequence of int, optional
        Category count for all polytomous items, or one count per item.
    n_quadpts : int
        Number of quadrature points.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Print progress.
    reference_group : int or str, optional
        Reference group for identification, as in :func:`fit_multigroup`.
    alpha : float
        Significance level for LRT tests.

    Returns
    -------
    dict
        Dictionary with 'results' (fit results per level) and
        'comparisons' (LRT test results).
    """
    from mirt.multigroup.invariance import (
        get_invariance_hierarchy_pairs,
        test_invariance_step,
    )

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
    "multigroup_dif",
    "select_dif_anchors",
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


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_name, symbol_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, symbol_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'mirt.multigroup' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
