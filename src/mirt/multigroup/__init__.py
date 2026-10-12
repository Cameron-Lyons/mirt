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
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

    from mirt.model_syntax import ModelSpec
    from mirt.models.base import BaseItemModel
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
    n_quadpts: int | None = None,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
    reference_group: int | str | None = None,
    free_items: dict[str, list[int]] | None = None,
    item_names: list[str] | None = None,
    *,
    n_factors: int = 1,
    spec: ModelSpec | str | None = None,
    free_means: bool | None = None,
    free_variances: bool | None = None,
    free_correlations: bool | NDArray[np.bool_] | None = None,
    quadrature: Literal["auto", "rectangular", "gauss-hermite"] = "auto",
    accelerate: Literal["none", "squarem"] = "none",
    compute_standard_errors: bool = True,
) -> MultigroupFitResult:
    """Fit a multiple group IRT model with measurement invariance constraints.

    This function performs simultaneous estimation of IRT parameters across
    multiple groups, with options for testing measurement invariance at
    different levels.

    Parameters
    ----------
    data : ndarray or DataFrame of shape (n_persons, n_items)
        Combined response matrix for all groups. Missing responses are coded
        as negative values or ``NaN``. An item that a group never answered
        has no group-specific parameters in that group: they are neither
        estimated nor counted, and ``coef`` reports them as ``NaN``.
    groups : ndarray of shape (n_persons,)
        Group membership indicator for each person. Rows whose label is
        missing (``None`` or ``NaN``) are removed with a ``UserWarning``,
        as R's ``multipleGroup`` does.
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

        For "NRM" items, metric invariance constrains every category slope
        across groups, which is the nominal model's discrimination. R's
        ``multipleGroup(invariance="slopes")`` constrains only the overall
        slope ``a1`` and keeps the category scoring ``ak`` group-specific,
        so its metric model has more parameters. For families without
        parameters beyond slopes and intercepts, such as "2PL", "GRM",
        "GPCM" and "NRM", strict invariance equals scalar invariance.
    n_categories : int or sequence of int, optional
        Category count for all polytomous items, or one count per item. If
        None, each item's count is inferred from its largest code observed in
        any group, with a minimum of two. A group that never uses a category
        of an item whose category parameters are group-specific (thresholds
        under configural or metric invariance) raises
        ``MirtValidationError``, because those parameters are not
        identified; R's ``multipleGroup`` stops as well.
    n_quadpts : int, optional
        Quadrature points per latent dimension. The default depends on the
        number of factors, as in :func:`mirt.fit_mirt`: 61 for one factor
        and 31 for two (see :class:`MultigroupEMEstimator`).
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
    n_factors : int, default=1
        Number of latent factors of an exploratory model, as in
        :func:`mirt.fit_mirt`. Every item loads on every factor, and the
        uncorrelated reference factors identify the rotation.
    spec : ModelSpec or str, optional
        Confirmatory structure from :func:`mirt.mirt_model`, or model syntax
        parsed against the item names, as in :func:`mirt.fit_mirt`. Items
        load only on their factors, ``COV`` pairs of distinct factors free
        those correlations in every group, and ``FIXED`` holds coordinates
        at their starting values. Several factors are supported for "2PL",
        "GRM" and "GPCM" items. Group variances follow the invariance level
        and ``free_variances``, so ``COV = F*F`` terms are rejected, as are
        ``START``, ``PRIOR`` and ``CONSTRAIN`` groups.
    free_means : bool, optional
        Estimate (``True``) or fix at zero (``False``) the latent means of
        the nonreference groups, like ``"free_means"`` in R's
        ``invariance``. ``None`` estimates them whenever invariant
        intercepts identify them, from the scalar level on. ``True``
        requires such anchors. Comparing scalar fits with and without
        ``free_means`` tests the group mean differences.
    free_variances : bool, optional
        Estimate or fix at one the latent variances (covariances with
        several factors) of the nonreference groups, like R's
        ``"free_var"``. ``None`` estimates them whenever invariant slopes
        identify them, from the metric level on.
    free_correlations : bool or ndarray of bool, optional
        Factor correlations of a multidimensional model, estimated in every
        group with the reference variances held at one (see
        :meth:`MultigroupEMEstimator.fit`). ``None`` estimates them when
        the loading pattern identifies them. Use ``COV`` instead with
        ``spec``; without ``COV`` the factors of a ``spec`` model are
        uncorrelated, as in R.
    quadrature : {"auto", "rectangular", "gauss-hermite"}, default="auto"
        Quadrature rule, as in :func:`mirt.fit_mirt`. The default
        rectangular grid on ``[-6, 6]`` integrates every group with its own
        normal weights, so groups far from the reference group are fitted as
        accurately as it is, and a configural fit equals separate
        :func:`mirt.fit_mirt` fits of the groups. The grid grows to cover a
        group whose distribution reaches beyond it (see
        :class:`MultigroupEMEstimator`).
    accelerate : {"none", "squarem"}, default="none"
        EM acceleration, as in :func:`mirt.fit_mirt`. ``"squarem"``
        extrapolates every group's item parameters and latent moments
        together with a monotone safeguard (see
        :class:`MultigroupEMEstimator`); it needs several times fewer
        E-steps when latent means are weakly identified, as with a few
        anchor items. ``n_iterations`` then counts E-steps.
    compute_standard_errors : bool, default=True
        Compute observed-information standard errors of every free item
        parameter and latent mean and covariance, as R's
        ``multipleGroup(SE=TRUE)`` and :func:`mirt.fit_mirt` do (see
        :class:`MultigroupEMEstimator`). They fill
        ``MultigroupFitResult.standard_errors``,
        ``latent_standard_errors`` and ``vcov``, and the ``_se`` columns of
        ``coef(include_standard_errors=True)`` and
        ``latent_pars(include_standard_errors=True)``.

    Returns
    -------
    MultigroupFitResult
        Fitted model with per-group parameters, latent distributions,
        and fit statistics.

    Raises
    ------
    MirtValidationError
        If ``spec`` is invalid or combined with ``free_correlations``, or
        a latent moment requested with ``free_means``, ``free_variances``
        or ``free_correlations`` is not identified.

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

    >>> # Latent impact: scalar fits with and without free means
    >>> impact = fit_multigroup(data, groups, invariance="scalar")
    >>> no_impact = fit_multigroup(
    ...     data, groups, invariance="scalar", free_means=False
    ... )
    """
    from mirt.multigroup.estimator import MultigroupEMEstimator
    from mirt.multigroup.invariance import parse_invariance

    if spec is not None and free_correlations is not None:
        from mirt.exceptions import MirtValidationError

        raise MirtValidationError(
            "free_correlations cannot be combined with spec; free factor "
            "correlations with COV terms of the model syntax",
            parameter="free_correlations",
            value=free_correlations,
        )
    prepared = _prepare_multigroup(
        data,
        groups,
        model,
        n_categories=n_categories,
        reference_group=reference_group,
        item_names=item_names,
        n_factors=n_factors,
        spec=spec,
    )
    inv_spec = parse_invariance(
        invariance,
        free_items,
        free_means=free_means,
        free_variances=free_variances,
    )

    estimator = MultigroupEMEstimator(
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        quadrature=quadrature,
        accelerate=accelerate,
        compute_standard_errors=compute_standard_errors,
    )

    return estimator.fit(
        model=prepared.model,
        responses=prepared.responses,
        invariance=inv_spec,
        reference_group=prepared.reference_group,
        free_correlations=(
            free_correlations if spec is None else prepared.correlations
        ),
    )


class _PreparedMultigroup(NamedTuple):
    """Unfitted multigroup model with its data, from :func:`_prepare_multigroup`."""

    model: MultigroupModel
    responses: list[NDArray[np.int_]]
    reference_group: int
    # Factor correlations freed by a spec's COV terms; None without spec.
    correlations: NDArray[np.bool_] | bool | None


def _missing_group_labels(groups: NDArray[Any]) -> NDArray[np.bool_]:
    """Flag ``None``, ``NaN`` and other missing-value group labels."""
    import numpy as np

    if groups.dtype.kind in "fc":
        return np.isnan(groups)
    if groups.dtype.kind in "mM":
        return np.isnat(groups)
    if groups.dtype.kind != "O":
        return np.zeros(groups.shape, dtype=np.bool_)

    def missing(label: Any) -> bool:
        if label is None:
            return True
        try:
            # NaN-like values differ from themselves; pandas.NA is ambiguous.
            return bool(label != label)
        except (TypeError, ValueError):
            return True

    return np.fromiter((missing(label) for label in groups), np.bool_, groups.size)


def _prepare_multigroup(
    data: NDArray[np.int_] | Any,
    groups: NDArray,
    model: str | Sequence[str],
    *,
    n_categories: int | Sequence[int] | None,
    reference_group: int | str | None,
    item_names: list[str] | None,
    n_factors: int = 1,
    spec: ModelSpec | str | None = None,
) -> _PreparedMultigroup:
    """Build an unfitted multigroup model, per-group responses and reference index.

    Rows with missing group labels are dropped with a ``UserWarning``.
    Groups are ordered by their sorted unique labels. A string
    ``reference_group`` is matched against ``str(label)``; an integer is a
    group index and must not be the label of a different group; ``None`` is
    the first group.
    """
    import warnings

    import numpy as np

    from mirt.models._factory import single_item_family, validate_n_factors
    from mirt.multigroup.model import MultigroupModel
    from mirt.utils.data import response_column_names, validate_responses

    if item_names is None:
        item_names = response_column_names(data)
    data = validate_responses(data)
    # Like fit_mirt, a per-item sequence of one family fits that family.
    model = single_item_family(model, data.shape[1], operation="multigroup models")
    n_factors = validate_n_factors(n_factors)
    groups = np.asarray(groups)

    if groups.ndim != 1 or groups.shape[0] != data.shape[0]:
        raise ValueError(
            f"groups length ({groups.shape[0] if groups.ndim else 0}) must "
            f"match data rows ({data.shape[0]})"
        )
    missing = _missing_group_labels(groups)
    if np.any(missing):
        warnings.warn(
            f"{int(np.count_nonzero(missing))} rows with missing group labels "
            "were removed along with their responses, as in R's multipleGroup",
            UserWarning,
            stacklevel=3,
        )
        data, groups = data[~missing], groups[~missing]

    try:
        unique_groups = np.unique(groups)
    except TypeError as exc:
        raise ValueError("group labels must be mutually comparable") from exc
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
    base_model, correlations = _base_item_model(
        model,
        data,
        n_categories=n_categories,
        item_names=item_names,
        n_factors=n_factors,
        spec=spec,
    )

    mg_model = MultigroupModel(
        base_model=base_model,
        n_groups=n_groups,
        group_labels=group_labels,
    )
    group_responses = [data[groups == g_val] for g_val in unique_groups]
    return _PreparedMultigroup(mg_model, group_responses, ref_idx, correlations)


def _base_item_model(
    model: str,
    data: NDArray[np.int_],
    *,
    n_categories: int | Sequence[int] | None,
    item_names: list[str] | None,
    n_factors: int,
    spec: ModelSpec | str | None,
) -> tuple[BaseItemModel, NDArray[np.bool_] | bool | None]:
    """Build the shared item model and the factor correlations ``spec`` frees.

    Without ``spec`` the model is exploratory with ``n_factors`` factors and
    no correlation choice is returned. A ``spec`` model loads items only on
    their factors and frees the correlations of its ``COV`` pairs (``False``
    when it has none).
    """
    import numpy as np

    from mirt.exceptions import MirtValidationError
    from mirt.model_syntax import (
        _CONFIRMATORY_FAMILIES,
        ModelSpec,
        _ParameterTargets,
        mirt_model,
    )
    from mirt.models._factory import build_item_model

    n_items = data.shape[1]
    if spec is None:
        return (
            build_item_model(
                model,
                n_items,
                n_factors=n_factors,
                n_categories=n_categories,
                item_names=item_names,
                responses=data,
            ),
            None,
        )
    if isinstance(spec, str):
        spec = mirt_model(spec, item_names=item_names)
    elif not isinstance(spec, ModelSpec):
        raise MirtValidationError(
            "spec must be a ModelSpec or a model syntax string",
            parameter="spec",
            value=type(spec).__name__,
        )
    if spec.item_names is not None:
        if len(spec.item_names) != n_items or (
            item_names is not None and tuple(item_names) != spec.item_names
        ):
            raise MirtValidationError(
                "spec was parsed with item names that differ from the data's",
                parameter="spec",
            )
        item_names = list(spec.item_names)
    if n_factors not in (1, spec.n_factors):
        raise MirtValidationError(
            f"n_factors={n_factors} conflicts with the {spec.n_factors} factors "
            "of spec",
            parameter="n_factors",
            value=n_factors,
            expected=str(spec.n_factors),
        )
    unsupported = [
        keyword
        for keyword, entries in (
            ("START", spec.start),
            ("PRIOR", spec.priors),
            ("CONSTRAIN", spec.constraints),
        )
        if entries
    ]
    if unsupported:
        raise MirtValidationError(
            f"multigroup model syntax supports factor loadings, COV and FIXED, "
            f"not {', '.join(unsupported)}; constrain parameters across groups "
            "with the invariance level instead",
            parameter="spec",
        )
    covariances = spec.covariance_pattern()
    if np.any(np.diag(covariances)):
        raise MirtValidationError(
            "COV = F*F frees a factor variance, which multigroup fits set with "
            "the invariance level and free_variances",
            parameter="spec",
        )
    pattern = spec.loading_pattern(n_items)
    unloaded = np.flatnonzero(~pattern.any(axis=1))
    if unloaded.size:
        names = ", ".join(
            item_names[index] if item_names is not None else f"Item_{index + 1}"
            for index in unloaded[:5]
        )
        raise MirtValidationError(
            f"every item must load on a factor; {names} load on none",
            parameter="spec",
        )
    if spec.n_factors > 1 and model not in _CONFIRMATORY_FAMILIES:
        raise MirtValidationError(
            "multidimensional multigroup model syntax supports "
            + ", ".join(_CONFIRMATORY_FAMILIES)
            + " models",
            parameter="model",
            value=model,
        )
    base = build_item_model(
        model,
        n_items,
        n_factors=spec.n_factors,
        n_categories=n_categories,
        item_names=item_names,
        responses=data,
    )
    if spec.n_factors > 1:
        base.set_parameters(discrimination=base.parameters["discrimination"] * pattern)
        base.set_free_parameter_masks({"discrimination": pattern})
    if spec.fixed:
        targets = _ParameterTargets(base, pattern)
        free = base.free_parameter_masks
        for entry in spec.fixed:
            for name, mask in targets.masks(entry).items():
                free[name] &= ~mask
        base.set_free_parameter_masks(free)
    return base, (covariances if np.any(covariances) else False)


def compare_invariance(
    data: NDArray[np.int_],
    groups: NDArray,
    model: Literal["1PL", "2PL", "3PL", "GRM", "GPCM", "PCM", "NRM"] = "2PL",
    n_categories: int | Sequence[int] | None = None,
    n_quadpts: int | None = None,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
    reference_group: int | str | None = None,
    *,
    n_factors: int = 1,
    spec: ModelSpec | str | None = None,
    free_correlations: bool | NDArray[np.bool_] | None = None,
    quadrature: Literal["auto", "rectangular", "gauss-hermite"] = "auto",
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
    n_quadpts : int, optional
        Quadrature points per dimension, as in :func:`fit_multigroup`.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Print progress.
    reference_group : int or str, optional
        Reference group for identification, as in :func:`fit_multigroup`.
    n_factors, spec, free_correlations, quadrature
        Model structure and quadrature, as in :func:`fit_multigroup`.

    Returns
    -------
    dict
        Dictionary mapping invariance level to MultigroupFitResult. A level
        that constrains no parameter beyond the previous one is not refitted
        and maps to the previous level's result object; strict invariance
        equals scalar invariance for every family except "3PL" and "4PL",
        whose guessing and upper asymptotes it also constrains.
    """
    results: dict[str, MultigroupFitResult] = {}
    levels: list[Literal["configural", "metric", "scalar", "strict"]] = [
        "configural",
        "metric",
        "scalar",
        "strict",
    ]

    previous: str | None = None
    for level in levels:
        if previous is not None and _adds_no_constraint(
            results[previous].model, previous, level
        ):
            results[level] = results[previous]
            if verbose:
                print(
                    f"\n{level} invariance constrains no parameter beyond "
                    f"{previous} for this model; reusing the {previous} fit"
                )
            continue
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
            n_factors=n_factors,
            spec=spec,
            free_correlations=free_correlations,
            quadrature=quadrature,
        )
        previous = level

        if verbose:
            r = results[level]
            print(f"  LL={r.log_likelihood:.4f}, AIC={r.aic:.4f}, BIC={r.bic:.4f}")

    return results


def _adds_no_constraint(model: MultigroupModel, free: str, constrained: str) -> bool:
    """Whether ``constrained`` shares exactly the parameters ``free`` shares."""
    from mirt.multigroup.invariance import parse_invariance

    return set(parse_invariance(free).get_shared_parameters(model)) == set(
        parse_invariance(constrained).get_shared_parameters(model)
    )


def test_invariance_hierarchy(
    data: NDArray[np.int_],
    groups: NDArray,
    model: Literal["1PL", "2PL", "3PL", "GRM", "GPCM", "PCM", "NRM"] = "2PL",
    n_categories: int | Sequence[int] | None = None,
    n_quadpts: int | None = None,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
    reference_group: int | str | None = None,
    alpha: float = 0.05,
    *,
    n_factors: int = 1,
    spec: ModelSpec | str | None = None,
    free_correlations: bool | NDArray[np.bool_] | None = None,
    quadrature: Literal["auto", "rectangular", "gauss-hermite"] = "auto",
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
    n_quadpts : int, optional
        Quadrature points per dimension, as in :func:`fit_multigroup`.
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
    n_factors, spec, free_correlations, quadrature
        Model structure and quadrature, as in :func:`fit_multigroup`.

    Returns
    -------
    dict
        Dictionary with 'results' (fit results per level, as returned by
        :func:`compare_invariance`) and 'comparisons', one
        ``InvarianceTestResult`` per adjacent pair of levels. A level that
        adds no constraint, such as strict after scalar for a 2PL model,
        gets a row with ``chi2=0``, ``df=0``, a ``NaN`` p-value and a
        ``note``. So does a pair whose test fails, for example because
        the constrained fit has the higher log-likelihood; it also warns.
    """
    import warnings

    from mirt.multigroup.invariance import (
        InvarianceTestResult,
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
        n_factors=n_factors,
        spec=spec,
        free_correlations=free_correlations,
        quadrature=quadrature,
    )

    comparisons = []
    pairs = get_invariance_hierarchy_pairs()

    for free_level, constrained_level in pairs:
        name = f"{free_level} vs {constrained_level}"
        free_result, constrained_result = (
            results[free_level],
            results[constrained_level],
        )
        if constrained_result is free_result:
            comparisons.append(
                InvarianceTestResult(
                    comparison=name,
                    chi2=0.0,
                    df=0,
                    p_value=float("nan"),
                    delta_aic=0.0,
                    delta_bic=0.0,
                    significant=False,
                    note=(
                        f"{constrained_level} invariance constrains no parameter "
                        f"beyond {free_level} for this model; not tested"
                    ),
                )
            )
            continue
        try:
            test_result = test_invariance_step(
                constrained=constrained_result,
                free=free_result,
                comparison_name=name,
                alpha=alpha,
            )
        except ValueError as e:
            warnings.warn(f"could not compare {name}: {e}", UserWarning, stacklevel=2)
            test_result = InvarianceTestResult(
                comparison=name,
                chi2=float("nan"),
                df=int(free_result.n_parameters - constrained_result.n_parameters),
                p_value=float("nan"),
                delta_aic=float(constrained_result.aic - free_result.aic),
                delta_bic=float(constrained_result.bic - free_result.bic),
                significant=False,
                note=f"not tested: {e}",
            )
        comparisons.append(test_result)

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
            if c.note:
                print(f"  note: {c.note}")
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
