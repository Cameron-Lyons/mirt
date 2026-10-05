"""Likelihood-ratio DIF tests and anchor selection for multiple groups.

Each test compares two nested multiple-group fits that differ only in
whether one studied item's parameters are constrained equal across groups.
Latent means and variances of nonreference groups are estimated whenever
invariant anchors identify them, so group impact is not mistaken for DIF.
The workflow follows ``mirt::DIF`` (Chalmers, 2012) with the add, drop and
sequential schemes, and all-other-as-anchor selection (Kopf, Zeileis and
Strobl, 2015).
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from mirt.diagnostics.multiple_testing import (
    PValueAdjustment,
    adjust_p_values,
    validate_p_value_adjustment,
)
from mirt.multigroup.estimator import MultigroupEMEstimator
from mirt.multigroup.invariance import (
    DISCRIMINATION_PARAMS,
    INTERCEPT_PARAMS,
    InvarianceSpec,
)
from mirt.utils.bootstrap import _run_bootstrap_tasks, _validate_n_jobs

if TYPE_CHECKING:
    from mirt.multigroup.latent import GroupLatentDistribution
    from mirt.multigroup.model import MultigroupModel
    from mirt.multigroup.results import MultigroupFitResult


DIFScheme = Literal["drop", "add", "drop_sequential", "add_sequential"]
DIFParameterFamily = Literal["discrimination", "intercepts"]
AnchorSelectionMethod = Literal["aoaa_iterative", "rank"]

DIF_SCHEMES: tuple[str, ...] = get_args(DIFScheme)
_PARAMETER_FAMILIES: tuple[str, ...] = get_args(DIFParameterFamily)
_FAMILY_PARAMETERS = {
    "discrimination": DISCRIMINATION_PARAMS,
    "intercepts": INTERCEPT_PARAMS,
}
_FIT_EXCEPTIONS = (
    ValueError,
    RuntimeError,
    ArithmeticError,
    np.linalg.LinAlgError,
)


@dataclass(frozen=True)
class _FitSettings:
    n_quadpts: int
    max_iter: int
    tol: float
    reference_group: int
    families: tuple[str, ...]


@dataclass(frozen=True)
class _FitSummary:
    log_likelihood: float
    n_parameters: int
    aic: float
    bic: float
    converged: bool
    item_parameters: tuple[dict[str, Any], ...]


@dataclass(slots=True)
class _RefitTask:
    start_model: MultigroupModel
    start_latent: list[GroupLatentDistribution]
    responses: list[NDArray[np.int_]]
    free_items: tuple[int, ...]
    item: int
    settings: _FitSettings


@dataclass(frozen=True)
class _DIFTestRow:
    """One studied item's likelihood-ratio test (internal result row)."""

    item: int
    chi2: float
    df: float
    p_value: float
    p_value_adjusted: float
    delta_aic: float
    delta_bic: float
    converged: bool
    round: int
    group_parameters: tuple[dict[str, Any], ...] | None


@dataclass(frozen=True)
class _DIFTestTable:
    """Rows of a multiple-group DIF analysis with model metadata."""

    rows: list[_DIFTestRow]
    item_names: list[str]
    group_labels: list[str]
    reference_group: int
    n_categories: list[int] | None
    alpha: float
    p_adjust: str

    def flagged(self) -> list[int]:
        """Studied items whose adjusted p-value is below ``alpha``."""
        return [row.item for row in self.rows if row.p_value_adjusted < self.alpha]

    def to_dataframe(self) -> Any:
        """Return one row per studied item as a DataFrame."""
        from mirt.utils.dataframe import create_dataframe

        def column(name: str, dtype: Any = np.float64) -> NDArray[Any]:
            return np.array([getattr(row, name) for row in self.rows], dtype=dtype)

        statistics = ("chi2", "df", "p_value", "p_value_adjusted")
        return create_dataframe(
            {
                "item": [self.item_names[row.item] for row in self.rows],
                **{name: column(name) for name in statistics},
                "delta_aic": column("delta_aic"),
                "delta_bic": column("delta_bic"),
                "flagged": column("p_value_adjusted") < self.alpha,
                "converged": column("converged", np.bool_),
                "round": column("round", np.int64),
            }
        )


def multigroup_dif(
    data: NDArray[np.int_] | Any,
    groups: NDArray[Any],
    model: str = "2PL",
    *,
    items: Sequence[int | str] | None = None,
    anchors: Sequence[int | str] | None = None,
    scheme: DIFScheme = "drop",
    parameters: Sequence[DIFParameterFamily] = ("discrimination", "intercepts"),
    p_adjust: PValueAdjustment = "none",
    alpha: float = 0.05,
    n_categories: int | Sequence[int] | None = None,
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    reference_group: int | str | None = None,
    max_rounds: int = 10,
    n_jobs: int = 1,
) -> Any:
    """Test items for DIF with nested multiple-group likelihood-ratio tests.

    Every test compares a model in which the studied item's ``parameters``
    are constrained equal across groups with one in which they are free.
    Nonreference latent means and variances are estimated whenever invariant
    items identify them, so the tests operate on a common latent scale.
    Parameters outside ``parameters`` (for example 3PL guessing) are always
    held equal across groups.

    Parameters
    ----------
    data : ndarray or DataFrame of shape (n_persons, n_items)
        Combined response matrix. Missing responses are negative or ``NaN``.
    groups : ndarray of shape (n_persons,)
        Group membership. Two or more groups are supported.
    model : str
        Item model passed to :func:`mirt.multigroup.fit_multigroup`.
    items : sequence of int or str, optional
        Items to test, by index or name. Defaults to every non-anchor item.
    anchors : sequence of int or str, optional
        Items assumed free of DIF. They are constrained in every model and
        never tested. Required by the ``"add"`` schemes.
    scheme : {"drop", "add", "drop_sequential", "add_sequential"}
        ``"drop"`` starts from a fully constrained model and frees one
        studied item at a time; every other item acts as an anchor.
        ``"add"`` starts from a model that constrains only ``anchors`` and
        constrains one studied item at a time. ``"drop_sequential"`` repeats
        the drop step, leaving items flagged in earlier rounds free, until no
        new item is flagged. ``"add_sequential"`` repeats the add step,
        adding items that showed no DIF to the anchors, until no new
        invariant item is found.
    parameters : sequence of {"discrimination", "intercepts"}
        Parameter families tested for each studied item. ``"intercepts"``
        covers difficulties, thresholds and step parameters.
    p_adjust : {"none", "holm", "bonferroni", "fdr_bh"}
        Multiple-testing adjustment over the items tested in each round.
        Default ``"none"``, as in :func:`mirt.dif` and R's ``mirt::DIF``.
    alpha : float
        Significance level for flagging and sequential decisions.
    n_categories : int or sequence of int, optional
        Category counts for polytomous items.
    n_quadpts, max_iter, tol : int, int, float
        EM settings for every fit.
    reference_group : int or str, optional
        Reference group index, or label matched against ``str(label)``. An
        integer that is also the label of another group raises
        ``ValueError``; pass such a label as a string. Defaults to the first
        group in sorted order.
    max_rounds : int
        Maximum rounds of the sequential schemes.
    n_jobs : int
        Worker processes for the independent per-item refits. ``-1`` uses
        all cores.

    Returns
    -------
    DataFrame
        One row per studied item with columns ``item``, ``chi2``, ``df``,
        ``p_value``, ``p_value_adjusted``, ``delta_aic``, ``delta_bic``,
        ``flagged``, ``converged`` and ``round``. ``chi2`` is twice the
        log-likelihood gain of the model that frees the item and ``df`` the
        number of parameters it adds. ``delta_aic`` and ``delta_bic`` are
        the constrained-minus-free criteria, so positive values favor DIF.
        Sequential schemes report each item's test from the last round in
        which it was tested. Failed refits give ``NaN`` statistics.

    Raises
    ------
    ValueError
        If an option is invalid, or if ``parameters`` has no free
        coordinate in any studied item, as with ``"discrimination"`` for a
        1PL model. Studied items without one are tested with ``df=0`` and
        ``NaN`` p-values after a ``UserWarning``.

    Notes
    -----
    A drop analysis needs one baseline fit plus one refit per studied item;
    sequential schemes repeat this for each round. Refits are warm-started
    from the baseline estimates and run in ``n_jobs`` worker processes.
    Built-in 1PL and 2PL items take batched Newton M-steps, so a drop
    analysis of 30 binary items with 1,000 persons per group takes one to
    two seconds; 3PL and polytomous items take an itemwise optimizer and are
    several times slower per item. :func:`mirt.diagnostics.compute_grdif`
    offers a faster residual-based screen that does not refit models.

    References
    ----------
    Chalmers, R. P. (2012). mirt: A multidimensional item response theory
    package for the R environment. Journal of Statistical Software, 48(6).

    Kopf, J., Zeileis, A., & Strobl, C. (2015). Anchor selection strategies
    for DIF analysis: Review, assessment, and new approaches. Educational and
    Psychological Measurement, 75(1), 22-56.

    Examples
    --------
    >>> from mirt.multigroup import multigroup_dif
    >>> table = multigroup_dif(data, groups, model="2PL", scheme="drop")
    """
    table = run_multigroup_dif(
        data,
        groups,
        model,
        items=items,
        anchors=anchors,
        scheme=scheme,
        parameters=parameters,
        p_adjust=p_adjust,
        alpha=alpha,
        n_categories=n_categories,
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        reference_group=reference_group,
        max_rounds=max_rounds,
        n_jobs=n_jobs,
    )
    return table.to_dataframe()


def select_dif_anchors(
    data: NDArray[np.int_] | Any,
    groups: NDArray[Any],
    model: str = "2PL",
    *,
    method: AnchorSelectionMethod = "aoaa_iterative",
    n_anchors: int | None = 4,
    parameters: Sequence[DIFParameterFamily] = ("discrimination", "intercepts"),
    p_adjust: PValueAdjustment = "none",
    alpha: float = 0.05,
    n_categories: int | Sequence[int] | None = None,
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    reference_group: int | str | None = None,
    max_rounds: int = 10,
    n_jobs: int = 1,
) -> list[int]:
    """Select DIF-free anchor items with all-other-as-anchor LR tests.

    Parameters
    ----------
    data, groups, model
        As for :func:`multigroup_dif`.
    method : {"aoaa_iterative", "rank"}
        ``"aoaa_iterative"`` runs the ``"drop_sequential"`` scheme: every
        item is tested with all other retained items as anchors, flagged
        items are freed, and the remaining items are retested until no new
        item is flagged. ``"rank"`` runs a single ``"drop"`` pass.
        Candidates are ranked by p-value, largest first.
    n_anchors : int or None
        Number of anchors to return. ``None`` returns every unflagged item.
        ``"aoaa_iterative"`` never returns a flagged item, so it may return
        fewer than ``n_anchors``.
    parameters, p_adjust, alpha, n_categories, n_quadpts, max_iter, tol
        As for :func:`multigroup_dif`.
    reference_group, max_rounds, n_jobs
        As for :func:`multigroup_dif`.

    Returns
    -------
    list of int
        Sorted indices of the selected anchor items.

    Raises
    ------
    ValueError
        If no item qualifies as an anchor.
    """
    if method not in {"aoaa_iterative", "rank"}:
        raise ValueError("method must be 'aoaa_iterative' or 'rank'")
    if n_anchors is not None and (
        isinstance(n_anchors, (bool, np.bool_))
        or not isinstance(n_anchors, (int, np.integer))
        or n_anchors < 1
    ):
        raise ValueError("n_anchors must be a positive integer or None")

    table = run_multigroup_dif(
        data,
        groups,
        model,
        scheme="drop_sequential" if method == "aoaa_iterative" else "drop",
        parameters=parameters,
        p_adjust=p_adjust,
        alpha=alpha,
        n_categories=n_categories,
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        reference_group=reference_group,
        max_rounds=max_rounds,
        n_jobs=n_jobs,
    )
    tested = [row for row in table.rows if np.isfinite(row.p_value)]
    ranked = sorted(tested, key=lambda row: (-row.p_value, row.chi2, row.item))
    if method == "aoaa_iterative" or n_anchors is None:
        ranked = [row for row in ranked if not row.p_value_adjusted < alpha]
    selected = ranked if n_anchors is None else ranked[: int(n_anchors)]
    if not selected:
        raise ValueError("no item qualified as a DIF-free anchor")
    return sorted(row.item for row in selected)


def run_multigroup_dif(
    data: NDArray[np.int_] | Any,
    groups: NDArray[Any],
    model: str = "2PL",
    *,
    items: Sequence[int | str] | None = None,
    anchors: Sequence[int | str] | None = None,
    scheme: DIFScheme = "drop",
    parameters: Sequence[DIFParameterFamily] = ("discrimination", "intercepts"),
    p_adjust: PValueAdjustment = "none",
    alpha: float = 0.05,
    n_categories: int | Sequence[int] | None = None,
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    reference_group: int | str | None = None,
    max_rounds: int = 10,
    n_jobs: int = 1,
) -> _DIFTestTable:
    """Run :func:`multigroup_dif` and return rows with group parameters.

    This is the entry point of :func:`mirt.diagnostics.compute_dif`. The
    per-item ``group_parameters`` come from the model in which the studied
    item is free, ordered like ``group_labels``.
    """
    from mirt.multigroup import _prepare_multigroup

    if scheme not in DIF_SCHEMES:
        raise ValueError(f"scheme must be one of: {', '.join(DIF_SCHEMES)}")
    families = _validate_parameter_families(parameters)
    p_adjust = validate_p_value_adjustment(p_adjust, name="p_adjust")
    if (
        isinstance(alpha, (bool, np.bool_))
        or not isinstance(alpha, (int, float, np.integer, np.floating))
        or not 0.0 < float(alpha) < 1.0
    ):
        raise ValueError("alpha must be finite and in (0, 1)")
    if (
        isinstance(max_rounds, (bool, np.bool_))
        or not isinstance(max_rounds, (int, np.integer))
        or max_rounds < 1
    ):
        raise ValueError("max_rounds must be a positive integer")
    n_jobs = _validate_n_jobs(n_jobs)

    start_model, responses, ref_idx = _prepare_multigroup(
        data,
        groups,
        model,
        n_categories=n_categories,
        reference_group=reference_group,
        item_names=None,
    )
    if (
        isinstance(ref_idx, (bool, np.bool_))
        or not isinstance(ref_idx, (int, np.integer))
        or not 0 <= ref_idx < start_model.n_groups
    ):
        raise ValueError("reference_group must be a valid group index or label")
    n_items = start_model.n_items
    item_names = list(start_model.item_names)
    anchor_set = resolve_items(anchors, item_names, "anchors") or []
    tested = resolve_items(items, item_names, "items")
    if tested is None:
        tested = [item for item in range(n_items) if item not in anchor_set]
    if not tested:
        raise ValueError("at least one item must be tested for DIF")
    overlap = sorted(set(tested) & set(anchor_set))
    if overlap:
        raise ValueError(f"items and anchors must not overlap: {overlap}")
    adding = scheme.startswith("add")
    if adding and not anchor_set:
        raise ValueError(f"scheme={scheme!r} requires at least one anchor item")
    _check_testable(start_model, int(ref_idx), tested, families, model)

    settings = _FitSettings(
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        reference_group=int(ref_idx),
        families=families,
    )
    rows = _run_rounds(
        start_model,
        responses,
        tested=sorted(tested),
        anchors=set(anchor_set),
        adding=adding,
        sequential=scheme.endswith("_sequential"),
        settings=settings,
        p_adjust=p_adjust,
        alpha=float(alpha),
        max_rounds=int(max_rounds),
        n_jobs=n_jobs,
    )
    template = start_model.get_group_model(0)
    return _DIFTestTable(
        rows=sorted(rows.values(), key=lambda row: row.item),
        item_names=item_names,
        group_labels=list(start_model.group_labels),
        reference_group=int(ref_idx),
        n_categories=(
            [int(count) for count in template.n_categories]
            if start_model.is_polytomous
            else None
        ),
        alpha=float(alpha),
        p_adjust=p_adjust,
    )


def _run_rounds(
    start_model: MultigroupModel,
    responses: list[NDArray[np.int_]],
    *,
    tested: list[int],
    anchors: set[int],
    adding: bool,
    sequential: bool,
    settings: _FitSettings,
    p_adjust: PValueAdjustment,
    alpha: float,
    max_rounds: int,
    n_jobs: int,
) -> dict[int, _DIFTestRow]:
    """Fit each round's baseline and per-item alternatives."""
    all_items = set(range(start_model.n_items))
    constrained = set(anchors)
    freed: set[int] = set()
    remaining = list(tested)
    start_latent: list[GroupLatentDistribution] | None = None
    rows: dict[int, _DIFTestRow] = {}

    for round_number in range(1, max_rounds + 1):
        baseline_free = all_items - constrained if adding else set(freed)
        baseline = _fit(start_model, responses, baseline_free, settings, start_latent)
        tasks = [
            _RefitTask(
                start_model=baseline.model,
                start_latent=baseline.latent_distributions,
                responses=responses,
                free_items=tuple(sorted(baseline_free ^ {item})),
                item=item,
                settings=settings,
            )
            for item in remaining
        ]
        summaries = _run_bootstrap_tasks(_refit, tasks, n_jobs)
        base_summary = _summarize(baseline, None)
        round_rows = [
            _likelihood_ratio_row(
                item,
                base_summary,
                summary,
                baseline_parameters=(
                    _item_parameters(baseline.model, item) if adding else None
                ),
                round_number=round_number,
            )
            for item, summary in zip(remaining, summaries, strict=True)
        ]
        adjusted = adjust_p_values(
            np.array([row.p_value for row in round_rows], dtype=np.float64), p_adjust
        )
        significant: dict[int, bool] = {}
        for row, p_adjusted in zip(round_rows, adjusted, strict=True):
            rows[row.item] = replace(row, p_value_adjusted=float(p_adjusted))
            significant[row.item] = bool(p_adjusted < alpha)

        if not sequential:
            break
        if adding:
            # Items that pass join the anchors; failed refits stay under test.
            settled = [
                item
                for item in remaining
                if not significant[item] and np.isfinite(rows[item].p_value)
            ]
            constrained.update(settled)
        else:
            settled = [item for item in remaining if significant[item]]
            freed.update(settled)
        remaining = [item for item in remaining if item not in settled]
        if not settled or not remaining:
            break
        start_model, start_latent = baseline.model, baseline.latent_distributions
    else:
        warnings.warn(
            f"sequential DIF scheme did not settle within {max_rounds} rounds; "
            "rows report each item's last test",
            UserWarning,
            stacklevel=4,
        )
    return rows


def _check_testable(
    model: MultigroupModel,
    reference: int,
    tested: list[int],
    families: tuple[str, ...],
    name: Any,
) -> None:
    """Reject tests that free nothing, and warn about items that free nothing."""
    masks = model.effective_free_parameter_masks(reference)
    free = np.zeros(model.n_items, dtype=np.bool_)
    for family in families:
        for parameter in _FAMILY_PARAMETERS[family] & masks.keys():
            mask = masks[parameter]
            if mask.ndim and mask.shape[0] == model.n_items:
                free |= mask.reshape(model.n_items, -1).any(axis=1)
    untestable = [item for item in tested if not free[item]]
    if len(untestable) == len(tested):
        raise ValueError(
            f"parameters {list(families)} have no free coordinates in the tested "
            f"items of model {name!r}; nothing to test"
        )
    if untestable:
        warnings.warn(
            f"parameters {list(families)} have no free coordinates in items "
            f"{[model.item_names[item] for item in untestable]}; their tests "
            "have df=0 and NaN p-values",
            UserWarning,
            stacklevel=4,
        )


def _fit(
    start_model: MultigroupModel,
    responses: list[NDArray[np.int_]],
    free_items: set[int] | tuple[int, ...],
    settings: _FitSettings,
    start_latent: list[GroupLatentDistribution] | None,
) -> MultigroupFitResult:
    """Fit a copy of ``start_model`` with the given items freed."""
    free = sorted(free_items) or None
    spec = InvarianceSpec(
        "strict",
        free_discrimination=free if "discrimination" in settings.families else None,
        free_intercepts=free if "intercepts" in settings.families else None,
    )
    estimator = MultigroupEMEstimator(
        n_quadpts=settings.n_quadpts,
        max_iter=settings.max_iter,
        tol=settings.tol,
    )
    return estimator.fit(
        copy.deepcopy(start_model),
        responses,
        invariance=spec,
        reference_group=settings.reference_group,
        initial_latent=start_latent,
    )


def _refit(task: _RefitTask) -> _FitSummary | None:
    """Warm-start one alternative model; failures yield ``None``."""
    try:
        result = _fit(
            task.start_model,
            task.responses,
            task.free_items,
            task.settings,
            task.start_latent,
        )
    except _FIT_EXCEPTIONS:
        return None
    return _summarize(result, task.item)


def _summarize(result: MultigroupFitResult, item: int | None) -> _FitSummary:
    return _FitSummary(
        log_likelihood=float(result.log_likelihood),
        n_parameters=int(result.n_parameters),
        aic=float(result.aic),
        bic=float(result.bic),
        converged=bool(result.converged),
        item_parameters=() if item is None else _item_parameters(result.model, item),
    )


def _item_parameters(model: MultigroupModel, item: int) -> tuple[dict[str, Any], ...]:
    return tuple(group.get_item_parameters(item) for group in model.group_models)


def _likelihood_ratio_row(
    item: int,
    baseline: _FitSummary,
    alternative: _FitSummary | None,
    *,
    baseline_parameters: tuple[dict[str, Any], ...] | None,
    round_number: int,
) -> _DIFTestRow:
    """Orient a baseline/alternative pair as constrained versus free.

    ``baseline_parameters`` holds the studied item's group parameters when
    the item is free in the baseline (add schemes) and is None otherwise.
    """
    if alternative is None:
        return _DIFTestRow(
            item=item,
            chi2=np.nan,
            df=np.nan,
            p_value=np.nan,
            p_value_adjusted=np.nan,
            delta_aic=np.nan,
            delta_bic=np.nan,
            converged=False,
            round=round_number,
            group_parameters=None,
        )
    if baseline_parameters is not None:
        free, constrained = baseline, alternative
        group_parameters = baseline_parameters
    else:
        free, constrained = alternative, baseline
        group_parameters = alternative.item_parameters
    # EM stops slightly short of each maximum; a small negative gain is noise.
    chi2 = max(2.0 * (free.log_likelihood - constrained.log_likelihood), 0.0)
    df = free.n_parameters - constrained.n_parameters
    p_value = float(stats.chi2.sf(chi2, df)) if df > 0 else np.nan
    return _DIFTestRow(
        item=item,
        chi2=float(chi2),
        df=float(df),
        p_value=p_value,
        p_value_adjusted=np.nan,
        delta_aic=constrained.aic - free.aic,
        delta_bic=constrained.bic - free.bic,
        converged=baseline.converged and alternative.converged,
        round=round_number,
        group_parameters=group_parameters,
    )


def _validate_parameter_families(
    parameters: Sequence[str] | str,
) -> tuple[str, ...]:
    values = (parameters,) if isinstance(parameters, str) else tuple(parameters)
    if (
        not values
        or len(set(values)) != len(values)
        or any(value not in _PARAMETER_FAMILIES for value in values)
    ):
        raise ValueError(
            "parameters must be a nonempty subset of "
            f"{', '.join(repr(name) for name in _PARAMETER_FAMILIES)}"
        )
    return values


def resolve_items(
    values: Sequence[int | str] | None,
    item_names: list[str],
    name: str,
) -> list[int] | None:
    """Resolve item indices or names into unique in-range indices.

    ``None`` passes through. Shared with :func:`mirt.diagnostics.compute_dif`.
    """
    if values is None:
        return None
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        values = list(np.atleast_1d(np.asarray(values, dtype=object)))
    resolved: list[int] = []
    for value in values:
        if isinstance(value, str):
            if value not in item_names:
                raise ValueError(f"{name} contains unknown item name {value!r}")
            resolved.append(item_names.index(value))
        elif isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must contain item indices or names")
        elif not 0 <= int(value) < len(item_names):
            raise ValueError(
                f"{name} index {int(value)} out of range [0, {len(item_names)})"
            )
        else:
            resolved.append(int(value))
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} must not contain duplicate items")
    return resolved


__all__ = ["multigroup_dif", "select_dif_anchors"]
