"""Differential Item Functioning (DIF) analysis."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.integrate import trapezoid

from mirt.constants import PROB_EPSILON
from mirt.diagnostics._utils import (
    fit_linked_group_models,
    resolve_anchor_items,
    split_groups,
)
from mirt.diagnostics.multiple_testing import (
    PValueAdjustment,
    adjust_p_values,
    validate_p_value_adjustment,
)
from mirt.utils._parallel import resolve_n_jobs

if TYPE_CHECKING:
    from mirt.diagnostics._utils import LinkedGroupModels
    from mirt.multigroup.dif import DIFScheme


_DIF_METHODS = frozenset({"likelihood_ratio", "wald", "lord", "raju"})
_DIF_MODELS = frozenset({"1PL", "2PL", "3PL", "GRM", "GPCM"})
_SLOPE_PARAMETERS = frozenset({"discrimination", "slopes"})
_LOCATION_PARAMETERS = ("difficulty", "thresholds", "steps")
_POLYTOMOUS_DIF_MODELS = frozenset({"GRM", "GPCM"})
_ETS_ALPHA = 0.05
# ETS cut-offs (Zwick, 2012): |MH D-DIF| for binary items, |SMD| / SD for
# polytomous items. MH D-DIF is -2.35 times the MH log odds ratio.
_ETS_CUTS = {"delta": (1.0, 1.5), "smd": (0.17, 0.25)}
_MH_DELTA_SCALE = 2.35
# Reference-scale abilities for the model-implied ETS effects.
_ETS_THETA = np.linspace(-6.0, 6.0, 121)
_GRDIF_MODELS = frozenset({"1PL", "2PL", "3PL", "GRM", "GPCM"})
_GRDIF_SCORING_METHODS = frozenset({"EAP", "MAP", "ML", "WLE"})
_GRDIF_PURIFICATION_METHODS = frozenset({"grdif_rs", "grdif_r", "grdif_s"})
_GRDIF_SCALING_METHODS = frozenset({"mean", "mad", "iqr"})
_GRDIF_EFFECT_TYPES = frozenset({"delta_mrr", "delta_msr", "max_diff"})


def compute_dif(
    data: NDArray[np.int_],
    groups: NDArray,
    model: Literal["1PL", "2PL", "3PL", "GRM", "GPCM"] = "2PL",
    method: Literal["likelihood_ratio", "wald", "lord", "raju"] = "likelihood_ratio",
    n_categories: int | None = None,
    n_quadpts: int | None = None,
    max_iter: int = 500,
    tol: float = 1e-4,
    focal_group: str | int | None = None,
    p_adjust: PValueAdjustment = "none",
    *,
    anchors: Sequence[int | str] | None = None,
    scheme: DIFScheme = "drop",
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Compute Differential Item Functioning statistics on a common scale.

    DIF analysis tests whether items function differently across groups
    after controlling for ability. Every method compares the groups on one
    latent scale, so a difference in group ability (impact) is not reported
    as DIF.

    The default likelihood-ratio method fits one baseline multiple-group
    model plus one refit per tested item. For 30 binary 2PL items and 1,000
    persons per group this takes one to two seconds; 3PL and polytomous
    items are several times slower per item. ``n_jobs=-1`` runs the refits
    in parallel, and ``method="wald"`` or
    :func:`mirt.diagnostics.compute_grdif` are fast screens without refits.

    Args:
        data: Response matrix (n_persons x n_items).
        groups: Group membership array (n_persons,). Must have exactly 2 groups.
        model: IRT model type.
        method: DIF detection method:
            - 'likelihood_ratio': Nested multiple-group likelihood-ratio test
              (see :func:`mirt.multigroup.multigroup_dif`). Each studied item
              is compared constrained versus free across groups while the
              focal latent mean and variance are estimated. It needs one
              baseline fit plus one refit per tested item.
            - 'wald': Wald test of item-parameter differences after linking
              separately calibrated groups by Stocking-Lord over the anchors.
              Focal estimates and standard errors are rescaled by the linking
              constants. The statistic uses each parameter's standard error
              from the group fits and ignores parameter covariances and
              linking error, so it can be liberal, particularly in small
              samples and for polytomous items; prefer 'likelihood_ratio'
              for inference. Separate 3PL calibrations estimate guessing
              poorly, so 'wald' and 'raju' are unreliable for 3PL, whereas
              'likelihood_ratio' holds guessing equal across groups.
            - 'lord': Lord's chi-square; an alias of 'wald'.
            - 'raju': Raju's signed and unsigned areas between the linked item
              response curves. Areas are descriptive: ``p_value`` is ``NaN``
              and the ETS class uses the signed area alone.
        n_categories: Number of categories for polytomous models.
        n_quadpts: Quadrature points per dimension for EM. ``None`` uses the
            default grid of :func:`mirt.fit_mirt` and
            :func:`mirt.multigroup.fit_multigroup` (61 rectangular points).
        max_iter: Maximum EM iterations.
        tol: Convergence tolerance.
        focal_group: Which group to use as focal (default: second unique group).
        p_adjust: Multiple-testing adjustment across tested items. Supported
            values are 'none', 'bonferroni', 'holm', and 'fdr_bh'. Default
            'none', as in :func:`mirt.multigroup.multigroup_dif` and R's
            ``mirt::DIF``.
        anchors: Items assumed free of DIF, by 0-based index or by name
            (column names of a DataFrame, otherwise ``Item_1``, ``Item_2``,
            ... as in :func:`mirt.fit_mirt` and :func:`mirt.itemfit`). They
            are not tested and get ``NaN`` statistics. Likelihood-ratio tests
            constrain them in every model; the other methods link the groups
            over them. ``None`` treats every other item as an anchor in a
            likelihood-ratio test and links on all items otherwise, which
            assumes DIF that balances across items.
        scheme: Likelihood-ratio scheme: 'drop', 'add', 'drop_sequential' or
            'add_sequential' (see :func:`mirt.multigroup.multigroup_dif`).
            The 'add' schemes require ``anchors``.
        n_jobs: Worker processes for likelihood-ratio refits. ``-1`` uses all
            cores.

    Returns:
        Dictionary with one value per item for:
            - 'statistic': LR or Wald chi-square, or Raju's unsigned area
            - 'df': Degrees of freedom of the test (``NaN`` for Raju)
            - 'p_value': P-value for each item
            - 'p_value_adjusted': Multiplicity-adjusted P-value for each item
            - 'effect_size': Focal-minus-reference item location on the
              common scale (signed area for Raju); positive values mean the
              item is harder for the focal group
            - 'ets_effect': Model-implied ETS effect: for binary items the
              MH D-DIF, ``-2.35`` times the log Mantel-Haenszel odds ratio
              of the two groups' response curves matched on ability; for
              GRM and GPCM items the standardized mean difference (focal
              minus reference expected score, averaged over the focal
              ability distribution) divided by the total-group item-score
              SD. Negative values disadvantage the focal group. Under a 1PL
              model the MH D-DIF is ``-2.35 * effect_size``.
            - 'classification': ETS A/B/C class from ``ets_effect`` and the
              adjusted P-value: A unless the P-value is at most 0.05 and
              ``|ets_effect|`` reaches 1.0 (B) or 1.5 (C) for binary items,
              or exceeds 0.17 (B) or 0.25 (C) for polytomous items (Zwick,
              2012). Raju areas carry no P-value, so the effect alone decides.
            - 'adjustment': Adjustment method for each row
            - 'tested': Whether the item was tested (False for anchors)
            - 'converged': Whether the fits behind each row converged; for
              untested items, whether every fit converged
        and the metadata keys 'item_names', 'method', 'anchors',
        'ets_scale' (``"delta"`` for binary and ``"smd"`` for polytomous
        models) and 'linking_constants' (``(A, B)`` placing the focal group
        on the reference scale, or None for likelihood-ratio tests).

    Raises:
        ValueError: If the method, model, scheme or anchors are invalid.

    References:
        Holland, P. W., & Thayer, D. T. (1988). Differential item
        performance and the Mantel-Haenszel procedure. In H. Wainer &
        H. I. Braun (Eds.), Test validity (pp. 129-145). Erlbaum.

        Zwick, R. (1990). When do item response function and
        Mantel-Haenszel definitions of differential item functioning
        coincide? Journal of Educational Statistics, 15(3), 185-197.

        Zwick, R., Thayer, D. T., & Mazzeo, J. (1997). Descriptive and
        inferential procedures for assessing differential item functioning
        in polytomous items. Applied Measurement in Education, 10(4),
        321-344.

        Zwick, R. (2012). A review of ETS differential item functioning
        assessment procedures (ETS Research Report RR-12-08). ETS.
    """
    from mirt.multigroup.dif import DIF_SCHEMES, default_item_names, resolve_items

    if method not in _DIF_METHODS:
        raise ValueError(f"Unknown DIF method: {method}")
    if model not in _DIF_MODELS:
        raise ValueError(f"model must be one of: {', '.join(sorted(_DIF_MODELS))}")
    if scheme not in DIF_SCHEMES:
        raise ValueError(f"scheme must be one of: {', '.join(sorted(DIF_SCHEMES))}")
    p_adjust = validate_p_value_adjustment(p_adjust, name="p_adjust")
    n_jobs = resolve_n_jobs(n_jobs)

    item_names = default_item_names(data)
    data = np.asarray(data)
    groups = np.asarray(groups)
    n_items = data.shape[1]
    if item_names is None:
        item_names = [f"Item_{item + 1}" for item in range(n_items)]
    likelihood_ratio = method == "likelihood_ratio"
    anchor_items = resolve_anchor_items(
        resolve_items(anchors, item_names, "anchors"),
        n_items,
        name="anchors",
        minimum=1 if likelihood_ratio else 2,
    )
    tested = np.ones(n_items, dtype=np.bool_)
    if anchor_items is not None:
        tested[anchor_items] = False

    ref_data, focal_data, _, _, ref_group, _ = split_groups(data, groups, focal_group)
    group_sizes = (ref_data.shape[0], focal_data.shape[0])
    ets_scale: Literal["delta", "smd"] = (
        "smd" if model in _POLYTOMOUS_DIF_MODELS else "delta"
    )
    item_sd = _item_score_sd(data) if ets_scale == "smd" else None
    fit_options = {
        "n_categories": n_categories,
        "n_quadpts": n_quadpts,
        "max_iter": max_iter,
        "tol": tol,
    }

    result: dict[str, Any]
    if likelihood_ratio:
        result = _dif_likelihood_ratio(
            data,
            groups,
            ref_group,
            model,
            anchors=anchor_items,
            scheme=scheme,
            p_adjust=p_adjust,
            n_jobs=n_jobs,
            group_sizes=group_sizes,
            item_sd=item_sd,
            **fit_options,
        )
        result["linking_constants"] = None
    else:
        linked = fit_linked_group_models(
            ref_data,
            focal_data,
            model=model,
            anchor_items=anchor_items,
            compute_standard_errors=method != "raju",
            **fit_options,
        )
        result = (
            _dif_raju(linked, tested) if method == "raju" else _dif_wald(linked, tested)
        )
        result["p_value_adjusted"] = adjust_p_values(result["p_value"], p_adjust)
        result["converged"] = np.full(
            n_items, bool(linked.reference.converged and linked.focal.converged)
        )
        result["linking_constants"] = (linked.A, linked.B)
        # The focal abilities, N(0, 1) in their own calibration, follow
        # N(B, A^2) on the reference scale.
        ets_effect = _ets_effects(
            _expected_response_matrix(
                linked.reference.model, _ETS_THETA[:, None], n_items
            ),
            _expected_response_matrix(
                linked.focal_on_reference, _ETS_THETA[:, None], n_items
            ),
            latent=((0.0, 1.0), (linked.B, linked.A**2)),
            group_sizes=group_sizes,
            item_sd=item_sd,
        )
        result["ets_effect"] = np.where(tested, ets_effect, np.nan)

    result["classification"] = _ets_classify(
        result["ets_effect"],
        None if method == "raju" else result["p_value_adjusted"],
        ets_scale,
    )
    result["adjustment"] = np.full(n_items, p_adjust)
    result["tested"] = tested
    result["item_names"] = list(item_names)
    result["method"] = method
    result["anchors"] = anchor_items
    result["ets_scale"] = ets_scale
    return result


def _dif_likelihood_ratio(
    data: NDArray[np.int_],
    groups: NDArray[Any],
    reference_group: Any,
    model: str,
    *,
    anchors: list[int] | None,
    scheme: DIFScheme,
    p_adjust: PValueAdjustment,
    n_jobs: int,
    n_categories: int | None,
    n_quadpts: int | None,
    max_iter: int,
    tol: float,
    group_sizes: tuple[int, int],
    item_sd: NDArray[np.float64] | None,
) -> dict[str, Any]:
    """Nested multiple-group likelihood-ratio tests for two groups."""
    from mirt.multigroup.dif import run_multigroup_dif

    labels = np.unique(groups)
    reference_index = int(np.flatnonzero(labels == reference_group)[0])
    table = run_multigroup_dif(
        data,
        groups,
        model,
        anchors=anchors,
        scheme=scheme,
        p_adjust=p_adjust,
        alpha=_ETS_ALPHA,
        n_categories=n_categories,
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        # A label string is unambiguous when labels are integers too.
        reference_group=str(labels[reference_index]),
        n_jobs=n_jobs,
    )

    n_items = data.shape[1]
    statistic = np.full(n_items, np.nan)
    df = np.full(n_items, np.nan)
    p_value = np.full(n_items, np.nan)
    p_value_adjusted = np.full(n_items, np.nan)
    effect_size = np.full(n_items, np.nan)
    ets_effect = np.full(n_items, np.nan)
    focal_index = 1 - reference_index
    # Untested anchors report whether every fit of the analysis converged.
    converged = np.full(n_items, all(row.converged for row in table.rows))
    for row in table.rows:
        statistic[row.item] = row.chi2
        df[row.item] = row.df
        p_value[row.item] = row.p_value
        p_value_adjusted[row.item] = row.p_value_adjusted
        converged[row.item] = row.converged
        if row.group_parameters is not None:
            n_active = (
                None if table.n_categories is None else table.n_categories[row.item] - 1
            )
            effect_size[row.item] = _item_location(
                row.group_parameters[focal_index], n_active
            ) - _item_location(row.group_parameters[reference_index], n_active)
            latent = row.group_latent or ((0.0, 1.0),) * len(labels)
            scores = [
                _parameter_scores(model, row.group_parameters[index], n_active)[:, None]
                for index in (reference_index, focal_index)
            ]
            ets_effect[row.item] = _ets_effects(
                *scores,
                latent=(latent[reference_index], latent[focal_index]),
                group_sizes=group_sizes,
                item_sd=None if item_sd is None else item_sd[[row.item]],
            )[0]

    return {
        "statistic": statistic,
        "df": df,
        "p_value": p_value,
        "p_value_adjusted": p_value_adjusted,
        "effect_size": effect_size,
        "ets_effect": ets_effect,
        "converged": converged,
    }


def _dif_wald(
    linked: LinkedGroupModels,
    tested: NDArray[np.bool_],
) -> dict[str, NDArray[np.float64]]:
    """Wald test of linked item-parameter differences (Lord's chi-square)."""
    reference = linked.reference.model
    focal = linked.focal_on_reference
    n_items = int(reference.n_items)
    reference_errors = linked.reference.standard_errors
    focal_errors = _linked_standard_errors(linked.focal.standard_errors, linked.A)
    n_active = _active_locations(reference)

    statistic = np.full(n_items, np.nan)
    df = np.full(n_items, np.nan)
    effect_size = np.full(n_items, np.nan)
    for item in np.flatnonzero(tested):
        reference_parameters = reference.get_item_parameters(item)
        focal_parameters = focal.get_item_parameters(item)
        wald = 0.0
        n_compared = 0
        for name, reference_value in reference_parameters.items():
            if name not in reference_errors or name not in focal_errors:
                continue
            difference = np.ravel(reference_value) - np.ravel(focal_parameters[name])
            variance = (
                _item_row(reference_errors[name], item, n_items) ** 2
                + _item_row(focal_errors[name], item, n_items) ** 2
            )
            valid = (
                np.isfinite(difference)
                & np.isfinite(variance)
                & (variance > PROB_EPSILON)
            )
            wald += float(np.sum(difference[valid] ** 2 / variance[valid]))
            n_compared += int(np.count_nonzero(valid))
        if n_compared:
            statistic[item] = wald
            df[item] = n_compared
        effect_size[item] = _item_location(
            focal_parameters, n_active[item]
        ) - _item_location(reference_parameters, n_active[item])

    p_value = np.full(n_items, np.nan)
    has_test = np.isfinite(statistic)
    p_value[has_test] = stats.chi2.sf(statistic[has_test], df[has_test])
    return {
        "statistic": statistic,
        "df": df,
        "p_value": p_value,
        "effect_size": effect_size,
    }


def _dif_raju(
    linked: LinkedGroupModels,
    tested: NDArray[np.bool_],
) -> dict[str, NDArray[np.float64]]:
    """Raju's signed and unsigned areas between linked response curves.

    Polytomous expected scores are divided by the item's maximum score.
    Areas are integrated over theta in [-4, 4] on the reference scale.
    """
    theta = np.linspace(-4.0, 4.0, 100)
    reference = linked.reference.model
    n_items = int(reference.n_items)
    scale = np.ones(n_items)
    if reference.is_polytomous:
        scale = np.asarray(reference.n_categories, dtype=np.float64) - 1.0
    reference_curves = (
        _expected_response_matrix(reference, theta[:, None], n_items) / scale
    )
    focal_curves = (
        _expected_response_matrix(linked.focal_on_reference, theta[:, None], n_items)
        / scale
    )
    difference = reference_curves - focal_curves

    statistic = np.full(n_items, np.nan)
    effect_size = np.full(n_items, np.nan)
    statistic[tested] = trapezoid(np.abs(difference), theta, axis=0)[tested]
    effect_size[tested] = trapezoid(difference, theta, axis=0)[tested]
    return {
        "statistic": statistic,
        "df": np.full(n_items, np.nan),
        "p_value": np.full(n_items, np.nan),
        "effect_size": effect_size,
    }


def _linked_standard_errors(
    standard_errors: dict[str, NDArray[np.float64]],
    A: float,
) -> dict[str, NDArray[np.float64]]:
    """Rescale focal standard errors like the linked parameters.

    Slopes become ``a / A`` and locations ``A * b + B``; asymptotes are
    unchanged.
    """
    scaled = {}
    for name, values in standard_errors.items():
        errors = np.asarray(values, dtype=np.float64)
        if name in _SLOPE_PARAMETERS:
            errors = errors / A
        elif name in _LOCATION_PARAMETERS:
            errors = errors * A
        scaled[name] = errors
    return scaled


def _item_row(
    values: NDArray[np.float64], item: int, n_items: int
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim and array.shape[0] == n_items:
        return np.ravel(array[item])
    return np.ravel(array)


def _active_locations(model: Any) -> list[int | None]:
    """Number of active category locations per item (None if dichotomous)."""
    if not model.is_polytomous:
        return [None] * int(model.n_items)
    return [int(count) - 1 for count in model.n_categories]


def _item_location(parameters: dict[str, Any], n_active: int | None) -> float:
    """Item location: difficulty, or the mean active threshold or step."""
    for name in _LOCATION_PARAMETERS:
        if name in parameters:
            values = np.ravel(np.asarray(parameters[name], dtype=np.float64))
            if n_active is not None:
                values = values[:n_active]
            values = values[np.isfinite(values)]
            return float(np.mean(values)) if values.size else np.nan
    return np.nan


def _ets_effects(
    reference_scores: NDArray[np.float64],
    focal_scores: NDArray[np.float64],
    *,
    latent: tuple[tuple[float, float], tuple[float, float]],
    group_sizes: tuple[int, int],
    item_sd: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Model-implied ETS effect of each item, matched on the latent trait.

    ``reference_scores`` and ``focal_scores`` are expected item scores at
    ``_ETS_THETA`` on the reference scale, one column per item, and
    ``latent`` holds each group's ability mean and variance on that scale.

    Binary items get the MH D-DIF ``-2.35 log(alpha)`` of the ability-matched
    Mantel-Haenszel common odds ratio
    ``alpha = int w P_R Q_F / int w P_F Q_R``, with the MH weights
    ``w = n_R f_R n_F f_F / (n_R f_R + n_F f_F)`` (Holland and Thayer, 1988;
    Zwick, 1990). Polytomous items get the standardized mean difference
    ``int f_F (E_F - E_R)`` divided by ``item_sd``, the total-group
    item-score standard deviation (Zwick, Thayer and Mazzeo, 1997). Either
    is negative when the item disadvantages the focal group.
    """
    n_items = reference_scores.shape[1]
    (mean_r, var_r), (mean_f, var_f) = latent
    if not (
        np.isfinite([mean_r, var_r, mean_f, var_f]).all() and min(var_r, var_f) > 0
    ):
        return np.full(n_items, np.nan)
    density_r = stats.norm.pdf(_ETS_THETA, mean_r, np.sqrt(var_r))
    density_f = stats.norm.pdf(_ETS_THETA, mean_f, np.sqrt(var_f))
    if item_sd is None:
        n_r, n_f = group_sizes
        pooled = n_r * density_r + n_f * density_f
        weights = np.divide(
            n_r * density_r * n_f * density_f,
            pooled,
            out=np.zeros_like(pooled),
            where=pooled > 0.0,
        )
        p_r = np.clip(reference_scores, PROB_EPSILON, 1.0 - PROB_EPSILON)
        p_f = np.clip(focal_scores, PROB_EPSILON, 1.0 - PROB_EPSILON)
        odds_ratio = (weights @ (p_r * (1.0 - p_f))) / (weights @ (p_f * (1.0 - p_r)))
        return -_MH_DELTA_SCALE * np.log(odds_ratio)
    focal_weights = density_f / np.sum(density_f)
    difference = focal_weights @ (focal_scores - reference_scores)
    sd = np.asarray(item_sd, dtype=np.float64)
    return np.divide(difference, sd, out=np.full(n_items, np.nan), where=sd > 0.0)


def _item_score_sd(data: NDArray[np.int_]) -> NDArray[np.float64]:
    """Total-group standard deviation of each item's valid scores."""
    values = np.where(data >= 0, data, np.nan).astype(np.float64)
    counts = np.count_nonzero(data >= 0, axis=0)
    sd = np.full(data.shape[1], np.nan)
    enough = counts >= 2
    sd[enough] = np.nanstd(values[:, enough], axis=0, ddof=1)
    return sd


def _parameter_scores(
    model: str, parameters: dict[str, Any], n_active: int | None
) -> NDArray[np.float64]:
    """Expected scores of one item at ``_ETS_THETA`` from its parameters."""
    from mirt.models._factory import build_item_model

    item = build_item_model(
        model, 1, n_categories=None if n_active is None else [n_active + 1]
    )
    values = {}
    for name, value in parameters.items():
        if model == "1PL" and name in _SLOPE_PARAMETERS:
            continue  # fixed at one
        array = np.asarray(value, dtype=np.float64)
        if n_active is not None and name in _LOCATION_PARAMETERS:
            array = np.ravel(array)[:n_active]
        values[name] = array.reshape((1, *array.shape))
    item.set_parameters(**values)
    return _expected_response_matrix(item, _ETS_THETA[:, None], 1)[:, 0]


def _ets_classify(
    ets_effects: NDArray[np.float64],
    p_values: NDArray[np.float64] | None,
    scale: Literal["delta", "smd"] = "delta",
) -> NDArray[np.str_]:
    """Classify DIF using ETS guidelines (A/B/C).

    ``scale="delta"`` takes MH D-DIF values: class A below 1, B below 1.5
    and C from 1.5 in absolute value. ``scale="smd"`` takes standardized
    mean differences over the item SD: A up to 0.17, B up to 0.25 and C
    above. Items are class A unless the p-value is also at most 0.05.
    Without p-values (descriptive methods) the effect alone decides, and a
    missing effect or p-value gives class A.
    """
    low, high = _ETS_CUTS[scale]
    magnitude = np.abs(np.asarray(ets_effects, dtype=np.float64))
    if scale == "delta":
        classes = np.where(magnitude < low, "A", np.where(magnitude < high, "B", "C"))
    else:
        classes = np.where(magnitude <= low, "A", np.where(magnitude <= high, "B", "C"))
    classification = classes.astype("U1")
    negligible = ~np.isfinite(magnitude)
    if p_values is not None:
        negligible |= ~(np.asarray(p_values, dtype=np.float64) <= _ETS_ALPHA)
    classification[negligible] = "A"
    return classification


def flag_dif_items(
    dif_results: dict[str, Any],
    alpha: float = 0.05,
    min_effect_size: float = 0.426,
    classification: str | None = None,
    p_adjust: PValueAdjustment = "none",
) -> NDArray[np.bool_]:
    """Flag items showing significant DIF.

    Args:
        dif_results: Output from compute_dif().
        alpha: Significance level for p-value.
        min_effect_size: Minimum absolute ``effect_size`` (the location
            difference in theta units, or Raju's signed area) to flag.
        classification: If specified, flag items with this ETS class or worse.
            'B' flags B and C items, 'C' flags only C items. Classes are
            recomputed from ``dif_results['ets_effect']`` with the adjusted
            p-values; results without that entry are classified from
            ``effect_size`` read as a 1PL log odds ratio.
        p_adjust: Multiple-testing adjustment applied before flagging. Supported
            values are 'none', 'bonferroni', 'holm', and 'fdr_bh'.

    Returns:
        Boolean array indicating flagged items. Results of the descriptive
        'raju' method carry no p-values, so they are flagged on effect size
        alone; untested items and failed tests are never flagged.
    """
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError("alpha must be finite and in (0, 1)")
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha must be finite and in (0, 1)") from exc
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    if isinstance(min_effect_size, (bool, np.bool_)):
        raise ValueError("min_effect_size must be finite and nonnegative")
    try:
        min_effect_size = float(min_effect_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_effect_size must be finite and nonnegative") from exc
    if not np.isfinite(min_effect_size) or min_effect_size < 0.0:
        raise ValueError("min_effect_size must be finite and nonnegative")
    if classification not in {None, "B", "C"}:
        raise ValueError("classification must be 'B', 'C', or None")

    p_adjust = validate_p_value_adjustment(p_adjust, name="p_adjust")
    effect_sizes = np.asarray(dif_results["effect_size"], dtype=np.float64)
    if "ets_effect" in dif_results:
        ets_effects = np.asarray(dif_results["ets_effect"], dtype=np.float64)
        ets_scale = dif_results.get("ets_scale", "delta")
    else:
        # Under the 1PL a location difference is the MH log odds ratio.
        ets_effects = -_MH_DELTA_SCALE * effect_sizes
        ets_scale = "delta"
    if dif_results.get("method") == "raju":
        significant = np.isfinite(effect_sizes)
        classes = _ets_classify(ets_effects, None, ets_scale)
    else:
        p_values = adjust_p_values(dif_results["p_value"], p_adjust)
        significant = p_values <= alpha
        classes = _ets_classify(ets_effects, p_values, ets_scale)

    flags = significant & (np.abs(effect_sizes) >= min_effect_size)

    if classification is not None:
        if classification == "B":
            flags &= (classes == "B") | (classes == "C")
        else:
            flags &= classes == "C"

    return flags


def _validate_grdif_inputs(
    *,
    data: object,
    groups: object,
    model: str,
    scoring_method: str,
    alpha: float,
    purify: object,
    purify_by: str,
    max_purify_iter: int,
    n_quadpts: int,
    max_iter: int,
    tol: float,
    scaling_method: str,
) -> tuple[NDArray[np.int_], NDArray[Any], NDArray[Any]]:
    if model not in _GRDIF_MODELS:
        valid = ", ".join(sorted(_GRDIF_MODELS))
        raise ValueError(f"model must be one of: {valid}")
    if scoring_method not in _GRDIF_SCORING_METHODS:
        valid = ", ".join(sorted(_GRDIF_SCORING_METHODS))
        raise ValueError(f"scoring_method must be one of: {valid}")
    if purify_by not in _GRDIF_PURIFICATION_METHODS:
        valid = ", ".join(sorted(_GRDIF_PURIFICATION_METHODS))
        raise ValueError(f"purify_by must be one of: {valid}")
    if scaling_method not in _GRDIF_SCALING_METHODS:
        valid = ", ".join(sorted(_GRDIF_SCALING_METHODS))
        raise ValueError(f"scaling_method must be one of: {valid}")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    if not isinstance(purify, (bool, np.bool_)):
        raise ValueError("purify must be a boolean")

    integer_controls = (
        ("max_purify_iter", max_purify_iter, 1),
        ("n_quadpts", n_quadpts, 2),
        ("max_iter", max_iter, 1),
    )
    for name, value, minimum in integer_controls:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer of at least {minimum}")
        if value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")

    values = np.asarray(data)
    labels = np.asarray(groups)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("data must be a nonempty two-dimensional response matrix")
    if values.dtype.kind not in "biuf" or not np.all(np.isfinite(values)):
        raise ValueError("data must contain finite numeric response codes")
    if np.any(values < -1) or np.any(values != np.floor(values)):
        raise ValueError("responses must be integer coded with -1 reserved for missing")
    if labels.ndim != 1:
        raise ValueError("groups must be one-dimensional")
    if labels.shape[0] != values.shape[0]:
        raise ValueError("groups length must match the number of response-matrix rows")
    if labels.dtype.kind in "fc" and not np.all(np.isfinite(labels)):
        raise ValueError("groups must not contain missing or non-finite labels")
    if labels.dtype.kind == "O" and any(
        label is None
        or (isinstance(label, (float, np.floating)) and not np.isfinite(label))
        for label in labels
    ):
        raise ValueError("groups must not contain missing labels")
    try:
        unique_groups = np.unique(labels)
    except TypeError as exc:
        raise ValueError("group labels must be mutually comparable") from exc
    if unique_groups.size < 2:
        raise ValueError(
            f"GRDIF requires at least 2 groups, found {unique_groups.size}"
        )
    return values.astype(np.int64, copy=False), labels, unique_groups


def _score_grdif_responses(
    model: Any,
    data: NDArray[np.int_],
    anchor_items: NDArray[np.bool_],
    *,
    scoring_method: str,
    n_quadpts: int,
) -> NDArray[np.float64]:
    """Estimate abilities from the current purified item set."""
    from mirt.scoring import fscores

    anchors = np.asarray(anchor_items, dtype=np.bool_)
    if anchors.shape != (data.shape[1],):
        raise ValueError("anchor_items must match the number of response columns")
    if not np.any(anchors):
        raise ValueError("at least one anchor item is required for scoring")

    if np.all(anchors):
        scoring_data = data
    else:
        scoring_data = data.copy()
        scoring_data[:, ~anchors] = -1
    score_result = fscores(
        model,
        scoring_data,
        method=scoring_method,
        n_quadpts=n_quadpts,
    )
    theta = np.asarray(score_result.theta, dtype=np.float64)
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)
    if theta.ndim != 2 or theta.shape[0] != data.shape[0]:
        raise ValueError("ability estimates must have one row per response record")
    if not np.all(np.isfinite(theta)):
        raise ValueError("ability estimates must contain only finite values")
    return theta


def _expected_response_matrix(
    model: Any,
    theta: NDArray[np.float64],
    n_items: int,
) -> NDArray[np.float64]:
    """Evaluate every expected item score in one model call."""
    return _expected_response_moments(model, theta, n_items)[0]


def _expected_response_moments(
    model: Any,
    theta: NDArray[np.float64],
    n_items: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Expected item scores and their model variances in one model call.

    The variance is ``P(1 - P)`` for binary items and
    ``sum_k (k - E)^2 P_k`` for polytomous items, the expected squared
    residual of each response given its person's ability.
    """
    probabilities = np.asarray(model.probability(theta), dtype=np.float64)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("model probabilities must contain only finite values")
    if np.any(probabilities < -PROB_EPSILON) or np.any(
        probabilities > 1.0 + PROB_EPSILON
    ):
        raise ValueError("model probabilities must lie in [0, 1]")

    expected_shape = (theta.shape[0], n_items)
    if probabilities.ndim == 2:
        if probabilities.shape != expected_shape:
            raise ValueError(
                f"dichotomous probabilities must have shape {expected_shape}"
            )
        expected = probabilities
        variance = probabilities * (1.0 - probabilities)
    elif probabilities.ndim == 3:
        if probabilities.shape[:2] != expected_shape:
            raise ValueError(
                "polytomous probabilities must have shape "
                f"({theta.shape[0]}, {n_items}, n_categories)"
            )
        category_scores = np.arange(probabilities.shape[2], dtype=np.float64)
        expected = probabilities @ category_scores
        variance = probabilities @ category_scores**2 - expected**2
    else:
        raise ValueError("model probabilities must be two- or three-dimensional")
    if not np.all(np.isfinite(expected)):
        raise ValueError("expected item scores must contain only finite values")
    return expected, np.maximum(variance, 0.0)


def _column_scale(
    values: NDArray[np.float64],
    valid: NDArray[np.bool_],
    counts: NDArray[np.int_],
    means: NDArray[np.float64],
    method: Literal["mean", "mad", "iqr"],
) -> NDArray[np.float64]:
    """Compute one variance-like scale per item column."""
    n_items = values.shape[1]
    scales = np.ones(n_items, dtype=np.float64)
    sufficient = counts >= 2
    if method == "mean":
        deviations = np.where(valid, values - means[None, :], 0.0)
        scales[sufficient] = np.sum(deviations**2, axis=0)[sufficient] / (
            counts[sufficient] - 1
        )
        return scales

    for item_index in np.flatnonzero(sufficient):
        scales[item_index] = _compute_robust_scale(
            values[valid[:, item_index], item_index], method
        )
    return scales


def _group_residual_moments(
    residuals: NDArray[np.float64],
    excess: NDArray[np.float64],
    valid: NDArray[np.bool_],
    group_masks: dict[Any, NDArray[np.bool_]],
    unique_groups: NDArray[Any],
    scaling_method: Literal["mean", "mad", "iqr"],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int_],
]:
    """Compute group-by-item residual moments with itemwise missingness.

    ``residuals`` are raw residuals and ``excess`` squared residuals minus
    their model variance; both must be zero where responses are missing.
    Returns the group means of both, the variances of those means and
    their covariance, and the valid-response counts. Groups with fewer
    than two valid responses to an item get zero means and unit variances.
    """
    n_groups, n_items = len(unique_groups), residuals.shape[1]
    mrr = np.zeros((n_groups, n_items), dtype=np.float64)
    msr = np.zeros((n_groups, n_items), dtype=np.float64)
    var_mrr = np.ones((n_groups, n_items), dtype=np.float64)
    var_msr = np.ones((n_groups, n_items), dtype=np.float64)
    covariance = np.zeros((n_groups, n_items), dtype=np.float64)
    counts = np.zeros((n_groups, n_items), dtype=np.int64)

    for group_index, group in enumerate(unique_groups):
        mask = group_masks[group]
        group_valid = valid[mask]
        group_residuals = residuals[mask]
        group_excess = excess[mask]
        group_counts = np.count_nonzero(group_valid, axis=0)
        counts[group_index] = group_counts
        sufficient = group_counts >= 2
        divisor = np.maximum(group_counts, 1)
        mrr[group_index] = np.where(
            sufficient, group_residuals.sum(axis=0) / divisor, 0.0
        )
        msr[group_index] = np.where(sufficient, group_excess.sum(axis=0) / divisor, 0.0)

        raw_scale = _column_scale(
            group_residuals,
            group_valid,
            group_counts,
            mrr[group_index],
            scaling_method,
        )
        squared_scale = _column_scale(
            group_excess,
            group_valid,
            group_counts,
            msr[group_index],
            scaling_method,
        )
        var_mrr[group_index, sufficient] = np.maximum(
            raw_scale[sufficient] / group_counts[sufficient], PROB_EPSILON
        )
        var_msr[group_index, sufficient] = np.maximum(
            squared_scale[sufficient] / group_counts[sufficient], PROB_EPSILON
        )
        # The sample correlation times the chosen scales; for "mean" scaling
        # this is the sample covariance of the two group means.
        raw_deviations = np.where(group_valid, group_residuals - mrr[group_index], 0.0)
        excess_deviations = np.where(group_valid, group_excess - msr[group_index], 0.0)
        cross = np.sum(raw_deviations * excess_deviations, axis=0)
        norm = np.sqrt(
            np.sum(raw_deviations**2, axis=0) * np.sum(excess_deviations**2, axis=0)
        )
        correlation = np.clip(
            np.divide(cross, norm, out=np.zeros(n_items), where=norm > 0.0),
            -1.0,
            1.0,
        )
        covariance[group_index] = np.where(
            sufficient,
            correlation * np.sqrt(var_mrr[group_index] * var_msr[group_index]),
            0.0,
        )

    return mrr, msr, var_mrr, var_msr, covariance, counts


def compute_grdif(
    data: NDArray[np.int_],
    groups: NDArray,
    model: Literal["1PL", "2PL", "3PL", "GRM", "GPCM"] = "2PL",
    scoring_method: Literal["EAP", "MAP", "ML", "WLE"] = "ML",
    alpha: float = 0.05,
    purify: bool = False,
    purify_by: Literal["grdif_rs", "grdif_r", "grdif_s"] = "grdif_rs",
    max_purify_iter: int = 10,
    n_categories: int | None = None,
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    scaling_method: Literal["mean", "mad", "iqr"] = "mean",
    p_adjust: PValueAdjustment = "none",
) -> dict[str, Any]:
    """Compute Generalized Residual DIF (GRDIF) statistics for multiple groups.

    GRDIF is a generalized version of the RDIF detection framework designed
    to assess DIF across multiple groups simultaneously. It computes three
    chi-square distributed test statistics based on IRT residuals.

    One model is calibrated on the pooled groups and every person is scored
    on it. GRDIF_R compares the group mean raw residuals (MRR). GRDIF_S
    compares the group mean squared residuals (MSR) after subtracting their
    model expectation, the mean of each response's variance given the
    person's ability (``P(1 - P)`` for binary items), so groups of
    different ability (impact) are not flagged for that reason alone.
    GRDIF_RS tests both with their joint within-group covariance. Each is a
    Wald test of equal group means with ``G - 1`` (GRDIF_RS: ``2(G - 1)``)
    degrees of freedom, as in Lim et al. (2024) and R's
    ``irtQ::grdif``. Group means and their variances are estimated from the
    residuals themselves, by ``scaling_method``; ``irtQ`` uses
    model-implied variances instead.

    This method has several advantages over traditional DIF approaches:
    - Works with any number of groups (G >= 2)
    - No separate calibration per group required
    - No matching variable or theta bins needed
    - Computationally efficient

    Ability estimates matter under impact. EAP and MAP shrink every
    estimate toward the pooled prior mean, and WLE does so slightly, which
    biases the residuals of groups above and below that mean in opposite
    directions, so GRDIF_R is liberal with them when group abilities
    differ, and purification compounds this. ML scoring, the default as in
    ``irtQ``, avoids the bias.

    Args:
        data: Response matrix (n_persons x n_items).
        groups: Group membership array (n_persons,). Can have 2+ groups.
        model: IRT model type for aggregate calibration.
        scoring_method: Method for computing ability estimates. Default
            'ML' (see above).
        alpha: Significance level for flagging DIF items.
        purify: Whether to iteratively remove flagged items from ability scoring
            and re-estimate abilities from the remaining anchors.
        purify_by: Which statistic to use for purification decisions.
        max_purify_iter: Maximum purification iterations.
        n_categories: Number of categories for polytomous models.
        n_quadpts: Number of quadrature points for EM.
        max_iter: Maximum EM iterations.
        tol: Convergence tolerance.
        scaling_method: Method for variance estimation:
            - 'mean': Standard sample variance (default)
            - 'mad': Median absolute deviation (robust to outliers)
            - 'iqr': Interquartile range (robust to outliers)
        p_adjust: Multiple-testing adjustment applied separately across items
            for each of the GRDIF_R, GRDIF_S, and GRDIF_RS test families.

    Returns:
        Dictionary with GRDIF results:
            - 'grdif_r': GRDIF_R statistics (uniform DIF)
            - 'grdif_s': GRDIF_S statistics (nonuniform DIF)
            - 'grdif_rs': GRDIF_RS statistics (mixed DIF)
            - 'p_value_r': P-values for GRDIF_R
            - 'p_value_s': P-values for GRDIF_S
            - 'p_value_rs': P-values for GRDIF_RS
            - 'p_value_r_adjusted': Adjusted P-values for GRDIF_R
            - 'p_value_s_adjusted': Adjusted P-values for GRDIF_S
            - 'p_value_rs_adjusted': Adjusted P-values for GRDIF_RS
            - 'flagged_r': Items flagged by GRDIF_R
            - 'flagged_s': Items flagged by GRDIF_S
            - 'flagged_rs': Items flagged by GRDIF_RS
            - 'n_groups': Number of groups
            - 'group_labels': Unique group labels
            - 'group_sizes': Sample size per group
            - 'anchor_items': Boolean mask of the final purified item set
            - 'purification_history': Iteration details if purify=True
            - 'purification_complete': Whether the anchor set converged
            - 'purification_stop_reason': Convergence or stopping condition
            - 'theta': Final ability estimates used by the reported statistics
            - 'mrr': Mean raw residual per group and item (n_groups x n_items)
            - 'msr': Mean squared residual per group and item
            - 'msr_expected': Model-implied mean squared residual per group
              and item; GRDIF_S compares ``msr - msr_expected``
            - 'group_item_counts': Valid responses per group and item

        Items with fewer than two groups that have at least two valid
        responses get ``NaN`` statistics and p-values; otherwise groups
        below that count are left out and the degrees of freedom shrink.

    References:
        Lim, H., et al. (2024). Detecting Differential Item Functioning among
        Multiple Groups Using IRT Residual DIF Framework. Journal of
        Educational Measurement.
    """
    from mirt import fit_mirt

    p_adjust = validate_p_value_adjustment(p_adjust, name="p_adjust")
    data, groups, unique_groups = _validate_grdif_inputs(
        data=data,
        groups=groups,
        model=model,
        scoring_method=scoring_method,
        alpha=alpha,
        purify=purify,
        purify_by=purify_by,
        max_purify_iter=max_purify_iter,
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        scaling_method=scaling_method,
    )
    _, n_items = data.shape
    n_groups = len(unique_groups)

    group_masks = {g: groups == g for g in unique_groups}
    group_sizes = {g: int(np.count_nonzero(mask)) for g, mask in group_masks.items()}

    fit_result = fit_mirt(
        data,
        model=model,
        n_categories=n_categories,
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
    )

    anchor_items = np.ones(n_items, dtype=np.bool_)
    theta = _score_grdif_responses(
        fit_result.model,
        data,
        anchor_items,
        scoring_method=scoring_method,
        n_quadpts=n_quadpts,
    )

    purification_history: list[dict[str, Any]] = []
    purification_complete: bool | None = None
    purification_stop_reason: str | None = None

    if purify:
        for iteration in range(max_purify_iter):
            grdif_r, grdif_s, grdif_rs, p_r, p_s, p_rs = _compute_grdif_statistics(
                data,
                theta,
                fit_result.model,
                group_masks,
                unique_groups,
                scaling_method,
            )

            p_values = {
                "grdif_rs": p_rs,
                "grdif_r": p_r,
                "grdif_s": p_s,
            }[purify_by]
            flagged = adjust_p_values(p_values, p_adjust) < alpha

            new_anchors = ~flagged
            n_anchors = int(np.count_nonzero(new_anchors))
            purification_history.append(
                {
                    "iteration": iteration + 1,
                    "n_flagged": int(np.count_nonzero(flagged)),
                    "flagged_items": np.flatnonzero(flagged).tolist(),
                    "n_anchors": n_anchors,
                }
            )

            if n_anchors < 2:
                purification_complete = False
                purification_stop_reason = "insufficient_anchors"
                break

            if np.array_equal(anchor_items, new_anchors):
                purification_complete = True
                purification_stop_reason = "converged"
                break

            anchor_items = new_anchors
            theta = _score_grdif_responses(
                fit_result.model,
                data,
                anchor_items,
                scoring_method=scoring_method,
                n_quadpts=n_quadpts,
            )
        else:
            purification_complete = False
            purification_stop_reason = "max_iterations"

    expected, variance = _expected_response_moments(fit_result.model, theta, n_items)
    grdif_r, grdif_s, grdif_rs, p_r, p_s, p_rs = _compute_grdif_statistics(
        data,
        theta,
        fit_result.model,
        group_masks,
        unique_groups,
        scaling_method,
        expected_responses=expected,
        response_variances=variance,
    )
    mrr, msr, msr_expected, group_item_counts = _group_residual_means(
        data, expected, variance, group_masks, unique_groups
    )
    p_r_adjusted, p_s_adjusted, p_rs_adjusted = _adjust_grdif_families(
        p_r,
        p_s,
        p_rs,
        p_adjust,
    )

    return {
        "grdif_r": grdif_r,
        "grdif_s": grdif_s,
        "grdif_rs": grdif_rs,
        "p_value_r": p_r,
        "p_value_s": p_s,
        "p_value_rs": p_rs,
        "p_value_r_adjusted": p_r_adjusted,
        "p_value_s_adjusted": p_s_adjusted,
        "p_value_rs_adjusted": p_rs_adjusted,
        "flagged_r": p_r_adjusted < alpha,
        "flagged_s": p_s_adjusted < alpha,
        "flagged_rs": p_rs_adjusted < alpha,
        "p_adjustment": p_adjust,
        "n_groups": n_groups,
        "group_labels": unique_groups.tolist(),
        "group_sizes": group_sizes,
        "anchor_items": anchor_items,
        "purification_history": purification_history if purify else None,
        "purification_complete": purification_complete,
        "purification_stop_reason": purification_stop_reason,
        "theta": theta.copy(),
        "mrr": mrr,
        "msr": msr,
        "msr_expected": msr_expected,
        "group_item_counts": group_item_counts,
    }


def _group_residual_means(
    data: NDArray[np.int_],
    expected: NDArray[np.float64],
    variance: NDArray[np.float64],
    group_masks: dict[Any, NDArray[np.bool_]],
    unique_groups: NDArray[Any],
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int_]
]:
    """Group-by-item mean raw and squared residuals with valid-response counts.

    Also returns each group's model-implied mean squared residual, the mean
    of the response variances. Means are zero where a group has fewer than
    two valid responses.
    """
    valid = data >= 0
    residuals = np.where(valid, data - expected, 0.0)
    columns = (residuals, residuals**2, np.where(valid, variance, 0.0))
    n_groups, n_items = len(unique_groups), data.shape[1]
    means = np.zeros((len(columns), n_groups, n_items), dtype=np.float64)
    counts = np.zeros((n_groups, n_items), dtype=np.int64)
    for index, group in enumerate(unique_groups):
        mask = group_masks[group]
        counts[index] = np.count_nonzero(valid[mask], axis=0)
        sufficient = counts[index] >= 2
        divisor = np.maximum(counts[index], 1)
        for column, values in enumerate(columns):
            means[column, index] = np.where(
                sufficient, values[mask].sum(axis=0) / divisor, 0.0
            )
    return means[0], means[1], means[2], counts


def _adjust_grdif_families(
    p_r: NDArray[np.float64],
    p_s: NDArray[np.float64],
    p_rs: NDArray[np.float64],
    p_adjust: PValueAdjustment,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Adjust the three GRDIF test families independently."""
    return (
        adjust_p_values(p_r, p_adjust),
        adjust_p_values(p_s, p_adjust),
        adjust_p_values(p_rs, p_adjust),
    )


def _compute_grdif_statistics(
    data: NDArray[np.int_],
    theta: NDArray[np.float64],
    model: Any,
    group_masks: dict[Any, NDArray[np.bool_]],
    unique_groups: NDArray,
    scaling_method: Literal["mean", "mad", "iqr"] = "mean",
    *,
    expected_responses: NDArray[np.float64] | None = None,
    response_variances: NDArray[np.float64] | None = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Compute GRDIF_R, GRDIF_S, GRDIF_RS statistics.

    The statistics are Wald tests of equal group means of the raw residuals
    (MRR) and of the squared residuals in excess of their model expectation
    (MSR minus the mean response variance), as in Lim et al. (2024). Each
    person's squared residual is centred at its variance given that
    person's ability, so groups of different ability are compared at a
    common null value of zero.

    GRDIF_R detects uniform DIF (differences in difficulty)
    GRDIF_S detects nonuniform DIF (differences in discrimination)
    GRDIF_RS detects mixed DIF (both types)

    The group means are independent across groups, so each statistic is
    the sum over groups of squared deviations from the precision-weighted
    pooled mean, which equals the contrast form of Lim et al. GRDIF_RS uses
    the joint covariance of the two means within each group. Only groups
    with at least two valid responses to an item count; with ``k`` such
    groups GRDIF_R and GRDIF_S have ``k - 1`` and GRDIF_RS ``2(k - 1)``
    degrees of freedom, and items with ``k < 2`` get ``NaN``.

    Expected responses and their variances are evaluated in one batched
    model call unless both are passed. Residual moments are then reduced by
    group while retaining itemwise missingness.
    """
    if scaling_method not in _GRDIF_SCALING_METHODS:
        valid = ", ".join(sorted(_GRDIF_SCALING_METHODS))
        raise ValueError(f"scaling_method must be one of: {valid}")
    values = np.asarray(data)
    theta_values = np.asarray(theta, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("data must be a two-dimensional response matrix")
    n_items = values.shape[1]
    if theta_values.ndim != 2 or theta_values.shape[0] != values.shape[0]:
        raise ValueError("theta must be two-dimensional with one row per person")

    if (expected_responses is None) != (response_variances is None):
        raise ValueError(
            "expected_responses and response_variances must be passed together"
        )
    if expected_responses is None or response_variances is None:
        expected, variance = _expected_response_moments(model, theta_values, n_items)
    else:
        expected = np.asarray(expected_responses, dtype=np.float64)
        variance = np.asarray(response_variances, dtype=np.float64)
        if expected.shape != values.shape or not np.all(np.isfinite(expected)):
            raise ValueError(
                "expected_responses must be finite and match the response matrix"
            )
        if variance.shape != values.shape or not np.all(
            np.isfinite(variance) & (variance >= 0.0)
        ):
            raise ValueError(
                "response_variances must be finite, nonnegative and match the "
                "response matrix"
            )

    valid_responses = values >= 0
    residuals = np.where(valid_responses, values - expected, 0.0)
    excess = np.where(valid_responses, residuals**2 - variance, 0.0)
    mrr, msr, var_mrr, var_msr, covariance, counts = _group_residual_moments(
        residuals,
        excess,
        valid_responses,
        group_masks,
        unique_groups,
        scaling_method,
    )
    qualifying = counts >= 2

    def univariate(
        means: NDArray[np.float64], variances: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        precision = np.where(qualifying, 1.0 / variances, 0.0)
        total = np.sum(precision, axis=0)
        pooled = np.sum(precision * means, axis=0) / np.where(total > 0, total, 1.0)
        return np.sum(precision * (means - pooled) ** 2, axis=0)

    grdif_r = univariate(mrr, var_mrr)
    grdif_s = univariate(msr, var_msr)

    # Inverse of each group's 2 x 2 covariance [[a, b], [b, d]].
    determinant = np.maximum(
        var_mrr * var_msr - covariance**2, PROB_EPSILON * var_mrr * var_msr
    )
    a = np.where(qualifying, var_msr / determinant, 0.0)
    b = np.where(qualifying, -covariance / determinant, 0.0)
    d = np.where(qualifying, var_mrr / determinant, 0.0)
    sum_a, sum_b, sum_d = (np.sum(term, axis=0) for term in (a, b, d))
    rhs_r = np.sum(a * mrr + b * msr, axis=0)
    rhs_s = np.sum(b * mrr + d * msr, axis=0)
    pooled_determinant = sum_a * sum_d - sum_b**2
    safe_determinant = np.where(pooled_determinant > 0.0, pooled_determinant, 1.0)
    pooled_r = (sum_d * rhs_r - sum_b * rhs_s) / safe_determinant
    pooled_s = (sum_a * rhs_s - sum_b * rhs_r) / safe_determinant
    deviation_r = mrr - pooled_r
    deviation_s = msr - pooled_s
    grdif_rs = np.sum(
        a * deviation_r**2 + 2.0 * b * deviation_r * deviation_s + d * deviation_s**2,
        axis=0,
    )

    df = np.count_nonzero(qualifying, axis=0) - 1.0
    testable = df >= 1
    statistics = []
    for statistic, scale in ((grdif_r, 1.0), (grdif_s, 1.0), (grdif_rs, 2.0)):
        statistic = np.where(testable, statistic, np.nan)
        p_value = np.full(n_items, np.nan)
        p_value[testable] = stats.chi2.sf(statistic[testable], df=scale * df[testable])
        statistics.append((statistic, p_value))
    (grdif_r, p_r), (grdif_s, p_s), (grdif_rs, p_rs) = statistics
    return grdif_r, grdif_s, grdif_rs, p_r, p_s, p_rs


def _compute_robust_scale(
    data: NDArray[np.float64],
    method: Literal["mean", "mad", "iqr"] = "mean",
) -> float:
    """Compute scale estimate (variance-like) using specified method.

    Args:
        data: Array of values to compute scale for.
        method: Scaling method:
            - 'mean': Standard sample variance
            - 'mad': Median absolute deviation squared (robust)
            - 'iqr': Interquartile range squared (robust)

    Returns:
        Scale estimate (variance-like quantity).
    """
    if method == "mean":
        return float(np.var(data, ddof=1)) if len(data) > 1 else 1.0
    elif method == "mad":
        median = np.median(data)
        mad = np.median(np.abs(data - median)) * 1.4826
        return max(float(mad**2), PROB_EPSILON)
    elif method == "iqr":
        q75, q25 = np.percentile(data, [75, 25])
        iqr_scale = (q75 - q25) / 1.349
        return max(float(iqr_scale**2), PROB_EPSILON)
    raise ValueError(f"Unknown scaling method: {method}")


def compute_pairwise_rdif(
    data: NDArray[np.int_],
    groups: NDArray,
    model: Literal["1PL", "2PL", "3PL", "GRM", "GPCM"] = "2PL",
    scoring_method: Literal["EAP", "MAP", "ML", "WLE"] = "ML",
    alpha: float = 0.05,
    n_categories: int | None = None,
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    p_adjust: PValueAdjustment = "none",
) -> dict[str, Any]:
    """Compute pairwise RDIF statistics for post-hoc analysis.

    After finding significant GRDIF, this function performs pairwise
    comparisons between all group pairs to identify which specific
    groups differ on each item. Each pair is tested with the two-group
    GRDIF statistics of :func:`compute_grdif` (RDIF; Lim, Choe and Han,
    2022) from one pooled calibration, so squared residuals are compared
    net of their model expectation.

    Args:
        data: Response matrix (n_persons x n_items).
        groups: Group membership array.
        model: IRT model type.
        scoring_method: Method for computing ability estimates. Default
            'ML', as for :func:`compute_grdif`.
        alpha: Significance level.
        n_categories: Number of categories for polytomous models.
        n_quadpts: Number of quadrature points.
        max_iter: Maximum EM iterations.
        tol: Convergence tolerance.
        p_adjust: Multiple-testing adjustment. For each RDIF statistic, all
            group-pair and item combinations form one testing family.

    Returns:
        Dictionary with pairwise results:
            - 'pairs': List of group pairs compared
            - 'rdif_r': RDIF_R statistics per pair per item
            - 'rdif_s': RDIF_S statistics per pair per item
            - 'rdif_rs': RDIF_RS statistics per pair per item
            - 'p_values_r': P-values for RDIF_R
            - 'p_values_s': P-values for RDIF_S
            - 'p_values_rs': P-values for RDIF_RS
            - 'p_values_r_adjusted': Adjusted P-values for RDIF_R
            - 'p_values_s_adjusted': Adjusted P-values for RDIF_S
            - 'p_values_rs_adjusted': Adjusted P-values for RDIF_RS
    """
    from mirt import fit_mirt

    p_adjust = validate_p_value_adjustment(p_adjust, name="p_adjust")
    data, groups, unique_groups = _validate_grdif_inputs(
        data=data,
        groups=groups,
        model=model,
        scoring_method=scoring_method,
        alpha=alpha,
        purify=False,
        purify_by="grdif_rs",
        max_purify_iter=1,
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        scaling_method="mean",
    )
    n_items = data.shape[1]

    fit_result = fit_mirt(
        data,
        model=model,
        n_categories=n_categories,
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
    )

    theta = _score_grdif_responses(
        fit_result.model,
        data,
        np.ones(n_items, dtype=np.bool_),
        scoring_method=scoring_method,
        n_quadpts=n_quadpts,
    )
    expected_responses, response_variances = _expected_response_moments(
        fit_result.model, theta, n_items
    )

    pairs = list(combinations(unique_groups, 2))
    n_pairs = len(pairs)

    rdif_r = np.zeros((n_pairs, n_items))
    rdif_s = np.zeros((n_pairs, n_items))
    rdif_rs = np.zeros((n_pairs, n_items))

    for pair_idx, (g1, g2) in enumerate(pairs):
        mask1 = groups == g1
        mask2 = groups == g2

        pair_masks = {g1: mask1, g2: mask2}
        pair_groups = np.array([g1, g2])
        r, s, rs, _, _, _ = _compute_grdif_statistics(
            data,
            theta,
            fit_result.model,
            pair_masks,
            pair_groups,
            expected_responses=expected_responses,
            response_variances=response_variances,
        )

        rdif_r[pair_idx] = r
        rdif_s[pair_idx] = s
        rdif_rs[pair_idx] = rs

    p_r = stats.chi2.sf(rdif_r, df=1)
    p_s = stats.chi2.sf(rdif_s, df=1)
    p_rs = stats.chi2.sf(rdif_rs, df=2)
    p_r_adjusted, p_s_adjusted, p_rs_adjusted = _adjust_grdif_families(
        p_r,
        p_s,
        p_rs,
        p_adjust,
    )

    return {
        "pairs": pairs,
        "rdif_r": rdif_r,
        "rdif_s": rdif_s,
        "rdif_rs": rdif_rs,
        "p_values_r": p_r,
        "p_values_s": p_s,
        "p_values_rs": p_rs,
        "p_values_r_adjusted": p_r_adjusted,
        "p_values_s_adjusted": p_s_adjusted,
        "p_values_rs_adjusted": p_rs_adjusted,
        "flagged_r": p_r_adjusted < alpha,
        "flagged_s": p_s_adjusted < alpha,
        "flagged_rs": p_rs_adjusted < alpha,
        "p_adjustment": p_adjust,
    }


def grdif_effect_size(
    data: NDArray[np.int_],
    groups: NDArray,
    grdif_results: dict[str, Any],
    effect_type: Literal["delta_mrr", "delta_msr", "max_diff"] = "delta_mrr",
    *,
    model: Any = None,
) -> NDArray[np.float64]:
    """Compute effect sizes for GRDIF flagged items.

    Effect sizes are the spread (maximum minus minimum) across groups of the
    residual moments behind the GRDIF test, so they use the same calibration
    and final ability estimates as :func:`compute_grdif`. No model is
    refitted. Only groups with at least two valid responses to an item
    count; items with fewer than two such groups get zero.

    Args:
        data: Response matrix used for ``grdif_results``.
        groups: Group membership array used for ``grdif_results``.
        grdif_results: Output from compute_grdif().
        effect_type: Type of effect size:
            - 'delta_mrr': Maximum difference in mean raw residuals
            - 'delta_msr': Maximum difference in mean squared residuals
              net of their model expectation (``msr - msr_expected``), the
              moment GRDIF_S tests, so group impact alone gives no spread
            - 'max_diff': Maximum of both
        model: Fitted item model. Required only when ``grdif_results`` lacks
            the 'mrr', 'msr', 'msr_expected' and 'group_item_counts'
            entries; the moments are then recomputed from
            ``grdif_results['theta']``.

    Returns:
        Effect size array for each item.

    Raises:
        ValueError: If the inputs do not match the results, or if the moments
            are missing and no model is given.
    """
    if effect_type not in _GRDIF_EFFECT_TYPES:
        valid = ", ".join(sorted(_GRDIF_EFFECT_TYPES))
        raise ValueError(f"effect_type must be one of: {valid}")
    values = np.asarray(data)
    labels = np.asarray(groups)
    if values.ndim != 2 or labels.ndim != 1 or labels.shape[0] != values.shape[0]:
        raise ValueError(
            "data must be two-dimensional with one group label per response row"
        )
    n_items = values.shape[1]

    moment_keys = ("mrr", "msr", "msr_expected", "group_item_counts")
    if all(key in grdif_results for key in moment_keys):
        mrr = np.asarray(grdif_results["mrr"], dtype=np.float64)
        msr = np.asarray(grdif_results["msr"], dtype=np.float64)
        msr_expected = np.asarray(grdif_results["msr_expected"], dtype=np.float64)
        counts = np.asarray(grdif_results["group_item_counts"])
    elif model is not None:
        theta = np.asarray(grdif_results["theta"], dtype=np.float64)
        if theta.ndim == 1:
            theta = theta[:, None]
        if theta.shape[0] != values.shape[0]:
            raise ValueError("grdif_results['theta'] must have one row per person")
        unique_groups = np.asarray(grdif_results["group_labels"])
        expected, variance = _expected_response_moments(model, theta, n_items)
        mrr, msr, msr_expected, counts = _group_residual_means(
            values,
            expected,
            variance,
            {group: labels == group for group in unique_groups},
            unique_groups,
        )
    else:
        raise ValueError(
            "grdif_results lacks residual moments; pass the fitted model= used "
            "by compute_grdif"
        )
    if (
        mrr.ndim != 2
        or mrr.shape[1] != n_items
        or msr.shape != mrr.shape
        or msr_expected.shape != mrr.shape
    ):
        raise ValueError("grdif_results moments must have one column per item")
    excess_msr = msr - msr_expected
    if counts.shape != mrr.shape:
        raise ValueError("group_item_counts must match the residual moments")

    qualifying = counts >= 2
    enough_groups = np.count_nonzero(qualifying, axis=0) >= 2

    def spread(moments: NDArray[np.float64]) -> NDArray[np.float64]:
        upper = np.max(np.where(qualifying, moments, -np.inf), axis=0)
        lower = np.min(np.where(qualifying, moments, np.inf), axis=0)
        return np.where(enough_groups, upper - lower, 0.0)

    if effect_type == "delta_mrr":
        return spread(mrr)
    if effect_type == "delta_msr":
        return spread(excess_msr)
    return np.maximum(spread(mrr), spread(excess_msr))
