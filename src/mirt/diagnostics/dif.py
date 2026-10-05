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
    _validate_p_value_adjustment,
    adjust_p_values,
)
from mirt.utils.bootstrap import _validate_n_jobs

if TYPE_CHECKING:
    from mirt.diagnostics._utils import LinkedGroupModels
    from mirt.multigroup.dif import DIFScheme


_DIF_METHODS = frozenset({"likelihood_ratio", "wald", "lord", "raju"})
_DIF_MODELS = frozenset({"1PL", "2PL", "3PL", "GRM", "GPCM"})
_DIF_SCHEMES = frozenset({"drop", "add", "drop_sequential", "add_sequential"})
_SLOPE_PARAMETERS = frozenset({"discrimination", "slopes"})
_LOCATION_PARAMETERS = ("difficulty", "thresholds", "steps")
_ETS_ALPHA = 0.05
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
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    focal_group: str | int | None = None,
    p_adjust: PValueAdjustment = "none",
    *,
    anchors: Sequence[int] | None = None,
    scheme: DIFScheme = "drop",
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Compute Differential Item Functioning statistics on a common scale.

    DIF analysis tests whether items function differently across groups
    after controlling for ability. Every method compares the groups on one
    latent scale, so a difference in group ability (impact) is not reported
    as DIF.

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
        n_quadpts: Number of quadrature points for EM.
        max_iter: Maximum EM iterations.
        tol: Convergence tolerance.
        focal_group: Which group to use as focal (default: second unique group).
        p_adjust: Multiple-testing adjustment across tested items. Supported
            values are 'none', 'bonferroni', 'holm', and 'fdr_bh'. Default
            'none'.
        anchors: Items assumed free of DIF. They are not tested and get
            ``NaN`` statistics. Likelihood-ratio tests constrain them in every
            model; the other methods link the groups over them. ``None``
            treats every other item as an anchor in a likelihood-ratio test
            and links on all items otherwise, which assumes DIF that
            balances across items.
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
            - 'classification': ETS A/B/C class from the effect size and the
              adjusted P-value
            - 'adjustment': Adjustment method for each row
            - 'tested': Whether the item was tested (False for anchors)
            - 'converged': Whether the fits behind each row converged; for
              untested items, whether every fit converged
        and the metadata keys 'method', 'anchors' and 'linking_constants'
        (``(A, B)`` placing the focal group on the reference scale, or None
        for likelihood-ratio tests).

    Raises:
        ValueError: If the method, model, scheme or anchors are invalid.
    """
    if method not in _DIF_METHODS:
        raise ValueError(f"Unknown DIF method: {method}")
    if model not in _DIF_MODELS:
        raise ValueError(f"model must be one of: {', '.join(sorted(_DIF_MODELS))}")
    if scheme not in _DIF_SCHEMES:
        raise ValueError(f"scheme must be one of: {', '.join(sorted(_DIF_SCHEMES))}")
    p_adjust = _validate_p_value_adjustment(p_adjust, name="p_adjust")
    n_jobs = _validate_n_jobs(n_jobs)

    data = np.asarray(data)
    groups = np.asarray(groups)
    n_items = data.shape[1]
    likelihood_ratio = method == "likelihood_ratio"
    anchor_items = resolve_anchor_items(
        anchors, n_items, name="anchors", minimum=1 if likelihood_ratio else 2
    )
    tested = np.ones(n_items, dtype=np.bool_)
    if anchor_items is not None:
        tested[anchor_items] = False

    ref_data, focal_data, _, _, ref_group, _ = split_groups(data, groups, focal_group)
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

    result["classification"] = _ets_classify(
        result["effect_size"],
        None if method == "raju" else result["p_value_adjusted"],
    )
    result["adjustment"] = np.full(n_items, p_adjust)
    result["tested"] = tested
    result["method"] = method
    result["anchors"] = anchor_items
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
    n_quadpts: int,
    max_iter: int,
    tol: float,
) -> dict[str, Any]:
    """Nested multiple-group likelihood-ratio tests for two groups."""
    from mirt.multigroup.dif import _run_multigroup_dif

    labels = np.unique(groups)
    reference_index = int(np.flatnonzero(labels == reference_group)[0])
    table = _run_multigroup_dif(
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
        reference_group=reference_index,
        n_jobs=n_jobs,
    )

    n_items = data.shape[1]
    statistic = np.full(n_items, np.nan)
    df = np.full(n_items, np.nan)
    p_value = np.full(n_items, np.nan)
    p_value_adjusted = np.full(n_items, np.nan)
    effect_size = np.full(n_items, np.nan)
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
                row.group_parameters[1 - reference_index], n_active
            ) - _item_location(row.group_parameters[reference_index], n_active)

    return {
        "statistic": statistic,
        "df": df,
        "p_value": p_value,
        "p_value_adjusted": p_value_adjusted,
        "effect_size": effect_size,
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


def _ets_classify(
    effect_sizes: NDArray[np.float64],
    p_values: NDArray[np.float64] | None,
) -> NDArray[np.str_]:
    """Classify DIF using ETS guidelines (A/B/C).

    Items are class A unless the absolute effect reaches 0.426 and the
    p-value is at most 0.05; classes B and C split at 0.638. Without
    p-values (descriptive methods) the effect size alone decides, and a
    missing effect size or p-value gives class A.
    """
    magnitude = np.abs(np.asarray(effect_sizes, dtype=np.float64))
    classification = np.where(
        magnitude < 0.426, "A", np.where(magnitude < 0.638, "B", "C")
    ).astype("U1")
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
        min_effect_size: Minimum effect size to flag.
        classification: If specified, flag items with this ETS class or worse.
            'B' flags B and C items, 'C' flags only C items.
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

    p_adjust = _validate_p_value_adjustment(p_adjust, name="p_adjust")
    effect_sizes = np.asarray(dif_results["effect_size"], dtype=np.float64)
    if dif_results.get("method") == "raju":
        significant = np.isfinite(effect_sizes)
        classes = _ets_classify(effect_sizes, None)
    else:
        p_values = adjust_p_values(dif_results["p_value"], p_adjust)
        significant = p_values <= alpha
        classes = _ets_classify(effect_sizes, p_values)

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
    elif probabilities.ndim == 3:
        if probabilities.shape[:2] != expected_shape:
            raise ValueError(
                "polytomous probabilities must have shape "
                f"({theta.shape[0]}, {n_items}, n_categories)"
            )
        category_scores = np.arange(probabilities.shape[2], dtype=np.float64)
        expected = probabilities @ category_scores
    else:
        raise ValueError("model probabilities must be two- or three-dimensional")
    if not np.all(np.isfinite(expected)):
        raise ValueError("expected item scores must contain only finite values")
    return expected


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
    valid: NDArray[np.bool_],
    group_masks: dict[Any, NDArray[np.bool_]],
    unique_groups: NDArray[Any],
    scaling_method: Literal["mean", "mad", "iqr"],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int_],
]:
    """Compute group-by-item residual moments with itemwise missingness.

    ``residuals`` must be zero where responses are missing.
    """
    mrr, msr, group_counts = _group_residual_means(
        residuals, valid, group_masks, unique_groups
    )
    effective_counts = np.where(group_counts >= 2, group_counts, 1)
    var_mrr = np.ones_like(mrr)
    var_msr = np.ones_like(msr)

    for group_index, group in enumerate(unique_groups):
        mask = group_masks[group]
        group_valid = valid[mask]
        group_residuals = residuals[mask]
        squared = group_residuals**2
        counts = group_counts[group_index]
        sufficient = counts >= 2

        raw_scale = _column_scale(
            group_residuals,
            group_valid,
            counts,
            mrr[group_index],
            scaling_method,
        )
        squared_scale = _column_scale(
            squared,
            group_valid,
            counts,
            msr[group_index],
            scaling_method,
        )
        var_mrr[group_index, sufficient] = np.maximum(
            raw_scale[sufficient] / counts[sufficient], PROB_EPSILON
        )
        var_msr[group_index, sufficient] = np.maximum(
            squared_scale[sufficient] / counts[sufficient], PROB_EPSILON
        )

    return mrr, msr, var_mrr, var_msr, effective_counts


def compute_grdif(
    data: NDArray[np.int_],
    groups: NDArray,
    model: Literal["1PL", "2PL", "3PL", "GRM", "GPCM"] = "2PL",
    scoring_method: Literal["EAP", "MAP", "ML", "WLE"] = "EAP",
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

    This method has several advantages over traditional DIF approaches:
    - Works with any number of groups (G >= 2)
    - No separate calibration per group required
    - No matching variable or theta bins needed
    - Computationally efficient
    - Well-controlled Type I error rates

    Args:
        data: Response matrix (n_persons x n_items).
        groups: Group membership array (n_persons,). Can have 2+ groups.
        model: IRT model type for aggregate calibration.
        scoring_method: Method for computing ability estimates.
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
            - 'group_item_counts': Valid responses per group and item

    References:
        Lim, H., et al. (2024). Detecting Differential Item Functioning among
        Multiple Groups Using IRT Residual DIF Framework. Journal of
        Educational Measurement.
    """
    from mirt import fit_mirt

    p_adjust = _validate_p_value_adjustment(p_adjust, name="p_adjust")
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

    expected = _expected_response_matrix(fit_result.model, theta, n_items)
    grdif_r, grdif_s, grdif_rs, p_r, p_s, p_rs = _compute_grdif_statistics(
        data,
        theta,
        fit_result.model,
        group_masks,
        unique_groups,
        scaling_method,
        expected_responses=expected,
    )
    valid = data >= 0
    mrr, msr, group_item_counts = _group_residual_means(
        np.where(valid, data - expected, 0.0), valid, group_masks, unique_groups
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
        "group_item_counts": group_item_counts,
    }


def _group_residual_means(
    residuals: NDArray[np.float64],
    valid: NDArray[np.bool_],
    group_masks: dict[Any, NDArray[np.bool_]],
    unique_groups: NDArray[Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int_]]:
    """Group-by-item mean raw and squared residuals with valid-response counts.

    ``residuals`` must be zero where responses are missing. Means are zero
    where a group has fewer than two valid responses.
    """
    n_groups, n_items = len(unique_groups), residuals.shape[1]
    mrr = np.zeros((n_groups, n_items), dtype=np.float64)
    msr = np.zeros((n_groups, n_items), dtype=np.float64)
    counts = np.zeros((n_groups, n_items), dtype=np.int64)
    for index, group in enumerate(unique_groups):
        mask = group_masks[group]
        counts[index] = np.count_nonzero(valid[mask], axis=0)
        sufficient = counts[index] >= 2
        divisor = np.maximum(counts[index], 1)
        mrr[index] = np.where(sufficient, residuals[mask].sum(axis=0) / divisor, 0.0)
        msr[index] = np.where(
            sufficient, (residuals[mask] ** 2).sum(axis=0) / divisor, 0.0
        )
    return mrr, msr, counts


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
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Compute GRDIF_R, GRDIF_S, GRDIF_RS statistics.

    The statistics are based on the asymptotic multivariate normality of
    the mean raw residuals (MRR) and mean squared residuals (MSR).

    GRDIF_R detects uniform DIF (differences in difficulty)
    GRDIF_S detects nonuniform DIF (differences in discrimination)
    GRDIF_RS detects mixed DIF (both types)

    Expected responses are evaluated in one batched model call. Residual
    moments are then reduced by group while retaining itemwise missingness.
    """
    n_groups = len(unique_groups)
    df_r = n_groups - 1
    df_s = n_groups - 1
    df_rs = 2 * (n_groups - 1)

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

    if expected_responses is None:
        expected = _expected_response_matrix(model, theta_values, n_items)
    else:
        expected = np.asarray(expected_responses, dtype=np.float64)
        if expected.shape != values.shape or not np.all(np.isfinite(expected)):
            raise ValueError(
                "expected_responses must be finite and match the response matrix"
            )

    valid_responses = values >= 0
    residuals = np.where(valid_responses, values - expected, 0.0)
    mrr, msr, var_mrr, var_msr, effective_counts = _group_residual_moments(
        residuals,
        valid_responses,
        group_masks,
        unique_groups,
        scaling_method,
    )

    weights = effective_counts / np.sum(effective_counts, axis=0, keepdims=True)
    pooled_mrr = np.sum(weights * mrr, axis=0)
    pooled_msr = np.sum(weights * msr, axis=0)
    centered_mrr = mrr - pooled_mrr
    centered_msr = msr - pooled_msr

    grdif_r = np.sum(centered_mrr**2 / var_mrr, axis=0)
    grdif_s = np.sum(centered_msr**2 / var_msr, axis=0)
    grdif_rs = grdif_r + grdif_s

    p_r = stats.chi2.sf(grdif_r, df=df_r)
    p_s = stats.chi2.sf(grdif_s, df=df_s)
    p_rs = stats.chi2.sf(grdif_rs, df=df_rs)

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
    scoring_method: Literal["EAP", "MAP", "ML", "WLE"] = "EAP",
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
    groups differ on each item.

    Args:
        data: Response matrix (n_persons x n_items).
        groups: Group membership array.
        model: IRT model type.
        scoring_method: Method for computing ability estimates.
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

    p_adjust = _validate_p_value_adjustment(p_adjust, name="p_adjust")
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
    expected_responses = _expected_response_matrix(fit_result.model, theta, n_items)

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
            - 'max_diff': Maximum of both
        model: Fitted item model. Required only when ``grdif_results`` lacks
            the 'mrr', 'msr' and 'group_item_counts' entries; the moments are
            then recomputed from ``grdif_results['theta']``.

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

    if all(key in grdif_results for key in ("mrr", "msr", "group_item_counts")):
        mrr = np.asarray(grdif_results["mrr"], dtype=np.float64)
        msr = np.asarray(grdif_results["msr"], dtype=np.float64)
        counts = np.asarray(grdif_results["group_item_counts"])
    elif model is not None:
        theta = np.asarray(grdif_results["theta"], dtype=np.float64)
        if theta.ndim == 1:
            theta = theta[:, None]
        if theta.shape[0] != values.shape[0]:
            raise ValueError("grdif_results['theta'] must have one row per person")
        unique_groups = np.asarray(grdif_results["group_labels"])
        valid = values >= 0
        mrr, msr, counts = _group_residual_means(
            np.where(
                valid, values - _expected_response_matrix(model, theta, n_items), 0.0
            ),
            valid,
            {group: labels == group for group in unique_groups},
            unique_groups,
        )
    else:
        raise ValueError(
            "grdif_results lacks residual moments; pass the fitted model= used "
            "by compute_grdif"
        )
    if mrr.ndim != 2 or mrr.shape[1] != n_items or msr.shape != mrr.shape:
        raise ValueError("grdif_results moments must have one column per item")
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
        return spread(msr)
    return np.maximum(spread(mrr), spread(msr))
