"""SIBTEST (Simultaneous Item Bias Test) procedure.

SIBTEST is a nonparametric DIF detection method that uses a matching
criterion based on valid subtest scores.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from itertools import combinations
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from mirt.constants import PROB_EPSILON
from mirt.diagnostics._utils import split_groups
from mirt.diagnostics.multiple_testing import (
    PValueAdjustment,
    adjust_p_values,
    validate_p_value_adjustment,
)

SIBTESTMethod: TypeAlias = Literal["original", "crossing"]
SIBTESTConvention: TypeAlias = Literal["default", "mirt"]
SIBTESTResult: TypeAlias = dict[
    str, NDArray[Any] | list[tuple[Any, Any]] | float | int | str
]


@dataclass(frozen=True)
class _SIBTESTStatistics:
    beta: float = np.nan
    standard_error: float = np.nan
    chi2: float = np.nan
    df: int = 0
    crossing_point: float = np.nan
    n_strata: int = 0
    permutation_p_value: float | None = None

    @property
    def z(self) -> float:
        if self.standard_error > PROB_EPSILON:
            return self.beta / self.standard_error
        return float("nan")

    @property
    def p_value(self) -> float:
        if self.permutation_p_value is not None:
            return self.permutation_p_value
        if self.df == 0:
            return float("nan")
        return float(stats.chi2.sf(self.chi2, self.df))


@dataclass(frozen=True)
class _MultiGroupStatistics:
    beta: NDArray[np.float64]
    covariance: NDArray[np.float64]
    chi2: float
    df: int
    crossing_points: NDArray[np.float64]
    n_strata: int
    permutation_p_value: float | None = None

    @property
    def standard_errors(self) -> NDArray[np.float64]:
        return np.sqrt(np.diag(self.covariance))

    @property
    def z(self) -> NDArray[np.float64]:
        errors = self.standard_errors
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(errors > PROB_EPSILON, self.beta / errors, np.nan)

    @property
    def p_value(self) -> float:
        if self.permutation_p_value is not None:
            return self.permutation_p_value
        if self.df == 0:
            return float("nan")
        return float(stats.chi2.sf(self.chi2, self.df))


@dataclass(frozen=True)
class _SIBTESTOptions:
    """Validated inference options shared by every comparison."""

    method: SIBTESTMethod
    correction: bool
    min_cell_size: int
    convention: SIBTESTConvention
    n_permutations: int
    rng: np.random.Generator | None


def _validate_min_cell_size(min_cell_size: int) -> int:
    if (
        isinstance(min_cell_size, (bool, np.bool_))
        or not isinstance(min_cell_size, (int, np.integer))
        or min_cell_size < 2
    ):
        raise ValueError("min_cell_size must be an integer of at least 2")
    return int(min_cell_size)


def _validate_options(
    method: object,
    correction: object,
    min_cell_size: int,
    convention: object,
    n_permutations: object,
    random_state: int | np.random.Generator | None,
) -> _SIBTESTOptions:
    if not isinstance(method, str) or method not in {"original", "crossing"}:
        raise ValueError(f"Unknown SIBTEST method: {method}")
    if not isinstance(correction, (bool, np.bool_)):
        raise ValueError("correction must be boolean")
    minimum = _validate_min_cell_size(min_cell_size)
    if not isinstance(convention, str) or convention not in {"default", "mirt"}:
        raise ValueError("convention must be 'default' or 'mirt'")
    if (
        isinstance(n_permutations, (bool, np.bool_))
        or not isinstance(n_permutations, (int, np.integer))
        or n_permutations < 0
    ):
        raise ValueError("n_permutations must be a non-negative integer")
    if n_permutations and method != "crossing":
        raise ValueError("n_permutations requires method='crossing'")
    rng = np.random.default_rng(random_state) if n_permutations else None
    return _SIBTESTOptions(
        cast(SIBTESTMethod, method),
        bool(correction),
        minimum,
        cast(SIBTESTConvention, convention),
        int(n_permutations),
        rng,
    )


def _validate_flag(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


def _binary_item_variances(data: NDArray[np.int64]) -> NDArray[np.float64]:
    """Unbiased binary variances without allocating a centered response matrix."""
    proportions = np.mean(data, axis=0)
    return proportions * (1.0 - proportions) * (data.shape[0] / (data.shape[0] - 1))


def _item_variances(data: NDArray[np.int64]) -> NDArray[np.float64]:
    """Unbiased item-score variances for coefficient alpha.

    Binary items keep the closed form; items scored above one use the
    ordinary sample variance, so alpha reduces to KR-20 for binary items.
    """
    variances = _binary_item_variances(data)
    polytomous = np.flatnonzero(np.max(data, axis=0) > 1)
    if polytomous.size:
        variances[polytomous] = np.var(data[:, polytomous], axis=0, ddof=1)
    return variances


def _prepare_responses(
    data: NDArray[np.int_], groups: NDArray, *, na_rm: bool
) -> tuple[NDArray[np.int64], NDArray]:
    """Return complete non-negative integer item scores and their group labels.

    ``NaN`` and negative codes are missing. Like R mirt's ``na.rm``, removal
    is listwise over the whole response matrix.
    """
    values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError(f"data must be a 2D response matrix, got {values.ndim}D")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("data must contain at least one person and one item")
    if values.dtype.kind not in "biuf":
        raise ValueError("data must contain numeric item scores")
    labels = _validate_groups(groups, values.shape[0])
    if values.dtype.kind == "f":
        if np.any(np.isinf(values)):
            raise ValueError("data must contain only finite responses")
        missing = np.isnan(values) | (values < 0)
    else:
        missing = values < 0
    incomplete = np.any(missing, axis=1)
    if np.any(incomplete):
        if not na_rm:
            raise ValueError(
                "data contain missing responses (NaN or negative codes); "
                "set na_rm=True to remove incomplete persons"
            )
        values = values[~incomplete]
        labels = labels[~incomplete]
        if values.shape[0] == 0:
            raise ValueError("no complete persons remain after removing missing data")
    if values.dtype.kind == "f" and np.any(values != np.floor(values)):
        raise ValueError("data must contain non-negative integer item scores")
    return values.astype(np.int64, copy=False), labels


def _validate_groups(groups: NDArray, n_persons: int) -> NDArray:
    """Validate group labels before attempting a group split."""
    labels = np.asarray(groups)
    if labels.ndim != 1 or labels.shape[0] != n_persons:
        raise ValueError(f"groups must have shape ({n_persons},)")
    if labels.dtype.kind in "fc" and not np.all(np.isfinite(labels)):
        raise ValueError("groups must contain only finite labels")
    if labels.dtype.kind == "O" and any(
        label is None
        or (isinstance(label, (float, np.floating)) and not np.isfinite(label))
        for label in labels
    ):
        raise ValueError("groups must not contain missing labels")
    try:
        np.unique(labels)
    except (TypeError, ValueError) as exc:
        raise ValueError("groups must contain comparable labels") from exc
    return labels


def _validate_item_indices(
    items: list[int] | NDArray[np.int_],
    *,
    name: str,
    n_items: int,
) -> NDArray[np.int64]:
    """Validate a one-dimensional, unique item-index collection."""
    indices = np.asarray(items)
    if indices.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional collection of indices")
    if indices.size == 0:
        raise ValueError(f"{name} must contain at least one item")
    if indices.dtype.kind not in "iu" or indices.dtype.kind == "b":
        raise ValueError(f"{name} must contain integer indices")
    normalized = indices.astype(np.int64, copy=False)
    if np.any((normalized < 0) | (normalized >= n_items)):
        raise ValueError(f"{name} contains an item index outside [0, {n_items})")
    if np.unique(normalized).size != normalized.size:
        raise ValueError(f"{name} must not contain duplicate indices")
    return normalized


def _group_layout(
    labels: NDArray,
    *,
    focal_group: Any | None,
    contrasts: ArrayLike | None,
    pairwise: bool,
) -> tuple[NDArray, NDArray[np.float64] | None]:
    """Return sorted group labels and, for omnibus tests, the contrast matrix."""
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        raise ValueError(f"Expected at least 2 groups, found {unique_labels.size}")
    for label in unique_labels:
        if np.count_nonzero(labels == label) < 2:
            raise ValueError("each group must contain at least two persons")
    if pairwise:
        if focal_group is not None or contrasts is not None:
            raise ValueError(
                "pairwise=True compares every pair of groups; "
                "omit focal_group and contrasts"
            )
        return unique_labels, None
    if unique_labels.size == 2:
        if contrasts is not None:
            raise ValueError(
                "contrasts apply to more than two groups; use focal_group instead"
            )
        return unique_labels, None
    if focal_group is not None:
        raise ValueError(
            "focal_group applies to two groups; use contrasts or pairwise=True"
        )
    return unique_labels, _validate_contrasts(contrasts, unique_labels.size)


def _validate_contrasts(
    contrasts: ArrayLike | None, n_groups: int
) -> NDArray[np.float64]:
    """Return a full-rank contrast matrix over the sorted group labels.

    The default, like R mirt's ``cbind(1, -diag(G - 1))``, compares the first
    group with each other group (first minus other).
    """
    if contrasts is None:
        return np.hstack(
            (np.ones((n_groups - 1, 1)), -np.eye(n_groups - 1, dtype=np.float64))
        )
    try:
        matrix = np.array(contrasts, dtype=np.float64, ndmin=2)
    except (TypeError, ValueError) as exc:
        raise ValueError("contrasts must be a numeric matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != n_groups:
        raise ValueError(
            f"contrasts must have {n_groups} columns, one per sorted group label"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("contrasts must be finite")
    scale = np.maximum(np.max(np.abs(matrix), axis=1), 1.0)
    if np.any(np.abs(matrix.sum(axis=1)) > 1e-10 * scale):
        raise ValueError("each contrast row must sum to zero")
    if np.linalg.matrix_rank(matrix) < matrix.shape[0]:
        raise ValueError("contrasts must have linearly independent rows")
    return matrix


def _split_validated_groups(
    data: NDArray[np.int64], groups: NDArray, focal_group: Any | None = None
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.bool_],
    NDArray[np.bool_],
]:
    """Split the response matrix and require estimable group sizes."""
    ref_data, focal_data, ref_mask, focal_mask, _, _ = split_groups(
        data, groups, focal_group=focal_group
    )
    if ref_data.shape[0] < 2 or focal_data.shape[0] < 2:
        raise ValueError("each group must contain at least two persons")
    return ref_data, focal_data, ref_mask, focal_mask


def sibtest(
    data: NDArray[np.int_],
    groups: NDArray,
    suspect_items: list[int] | NDArray[np.int_],
    matching_items: list[int] | NDArray[np.int_] | None = None,
    method: SIBTESTMethod = "original",
    correction: bool = True,
    *,
    min_cell_size: int = 2,
    focal_group: Any | None = None,
    na_rm: bool = False,
    convention: SIBTESTConvention = "default",
    contrasts: ArrayLike | None = None,
    pairwise: bool = False,
    n_permutations: int = 0,
    random_state: int | np.random.Generator | None = None,
) -> SIBTESTResult:
    """SIBTEST procedure for DIF detection.

    SIBTEST compares the performance of reference and focal groups on
    suspect items after matching on valid (anchor) items. Items may be
    binary or polytomous (Poly-SIBTEST), and more than two groups are tested
    jointly (GSIBTEST).

    Parameters
    ----------
    data : NDArray
        Response matrix (n_persons, n_items) of non-negative integer item
        scores. ``NaN`` and negative codes are missing; see ``na_rm``.
    groups : NDArray
        Group membership (n_persons,) with at least two unique values
    suspect_items : list or NDArray
        Indices of items to test for DIF. Several items test their summed
        score as a bundle (DBF).
    matching_items : list or NDArray, optional
        Indices of items to use for matching (anchor items).
        If None, uses all items except suspect items.
    method : str
        SIBTEST method:

        - 'original': Standard unidirectional SIBTEST (β_uni)
        - 'crossing': Crossing SIBTEST for non-uniform DIF (β_cross)
    correction : bool
        Apply the Shealy-Stout true-score regression correction using group
        coefficient-alpha reliability (KR-20 for binary items). Correction
        requires at least two matching items, positive reliability, and
        observed adjacent matching-score strata.
    min_cell_size : int, default=2
        Minimum persons in each group at a retained matching score. Larger
        values (for example, 5) exclude sparse strata from asymptotic inference.
        R mirt's ``Jmin`` keeps scores with more than ``Jmin`` persons, so
        its default ``Jmin = 5`` is ``min_cell_size=6``.
    focal_group : optional
        Group label treated as focal. By default, the second sorted label.
        Only for two groups.
    na_rm : bool, default=False
        Remove persons with any missing response, as R mirt's ``na.rm``.
        Otherwise missing responses raise an error.
    convention : {"default", "mirt"}, default="default"
        ``"mirt"`` reproduces R mirt's ``SIBTEST`` (with ``min_cell_size``
        set to ``Jmin + 1``). It also drops the lowest and highest observed
        matching scores and any score at which a compared group's suspect
        score is constant, keeps a score for all contrasts when the groups of
        any one contrast qualify, and locates crossings as R does (see
        Notes).
    contrasts : array-like, optional
        Contrast matrix (n_contrasts, n_groups) for more than two groups.
        Columns follow the sorted group labels and each row must sum to
        zero. By default the first group is compared with each other group.
    pairwise : bool, default=False
        Run a separate two-group test for every pair of groups instead, as
        R mirt's ``pairwise = TRUE``.
    n_permutations : int, default=0
        Number of sign-flip permutations for the crossing test's p-value
        (Li and Stout, 1996), as R mirt's ``randomize = TRUE`` with
        ``permute``. Zero uses the asymptotic test.
    random_state : int, Generator, or None
        Seed or NumPy random generator for the permutations.

    Returns
    -------
    dict
        Dictionary with:

        - 'beta': SIBTEST β statistic
        - 'beta_se': Standard error of β
        - 'z': Z-statistic
        - 'p_value': Two-sided p-value
        - 'effect_size': Standardized effect size
        - 'chi2', 'df': Chi-square test statistic and degrees of freedom.
        - 'crossing_point': Estimated crossing location for the crossing method
        - 'n_strata': Number of retained matching-score strata

        With more than two groups, 'beta', 'beta_se', 'z', 'effect_size' and
        'crossing_point' have one entry per contrast, 'chi2', 'df' and
        'p_value' are the joint test, and 'group_labels', 'contrasts' and
        'beta_covariance' are added. With ``pairwise=True`` every statistic
        has one entry per group pair, listed as (reference, focal) labels
        in 'pairs'.

    Notes
    -----
    Unestimable statistics are ``NaN`` with zero degrees of freedom. Stratum
    weights use the pooled population, and standard errors use within-group
    suspect-score sampling variances. Polytomous items follow Chang, Mazzeo
    and Roussos (1996): scores are item-score sums and the correction uses
    coefficient alpha. Crossing inference uses Chalmers (2018)'s one- or
    two-region chi-square test; its p-value is not a normal-tail
    transformation of ``z``. With ``n_permutations``, the crossing p-value is
    the share of sign-flipped strata whose standardized crossing effect is
    at least the observed ``z``, as in R; ``chi2`` is then ``NaN`` and
    ``df`` zero.

    More than two groups use the GSIBTEST of Chalmers and Zheng (2023):
    ``beta`` is each contrast of the groups' (corrected) conditional
    suspect-score means and ``chi2`` is the Wald statistic with one degree of
    freedom per contrast. By default a score is kept only when every compared
    group meets the stratum rules, and the correction extrapolates to the
    average estimated true score of all groups. The multi-group crossing
    statistic has no asymptotic distribution, so its p-value is ``NaN``
    unless ``n_permutations`` is positive, as in R.

    By default a crossing is located by regressing the retained conditional
    differences on the matching score, weighted by the larger group's count.
    ``convention="mirt"`` uses R's regression instead: it covers every score
    observed in both groups whose larger group holds over 1% of the
    persons, enters unretained scores with zero difference, and rounds the
    location to one significant digit. With more than two groups R averages
    the pairwise locations of a contrast's groups, weighted by pair size.
    """
    options = _validate_options(
        method, correction, min_cell_size, convention, n_permutations, random_state
    )
    pairwise = _validate_flag(pairwise, "pairwise")
    response_data, group_labels = _prepare_responses(
        data, groups, na_rm=_validate_flag(na_rm, "na_rm")
    )
    n_items = response_data.shape[1]
    suspect_indices = _validate_item_indices(
        suspect_items, name="suspect_items", n_items=n_items
    )

    if matching_items is None:
        selected = np.ones(n_items, dtype=np.bool_)
        selected[suspect_indices] = False
        matching_indices = np.flatnonzero(selected)
        if matching_indices.size == 0:
            raise ValueError("No matching items available")
    else:
        matching_indices = _validate_item_indices(
            matching_items, name="matching_items", n_items=n_items
        )
        if np.intersect1d(suspect_indices, matching_indices).size:
            raise ValueError("suspect_items and matching_items must not overlap")

    unique_labels, contrast_matrix = _group_layout(
        group_labels, focal_group=focal_group, contrasts=contrasts, pairwise=pairwise
    )

    two_group = partial(
        _two_group_sibtest,
        suspect_indices=suspect_indices,
        matching_indices=matching_indices,
        options=options,
    )
    if pairwise:
        return _pairwise_results(
            response_data,
            group_labels,
            unique_labels,
            two_group,
            shared=("method", "n_suspect_items", "n_matching_items"),
        )
    if contrast_matrix is not None:
        return _multigroup_sibtest(
            response_data,
            group_labels,
            unique_labels,
            contrast_matrix,
            suspect_indices,
            matching_indices,
            options,
        )
    return two_group(response_data, group_labels, focal_group)


def _two_group_sibtest(
    response_data: NDArray[np.int64],
    group_labels: NDArray,
    focal_group: Any | None,
    *,
    suspect_indices: NDArray[np.int64],
    matching_indices: NDArray[np.int64],
    options: _SIBTESTOptions,
) -> SIBTESTResult:
    """Run one reference-versus-focal SIBTEST."""
    ref_data, focal_data, ref_mask, focal_mask = _split_validated_groups(
        response_data, group_labels, focal_group
    )
    matching_scores = np.sum(response_data[:, matching_indices], axis=1)
    suspect_scores_ref = np.sum(ref_data[:, suspect_indices], axis=1)
    suspect_scores_focal = np.sum(focal_data[:, suspect_indices], axis=1)
    result = _compute_sibtest_statistics(
        suspect_scores_ref,
        suspect_scores_focal,
        matching_scores[ref_mask],
        matching_scores[focal_mask],
        options.method,
        options.correction,
        min_cell_size=options.min_cell_size,
        n_matching_items=int(matching_indices.size),
        matching_variances=(
            float(_item_variances(ref_data[:, matching_indices]).sum()),
            float(_item_variances(focal_data[:, matching_indices]).sum()),
        )
        if options.correction
        else None,
        convention=options.convention,
        n_permutations=options.n_permutations,
        rng=options.rng,
    )
    effect_size = _effect_size(result.beta, suspect_scores_ref, suspect_scores_focal)

    return {
        "beta": result.beta,
        "beta_se": result.standard_error,
        "z": result.z,
        "p_value": result.p_value,
        "chi2": result.chi2,
        "df": result.df,
        "crossing_point": result.crossing_point,
        "n_strata": result.n_strata,
        "effect_size": float(effect_size),
        "method": options.method,
        "n_suspect_items": int(suspect_indices.size),
        "n_matching_items": int(matching_indices.size),
    }


def _multigroup_sibtest(
    response_data: NDArray[np.int64],
    group_labels: NDArray,
    unique_labels: NDArray,
    contrasts: NDArray[np.float64],
    suspect_indices: NDArray[np.int64],
    matching_indices: NDArray[np.int64],
    options: _SIBTESTOptions,
) -> SIBTESTResult:
    """Run the joint GSIBTEST over all groups."""
    matching_scores = np.sum(response_data[:, matching_indices], axis=1)
    suspect_scores = np.sum(response_data[:, suspect_indices], axis=1)
    masks = [group_labels == label for label in unique_labels]
    result = _compute_multigroup_statistics(
        [suspect_scores[mask] for mask in masks],
        [matching_scores[mask] for mask in masks],
        contrasts,
        options,
        n_matching_items=int(matching_indices.size),
        matching_variances=[
            float(_item_variances(response_data[mask][:, matching_indices]).sum())
            for mask in masks
        ]
        if options.correction
        else None,
    )
    return {
        "beta": result.beta,
        "beta_se": result.standard_errors,
        "z": result.z,
        "p_value": result.p_value,
        "chi2": result.chi2,
        "df": result.df,
        "crossing_point": result.crossing_points,
        "n_strata": result.n_strata,
        "effect_size": _contrast_effect_sizes(
            result.beta, [suspect_scores[mask] for mask in masks], contrasts
        ),
        "method": options.method,
        "n_suspect_items": int(suspect_indices.size),
        "n_matching_items": int(matching_indices.size),
        "group_labels": unique_labels,
        "contrasts": contrasts,
        "beta_covariance": result.covariance,
    }


def _pairwise_results(
    response_data: NDArray[np.int64],
    group_labels: NDArray,
    unique_labels: NDArray,
    two_group: Callable[[NDArray[np.int64], NDArray, Any], SIBTESTResult],
    *,
    shared: tuple[str, ...],
) -> SIBTESTResult:
    """Run two-group tests per pair and stack each statistic along a last axis."""
    pairs = list(combinations(unique_labels.tolist(), 2))
    results = []
    for reference, focal in pairs:
        mask = (group_labels == reference) | (group_labels == focal)
        results.append(two_group(response_data[mask], group_labels[mask], focal))
    stacked: SIBTESTResult = {"pairs": pairs}
    for name, value in results[0].items():
        if name in shared:
            stacked[name] = value
        else:
            stacked[name] = np.stack(
                [np.asarray(result[name]) for result in results], axis=-1
            )
    return stacked


@dataclass(frozen=True)
class _StratumStatistics:
    ref_counts: NDArray[np.int64]
    focal_counts: NDArray[np.int64]
    ref_means: NDArray[np.float64]
    focal_means: NDArray[np.float64]
    sampling_variances: NDArray[np.float64]


def _group_moments(
    scores: NDArray[np.int64], suspect: NDArray[np.int64], n_levels: int
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """Counts, means and mean sampling variances of suspect scores per stratum."""
    counts = np.bincount(scores, minlength=n_levels)
    values = np.asarray(suspect, dtype=np.float64)
    sums = np.bincount(scores, weights=values, minlength=n_levels)
    seconds = np.bincount(scores, weights=values**2, minlength=n_levels)
    means = sums / np.maximum(counts, 1)
    # Unbiased within-stratum score variance, then variance of its mean.
    centered = np.maximum(seconds - sums * means, 0.0)
    mean_variances = centered / (np.maximum(counts - 1, 1) * np.maximum(counts, 1))
    return counts, means, mean_variances


def _stratum_statistics(
    ref_suspect: NDArray[np.int64],
    focal_suspect: NDArray[np.int64],
    ref_scores: NDArray[np.int64],
    focal_scores: NDArray[np.int64],
) -> _StratumStatistics:
    """Reduce group score moments in linear time without person/stratum scans."""
    n_levels = int(max(np.max(ref_scores), np.max(focal_scores))) + 1
    ref = _group_moments(ref_scores, ref_suspect, n_levels)
    focal = _group_moments(focal_scores, focal_suspect, n_levels)
    return _StratumStatistics(ref[0], focal[0], ref[1], focal[1], ref[2] + focal[2])


def _constant_strata(
    scores: NDArray[np.int64], suspect: NDArray[np.int64], n_levels: int
) -> NDArray[np.bool_]:
    """Scores at which a group's suspect score has zero or undefined variance."""
    lowest = np.full(n_levels, np.inf)
    highest = np.full(n_levels, -np.inf)
    np.minimum.at(lowest, scores, suspect)
    np.maximum.at(highest, scores, suspect)
    return lowest == highest


def _interior_strata(
    group_scores: list[NDArray[np.int64]], n_levels: int
) -> NDArray[np.bool_]:
    """Every score except the lowest and highest observed matching scores."""
    keep = np.ones(n_levels, dtype=np.bool_)
    keep[min(int(np.min(scores)) for scores in group_scores)] = False
    keep[max(int(np.max(scores)) for scores in group_scores)] = False
    return keep


def _matching_reliability(
    scores: NDArray[np.int64], n_items: int, item_variance_sum: float
) -> float:
    """Coefficient alpha (KR-20 for binary items) with sample variances."""
    if n_items < 2:
        return float("nan")
    score_variance = float(np.var(scores, ddof=1))
    if score_variance <= PROB_EPSILON:
        return float("nan")
    reliability = n_items / (n_items - 1) * (1.0 - item_variance_sum / score_variance)
    return float(reliability) if reliability > PROB_EPSILON else float("nan")


def _regression_corrected_differences(
    strata: _StratumStatistics,
    ref_scores: NDArray[np.int64],
    focal_scores: NDArray[np.int64],
    n_matching_items: int,
    matching_variances: tuple[float, float],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Extrapolate each conditional mean to the common estimated true score.

    Shealy and Stout (1993) regress true matched scores on observed scores
    using group-specific reliability, then estimate each suspect-score slope
    from the neighboring conditional means. Endpoints and absent neighbors
    cannot supply that central slope and are excluded.
    """
    n_levels = strata.ref_counts.size
    differences = np.full(n_levels, np.nan)
    usable = np.zeros(n_levels, dtype=bool)
    reliabilities = (
        _matching_reliability(ref_scores, n_matching_items, matching_variances[0]),
        _matching_reliability(focal_scores, n_matching_items, matching_variances[1]),
    )
    if n_levels < 3 or not np.all(np.isfinite(reliabilities)):
        return differences, usable

    ref_mean = float(np.mean(ref_scores))
    focal_mean = float(np.mean(focal_scores))
    levels = np.arange(1, n_levels - 1)
    ref_true = ref_mean + reliabilities[0] * (levels - ref_mean)
    focal_true = focal_mean + reliabilities[1] * (levels - focal_mean)
    common_true = (ref_true + focal_true) / 2.0
    ref_slope = (strata.ref_means[2:] - strata.ref_means[:-2]) / (2 * reliabilities[0])
    focal_slope = (strata.focal_means[2:] - strata.focal_means[:-2]) / (
        2 * reliabilities[1]
    )
    differences[levels] = (
        strata.ref_means[levels]
        + ref_slope * (common_true - ref_true)
        - strata.focal_means[levels]
        - focal_slope * (common_true - focal_true)
    )
    usable[levels] = (
        (strata.ref_counts[:-2] > 0)
        & (strata.ref_counts[2:] > 0)
        & (strata.focal_counts[:-2] > 0)
        & (strata.focal_counts[2:] > 0)
    )
    return differences, usable


def _crossing_location(
    levels: NDArray[np.int64],
    differences: NDArray[np.float64],
    regression_weights: NDArray[np.float64],
) -> float:
    """Estimate one crossing point by weighted conditional-difference regression."""
    normalized = regression_weights / np.sum(regression_weights)
    score_mean = float(np.dot(normalized, levels))
    difference_mean = float(np.dot(normalized, differences))
    centered_scores = levels - score_mean
    score_variance = float(np.dot(normalized, centered_scores**2))
    if score_variance <= PROB_EPSILON:
        return float("nan")
    slope = float(np.dot(normalized, centered_scores * differences) / score_variance)
    if abs(slope) <= PROB_EPSILON:
        return float("nan")
    return score_mean - difference_mean / slope


@dataclass(frozen=True)
class _CrossingRegression:
    """Weighted least-squares crossing location of conditional differences.

    The regression runs over ``scores`` with ``weights``. ``columns`` places
    each retained stratum among them (-1 when it is left out); a regression
    score without a retained stratum enters with zero difference, as in R
    mirt's two-group CSIBTEST.
    """

    scores: NDArray[np.float64]
    weights: NDArray[np.float64]
    columns: NDArray[np.intp]

    @classmethod
    def over(
        cls, levels: NDArray[np.int64], weights: NDArray[np.int64]
    ) -> _CrossingRegression:
        """Regress over the retained strata themselves."""
        return cls(
            levels.astype(np.float64),
            weights.astype(np.float64),
            np.arange(levels.size, dtype=np.intp),
        )

    def locate(self, differences: NDArray[np.float64]) -> NDArray[np.float64]:
        """Crossing of each row of retained-stratum ``differences``.

        Matches :func:`_crossing_location`, row by row.
        """
        crossing = np.full(differences.shape[0], np.nan)
        if self.scores.size == 0:
            return crossing
        placed = np.zeros((differences.shape[0], self.scores.size))
        kept = self.columns >= 0
        placed[:, self.columns[kept]] = differences[:, kept]
        normalized = self.weights / np.sum(self.weights)
        score_mean = float(np.dot(normalized, self.scores))
        centered_scores = self.scores - score_mean
        score_variance = float(np.dot(normalized, centered_scores**2))
        if score_variance <= PROB_EPSILON:
            return crossing
        slopes = placed @ (normalized * centered_scores) / score_variance
        sloped = np.abs(slopes) > PROB_EPSILON
        crossing[sloped] = score_mean - (placed @ normalized)[sloped] / slopes[sloped]
        return crossing


def _mirt_crossing_regression(
    counts: NDArray[np.int64],
    other_counts: NDArray[np.int64],
    eligible: NDArray[np.bool_],
) -> _CrossingRegression:
    """R mirt's crossing regression for two groups.

    R regresses over every score observed in both groups whose larger group
    holds over 1% of the two groups' persons, weighting by that size.
    """
    heavier = np.maximum(counts, other_counts)
    n_persons = float(np.sum(counts) + np.sum(other_counts))
    regression = (counts > 0) & (other_counts > 0) & (heavier / n_persons > 0.01)
    positions = np.full(eligible.size, -1, dtype=np.intp)
    positions[regression] = np.arange(np.count_nonzero(regression))
    return _CrossingRegression(
        np.flatnonzero(regression).astype(np.float64),
        heavier[regression].astype(np.float64),
        positions[eligible],
    )


def _round_to_one_digit(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """R's ``signif(x, 1)``: one significant digit, ties to even."""
    rounded = np.array(values, dtype=np.float64)
    finite = np.isfinite(rounded) & (rounded != 0)
    magnitude = np.abs(rounded[finite])
    exponent = -np.floor(np.log10(magnitude))
    scale = 10.0 ** np.abs(exponent)
    rounded[finite] = np.sign(rounded[finite]) * np.where(
        exponent > 0,
        np.rint(magnitude * scale) / scale,
        np.rint(magnitude / scale) * scale,
    )
    return rounded


def _signed_crossing_effects(
    levels: NDArray[np.int64],
    differences: NDArray[np.float64],
    weights: NDArray[np.float64],
    crossings: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Weighted difference below each row's crossing minus the one above it."""
    low = np.ones(differences.shape, dtype=np.bool_)
    located = np.isfinite(crossings)
    low[located] = levels <= crossings[located, None]
    return np.sum(np.where(low, weights, -weights) * differences, axis=-1)


def _sign_flips(
    rng: np.random.Generator, n_permutations: int, n_levels: int
) -> NDArray[np.float64]:
    """Independent random signs for every stratum, as in Li and Stout (1996)."""
    return rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, n_levels))


def _compute_sibtest_statistics(
    ref_suspect: NDArray[np.int64],
    focal_suspect: NDArray[np.int64],
    ref_scores: NDArray[np.int64],
    focal_scores: NDArray[np.int64],
    method: SIBTESTMethod,
    correction: bool,
    *,
    min_cell_size: int = 2,
    n_matching_items: int = 0,
    matching_variances: tuple[float, float] | None = None,
    convention: SIBTESTConvention = "default",
    n_permutations: int = 0,
    rng: np.random.Generator | None = None,
) -> _SIBTESTStatistics:
    """Compute pooled conditional effects and their sampling uncertainty.

    The variance is sum(p_k**2 * (s_Rk**2/n_Rk + s_Fk**2/n_Fk)).
    Crossing inference follows Chalmers (2018), equation 14, summing squared
    standardized effects on the independent sides of the estimated crossing.
    """
    strata = _stratum_statistics(ref_suspect, focal_suspect, ref_scores, focal_scores)
    eligible = (strata.ref_counts >= min_cell_size) & (
        strata.focal_counts >= min_cell_size
    )
    if convention == "mirt":
        eligible &= (
            _interior_strata([ref_scores, focal_scores], eligible.size)
            & ~_constant_strata(ref_scores, ref_suspect, eligible.size)
            & ~_constant_strata(focal_scores, focal_suspect, eligible.size)
        )
    differences = strata.ref_means - strata.focal_means
    if correction:
        if matching_variances is None:
            return _SIBTESTStatistics()
        differences, correction_usable = _regression_corrected_differences(
            strata, ref_scores, focal_scores, n_matching_items, matching_variances
        )
        eligible &= correction_usable
    levels = np.flatnonzero(eligible)
    if levels.size == 0:
        return _SIBTESTStatistics()

    differences = differences[eligible]
    pooled_counts = strata.ref_counts[eligible] + strata.focal_counts[eligible]
    weights = pooled_counts / np.sum(pooled_counts)
    variance_components = weights**2 * strata.sampling_variances[eligible]
    standard_error = float(np.sqrt(np.sum(variance_components)))
    beta = float(np.dot(weights, differences))
    if method == "original":
        chi2 = (beta / standard_error) ** 2 if standard_error > PROB_EPSILON else np.nan
        return _SIBTESTStatistics(
            beta,
            standard_error,
            chi2,
            int(standard_error > PROB_EPSILON),
            n_strata=int(levels.size),
        )

    regression_weights = np.maximum(
        strata.ref_counts[eligible], strata.focal_counts[eligible]
    )
    if convention == "mirt":
        regression = _mirt_crossing_regression(
            strata.ref_counts, strata.focal_counts, eligible
        )

        def locate(rows: NDArray[np.float64]) -> NDArray[np.float64]:
            return _round_to_one_digit(regression.locate(rows))

        crossing = float(locate(differences[None, :])[0])
    else:
        locate = _CrossingRegression.over(levels, regression_weights).locate
        crossing = _crossing_location(levels, differences, regression_weights)
    low = (
        levels <= crossing
        if np.isfinite(crossing)
        else np.ones(levels.size, dtype=bool)
    )
    region_betas = np.array(
        [
            np.dot(weights[low], differences[low]),
            np.dot(weights[~low], differences[~low]),
        ]
    )
    region_variances = np.array(
        [
            np.sum(variance_components[low]),
            np.sum(variance_components[~low]),
        ]
    )
    estimable = region_variances > PROB_EPSILON**2
    df = int(np.count_nonzero(estimable))
    chi2 = (
        float(np.sum(region_betas[estimable] ** 2 / region_variances[estimable]))
        if df
        else np.nan
    )
    crossing_beta = abs(float(region_betas[0] - region_betas[1]))
    if n_permutations and rng is not None:
        # R's randomize = TRUE: flip each stratum's difference, re-estimate
        # the crossing, and compare |beta_cross| / SE with the observed value.
        permutation_p_value = float("nan")
        if standard_error > PROB_EPSILON:
            observed = np.abs(
                _signed_crossing_effects(
                    levels, differences[None, :], weights, locate(differences[None, :])
                )
            )[0]
            flipped_rows = _sign_flips(rng, n_permutations, levels.size) * differences
            flipped = np.abs(
                _signed_crossing_effects(
                    levels, flipped_rows, weights, locate(flipped_rows)
                )
            )
            permutation_p_value = float(np.mean(flipped >= observed))
        return _SIBTESTStatistics(
            crossing_beta,
            standard_error,
            np.nan,
            0,
            crossing,
            int(levels.size),
            permutation_p_value,
        )
    return _SIBTESTStatistics(
        crossing_beta,
        standard_error,
        chi2,
        df,
        crossing,
        int(levels.size),
    )


def _corrected_group_means(
    counts: NDArray[np.int64],
    means: NDArray[np.float64],
    group_scores: list[NDArray[np.int64]],
    n_matching_items: int,
    matching_variances: list[float],
    compared: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Multi-group Shealy-Stout correction to the average estimated true score.

    As in R mirt's GSIBTEST, every group's estimated true score enters the
    common target; neighbouring scores must be observed in each compared
    group.
    """
    n_groups, n_levels = counts.shape
    corrected = np.full((n_groups, n_levels), np.nan)
    usable = np.zeros(n_levels, dtype=bool)
    reliabilities = np.array(
        [
            _matching_reliability(scores, n_matching_items, variance)
            for scores, variance in zip(group_scores, matching_variances, strict=True)
        ]
    )
    if n_levels < 3 or not np.all(np.isfinite(reliabilities)):
        return corrected, usable

    group_means = np.array([float(np.mean(scores)) for scores in group_scores])
    levels = np.arange(1, n_levels - 1)
    true_scores = group_means[:, None] + reliabilities[:, None] * (
        levels - group_means[:, None]
    )
    common_true = np.mean(true_scores, axis=0)
    slopes = (means[:, 2:] - means[:, :-2]) / (2 * reliabilities[:, None])
    corrected[:, levels] = means[:, levels] + slopes * (common_true - true_scores)
    usable[levels] = np.all(
        (counts[compared, :-2] > 0) & (counts[compared, 2:] > 0), axis=0
    )
    return corrected, usable


def _quadratic_form(
    covariance: NDArray[np.float64], vectors: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return v' S^-1 v for each row v of ``vectors``."""
    solved = np.linalg.solve(covariance, vectors.T)
    return np.sum(vectors.T * solved, axis=0)


def _compute_multigroup_statistics(
    group_suspect: list[NDArray[np.int64]],
    group_scores: list[NDArray[np.int64]],
    contrasts: NDArray[np.float64],
    options: _SIBTESTOptions,
    *,
    n_matching_items: int,
    matching_variances: list[float] | None,
) -> _MultiGroupStatistics:
    """GSIBTEST contrasts of conditional means and their Wald statistic.

    The contrast covariance is C diag(sum_k p_k**2 s_gk**2 / n_gk) C', which
    is shared by the uniform and the crossing contrasts.
    """
    n_contrasts = contrasts.shape[0]
    unestimable = _MultiGroupStatistics(
        np.full(n_contrasts, np.nan),
        np.full((n_contrasts, n_contrasts), np.nan),
        np.nan,
        0,
        np.full(n_contrasts, np.nan),
        0,
    )
    n_levels = max(int(np.max(scores)) for scores in group_scores) + 1
    moments = [
        _group_moments(scores, suspect, n_levels)
        for scores, suspect in zip(group_scores, group_suspect, strict=True)
    ]
    counts = np.array([moment[0] for moment in moments])
    means = np.array([moment[1] for moment in moments])
    mean_variances = np.array([moment[2] for moment in moments])
    compared = np.any(contrasts != 0, axis=0)
    mirt = options.convention == "mirt"
    if mirt:
        interior = _interior_strata(group_scores, n_levels)
        constant = np.array(
            [
                _constant_strata(scores, suspect, n_levels)
                for scores, suspect in zip(group_scores, group_suspect, strict=True)
            ]
        )

    def qualifying(members: NDArray[np.bool_]) -> NDArray[np.bool_]:
        keep = np.all(counts[members] >= options.min_cell_size, axis=0)
        if mirt:
            keep &= interior & ~np.any(constant[members], axis=0)
        return keep

    if mirt:
        # R keeps a score when the groups of any one contrast qualify.
        eligible = np.any([qualifying(row != 0) for row in contrasts], axis=0)
        eligible &= np.all(counts[compared] > 0, axis=0)
    else:
        eligible = qualifying(compared)
    conditional = means
    if options.correction:
        if matching_variances is None:
            return unestimable
        conditional, correction_usable = _corrected_group_means(
            counts,
            means,
            group_scores,
            n_matching_items,
            matching_variances,
            compared,
        )
        eligible &= correction_usable
    levels = np.flatnonzero(eligible)
    if levels.size == 0:
        return unestimable

    pooled_counts = np.sum(counts[:, eligible], axis=0)
    weights = pooled_counts / np.sum(pooled_counts)
    used_contrasts = contrasts[:, compared]
    # Rows are strata, columns are contrasts of the conditional group means.
    differences = (used_contrasts @ conditional[compared][:, eligible]).T
    group_variances = np.sum(weights**2 * mean_variances[compared][:, eligible], axis=1)
    covariance = (used_contrasts * group_variances) @ used_contrasts.T
    invertible = bool(np.linalg.eigvalsh(covariance).min() > PROB_EPSILON**2)
    beta = weights @ differences

    if options.method == "original":
        if not invertible:
            return _MultiGroupStatistics(
                beta, covariance, np.nan, 0, unestimable.crossing_points, levels.size
            )
        return _MultiGroupStatistics(
            beta,
            covariance,
            float(_quadratic_form(covariance, beta[None, :])[0]),
            n_contrasts,
            unestimable.crossing_points,
            int(levels.size),
        )

    if mirt:
        # R averages the pairwise crossings of a contrast's groups, weighted
        # by pair size, and rounds the average to one significant digit.
        pair_regressions = {
            (first, second): _mirt_crossing_regression(
                counts[first], counts[second], eligible
            )
            for first, second in combinations(np.flatnonzero(compared), 2)
        }

        def locate(index: int, signs: NDArray[np.float64]) -> NDArray[np.float64]:
            located = np.zeros(signs.shape[0])
            total = 0.0
            members = np.flatnonzero(contrasts[index] != 0)
            for first, second in combinations(members, 2):
                pair_size = float(np.sum(counts[first]) + np.sum(counts[second]))
                pair_crossing = pair_regressions[(first, second)].locate(
                    signs
                    * (conditional[first, eligible] - conditional[second, eligible])
                )
                located += pair_size * np.where(
                    np.isnan(pair_crossing), 0.0, pair_crossing
                )
                total += pair_size
            return _round_to_one_digit(located / total)

    else:
        regressions = [
            _CrossingRegression.over(
                levels, np.max(counts[contrast != 0][:, eligible], axis=0)
            )
            for contrast in contrasts
        ]

        def locate(index: int, signs: NDArray[np.float64]) -> NDArray[np.float64]:
            return regressions[index].locate(signs * differences[:, index])

    def crossing_effects(signs: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.column_stack(
            [
                _signed_crossing_effects(
                    levels,
                    signs * differences[:, index],
                    weights,
                    locate(index, signs),
                )
                for index in range(n_contrasts)
            ]
        )

    unflipped = np.ones((1, levels.size))
    crossing_points = np.array(
        [locate(index, unflipped)[0] for index in range(n_contrasts)]
    )
    crossing_beta = crossing_effects(unflipped)[0]
    if not invertible:
        return _MultiGroupStatistics(
            np.abs(crossing_beta), covariance, np.nan, 0, crossing_points, levels.size
        )
    chi2 = float(_quadratic_form(covariance, crossing_beta[None, :])[0])
    permutation_p_value = None
    if options.n_permutations and options.rng is not None:
        # R flips the sign of all group means in a stratum, which flips every
        # contrast there, then re-estimates each contrast's crossing.
        flipped = _quadratic_form(
            covariance,
            crossing_effects(
                _sign_flips(options.rng, options.n_permutations, levels.size)
            ),
        )
        permutation_p_value = float(np.mean(flipped >= chi2))
    return _MultiGroupStatistics(
        np.abs(crossing_beta),
        covariance,
        chi2,
        0,
        crossing_points,
        int(levels.size),
        permutation_p_value,
    )


def _effect_size(
    beta: float,
    ref_suspect: NDArray[np.int64],
    focal_suspect: NDArray[np.int64],
) -> float:
    """Standardize beta by the equal-weight pooled suspect-score deviation."""
    pooled_variance = (
        float(np.var(ref_suspect, ddof=1)) + float(np.var(focal_suspect, ddof=1))
    ) / 2.0
    if pooled_variance <= PROB_EPSILON:
        return float("nan")
    return beta / np.sqrt(pooled_variance)


def _contrast_effect_sizes(
    beta: NDArray[np.float64],
    group_suspect: list[NDArray[np.int64]],
    contrasts: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Standardize each contrast by its compared groups' equal-weight deviation."""
    variances = np.array([float(np.var(suspect, ddof=1)) for suspect in group_suspect])
    pooled = np.array([np.mean(variances[contrast != 0]) for contrast in contrasts])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(pooled > PROB_EPSILON, beta / np.sqrt(pooled), np.nan)


def sibtest_items(
    data: NDArray[np.int_],
    groups: NDArray,
    anchor_items: list[int] | NDArray[np.int_] | None = None,
    method: SIBTESTMethod = "original",
    correction: bool = True,
    alpha: float = 0.05,
    p_adjust: PValueAdjustment = "bonferroni",
    *,
    min_cell_size: int = 2,
    focal_group: Any | None = None,
    na_rm: bool = False,
    convention: SIBTESTConvention = "default",
    contrasts: ArrayLike | None = None,
    pairwise: bool = False,
    n_permutations: int = 0,
    random_state: int | np.random.Generator | None = None,
) -> SIBTESTResult:
    """Run SIBTEST for each item individually.

    Parameters
    ----------
    data : NDArray
        Response matrix of non-negative integer item scores
    groups : NDArray
        Group membership
    anchor_items : list or NDArray, optional
        Items to use for matching. The item under test is always excluded.
        If None, all other items are used.
    method : str
        SIBTEST method
    correction : bool
        Apply the Shealy-Stout true-score correction for either method.
    alpha : float
        Family-wise significance level. Default 0.05.
    p_adjust : {"bonferroni", "holm", "fdr_bh", "none"}
        Multiple-testing adjustment. Default "bonferroni".
    min_cell_size : int, default=2
        Minimum persons per group in a retained matching-score stratum.
    focal_group : optional
        Label treated as focal. See :func:`sibtest` for inference details.
    na_rm, convention, contrasts, n_permutations, random_state
        As for :func:`sibtest`. One generator draws the permutations of all
        items in turn.
    pairwise : bool, default=False
        Test every pair of groups separately. Each pair is its own
        multiple-testing family, as in R mirt.

    Returns
    -------
    dict
        Dictionary with arrays for each item:

        - 'beta': β statistics
        - 'beta_se': standard errors
        - 'z': Z-statistics
        - 'p_value': P-values
        - 'p_value_adjusted': multiplicity-adjusted P-values
        - 'effect_size': standardized effect sizes
        - 'chi2', 'df': chi-square statistics and degrees of freedom
        - 'crossing_point', 'n_strata': crossing locations and retained strata
        - 'flagged': Boolean flags for significant DIF

        With more than two groups, per-contrast statistics ('beta',
        'beta_se', 'z', 'effect_size', 'crossing_point') have shape
        (n_items, n_contrasts) and 'group_labels' and 'contrasts' are added.
        With ``pairwise=True`` every array gains a last axis over the group
        pairs in 'pairs', and 'alpha_corrected' has one entry per pair.
    """
    options = _validate_options(
        method, correction, min_cell_size, convention, n_permutations, random_state
    )
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError("alpha must be finite and between 0 and 1")
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha must be finite and between 0 and 1") from exc
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and between 0 and 1")
    p_adjust = validate_p_value_adjustment(p_adjust, name="p_adjust")
    pairwise = _validate_flag(pairwise, "pairwise")

    response_data, group_labels = _prepare_responses(
        data, groups, na_rm=_validate_flag(na_rm, "na_rm")
    )
    n_items = response_data.shape[1]
    if n_items < 2:
        raise ValueError("sibtest_items requires at least two items")
    if anchor_items is None:
        anchors = np.arange(n_items, dtype=np.int64)
    else:
        anchors = _validate_item_indices(
            anchor_items, name="anchor_items", n_items=n_items
        )
    unique_labels, contrast_matrix = _group_layout(
        group_labels, focal_group=focal_group, contrasts=contrasts, pairwise=pairwise
    )

    two_group = partial(
        _two_group_items,
        explicit_anchors=anchor_items is not None,
        anchors=anchors,
        options=options,
        alpha=alpha,
        p_adjust=p_adjust,
    )
    if pairwise:
        return _pairwise_results(
            response_data,
            group_labels,
            unique_labels,
            two_group,
            shared=("alpha", "adjustment"),
        )
    if contrast_matrix is not None:
        return _multigroup_items(
            response_data,
            group_labels,
            unique_labels,
            contrast_matrix,
            anchors,
            options,
            alpha,
            p_adjust,
        )
    return two_group(response_data, group_labels, focal_group)


def _two_group_items(
    response_data: NDArray[np.int64],
    group_labels: NDArray,
    focal_group: Any | None,
    *,
    explicit_anchors: bool,
    anchors: NDArray[np.int64],
    options: _SIBTESTOptions,
    alpha: float,
    p_adjust: PValueAdjustment,
) -> SIBTESTResult:
    """Run every item's reference-versus-focal SIBTEST on shared totals."""
    n_items = response_data.shape[1]
    correction = options.correction
    ref_data, focal_data, ref_mask, focal_mask = _split_validated_groups(
        response_data, group_labels, focal_group
    )

    if explicit_anchors:
        matching_totals = np.sum(response_data[:, anchors], axis=1)
    else:
        matching_totals = np.sum(response_data, axis=1)
    anchor_membership = np.zeros(n_items, dtype=np.bool_)
    anchor_membership[anchors] = True
    if correction:
        ref_item_variances = _item_variances(ref_data)
        focal_item_variances = _item_variances(focal_data)
        ref_anchor_variance = float(np.sum(ref_item_variances[anchors]))
        focal_anchor_variance = float(np.sum(focal_item_variances[anchors]))

    betas = np.full(n_items, np.nan)
    standard_errors = np.full(n_items, np.nan)
    zs = np.full(n_items, np.nan)
    p_values = np.full(n_items, np.nan)
    effect_sizes = np.full(n_items, np.nan)
    chi2s = np.full(n_items, np.nan)
    degrees = np.zeros(n_items, dtype=np.float64)
    crossing_points = np.full(n_items, np.nan)
    stratum_counts = np.zeros(n_items, dtype=np.float64)

    for item_index in range(n_items):
        if anchors.size == 1 and anchor_membership[item_index]:
            continue
        item_responses = response_data[:, item_index]
        matching_scores = (
            matching_totals - item_responses
            if anchor_membership[item_index]
            else matching_totals
        )
        ref_suspect = ref_data[:, item_index]
        focal_suspect = focal_data[:, item_index]
        matching_variances = None
        if correction:
            matching_variances = (
                ref_anchor_variance - ref_item_variances[item_index]
                if anchor_membership[item_index]
                else ref_anchor_variance,
                focal_anchor_variance - focal_item_variances[item_index]
                if anchor_membership[item_index]
                else focal_anchor_variance,
            )
        item_result = _compute_sibtest_statistics(
            ref_suspect,
            focal_suspect,
            matching_scores[ref_mask],
            matching_scores[focal_mask],
            options.method,
            correction,
            min_cell_size=options.min_cell_size,
            n_matching_items=int(anchors.size - anchor_membership[item_index]),
            matching_variances=matching_variances,
            convention=options.convention,
            n_permutations=options.n_permutations,
            rng=options.rng,
        )
        betas[item_index] = item_result.beta
        standard_errors[item_index] = item_result.standard_error
        effect_sizes[item_index] = _effect_size(
            item_result.beta, ref_suspect, focal_suspect
        )
        zs[item_index] = item_result.z
        p_values[item_index] = item_result.p_value
        chi2s[item_index] = item_result.chi2
        degrees[item_index] = item_result.df
        crossing_points[item_index] = item_result.crossing_point
        stratum_counts[item_index] = item_result.n_strata

    return _item_result(
        {
            "beta": betas,
            "beta_se": standard_errors,
            "z": zs,
            "p_value": p_values,
            "effect_size": effect_sizes,
            "chi2": chi2s,
            "df": degrees,
            "crossing_point": crossing_points,
            "n_strata": stratum_counts,
        },
        alpha,
        p_adjust,
    )


def _item_result(
    statistics: SIBTESTResult, alpha: float, p_adjust: PValueAdjustment
) -> SIBTESTResult:
    """Add adjusted p-values, flags and the Bonferroni per-test level."""
    p_values = cast(NDArray[np.float64], statistics["p_value"])
    adjusted = adjust_p_values(p_values, p_adjust)
    n_finite_tests = int(np.count_nonzero(np.isfinite(p_values)))
    corrected_alpha = (
        alpha / n_finite_tests
        if p_adjust == "bonferroni" and n_finite_tests > 0
        else alpha
    )
    result = {}
    for name, value in statistics.items():
        result[name] = value
        if name == "p_value":
            result["p_value_adjusted"] = adjusted
    result["flagged"] = adjusted < alpha
    result["alpha"] = float(alpha)
    result["alpha_corrected"] = float(corrected_alpha)
    result["adjustment"] = p_adjust
    return result


def _multigroup_items(
    response_data: NDArray[np.int64],
    group_labels: NDArray,
    unique_labels: NDArray,
    contrasts: NDArray[np.float64],
    anchors: NDArray[np.int64],
    options: _SIBTESTOptions,
    alpha: float,
    p_adjust: PValueAdjustment,
) -> SIBTESTResult:
    """Run every item's GSIBTEST on shared matching totals."""
    n_items = response_data.shape[1]
    n_contrasts = contrasts.shape[0]
    masks = [group_labels == label for label in unique_labels]
    matching_totals = np.sum(response_data[:, anchors], axis=1)
    anchor_membership = np.zeros(n_items, dtype=np.bool_)
    anchor_membership[anchors] = True
    if options.correction:
        item_variances = [_item_variances(response_data[mask]) for mask in masks]
        anchor_variances = [float(np.sum(item[anchors])) for item in item_variances]

    per_contrast = {
        name: np.full((n_items, n_contrasts), np.nan)
        for name in ("beta", "beta_se", "z", "effect_size", "crossing_point")
    }
    p_values = np.full(n_items, np.nan)
    chi2s = np.full(n_items, np.nan)
    degrees = np.zeros(n_items, dtype=np.float64)
    stratum_counts = np.zeros(n_items, dtype=np.float64)

    for item_index in range(n_items):
        is_anchor = bool(anchor_membership[item_index])
        if anchors.size == 1 and is_anchor:
            continue
        item_responses = response_data[:, item_index]
        matching_scores = (
            matching_totals - item_responses if is_anchor else matching_totals
        )
        suspects = [item_responses[mask] for mask in masks]
        result = _compute_multigroup_statistics(
            suspects,
            [matching_scores[mask] for mask in masks],
            contrasts,
            options,
            n_matching_items=int(anchors.size - is_anchor),
            matching_variances=[
                total - item[item_index] if is_anchor else total
                for total, item in zip(anchor_variances, item_variances, strict=True)
            ]
            if options.correction
            else None,
        )
        per_contrast["beta"][item_index] = result.beta
        per_contrast["beta_se"][item_index] = result.standard_errors
        per_contrast["z"][item_index] = result.z
        per_contrast["effect_size"][item_index] = _contrast_effect_sizes(
            result.beta, suspects, contrasts
        )
        per_contrast["crossing_point"][item_index] = result.crossing_points
        p_values[item_index] = result.p_value
        chi2s[item_index] = result.chi2
        degrees[item_index] = result.df
        stratum_counts[item_index] = result.n_strata

    result = _item_result(
        {
            "beta": per_contrast["beta"],
            "beta_se": per_contrast["beta_se"],
            "z": per_contrast["z"],
            "p_value": p_values,
            "effect_size": per_contrast["effect_size"],
            "chi2": chi2s,
            "df": degrees,
            "crossing_point": per_contrast["crossing_point"],
            "n_strata": stratum_counts,
        },
        alpha,
        p_adjust,
    )
    result["group_labels"] = unique_labels
    result["contrasts"] = contrasts
    return result
