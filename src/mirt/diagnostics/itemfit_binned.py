"""Ability-grouped item fit: Bock/Yen X2, likelihood-ratio G2 and PV-Q1.

Persons are grouped into ability quantiles with the same bins as
:func:`mirt.utils.empirical.empirical_rmsea` and ``empirical_plot``. Within
each group, observed category counts are compared with the model's category
probabilities at the group's mean ability.

References:
    Bock, R. D. (1972). Estimating item parameters and latent ability when
        responses are scored in two or more nominal categories.
        Psychometrika, 37(1), 29-51.
    Chalmers, R. P., & Ng, V. (2017). Plausible-value imputation statistics
        for detecting item misfit. Applied Psychological Measurement, 41(5),
        372-387.
    Yen, W. M. (1981). Using simulation results to choose a latent trait
        model. Applied Psychological Measurement, 5(2), 245-262.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import chdtrc

from mirt.diagnostics.multiple_testing import PValueAdjustment, adjust_p_values

if TYPE_CHECKING:
    from mirt.estimation._shared_step import EqualityConstraints
    from mirt.models.base import BaseItemModel


_BINNED_TARGET_CHUNK_ELEMENTS = 1_000_000


def _compute_binned_itemfit(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    statistics: Sequence[str],
    *,
    theta: NDArray[np.float64],
    posterior: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
    n_groups: int,
    n_plausible: int,
    seed: int | None,
    item_parameter_counts: ArrayLike | None,
    p_adjust: PValueAdjustment,
    constraints: EqualityConstraints | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute the requested X2, G2 and PV_Q1 item-fit statistics.

    Parameters
    ----------
    model : BaseItemModel
        Fitted unidimensional item response model.
    responses : ndarray of shape (n_persons, n_items)
        Category codes; negative codes and NaN are missing.
    statistics : sequence of str
        Subset of ``"X2"``, ``"G2"`` and ``"PV_Q1"``.
    theta : ndarray of shape (n_persons,)
        Ability estimates used to group persons for X2 and G2.
    posterior : tuple of ndarray or None
        EAP scores and posterior standard deviations, required for PV_Q1.
    n_groups : int
        Number of ability quantile groups.
    n_plausible : int
        Number of plausible-value draws for PV_Q1.
    seed : int or None
        Seed for the plausible-value draws.
    item_parameter_counts : array-like or None
        Estimated parameters per item; defaults to the free parameter masks.
    constraints : sequence, optional
        Equality constraints of the fit; see
        :func:`~mirt.diagnostics.itemfit.compute_itemfit`.
    p_adjust : {"none", "bonferroni", "holm", "fdr_bh"}
        Multiple-testing adjustment across items.

    Returns
    -------
    dict of str to ndarray
        ``<name>``, ``<name>_df`` and ``<name>_p`` for each statistic, plus
        ``<name>_p_adjusted`` when ``p_adjust`` is not ``"none"``.

    Notes
    -----
    ``X2 = sum_g sum_k (O_gk - N_g P_gk)^2 / (N_g P_gk)`` and
    ``G2 = 2 sum_g sum_k O_gk log(O_gk / (N_g P_gk))`` over nonempty groups,
    with ``df = groups * (categories - 1) - parameters``. PV_Q1 draws
    plausible abilities from the normal approximation ``N(EAP, PSD^2)`` of
    each posterior, recomputes X2 for every draw and reports the median
    statistic and degrees of freedom.
    """
    from mirt.diagnostics.itemfit import _sx2_categories, _sx2_parameter_counts

    categories = _sx2_categories(model)
    codes = _response_codes(responses, categories)
    parameters = _sx2_parameter_counts(
        model,
        item_parameter_counts,
        statistic="X2, G2 and PV_Q1",
        constraints=constraints,
    )

    result: dict[str, NDArray[np.float64]] = {}
    if "X2" in statistics or "G2" in statistics:
        x2, g2, df = _grouped_statistics(model, codes, theta, categories, n_groups)
        df = np.maximum(df - parameters, 0)
        for name, values in (("X2", x2), ("G2", g2)):
            if name in statistics:
                result.update(_with_p_values(name, values, df, p_adjust))

    if "PV_Q1" in statistics:
        if posterior is None:
            raise ValueError("PV_Q1 requires EAP scores and posterior deviations")
        center, spread = posterior
        rng = np.random.default_rng(seed)
        draws = np.empty((n_plausible, codes.shape[1]))
        draw_df = np.empty((n_plausible, codes.shape[1]))
        for index in range(n_plausible):
            plausible = center + spread * rng.standard_normal(center.shape)
            draws[index], _, draw_df[index] = _grouped_statistics(
                model, codes, plausible, categories, n_groups
            )
        df = np.maximum(np.floor(np.median(draw_df, axis=0)) - parameters, 0)
        result.update(_with_p_values("PV_Q1", np.median(draws, axis=0), df, p_adjust))
    return result


def _validate_binned_options(
    model: BaseItemModel,
    n_groups: int,
    n_plausible: int,
    *,
    plausible: bool,
) -> int:
    """Validate grouped item-fit settings before any ability scoring."""
    from mirt.diagnostics.itemfit import _validate_n_groups
    from mirt.models.cdm_advanced import HigherOrderCDM
    from mirt.models.mixture import MixtureIRT

    if isinstance(model, (MixtureIRT, HigherOrderCDM)):
        raise ValueError(
            f"X2, G2 and PV_Q1 do not support {type(model).__name__}: "
            "a single ability does not order its latent classes"
        )
    if getattr(model, "n_factors", 1) != 1:
        raise ValueError("X2, G2 and PV_Q1 require a unidimensional model")
    if plausible and (
        isinstance(n_plausible, (bool, np.bool_))
        or not isinstance(n_plausible, (int, np.integer))
        or n_plausible < 1
    ):
        raise ValueError("n_plausible must be a positive integer")
    return _validate_n_groups(n_groups)


def _response_codes(
    responses: NDArray[np.int_],
    categories: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Return integer category codes with ``-1`` marking missing responses."""
    if responses.dtype.kind not in "biuf":
        raise ValueError("responses must contain numeric category codes")
    if np.any(np.isinf(responses)):
        raise ValueError("responses must not contain infinite values")
    observed = ~np.isnan(responses) & (responses >= 0)
    values = np.where(observed, responses, 0)
    if np.any(values != np.floor(values)) or np.any(values >= categories):
        raise ValueError(
            "responses must be integer category codes within each item's range"
        )
    return np.where(observed, values, -1).astype(np.int64)


def _grouped_statistics(
    model: BaseItemModel,
    codes: NDArray[np.int64],
    theta: NDArray[np.float64],
    categories: NDArray[np.int64],
    n_groups: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Return X2, G2 and unadjusted degrees of freedom for one grouping."""
    from mirt.utils.empirical import _build_theta_bins

    group, group_theta = _build_theta_bins(theta, n_groups)
    observed = _group_category_counts(codes, group, n_groups, int(categories.max()))
    group_sizes = observed.sum(axis=2)
    expected = group_sizes[:, :, None] * _category_probabilities(
        model, group_theta, categories
    )

    exists = np.arange(observed.shape[2]) < categories[:, None]
    cells = (group_sizes > 0)[:, :, None] & exists[None, :, :]
    positive = cells & (expected > 0.0)
    impossible = np.any(cells & (expected <= 0.0) & (observed > 0), axis=(0, 2))
    safe_expected = np.where(positive, expected, 1.0)

    x2 = np.sum(
        np.where(positive, (observed - expected) ** 2 / safe_expected, 0.0),
        axis=(0, 2),
    )
    log_ratio = np.log(
        np.where(positive & (observed > 0), observed / safe_expected, 1.0)
    )
    g2 = 2.0 * np.sum(observed * log_ratio, axis=(0, 2))
    x2[impossible] = np.inf
    g2[impossible] = np.inf

    df = np.count_nonzero(group_sizes > 0, axis=0) * (categories - 1)
    return x2, g2, df


def _group_category_counts(
    codes: NDArray[np.int64],
    group: NDArray[np.intp],
    n_groups: int,
    n_categories: int,
) -> NDArray[np.float64]:
    """Count observed responses by ability group, item and category."""
    n_persons, n_items = codes.shape
    width = n_items * n_categories
    item_offsets = np.arange(n_items) * n_categories
    counts = np.zeros(n_groups * width)
    rows = max(1, _BINNED_TARGET_CHUNK_ELEMENTS // n_items)
    for start in range(0, n_persons, rows):
        block = codes[start : start + rows]
        cells = group[start : start + rows, None] * width + item_offsets + block
        counts += np.bincount(cells[block >= 0], minlength=counts.size)
    return counts.reshape(n_groups, n_items, n_categories)


def _category_probabilities(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    categories: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Evaluate zero-padded category probabilities at group abilities."""
    raw = np.asarray(model.probability(theta[:, None]), dtype=np.float64)
    shape = (theta.shape[0], categories.shape[0])
    if not model.is_polytomous:
        if raw.shape != shape:
            raise ValueError(
                "model probabilities must have one row per ability and one "
                "column per item"
            )
        raw = np.stack((1.0 - raw, raw), axis=2)
    if raw.shape != (*shape, int(categories.max())):
        raise ValueError(
            "model probabilities must match abilities, items, and maximum "
            "category count"
        )
    if not np.all(np.isfinite(raw)) or np.any((raw < 0.0) | (raw > 1.0)):
        raise ValueError("model probabilities must be finite and between 0 and 1")
    return raw


def _with_p_values(
    name: str,
    statistic: NDArray[np.float64],
    df: NDArray[np.int64] | NDArray[np.float64],
    p_adjust: PValueAdjustment,
) -> dict[str, NDArray[np.float64]]:
    """Attach chi-square degrees of freedom and (adjusted) p-values."""
    df = np.asarray(df, dtype=np.float64)
    p_value = np.full(statistic.shape, np.nan)
    testable = df > 0
    p_value[testable] = chdtrc(df[testable], statistic[testable])
    values = {name: statistic, f"{name}_df": df, f"{name}_p": p_value}
    if p_adjust != "none":
        values[f"{name}_p_adjusted"] = adjust_p_values(p_value, p_adjust)
    return values
