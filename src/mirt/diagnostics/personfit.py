from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, TypeAlias, cast, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

from mirt.constants import PROB_EPSILON
from mirt.diagnostics.itemfit import _validate_statistics
from mirt.diagnostics.multiple_testing import (
    PValueAdjustment,
    adjust_p_values,
    validate_p_value_adjustment,
)
from mirt.scoring._common import validate_scoring_responses
from mirt.typing import PersonFitStatistic
from mirt.utils._validation import validate_alpha as _validate_alpha
from mirt.utils.numeric import (
    _FitStatsAccumulator,
    _fourth_central_moment,
    compute_probability_moments,
)

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


_ZH_TARGET_CHUNK_ELEMENTS = 2_000_000
_PERSONFIT_TARGET_CHUNK_ELEMENTS = 262_144
PersonFitAlternative: TypeAlias = Literal["lower", "two-sided", "upper"]
_PERSON_FIT_ALTERNATIVES = frozenset({"lower", "two-sided", "upper"})
_PERSONFIT_STATISTICS: tuple[str, ...] = get_args(PersonFitStatistic)


def compute_personfit(
    model: BaseItemModel,
    responses: ArrayLike,
    theta: NDArray[np.float64],
    statistics: Sequence[str] | str | None = None,
    *,
    p_adjust: PValueAdjustment | None = None,
    alpha: float = 0.05,
    alternative: PersonFitAlternative = "lower",
) -> dict[str, NDArray[np.float64] | NDArray[np.bool_]]:
    """Compute person-fit statistics and optional calibrated significance.

    Parameters
    ----------
    model : BaseItemModel
        Fitted item response model.
    responses : array-like of shape (n_persons, n_items)
        Category codes. Negative codes, ``NaN`` and the nulls of nullable
        DataFrame columns denote missing responses.
    theta : ndarray of shape (n_persons,) or (n_persons, n_factors)
        Person abilities.
    statistics : list of str, optional
        Any of ``"infit"``, ``"outfit"``, ``"z_infit"``, ``"z_outfit"``,
        ``"Zh"`` and its alias ``"lz"``. Defaults to
        ``["infit", "outfit", "Zh"]``. Unknown names raise
        :class:`~mirt.exceptions.MirtValidationError`.
    p_adjust : {"none", "bonferroni", "holm", "fdr_bh"}, optional
        Enables respondent-level p-values derived from the standardized
        log-likelihood statistic. ``None`` preserves the legacy output;
        ``"none"`` adds unadjusted significance columns without correcting
        across respondents.
    alpha : float, default=0.05
        Significance threshold for the ``aberrant`` flag.
    alternative : {"lower", "two-sided", "upper"}, default="lower"
        Normal tail for the p-values. The lower tail targets unexpectedly
        improbable response patterns.

    Returns
    -------
    dict of str to ndarray
        One array of length ``n_persons`` per requested statistic, plus
        ``p_value``, ``p_value_adjusted`` and ``aberrant`` when ``p_adjust``
        is given.

    Notes
    -----
    ``z_infit`` and ``z_outfit`` are Wilson-Hilferty standardized mean
    squares (see :func:`mirt.diagnostics.itemfit.compute_itemfit`). Their
    standard normal reference assumes known abilities; with EAP abilities
    estimated from the same responses they are mildly conservative (in a
    20-item 2PL, ``z_infit`` has mean about -0.3 and standard deviation
    about 0.9). Model probabilities and intermediate statistics are
    evaluated in bounded row blocks. Multiplicity corrections use the full
    respondent population.
    """
    statistics = _validate_statistics(
        statistics, _PERSONFIT_STATISTICS, default=("infit", "outfit", "Zh")
    )

    validated_adjustment: PValueAdjustment | None = None
    if p_adjust is not None:
        validated_adjustment = validate_p_value_adjustment(
            p_adjust,
            name="p_adjust",
        )
        alpha = _validate_alpha(alpha)
        alternative = _validate_alternative(alternative)

    responses = validate_scoring_responses(model, responses)
    theta = np.asarray(theta)

    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    n_persons, n_items = responses.shape

    result: dict[str, NDArray[np.float64] | NDArray[np.bool_]] = {}
    standardized = "z_infit" in statistics or "z_outfit" in statistics
    compute_mean_squares = standardized or any(
        name in statistics for name in ("infit", "outfit")
    )
    compute_zh = (
        "Zh" in statistics or "lz" in statistics or validated_adjustment is not None
    )

    if theta.ndim != 2 or theta.shape[0] != n_persons or n_persons == 0:
        raise ValueError("theta must be a non-empty matrix with one row per person")

    category_width = max(model.n_categories) if model.is_polytomous else 1
    rows_per_chunk = max(
        1, _PERSONFIT_TARGET_CHUNK_ELEMENTS // max(n_items * category_width, 1)
    )
    accumulator = (
        _FitStatsAccumulator(n_persons, standardized=standardized)
        if compute_mean_squares
        else None
    )
    zh = np.empty(n_persons) if compute_zh else None
    for start in range(0, n_persons, rows_per_chunk):
        stop = min(start + rows_per_chunk, n_persons)
        block_responses = responses[start:stop]
        probabilities, expected, variance = compute_probability_moments(
            model, theta[start:stop], n_items
        )
        if accumulator is not None:
            accumulator.add(
                block_responses,
                expected,
                variance,
                axis=1,
                target=slice(start, stop),
                fourth_moment=(
                    _fourth_central_moment(probabilities, expected)
                    if standardized
                    else None
                ),
            )
        if zh is not None:
            zh[start:stop] = _compute_zh_vectorized(
                model, block_responses, probabilities, block_responses >= 0
            )

    if accumulator is not None:
        for name, values in accumulator.statistics().items():
            if name in statistics:
                result[name] = values

    if zh is not None:
        if "Zh" in statistics:
            result["Zh"] = zh
        if "lz" in statistics:
            result["lz"] = zh

        if validated_adjustment is not None:
            result.update(
                _personfit_significance(
                    zh,
                    alpha=alpha,
                    alternative=alternative,
                    p_adjust=validated_adjustment,
                )
            )

    return result


def compute_personfit_significance(
    zh: ArrayLike,
    *,
    alpha: float = 0.05,
    alternative: PersonFitAlternative = "lower",
    p_adjust: PValueAdjustment = "none",
) -> dict[str, NDArray[np.float64] | NDArray[np.bool_]]:
    """Convert standardized person-fit scores into calibrated decisions.

    Parameters
    ----------
    zh : array-like of shape (n_persons,)
        Standardized log-likelihood person-fit scores. ``NaN`` values are
        preserved in p-value outputs and never flagged.
    alpha : float, default=0.05
        Significance threshold used for the ``aberrant`` result.
    alternative : {"lower", "two-sided", "upper"}, default="lower"
        Tail used to convert scores to p-values. The lower-tail test is the
        conventional choice for unexpectedly improbable response patterns.
    p_adjust : {"none", "bonferroni", "holm", "fdr_bh"}, default="none"
        Multiplicity adjustment across respondents with finite scores.

    Returns
    -------
    dict
        Raw ``p_value``, corrected ``p_value_adjusted``, and boolean
        ``aberrant`` arrays.
    """
    values = _coerce_zh(zh)
    validated_alpha = _validate_alpha(alpha)
    validated_alternative = _validate_alternative(alternative)
    validated_adjustment = validate_p_value_adjustment(
        p_adjust,
        name="p_adjust",
    )
    return _personfit_significance(
        values,
        alpha=validated_alpha,
        alternative=validated_alternative,
        p_adjust=validated_adjustment,
    )


def _coerce_zh(zh: ArrayLike) -> NDArray[np.float64]:
    """Return a validated vector of standardized person-fit scores."""
    if np.iscomplexobj(zh):
        raise ValueError("zh must contain real values or NaN")
    try:
        values = np.asarray(zh, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("zh must contain real values or NaN") from exc
    if values.ndim != 1:
        raise ValueError("zh must be one-dimensional")
    return values


def _validate_alternative(alternative: object) -> PersonFitAlternative:
    """Return a supported normal-tail alternative."""
    if not isinstance(alternative, str) or alternative not in _PERSON_FIT_ALTERNATIVES:
        raise ValueError("alternative must be 'lower', 'two-sided', or 'upper'")
    return cast(PersonFitAlternative, alternative)


def _personfit_significance(
    zh: NDArray[np.float64],
    *,
    alpha: float,
    alternative: PersonFitAlternative,
    p_adjust: PValueAdjustment,
) -> dict[str, NDArray[np.float64] | NDArray[np.bool_]]:
    """Compute p-values and decisions from validated inputs."""
    if alternative == "lower":
        p_values = ndtr(zh)
    elif alternative == "upper":
        p_values = ndtr(-zh)
    else:
        p_values = 2.0 * ndtr(-np.abs(zh))

    adjusted = adjust_p_values(p_values, p_adjust)
    return {
        "p_value": p_values,
        "p_value_adjusted": adjusted,
        "aberrant": np.isfinite(adjusted) & (adjusted <= alpha),
    }


def _compute_zh_vectorized(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    probabilities: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Standardized log-likelihood ``Zh`` of every person (row)."""
    ll, expected_ll, var_ll = _log_likelihood_moments(
        model, responses, probabilities, valid_mask, axis=1
    )
    return _standardized_log_likelihood(ll, expected_ll, var_ll, valid_mask.sum(axis=1))


def _standardized_log_likelihood(
    ll: NDArray[np.float64],
    expected_ll: NDArray[np.float64],
    var_ll: NDArray[np.float64],
    counts: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Return ``(l0 - E[l0]) / sqrt(Var[l0])``, or NaN without information.

    Sums of fewer than two responses, or with (near) zero variance, are NaN.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            (counts >= 2) & (var_ll > PROB_EPSILON),
            (ll - expected_ll) / np.sqrt(var_ll),
            np.nan,
        )


def _log_likelihood_moments(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    probabilities: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    *,
    axis: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Sum response log-likelihoods, their expectations and variances.

    Each observed response contributes its log-likelihood ``l0``, the
    expectation ``E = sum_k P_k log P_k`` and the variance
    ``sum_k P_k log^2 P_k - E^2`` of its log-likelihood under the model
    (Drasgow, Levine & Williams, 1985). Sums run over items (``axis=1``,
    person fit) or over persons (``axis=0``, item fit). Missing responses
    contribute nothing.
    """
    n_persons, n_items = responses.shape
    size = n_persons if axis == 1 else n_items
    ll = np.zeros(size)
    expected_ll = np.zeros(size)
    var_ll = np.zeros(size)

    category_width = probabilities.shape[2] if probabilities.ndim == 3 else 1
    chunk_size = max(
        1,
        min(
            n_items,
            _ZH_TARGET_CHUNK_ELEMENTS // max(n_persons * category_width, 1),
        ),
    )

    def accumulate(
        start: int,
        stop: int,
        person_ll: NDArray[np.float64],
        item_expected_ll: NDArray[np.float64],
        item_var_ll: NDArray[np.float64],
        block_valid: NDArray[np.bool_],
    ) -> None:
        target = slice(None) if axis == 1 else slice(start, stop)
        for total, values in (
            (ll, person_ll),
            (expected_ll, item_expected_ll),
            (var_ll, item_var_ll),
        ):
            total[target] += np.sum(np.where(block_valid, values, 0.0), axis=axis)

    if probabilities.ndim == 2:
        for start in range(0, n_items, chunk_size):
            stop = min(start + chunk_size, n_items)
            probs = np.clip(
                probabilities[:, start:stop],
                PROB_EPSILON,
                1 - PROB_EPSILON,
            )
            block_responses = responses[:, start:stop]
            log_p = np.log(probs)
            log_q = np.log(1 - probs)
            accumulate(
                start,
                stop,
                np.where(block_responses == 1, log_p, log_q),
                probs * log_p + (1 - probs) * log_q,
                probs * (1 - probs) * (log_p - log_q) ** 2,
                valid_mask[:, start:stop],
            )
    else:
        category_counts = np.asarray(model.n_categories, dtype=np.int64)
        category_indices = np.arange(probabilities.shape[2])

        for start in range(0, n_items, chunk_size):
            stop = min(start + chunk_size, n_items)
            block_valid = valid_mask[:, start:stop]
            block_responses = responses[:, start:stop]
            category_mask = category_indices < category_counts[start:stop, None]
            clipped_probs = np.clip(
                probabilities[:, start:stop, :],
                PROB_EPSILON,
                1 - PROB_EPSILON,
            )
            probs = np.where(category_mask[None, :, :], clipped_probs, 0.0)
            log_probs = np.where(
                category_mask[None, :, :],
                np.log(clipped_probs),
                0.0,
            )

            safe_responses = np.where(block_valid, block_responses, 0)
            safe_responses = np.clip(
                safe_responses,
                0,
                category_counts[None, start:stop] - 1,
            ).astype(np.intp, copy=False)
            person_ll = np.take_along_axis(
                log_probs,
                safe_responses[..., None],
                axis=2,
            )[..., 0]
            item_expected_ll = np.sum(probs * log_probs, axis=2)
            accumulate(
                start,
                stop,
                person_ll,
                item_expected_ll,
                np.sum(probs * log_probs**2, axis=2) - item_expected_ll**2,
                block_valid,
            )

    return ll, expected_ll, var_ll
