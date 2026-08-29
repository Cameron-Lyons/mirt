from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

from mirt.constants import PROB_EPSILON
from mirt.diagnostics.multiple_testing import (
    PValueAdjustment,
    _validate_p_value_adjustment,
    adjust_p_values,
)
from mirt.utils.numeric import compute_fit_stats, compute_probability_moments

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


_ZH_TARGET_CHUNK_ELEMENTS = 2_000_000
PersonFitAlternative: TypeAlias = Literal["lower", "two-sided", "upper"]
_PERSON_FIT_ALTERNATIVES = frozenset({"lower", "two-sided", "upper"})


def compute_personfit(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    statistics: list[str] | None = None,
    *,
    p_adjust: PValueAdjustment | None = None,
    alpha: float = 0.05,
    alternative: PersonFitAlternative = "lower",
) -> dict[str, NDArray[np.float64] | NDArray[np.bool_]]:
    """Compute person-fit statistics and optional calibrated significance.

    Passing ``p_adjust`` enables respondent-level p-values derived from the
    standardized log-likelihood statistic. The default lower-tail test targets
    unexpectedly improbable response patterns. ``None`` preserves the legacy
    output; ``"none"`` adds unadjusted significance columns without correcting
    across respondents.
    """
    if statistics is None:
        statistics = ["infit", "outfit", "Zh"]

    validated_adjustment: PValueAdjustment | None = None
    if p_adjust is not None:
        validated_adjustment = _validate_p_value_adjustment(
            p_adjust,
            name="p_adjust",
        )
        alpha = _validate_alpha(alpha)
        alternative = _validate_alternative(alternative)

    responses = np.asarray(responses)
    theta = np.asarray(theta)

    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    n_persons, n_items = responses.shape

    result: dict[str, NDArray[np.float64] | NDArray[np.bool_]] = {}
    compute_mean_squares = "outfit" in statistics or "infit" in statistics
    compute_zh = (
        "Zh" in statistics or "lz" in statistics or validated_adjustment is not None
    )
    if not compute_mean_squares and not compute_zh:
        return result

    probabilities, expected, variance = compute_probability_moments(
        model,
        theta,
        n_items,
    )

    if compute_mean_squares:
        infit, outfit = compute_fit_stats(responses, expected, variance, axis=1)

        if "outfit" in statistics:
            result["outfit"] = outfit

        if "infit" in statistics:
            result["infit"] = infit

    if compute_zh:
        valid_mask = responses >= 0
        zh = _compute_zh_vectorized(model, responses, probabilities, valid_mask)

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
    validated_adjustment = _validate_p_value_adjustment(
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


def _validate_alpha(alpha: object) -> float:
    """Return a finite significance threshold strictly between zero and one."""
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError("alpha must be finite and in (0, 1)")
    try:
        value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha must be finite and in (0, 1)") from exc
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    return value


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
    n_persons, n_items = responses.shape

    ll = np.zeros(n_persons)
    expected_ll = np.zeros(n_persons)
    var_ll = np.zeros(n_persons)

    category_width = probabilities.shape[2] if probabilities.ndim == 3 else 1
    chunk_size = max(
        1,
        min(
            n_items,
            _ZH_TARGET_CHUNK_ELEMENTS // max(n_persons * category_width, 1),
        ),
    )

    if probabilities.ndim == 2:
        for start in range(0, n_items, chunk_size):
            stop = min(start + chunk_size, n_items)
            probs = np.clip(
                probabilities[:, start:stop],
                PROB_EPSILON,
                1 - PROB_EPSILON,
            )
            block_valid = valid_mask[:, start:stop]
            block_responses = responses[:, start:stop]
            log_p = np.log(probs)
            log_q = np.log(1 - probs)

            person_ll = np.where(block_responses == 1, log_p, log_q)
            ll += np.sum(np.where(block_valid, person_ll, 0.0), axis=1)

            item_expected_ll = probs * log_p + (1 - probs) * log_q
            expected_ll += np.sum(
                np.where(block_valid, item_expected_ll, 0.0),
                axis=1,
            )

            item_var_ll = probs * (1 - probs) * (log_p - log_q) ** 2
            var_ll += np.sum(
                np.where(block_valid, item_var_ll, 0.0),
                axis=1,
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
            )
            person_ll = np.take_along_axis(
                log_probs,
                safe_responses[..., None],
                axis=2,
            )[..., 0]
            ll += np.sum(np.where(block_valid, person_ll, 0.0), axis=1)

            item_expected_ll = np.sum(probs * log_probs, axis=2)
            expected_ll += np.sum(
                np.where(block_valid, item_expected_ll, 0.0),
                axis=1,
            )

            item_var_ll = np.sum(probs * log_probs**2, axis=2) - item_expected_ll**2
            var_ll += np.sum(
                np.where(block_valid, item_var_ll, 0.0),
                axis=1,
            )

    valid_count = valid_mask.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        zh = np.where(
            (valid_count >= 2) & (var_ll > PROB_EPSILON),
            (ll - expected_ll) / np.sqrt(var_ll),
            np.nan,
        )

    return zh


def flag_aberrant_persons(
    fit_stats: dict[str, NDArray[np.float64]],
    criteria: dict[str, tuple[float, float]] | None = None,
) -> NDArray[np.bool_]:
    if criteria is None:
        criteria = {
            "infit": (0.5, 1.5),
            "outfit": (0.5, 1.5),
            "Zh": (-2.0, 2.0),
            "lz": (-2.0, 2.0),
        }

    if not fit_stats:
        raise ValueError("fit_stats must contain at least one statistic")

    first_stat = next(iter(fit_stats.values()))
    n_persons = len(first_stat)

    flags = np.zeros(n_persons, dtype=bool)

    for stat_name, (lower, upper) in criteria.items():
        if stat_name in fit_stats:
            values = fit_stats[stat_name]
            flags |= (values < lower) | (values > upper)

    return flags
