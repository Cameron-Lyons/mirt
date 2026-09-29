"""Shared logistic variances and bounded multidimensional Fisher reductions."""

from collections.abc import Callable
from math import exp, fsum

import numpy as np
from numpy.typing import NDArray

_INFORMATION_CHUNK_ELEMENTS = 262_144
_AFFINE_CHUNK_ELEMENTS = 262_144
_Logits = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def _logistic_probability(
    logits: NDArray[np.float64],
    guessing: NDArray[np.float64] | None,
    upper: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Reuse an owned logit buffer for probabilities without dropping either tail."""
    if guessing is None and logits.min(initial=0.0) >= -700.0:
        # The usual logistic expression is safe here and needs no masks or
        # extra response buffer. Retain the tail-based path below -700.
        with np.errstate(under="ignore"):
            np.negative(logits, out=logits)
            np.exp(logits, out=logits)
            logits += 1.0
            np.reciprocal(logits, out=logits)
        return logits
    positive = logits >= 0.0
    np.abs(logits, out=logits)
    logits *= -1.0
    with np.errstate(under="ignore"):
        np.exp(logits, out=logits)
        logits /= 1.0 + logits
        if guessing is None:
            return np.where(positive, 1.0 - logits, logits)
        # Interpolate from the nearer asymptote. This preserves exact
        # endpoints and cannot round a saturated response above its upper.
        logits *= 1.0 - guessing if upper is None else upper - guessing
        high = (1.0 if upper is None else upper) - logits
        logits += guessing
        return np.where(positive, high, logits)


def _exact_affine_pairs(
    theta: NDArray[np.float64],
    slopes: NDArray[np.float64],
    intercepts: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Round an exact dot product plus intercept once, for finite inputs only."""
    from fractions import Fraction

    result = np.empty(len(theta))
    for row, (point, slope, intercept) in enumerate(
        zip(theta, slopes, intercepts, strict=True)
    ):
        value = Fraction(float(intercept)) + sum(
            Fraction(float(a)) * Fraction(float(t))
            for a, t in zip(slope, point, strict=True)
        )
        try:
            result[row] = float(value)
        except OverflowError:
            result[row] = -np.inf if value < 0 else np.inf
    return result


def _affine_logits(
    theta: NDArray[np.float64],
    slopes: NDArray[np.float64],
    intercepts: NDArray[np.float64],
    *,
    paired: bool = False,
    specific_indices: NDArray[np.intp] | None = None,
) -> NDArray[np.float64]:
    """Evaluate dense or bifactor logits, recovering only uncertain finite cells.

    Bifactor slopes have two columns; specific_indices identifies the second
    coefficient's factor after the general factor. Paired queries align each
    theta row with a slope row; a one-dimensional slope requests a single item.
    """
    specific_columns = None if specific_indices is None else 1 + specific_indices
    single_item = slopes.ndim == 1
    # Adaptive selection repeatedly requests a single small dot product. Avoid
    # array masks and reductions when its absolute sum has a safe bound.
    if len(theta) == 1 and (single_item or len(slopes) == 1):
        single_slopes = slopes.tolist() if single_item else slopes[0].tolist()
        if specific_columns is None:
            single_points = theta[0].tolist()
        else:
            column = int(specific_columns) if single_item else int(specific_columns[0])
            single_points = [float(theta[0, 0]), float(theta[0, column])]
        offset = float(intercepts) if single_item else float(intercepts[0])
        terms = [a * t for a, t in zip(single_slopes, single_points, strict=True)]
        terms.append(offset)
        if sum(abs(value) for value in terms) <= 1000.0:
            value = np.array([fsum(terms)])
            return value if single_item or paired else value[:, None]

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if specific_columns is None:
            if paired:
                logits = np.einsum("ij,ij->i", theta, slopes)
            else:
                logits = np.dot(theta, slopes.T)
        else:
            general = theta[:, 0] if single_item or paired else theta[:, 0, None]
            if paired:
                specific = theta[np.arange(len(theta)), specific_columns]
            else:
                specific = theta[:, specific_columns]
            logits = general * slopes[..., 0]
            logits += specific * slopes[..., 1]
        logits += intercepts
        n_coefficients = slopes.shape[-1]
        # For a small selected-item query, a second short dot product supplies
        # the bound with fewer reductions than separate global input scales.
        direct_bound = len(theta) == 1 or (single_item and len(theta) <= 1024)
        if not direct_bound:
            theta_scale = (
                np.abs(theta).max(initial=0.0)
                if specific_columns is None
                else np.maximum(
                    np.abs(general).max(initial=0.0),
                    np.abs(specific).max(initial=0.0),
                )
            )
            max_bound = theta_scale * np.abs(slopes).max(
                initial=0.0
            ) * n_coefficients + np.abs(intercepts).max(initial=0.0)
            if max_bound <= 1000.0:
                return logits
        if specific_columns is None:
            if direct_bound:
                bound = np.dot(np.abs(theta), np.abs(slopes).T)
                if paired:
                    bound = bound[:, 0]
            else:
                row_scale = np.max(np.abs(theta), axis=1)
                if logits.ndim == 2:
                    row_scale = row_scale[:, None]
                bound = row_scale * np.sum(np.abs(slopes), axis=-1)
        else:
            bound = np.abs(general) * np.abs(slopes[..., 0])
            bound += np.abs(specific) * np.abs(slopes[..., 1])
        bound += np.abs(intercepts)
        if bound.max(initial=0.0) <= 1000.0:
            return logits
        error = 4.0 * np.finfo(float).eps * n_coefficients * bound
        recover = ~np.isfinite(logits) | (
            (bound > 1000.0) & (np.abs(logits) <= 2500.0 + error)
        )
    # Beyond +/-2500, both logistic tails are saturated even after rescaling
    # by the largest finite slopes. Resolve only cells that could cross back.
    positions = np.flatnonzero(recover)
    block_size = max(1, _AFFINE_CHUNK_ELEMENTS // n_coefficients)
    for start in range(0, positions.size, block_size):
        selected = positions[start : start + block_size]
        if logits.ndim == 2:
            rows, items = np.divmod(selected, logits.shape[1])
        else:
            rows, items = selected, selected if paired else np.zeros_like(selected)
        if single_item:
            coefficients = np.broadcast_to(slopes, (len(rows), n_coefficients))
            offsets = np.full(len(rows), intercepts)
        else:
            coefficients, offsets = slopes[items], intercepts[items]
        if specific_columns is None:
            points = theta[rows]
        else:
            columns = specific_columns if single_item else specific_columns[items]
            points = np.column_stack((theta[rows, 0], theta[rows, columns]))
        finite = (
            np.all(np.isfinite(points), axis=1)
            & np.all(np.isfinite(coefficients), axis=1)
            & np.isfinite(offsets)
        )
        if np.any(finite):
            logits.flat[selected[finite]] = _exact_affine_pairs(
                points[finite], coefficients[finite], offsets[finite]
            )
    return logits


def _affine_probability(
    theta: NDArray[np.float64],
    slopes: NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]],
    intercepts: NDArray[np.float64],
    *,
    item_indices: NDArray[np.intp] | None = None,
    specific_indices: NDArray[np.intp] | None = None,
) -> NDArray[np.float64]:
    """Share bounded full, single-item, and aligned-pair logistic queries.

    Separate bifactor loading vectors avoid copying the complete item bank
    when an aligned query uses only a few items.
    """
    if isinstance(slopes, tuple):
        n_coefficients, full, width = 2, item_indices is None, len(slopes[0])
    else:
        n_coefficients = slopes.shape[-1]
        full = slopes.ndim == 2 and item_indices is None
        width = slopes.shape[0]
    width = width if full else 1
    shape = (len(theta), width) if full else (len(theta),)
    block_size = max(1, _AFFINE_CHUNK_ELEMENTS // max(width, n_coefficients))

    def evaluate(
        points: NDArray[np.float64], indices: NDArray[np.intp] | None
    ) -> NDArray[np.float64]:
        if indices is None:
            coefficients = (
                np.column_stack(slopes) if isinstance(slopes, tuple) else slopes
            )
            offsets, columns = intercepts, specific_indices
        else:
            coefficients = (
                np.column_stack((slopes[0][indices], slopes[1][indices]))
                if isinstance(slopes, tuple)
                else slopes[indices]
            )
            offsets = intercepts[indices]
            columns = None if specific_indices is None else specific_indices[indices]
        return _logistic_probability(
            _affine_logits(
                points,
                coefficients,
                offsets,
                paired=indices is not None,
                specific_indices=columns,
            ),
            None,
            None,
        )

    if len(theta) <= block_size:
        return evaluate(theta, item_indices)
    result = np.empty(shape)
    for start in range(0, len(theta), block_size):
        rows = slice(start, start + block_size)
        result[rows] = evaluate(
            theta[rows], None if item_indices is None else item_indices[rows]
        )
    return result


def _sigmoid_derivative(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate sigmoid(z) sigmoid(-z) without subtracting either tail."""
    result = np.abs(logits)
    result *= -1.0
    with np.errstate(under="ignore"):
        np.exp(result, out=result)
        denominator = 1.0 + result
        np.square(denominator, out=denominator)
        result /= denominator
    return result


def _log_variance(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    absolute = np.abs(logits)
    with np.errstate(under="ignore"):
        return -absolute - 2.0 * np.log1p(np.exp(-absolute))


def _scaled_information(
    logits: NDArray[np.float64],
    magnitude: NDArray[np.float64],
    *,
    norm_factor: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    information = _sigmoid_derivative(logits)
    exceptional = (
        (information < np.finfo(float).tiny) & np.isfinite(logits) & (magnitude != 0.0)
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if norm_factor is not None:
            information *= norm_factor
        information *= magnitude
        information *= magnitude
        if np.any(exceptional):
            scale = np.broadcast_to(magnitude, logits.shape)[exceptional]
            log_information = 2.0 * np.log(np.abs(scale))
            log_information += _log_variance(logits[exceptional])
            if norm_factor is not None:
                log_information += np.log(
                    np.broadcast_to(norm_factor, logits.shape)[exceptional]
                )
            information[exceptional] = np.exp(log_information)
    return information


def _information(
    theta: NDArray[np.float64], slopes: NDArray[np.float64], logits: _Logits
) -> NDArray[np.float64]:
    """Evaluate scalar information with factored slope norms in row blocks."""
    magnitude = np.max(np.abs(slopes), axis=-1)
    with np.errstate(under="ignore", invalid="ignore"):
        scaled = slopes / np.where(magnitude == 0.0, 1.0, magnitude)[..., None]
        norm = np.sum(scaled * scaled, axis=-1)
    shape = (len(theta), slopes.shape[0]) if slopes.ndim == 2 else (len(theta),)
    result = np.empty(shape)
    per_row = shape[1] if len(shape) == 2 else 1
    block_size = max(1, _INFORMATION_CHUNK_ELEMENTS // per_row)
    for start in range(0, len(theta), block_size):
        stop = start + block_size
        result[start:stop] = _scaled_information(
            logits(theta[start:stop]), magnitude, norm_factor=norm
        )
    return result


def _item_information(
    logits: NDArray[np.float64], slope: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Preserve signed cross-factor information without squaring large slopes."""
    # Adaptive item selection repeatedly requests one small matrix. Scalar
    # arithmetic avoids array masks when neither tails nor slopes need recovery.
    if logits.size == 1 and abs(float(logits[0])) < 700.0:
        if all(
            value == 0.0 or 1e-150 <= abs(value) <= 1e150 for value in slope.tolist()
        ):
            tail = exp(-abs(float(logits[0])))
            scalar_variance = tail / (1.0 + tail) ** 2
            with np.errstate(under="ignore"):
                return (scalar_variance * np.outer(slope, slope))[None, :, :]
    variance = _sigmoid_derivative(logits)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        coefficients = np.outer(slope, slope)
        result = variance[:, None, None] * coefficients
        active = (slope[:, None] != 0.0) & (slope[None, :] != 0.0)
        unsafe = active & (
            ~np.isfinite(coefficients) | (np.abs(coefficients) < np.finfo(float).tiny)
        )
        tiny_rows = (variance < np.finfo(float).tiny) & np.isfinite(logits)
        if np.any(unsafe) or np.any(tiny_rows):
            left, right = np.triu_indices(slope.size)
            recover = active[left, right][None, :] & (
                unsafe[left, right][None, :] | tiny_rows[:, None]
            )
            rows, pairs = np.nonzero(recover)
            log_values = _log_variance(logits[rows])
            log_values += np.log(np.abs(slope[left[pairs]]))
            log_values += np.log(np.abs(slope[right[pairs]]))
            sign = np.sign(slope[left[pairs]]) * np.sign(slope[right[pairs]])
            values = sign * np.exp(log_values)
            result[rows, left[pairs], right[pairs]] = values
            result[rows, right[pairs], left[pairs]] = values
    return result


def _test_information(
    theta: NDArray[np.float64], slopes: NDArray[np.float64], logits: _Logits
) -> NDArray[np.float64]:
    """Contract variances with active upper-triangle coefficients in bounded blocks."""
    n_items, n_factors = slopes.shape
    result = np.zeros((len(theta), n_factors, n_factors))
    left, right = np.triu_indices(n_factors)
    block_size = max(1, _INFORMATION_CHUNK_ELEMENTS // max(n_items, left.size))
    pair_size = max(1, _INFORMATION_CHUNK_ELEMENTS // n_items)
    for start in range(0, len(theta), block_size):
        stop = start + block_size
        z = logits(theta[start:stop])
        variance = _sigmoid_derivative(z)
        tiny_rows = np.any((variance < np.finfo(float).tiny) & np.isfinite(z), axis=1)
        log_variance = None
        for first in range(0, left.size, pair_size):
            left_factors, right_factors = (
                left[first : first + pair_size],
                right[first : first + pair_size],
            )
            a, b = slopes[:, left_factors], slopes[:, right_factors]
            active = (a != 0.0) & (b != 0.0)
            present = np.any(active, axis=0)
            left_factors, right_factors, a, b, active = (
                left_factors[present],
                right_factors[present],
                a[:, present],
                b[:, present],
                active[:, present],
            )
            if not left_factors.size:
                continue
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                coefficients = a * b
                values = variance @ coefficients
                unsafe = np.any(
                    active
                    & (
                        ~np.isfinite(coefficients)
                        | (np.abs(coefficients) < np.finfo(float).tiny)
                    ),
                    axis=0,
                )
            recover = tiny_rows[:, None] | unsafe[None, :] | ~np.isfinite(values)
            if np.any(recover):
                from scipy.special import logsumexp

                if log_variance is None:
                    log_variance = _log_variance(z)
                for pair in np.flatnonzero(np.any(recover, axis=0)):
                    rows = np.flatnonzero(recover[:, pair])
                    items = np.flatnonzero(active[:, pair])
                    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                        log_coefficients = np.log(np.abs(a[items, pair])) + np.log(
                            np.abs(b[items, pair])
                        )
                        sign = np.sign(a[items, pair]) * np.sign(b[items, pair])
                        log_sum, sum_sign = logsumexp(
                            log_variance[np.ix_(rows, items)] + log_coefficients,
                            b=sign,
                            axis=1,
                            return_sign=True,
                        )
                        values[rows, pair] = sum_sign * np.exp(log_sum)
            result[start:stop, left_factors, right_factors] = values
            result[start:stop, right_factors, left_factors] = values
        # Structural zero pairs still propagate undefined respondent logits.
        result[start:stop][np.any(np.isnan(z), axis=1)] = np.nan
    return result
