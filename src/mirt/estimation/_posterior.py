"""In-place normalization of owned log-likelihood arrays for EM estimation."""

import numpy as np
from numpy.typing import NDArray

from mirt.backends.rust._helpers import mirt_rs, rust_enabled

# Row blocks of at most this many entries, with their scratch copy, stay in
# cache across the passes over them.
_MAX_NORMALIZATION_ELEMENTS = 131_072
# Rows whose joint mass, scaled by the largest prior mass, falls below this
# are normalized again from their own maximum, so that no term underflows
# relative to it.
_MIN_SCALED_TOTAL = 1e-100


def normalize_log_posterior(
    log_joint: NDArray[np.float64],
    log_prior_mass: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Consume an owned likelihood buffer and return posterior weights/log marginals.

    Normalize shifted exponentials directly, so very large common log offsets
    cannot cancel the normalizing constant when constructing posterior weights.
    Each likelihood row is centered at its maximum before the prior is added,
    so small prior differences survive a large common likelihood offset. The
    joint is then shifted by the largest prior mass rather than by each row's
    own maximum, which saves two passes over the array; rows whose shifted
    mass underflows, and nonfinite rows, are normalized again from their own
    maxima. Row blocks bound scratch space; the posterior reuses its input.
    With no prior mass, normalize sample likelihood ratios directly.

    A writeable C-contiguous float64 buffer is normalized by a native kernel
    when the Rust backend is enabled. It performs the same operations row by
    row, in parallel, so its posteriors and log marginals, including those of
    underflowing and nonfinite rows, equal the NumPy path's up to rounding.
    """
    native = _native_normalization(log_joint, log_prior_mass)
    if native is not None:
        return log_joint, native
    n_rows, n_points = log_joint.shape
    log_marginal = np.empty(n_rows, dtype=np.float64)
    prior: NDArray[np.float64] | None = None
    peak = 0.0
    if log_prior_mass is not None:
        peak = float(np.max(log_prior_mass))
        if np.isfinite(peak):
            prior = log_prior_mass - peak
    block_size = max(1, _MAX_NORMALIZATION_ELEMENTS // n_points)
    scratch: NDArray[np.float64] | None = None
    for start in range(0, n_rows, block_size):
        stop = min(start + block_size, n_rows)
        block = log_joint[start:stop]
        offset = np.max(block, axis=1)
        shift = np.where(np.isfinite(offset), offset, 0.0)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            block -= shift[:, None]
            if log_prior_mass is None:
                np.exp(block, out=block)
                total = block.sum(axis=1)
                block /= total[:, None]
                log_marginal[start:stop] = shift + np.log(total)
                continue
            if prior is None:
                block += log_prior_mass[None, :]
                log_marginal[start:stop] = shift + _normalize_rows(block)
                continue
            block += prior[None, :]
            if scratch is None:
                scratch = np.empty((min(block_size, n_rows), n_points))
                ones = np.ones(n_points)
            work = scratch[: stop - start]
            np.exp(block, out=work)
            # A matrix-vector product sums the nonnegative rows fastest.
            total = work @ ones
            marginal = shift + (peak + np.log(total))
            redo = np.flatnonzero(~(np.isfinite(total) & (total >= _MIN_SCALED_TOTAL)))
            rows = block[redo]
            if redo.size:
                marginal[redo] = shift[redo] + (peak + _normalize_rows(rows))
            np.multiply(work, (1.0 / total)[:, None], out=block)
            block[redo] = rows
        log_marginal[start:stop] = marginal
    return log_joint, log_marginal


def _native_normalization(
    log_joint: NDArray[np.float64],
    log_prior_mass: NDArray[np.float64] | None,
) -> NDArray[np.float64] | None:
    """Normalize ``log_joint`` in place natively; None when that is unavailable."""
    if (
        not rust_enabled()
        or not isinstance(log_joint, np.ndarray)
        or log_joint.dtype != np.float64
        or log_joint.ndim != 2
        or log_joint.shape[1] == 0
        or not log_joint.flags.c_contiguous
        or not log_joint.flags.writeable
    ):
        return None
    prior = None
    if log_prior_mass is not None:
        prior = np.ascontiguousarray(log_prior_mass, dtype=np.float64)
        if prior.shape != (log_joint.shape[1],):
            return None
    return mirt_rs.normalize_log_posterior_inplace(log_joint, prior)


def _normalize_rows(block: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize log-joint rows in place from their maxima; return log totals.

    Nonfinite rows keep logsumexp's semantics: their posteriors are
    undefined, while an all-zero joint retains a ``-inf`` log total.
    """
    joint_offset = np.max(block, axis=1)
    block -= joint_offset[:, None]
    np.exp(block, out=block)
    total = block.sum(axis=1)
    block /= total[:, None]
    log_total = joint_offset + np.log(total)
    log_total[np.isposinf(joint_offset)] = np.inf
    log_total[np.isneginf(joint_offset)] = -np.inf
    return log_total
