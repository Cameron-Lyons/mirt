"""In-place normalization of owned log-likelihood arrays for EM estimation."""

import numpy as np
from numpy.typing import NDArray

_MAX_NORMALIZATION_ELEMENTS = 1_000_000


def normalize_log_posterior(
    log_joint: NDArray[np.float64],
    log_prior_mass: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Consume an owned likelihood buffer and return posterior weights/log marginals.

    Normalize shifted exponentials directly, so very large common log offsets
    cannot cancel the normalizing constant when constructing posterior weights.
    Row blocks bound reduction scratch space; the posterior reuses its input.
    With no prior mass, normalize sample likelihood ratios directly.
    """
    log_marginal = np.empty(log_joint.shape[0], dtype=np.float64)
    block_size = max(1, _MAX_NORMALIZATION_ELEMENTS // log_joint.shape[1])
    for start in range(0, log_joint.shape[0], block_size):
        stop = min(start + block_size, log_joint.shape[0])
        block = log_joint[start:stop]
        offset = np.max(block, axis=1)
        shift = np.where(np.isfinite(offset), offset, 0.0)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            # Center the likelihood before adding the prior, so small prior
            # differences survive a large common likelihood offset.
            block -= shift[:, None]
            if log_prior_mass is None:
                joint_offset = np.zeros(stop - start)
            else:
                block += log_prior_mass[None, :]
                joint_offset = np.max(block, axis=1)
                block -= joint_offset[:, None]
            np.exp(block, out=block)
            total = block.sum(axis=1)
            block /= total[:, None]
            marginal = shift + (joint_offset + np.log(total))
        # Preserve logsumexp's nonfinite-row semantics. Their posteriors remain
        # undefined, while an all-zero likelihood retains a -inf log marginal.
        marginal[np.isposinf(joint_offset)] = np.inf
        marginal[np.isneginf(joint_offset)] = -np.inf
        log_marginal[start:stop] = marginal
    return log_joint, log_marginal
