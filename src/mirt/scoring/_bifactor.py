"""EAP scoring of bifactor models by Gibbons-Hedeker dimension reduction.

Given the general factor, the specific factors of a bifactor model are
conditionally independent, so the posterior moments of every factor follow
from one two-dimensional (general by specific) grid per specific factor. The
results equal EAP on the full ``n_quadpts ** (1 + S)`` product grid at the
same number of points per dimension, at a cost linear rather than
exponential in the number of specific factors ``S``. Dichotomous and graded
items enter through the log-probabilities of their categories.

References
----------
Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor
    analysis. Psychometrika, 57(3), 423-436.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.scoring._common import resolve_prior_distribution

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.models.bifactor import BifactorModel, GradedBifactorModel

# Relative size of a specific-factor conditional covariance below which the
# specific factors count as conditionally independent given the general one.
_CONDITIONAL_COVARIANCE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ReducedBifactorGrid:
    """One-dimensional rule and the affine maps of each factor onto it.

    With standard-normal nodes ``z``, the general factor is
    ``mean[0] + general_scale * z_g`` and, given it, specific factor ``k`` is
    ``mean[1 + k] + cross[k] * z_g + scale[k] * z_k``. These are the nodes of
    the Cholesky-transformed product grid that EAP otherwise integrates.
    """

    model: BifactorModel | GradedBifactorModel
    nodes: NDArray[np.float64]
    log_weights: NDArray[np.float64]
    mean: NDArray[np.float64]
    general_scale: float
    cross: NDArray[np.float64]
    scale: NDArray[np.float64]


def reduced_bifactor_grid(
    model: BaseItemModel,
    n_quadpts: int,
    prior_mean: NDArray[np.float64] | None,
    prior_cov: NDArray[np.float64] | None,
) -> ReducedBifactorGrid | None:
    """Return the reduced grid, or ``None`` when the reduction does not apply.

    The reduction needs a built-in :class:`~mirt.models.BifactorModel` or
    :class:`~mirt.models.GradedBifactorModel` and a normal prior under which
    the specific factors are conditionally independent given the general
    factor, such as any diagonal covariance.
    """
    from mirt._model_defaults import uses_builtin_model_hooks
    from mirt.models.bifactor import BifactorModel, GradedBifactorModel

    if not isinstance(
        model, (BifactorModel, GradedBifactorModel)
    ) or not uses_builtin_model_hooks(model, likelihood=True):
        return None
    mean, cov = resolve_prior_distribution(
        n_factors=model.n_factors, prior_mean=prior_mean, prior_cov=prior_cov
    )
    general_scale = float(np.sqrt(cov[0, 0]))
    cross = cov[1:, 0] / general_scale
    conditional = cov[1:, 1:] - np.outer(cross, cross)
    variance = np.diag(conditional).copy()
    off_diagonal = conditional - np.diag(variance)
    if np.any(variance <= 0.0) or np.any(
        np.abs(off_diagonal)
        > _CONDITIONAL_COVARIANCE_TOLERANCE * np.sqrt(np.outer(variance, variance))
    ):
        return None
    quadrature = GaussHermiteQuadrature(n_points=n_quadpts, n_dimensions=1)
    return ReducedBifactorGrid(
        model=model,
        nodes=quadrature.nodes.ravel(),
        log_weights=np.log(quadrature.weights + 1e-300),
        mean=mean,
        general_scale=general_scale,
        cross=cross,
        scale=np.sqrt(variance),
    )


def _log_curves(
    grid: ReducedBifactorGrid,
) -> tuple[NDArray[np.float64], NDArray[np.intp], NDArray[np.intp]]:
    """Return clipped category log-probabilities on the ``(Q * Q)`` grid.

    The table has one row per node, general node first, and one column per
    item category, item-major; ``offsets`` and ``sizes`` locate each item's
    columns. Every specific column of theta holds its own factor's nodes, so
    each item is evaluated on its factor's grid, with the clipping of the
    model's likelihood.
    """
    nodes = grid.nodes
    general = np.repeat(nodes, nodes.size)
    specific = np.tile(nodes, nodes.size)
    theta = np.empty((general.size, grid.model.n_factors))
    theta[:, 0] = grid.mean[0] + grid.general_scale * general
    theta[:, 1:] = (
        grid.mean[1:] + np.outer(general, grid.cross) + np.outer(specific, grid.scale)
    )
    model = grid.model
    if model.is_polytomous:
        sizes = np.asarray(model.n_categories, dtype=np.intp)
        present = np.arange(int(sizes.max())) < sizes[:, None]
        probability = model.probability(theta).transpose(1, 2, 0)[present].T
        table = np.log(np.clip(probability, PROB_EPSILON, 1.0 - PROB_EPSILON))
    else:
        sizes = np.full(model.n_items, 2, dtype=np.intp)
        correct = np.clip(model.probability(theta), PROB_EPSILON, 1.0 - PROB_EPSILON)
        table = np.empty((general.size, 2 * model.n_items))
        table[:, 0::2] = np.log1p(-correct)
        table[:, 1::2] = np.log(correct)
    offsets = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.intp)
    return table, offsets, sizes


def bifactor_eap(
    grid: ReducedBifactorGrid,
    patterns: NDArray[np.int_],
    batch_size: Callable[[int], int],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return EAP estimates and posterior SDs of validated response patterns.

    ``batch_size(n_nodes)`` gives the number of patterns per batch for a
    working set of ``n_nodes`` values per pattern.
    """
    model = grid.model
    nodes = grid.nodes
    n_points = nodes.size
    nodes_squared = nodes**2
    table, offsets, sizes = _log_curves(grid)
    groups = [
        np.flatnonzero(model._specific_factor_indices == factor)
        for factor in range(model.n_specific_factors)
    ]
    # Each factor's category columns and their log-probability rows.
    curves = []
    for items in groups:
        columns = np.concatenate(
            [np.arange(offsets[item], offsets[item] + sizes[item]) for item in items]
        )
        curves.append((columns, table[:, columns].T))
    n_patterns = patterns.shape[0]
    theta = np.empty((n_patterns, model.n_factors))
    variance = np.empty_like(theta)
    rows = batch_size(n_points * n_points + 2 * len(groups) * n_points)
    for start in range(0, n_patterns, rows):
        stop = min(start + rows, n_patterns)
        block = patterns[start:stop]
        n = stop - start
        persons, items = np.nonzero(block >= 0)
        indicators = np.zeros((n, table.shape[1]))
        indicators[persons, offsets[items] + block[persons, items]] = 1.0
        log_general = np.tile(grid.log_weights, (n, 1))
        # Moments of z_k given each general node, per specific factor k.
        first = np.empty((len(groups), n, n_points))
        second = np.empty_like(first)
        for factor, (columns, log_curves) in enumerate(curves):
            log_joint = (indicators[:, columns] @ log_curves).reshape(
                n, n_points, n_points
            )
            log_joint += grid.log_weights
            shift = log_joint.max(axis=2, keepdims=True)
            log_joint -= shift
            conditional = np.exp(log_joint, out=log_joint)
            total = conditional.sum(axis=2)
            log_general += shift[:, :, 0] + np.log(total)
            first[factor] = (conditional @ nodes) / total
            second[factor] = (conditional @ nodes_squared) / total

        shift = log_general.max(axis=1, keepdims=True)
        general = np.exp(log_general - shift)
        total = general.sum(axis=1, keepdims=True)
        if not np.all(np.isfinite(shift)) or not np.all(np.isfinite(total)):
            raise ValueError("model likelihoods must produce finite posterior mass")
        general /= total

        # Mix the conditional moments over the general-factor posterior.
        z_mean = general @ nodes
        z_square = general @ nodes_squared
        mixed_first = np.einsum("bi,kbi->kb", general, first)
        mixed_cross = np.einsum("bi,i,kbi->kb", general, nodes, first)
        mixed_second = np.einsum("bi,kbi->kb", general, second)
        cross = grid.cross[:, None]
        scale = grid.scale[:, None]
        specific_mean = cross * z_mean + scale * mixed_first
        specific_square = (
            cross**2 * z_square
            + 2.0 * cross * scale * mixed_cross
            + scale**2 * mixed_second
        )
        theta[start:stop, 0] = grid.general_scale * z_mean
        variance[start:stop, 0] = grid.general_scale**2 * (z_square - z_mean**2)
        theta[start:stop, 1:] = specific_mean.T
        variance[start:stop, 1:] = (specific_square - specific_mean**2).T

    theta += grid.mean
    return theta, np.sqrt(np.maximum(variance, 0.0))
