"""Observed-information standard errors of multigroup fits.

The joint log-likelihood is the sum of the groups' marginal log-likelihoods,
each a function of that group's item parameters and normal latent moments.
Its observed information is therefore ``sum_g J_g' I_g J_g``, where ``I_g``
is group ``g``'s observed information in its own coordinates and the 0/1
matrix ``J_g`` maps the joint free vector (shared item coordinates once,
group-specific coordinates per group, then the estimated latent moments) to
them. The item block of ``I_g`` is the exact Louis (1982) information with
the group's own prior masses (see
:func:`mirt.estimation.standard_errors.estimate_covariance`). Its latent
columns are central differences of the exact marginal score in the latent
moments, as in Oakes' (1999) identity: the item score comes from Fisher's
identity and the latent score from the derivative of the normalized
quadrature masses.

References
----------
Louis, T. A. (1982). Finding the observed information matrix when using the
    EM algorithm. Journal of the Royal Statistical Society B, 44, 226-233.
Oakes, D. (1999). Direct calculation of the information matrix via the EM
    algorithm. Journal of the Royal Statistical Society B, 61, 479-482.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.utils.numeric import logsumexp

if TYPE_CHECKING:
    from mirt.estimation.quadrature import GaussHermiteQuadrature
    from mirt.models.base import BaseItemModel
    from mirt.multigroup.latent import GroupLatentDistribution, MultigroupLatentDensity
    from mirt.multigroup.model import MultigroupModel

# Relative central-difference step in the latent means and covariances.
_LATENT_STEP = 1e-4
# Step of the marginal-likelihood differences for models without exact
# item derivatives, as in single-group EM.
_ITEM_STEP = 1e-5


@dataclass(frozen=True)
class MultigroupCovariance:
    """Standard errors and covariance of a multigroup fit's free parameters.

    Attributes
    ----------
    standard_errors : dict
        Parameter name -> group index -> standard errors in the stored
        parameter shape; ``NaN`` for coordinates that are not estimated.
    latent_standard_errors : list of dict
        Per group, ``"mean"`` and ``"cov"`` standard errors in the shapes of
        the latent mean and covariance; ``NaN`` for fixed moments.
    covariance : ndarray of shape (P, P)
        Covariance of the joint free vector described by ``labels``.
    labels : list of str
        Name of each coordinate: ``"discrimination[Item_1]"`` for a shared
        coordinate, ``"G2: difficulty[Item_1]"`` for a group-specific one
        and ``"G2: mean[0]"`` or ``"G2: cov[0,0]"`` for latent moments.
    """

    standard_errors: dict[str, dict[int, NDArray[np.float64]]]
    latent_standard_errors: list[dict[str, NDArray[np.float64]]]
    covariance: NDArray[np.float64]
    labels: list[str]


def _coordinate_label(
    model: MultigroupModel, name: str, flat: int, prefix: str = ""
) -> str:
    shape = model.get_group_model(0).parameters[name].shape
    index = np.unravel_index(flat, shape)
    if model._parameter_is_item_major(name):
        parts = [model.item_names[int(index[0])], *(str(int(i)) for i in index[1:])]
    else:
        parts = [str(int(i)) for i in index]
    return f"{prefix}{name}[{', '.join(parts)}]"


def _latent_coordinates(
    distribution: GroupLatentDistribution,
) -> list[tuple[str, tuple[int, ...]]]:
    """Return the estimated means and upper covariance entries of a group."""
    coordinates: list[tuple[str, tuple[int, ...]]] = []
    if distribution.estimate_mean:
        coordinates += [("mean", (f,)) for f in range(distribution.n_factors)]
    free = distribution.free_covariance_entries()
    if free is not None:
        rows, columns = np.nonzero(np.triu(free))
        coordinates += [
            ("cov", (int(i), int(j))) for i, j in zip(rows, columns, strict=True)
        ]
    return coordinates


def _perturbed(
    distribution: GroupLatentDistribution,
    coordinate: tuple[str, tuple[int, ...]],
    step: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mean = np.array(distribution.mean, dtype=np.float64)
    covariance = np.array(distribution.cov, dtype=np.float64)
    kind, index = coordinate
    if kind == "mean":
        mean[index] += step
    else:
        i, j = index
        covariance[i, j] += step
        if i != j:
            covariance[j, i] += step
    return mean, covariance


def _latent_score(
    log_likelihood: NDArray[np.float64],
    nodes: NDArray[np.float64],
    log_weights: NDArray[np.float64],
    mean: NDArray[np.float64],
    covariance: NDArray[np.float64],
    coordinates: list[tuple[str, tuple[int, ...]]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the prior masses and the marginal score in the latent moments.

    Group masses are ``w_q phi(theta_q) / phi_0(theta_q)``, normalized, for
    the grid weights ``w`` and the standard normal ``phi_0``. By Fisher's
    identity the score is ``sum_q n_q d log pi_q``, with expected node counts
    ``n_q`` and ``d log pi_q = d log phi(theta_q) - sum_r pi_r d log
    phi(theta_r)``.
    """
    precision = np.linalg.inv(covariance)
    difference = nodes - mean
    solved = difference @ precision
    _, log_determinant = np.linalg.slogdet(covariance)
    log_density = -0.5 * (log_determinant + np.sum(difference * solved, axis=1))
    log_mass = log_weights + log_density + 0.5 * np.sum(nodes**2, axis=1)
    log_mass -= logsumexp(log_mass)
    mass = np.exp(log_mass)
    log_joint = log_likelihood + log_mass[None, :]
    counts = np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True)).sum(0)
    derivatives = np.empty((nodes.shape[0], len(coordinates)))
    for column, (kind, index) in enumerate(coordinates):
        if kind == "mean":
            derivatives[:, column] = solved[:, index[0]]
            continue
        i, j = index
        # d log phi / d Sigma = (P d d' P - P) / 2 for a symmetric entry pair.
        curvature = solved[:, i] * solved[:, j] - precision[i, j]
        derivatives[:, column] = curvature if i != j else 0.5 * curvature
    centered = derivatives - mass @ derivatives
    return mass, counts @ centered


def multigroup_covariance(
    model: MultigroupModel,
    responses: list[NDArray[np.int_]],
    quadrature: GaussHermiteQuadrature,
    density: MultigroupLatentDensity,
    bound: Callable[[BaseItemModel, str], tuple[float, float]],
) -> MultigroupCovariance:
    """Estimate the observed-information covariance of a fitted multigroup model.

    Parameters
    ----------
    model : MultigroupModel
        Fitted model with its invariance links and fixed parameters.
    responses : list of ndarray
        Each group's responses, as fitted.
    quadrature : GaussHermiteQuadrature or RectangularQuadrature
        Final quadrature grid of the fit.
    density : MultigroupLatentDensity
        Fitted latent distributions.
    bound : callable
        Maps a group model and a parameter name to the optimizer box. Free
        item coordinates on a bound are held fixed and get ``NaN`` errors.

    Returns
    -------
    MultigroupCovariance
        Standard errors and the joint free-parameter covariance.
    """
    from mirt.estimation.standard_errors import (
        _BOUND_TOLERANCE,
        _covariance,
        _information_and_meat,
        _score_and_information,
    )

    groups = model.group_models
    labels_of_groups = model.group_labels
    masks = [model.effective_free_parameter_masks(g) for g in range(len(groups))]
    columns: dict[tuple[int, str, int], int] = {}
    labels: list[str] = []
    held: list[bool] = []

    def add(label: str, at_bound: bool) -> int:
        labels.append(label)
        held.append(at_bound)
        return len(labels) - 1

    def on_bound(group: BaseItemModel, name: str, flat: int) -> bool:
        value = float(np.ravel(group.parameters[name])[flat])
        low, high = bound(group, name)
        return min(abs(value - low), abs(value - high)) <= _BOUND_TOLERANCE

    anchors = [_rotation_anchors(group, masks[g]) for g, group in enumerate(groups)]
    for name in model.parameter_names:
        shared = model._shared_coordinate_mask(name)
        free = shared & np.logical_or.reduce([mask[name] for mask in masks])
        for flat in np.flatnonzero(free.ravel()):
            fixed = on_bound(groups[0], name, int(flat)) or any(
                name in anchor and anchor[name].ravel()[flat] for anchor in anchors
            )
            column = add(_coordinate_label(model, name, int(flat)), fixed)
            for g in range(len(groups)):
                if masks[g][name].ravel()[flat]:
                    columns[(g, name, int(flat))] = column
    for g, group in enumerate(groups):
        prefix = f"{labels_of_groups[g]}: "
        for name in model.parameter_names:
            specific = masks[g][name] & ~model._shared_coordinate_mask(name)
            for flat in np.flatnonzero(specific.ravel()):
                fixed = on_bound(group, name, int(flat)) or bool(
                    name in anchors[g] and anchors[g][name].ravel()[flat]
                )
                label = _coordinate_label(model, name, int(flat), prefix)
                columns[(g, name, int(flat))] = add(label, fixed)
    latent_columns: list[list[int]] = []
    for g, distribution in enumerate(density.distributions):
        prefix = f"{labels_of_groups[g]}: "
        latent_columns.append(
            [
                add(f"{prefix}{kind}[{','.join(map(str, index))}]", False)
                for kind, index in _latent_coordinates(distribution)
            ]
        )

    size = len(labels)
    information = np.zeros((size, size))
    nodes = quadrature.nodes
    log_weights = np.log(np.maximum(quadrature.weights, np.finfo(float).tiny))
    for g, (group, data, distribution) in enumerate(
        zip(groups, responses, density.distributions, strict=True)
    ):
        mass = np.exp(density.log_quadrature_mass(nodes, quadrature.weights, g))
        item_information, _, layouts = _information_and_meat(
            group, data, quadrature, mass, _ITEM_STEP
        )
        assert item_information is not None
        local = np.array(
            [
                columns.get((g, name, int(flat)), -1)
                for name, layout in layouts.items()
                for flat in layout.free_indices
            ],
            dtype=np.intp,
        )
        kept = np.flatnonzero(local >= 0)
        information[np.ix_(local[kept], local[kept])] += item_information[
            np.ix_(kept, kept)
        ]
        coordinates = _latent_coordinates(distribution)
        if not coordinates:
            continue
        # Latent columns: central differences of the item and latent scores.
        log_likelihood = np.asarray(
            group.log_likelihood_batch(data, nodes), dtype=np.float64
        )
        derivative = np.empty((local.size + len(coordinates), len(coordinates)))
        for column, coordinate in enumerate(coordinates):
            kind, index = coordinate
            value = (
                distribution.mean[index[0]]
                if kind == "mean"
                else distribution.cov[index]
            )
            step = _LATENT_STEP * max(1.0, abs(float(value)))
            scores = []
            for sign in (1.0, -1.0):
                mean, covariance = _perturbed(distribution, coordinate, sign * step)
                perturbed_mass, latent = _latent_score(
                    log_likelihood, nodes, log_weights, mean, covariance, coordinates
                )
                item_score, _, _ = _score_and_information(
                    group, data, quadrature, perturbed_mass, _ITEM_STEP, observed=False
                )
                scores.append(np.concatenate([item_score, latent]))
            derivative[:, column] = (scores[0] - scores[1]) / (2.0 * step)
        own = np.asarray(latent_columns[g], dtype=np.intp)
        block = -derivative
        cross = block[: local.size][kept]
        information[np.ix_(local[kept], own)] += cross
        information[np.ix_(own, local[kept])] += cross.T
        information[np.ix_(own, own)] += block[local.size :]

    information = (information + information.T) / 2.0
    active = ~np.asarray(held, dtype=np.bool_)
    covariance = (
        _covariance("oakes", information, None, active) if size else information
    )
    variances = np.diag(covariance)
    errors = np.full(size, np.nan)
    valid = np.isfinite(variances) & (variances >= 0.0)
    errors[valid] = np.sqrt(variances[valid])

    standard_errors: dict[str, dict[int, NDArray[np.float64]]] = {}
    for name in model.parameter_names:
        shape = groups[0].parameters[name].shape
        standard_errors[name] = {g: np.full(shape, np.nan) for g in range(len(groups))}
    for (g, name, flat), column in columns.items():
        standard_errors[name][g].reshape(-1)[flat] = errors[column]
    latent_errors = []
    for distribution, own_columns in zip(
        density.distributions, latent_columns, strict=True
    ):
        n_factors = distribution.n_factors
        result = {
            "mean": np.full(n_factors, np.nan),
            "cov": np.full((n_factors, n_factors), np.nan),
        }
        for (kind, index), column in zip(
            _latent_coordinates(distribution), own_columns, strict=True
        ):
            if kind == "mean":
                result["mean"][index] = errors[column]
            else:
                i, j = index
                result["cov"][i, j] = result["cov"][j, i] = errors[column]
        latent_errors.append(result)
    return MultigroupCovariance(standard_errors, latent_errors, covariance, labels)


def _rotation_anchors(
    group: BaseItemModel, masks: dict[str, NDArray[np.bool_]]
) -> dict[str, NDArray[np.bool_]]:
    """Return the slopes held to identify an exploratory rotation.

    As in single-group EM, the upper triangle of the first ``F - 1`` fully
    loading items' slopes is held at its estimates, which R mirt fixes at
    zero; models whose loading pattern identifies the rotation hold none.
    """
    name = group._rotatable_slopes
    if name is None or group._rotational_indeterminacy(masks) == 0:
        return {}
    n_factors = group.n_factors
    rows = np.asarray(masks[name], dtype=np.bool_).reshape(-1, n_factors)
    held = np.zeros_like(rows)
    anchored = np.flatnonzero(rows.all(axis=1))[: n_factors - 1]
    for position, row in enumerate(anchored):
        held[row, position + 1 :] = True
    return {name: held.reshape(masks[name].shape)}
