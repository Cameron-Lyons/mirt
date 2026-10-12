"""SQUAREM state of a multigroup EM fit: free item coordinates and latent moments."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.estimation._acceleration import _keeps_threshold_order

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.multigroup.latent import MultigroupLatentDensity
    from mirt.multigroup.model import MultigroupModel

# Boxes of the latent coordinates: log-variances as in the ordered latent
# update, and Fisher-z correlations below 0.9999 in magnitude.
_LOG_VARIANCE_BOUND = 14.0
_FISHER_Z_BOUND = 5.0


class MultigroupParameters:
    """Pack every group's free item coordinates and latent moments in one vector.

    Item coordinates come first, in parameter order and C order within each
    array: the free coordinates of shared parameters once, then each group's
    free group-specific coordinates. Linked groups therefore move together,
    as synchronization keeps them. Coordinates that the fit holds fixed,
    such as fixed anchors, padding and the group-specific parameters of
    unadministered items, are left out. Latent coordinates follow, group by
    group: estimated means, the logarithms of estimated variances and the
    Fisher-z transforms of estimated correlations, so every point of the
    box is a valid variance and every correlation lies in ``(-1, 1)``.

    Parameters
    ----------
    model : MultigroupModel
        Model whose invariance links and fixed parameters are applied.
    density : MultigroupLatentDensity
        Latent distributions whose ``estimate_*`` flags and correlation
        masks are set.
    bound : callable
        Maps a group model and a parameter name to the optimizer box.
    mean_order : sequence of int, optional
        Groups in nondecreasing latent-mean order; points that break it are
        rejected.
    """

    def __init__(
        self,
        model: MultigroupModel,
        density: MultigroupLatentDensity,
        bound: Callable[[BaseItemModel, str], tuple[float, float]],
        mean_order: Sequence[int] | None = None,
    ) -> None:
        self._model = model
        self._density = density
        self._mean_order = None if mean_order is None else list(mean_order)
        groups = model.group_models
        masks = [model.effective_free_parameter_masks(g) for g in range(len(groups))]
        lower: list[NDArray[np.float64]] = []
        upper: list[NDArray[np.float64]] = []
        # Shared blocks: (name, mask); group blocks: (group, name, mask).
        self._shared: list[tuple[str, NDArray[np.bool_]]] = []
        self._specific: list[tuple[int, str, NDArray[np.bool_]]] = []
        for name in model.parameter_names:
            shared = model._shared_coordinate_mask(name)
            free = shared & np.logical_or.reduce([mask[name] for mask in masks])
            if np.any(free):
                self._shared.append((name, free))
                self._extend(lower, upper, bound(groups[0], name), free)
        for g, group in enumerate(groups):
            for name in model.parameter_names:
                free = masks[g][name] & ~model._shared_coordinate_mask(name)
                if np.any(free):
                    self._specific.append((g, name, free))
                    self._extend(lower, upper, bound(group, name), free)
        for distribution in density.distributions:
            if distribution.estimate_mean:
                size = distribution.n_factors
                lower.append(np.full(size, -np.inf))
                upper.append(np.full(size, np.inf))
            free = distribution.free_covariance_entries()
            if free is None:
                continue
            diagonal = int(np.count_nonzero(np.diag(free)))
            off_diagonal = int(np.count_nonzero(np.triu(free, 1)))
            lower.append(np.full(diagonal, -_LOG_VARIANCE_BOUND))
            upper.append(np.full(diagonal, _LOG_VARIANCE_BOUND))
            lower.append(np.full(off_diagonal, -_FISHER_Z_BOUND))
            upper.append(np.full(off_diagonal, _FISHER_Z_BOUND))
        self.lower = np.concatenate(lower) if lower else np.empty(0)
        self.upper = np.concatenate(upper) if upper else np.empty(0)

    @staticmethod
    def _extend(
        lower: list[NDArray[np.float64]],
        upper: list[NDArray[np.float64]],
        box: tuple[float, float],
        mask: NDArray[np.bool_],
    ) -> None:
        count = int(np.count_nonzero(mask))
        lower.append(np.full(count, box[0], dtype=np.float64))
        upper.append(np.full(count, box[1], dtype=np.float64))

    def get(self) -> NDArray[np.float64]:
        """Return the current coordinates."""
        groups = self._model.group_models
        chunks = [groups[0].parameters[name][free] for name, free in self._shared]
        chunks += [groups[g].parameters[name][free] for g, name, free in self._specific]
        for distribution in self._density.distributions:
            if distribution.estimate_mean:
                chunks.append(np.asarray(distribution.mean, dtype=np.float64))
            free = distribution.free_covariance_entries()
            if free is None:
                continue
            covariance = np.asarray(distribution.cov, dtype=np.float64)
            variances = np.diag(covariance)
            chunks.append(np.log(variances[np.diag(free)]))
            scale = np.sqrt(variances)
            correlation = covariance / np.outer(scale, scale)
            chunks.append(np.arctanh(correlation[np.triu(free, 1)]))
        return np.concatenate(chunks) if chunks else np.empty(0)

    def set(self, vector: NDArray[np.float64]) -> bool:
        """Install coordinates, returning ``False`` for invalid points.

        Invalid points leave the model and the latent distributions unchanged:
        graded thresholds that lose the ordering gap of the current values,
        covariances that are not positive definite, and means that break
        ``mean_order``.
        """
        groups = self._model.group_models
        current = [group.parameters for group in groups]
        updates: list[dict[str, NDArray[np.float64]]] = [{} for _ in groups]
        offset = 0

        def take(count: int) -> NDArray[np.float64]:
            nonlocal offset
            values = vector[offset : offset + count]
            offset += count
            return values

        for name, free in self._shared:
            values = take(int(np.count_nonzero(free)))
            for g in range(len(groups)):
                array = updates[g].setdefault(name, current[g][name].copy())
                array[free] = values
        for g, name, free in self._specific:
            array = updates[g].setdefault(name, current[g][name].copy())
            array[free] = take(int(np.count_nonzero(free)))
        moments = []
        for distribution in self._density.distributions:
            mean = np.array(distribution.mean, dtype=np.float64)
            covariance = np.array(distribution.cov, dtype=np.float64)
            if distribution.estimate_mean:
                mean = take(mean.size).copy()
            free = distribution.free_covariance_entries()
            if free is not None:
                variances = np.diag(covariance).copy()
                variances[np.diag(free)] = np.exp(take(int(np.diag(free).sum())))
                scale = np.sqrt(variances)
                upper_free = np.triu(free, 1)
                correlations = np.tanh(take(int(np.count_nonzero(upper_free))))
                updated = covariance.copy()
                updated[upper_free] = correlations * np.outer(scale, scale)[upper_free]
                updated = np.triu(updated, 1)
                updated = updated + updated.T + np.diag(variances)
                # Covariances held fixed keep their values.
                held = ~(free | free.T) & ~np.eye(len(variances), dtype=np.bool_)
                updated[held] = covariance[held]
                covariance = updated
            moments.append((mean, covariance))
        if self._mean_order is not None:
            ordered = np.array([moments[g][0][0] for g in self._mean_order])
            if np.any(np.diff(ordered) < 0.0):
                return False
        if not all(
            _keeps_threshold_order(group, update)
            for group, update in zip(groups, updates, strict=True)
        ):
            return False
        distributions = self._density.distributions
        saved = [(d.mean.copy(), d.cov.copy()) for d in distributions]
        try:
            for distribution, (mean, covariance) in zip(
                distributions, moments, strict=True
            ):
                if distribution.estimate_mean or (
                    distribution.free_covariance_entries() is not None
                ):
                    distribution._set_parameters(mean, covariance)
            for group, update in zip(groups, updates, strict=True):
                if update:
                    group.set_parameters(**update)
        except ValueError:
            for group, parameters in zip(groups, current, strict=True):
                group.set_parameters(**parameters)
            for distribution, (mean, covariance) in zip(
                distributions, saved, strict=True
            ):
                distribution._set_parameters(mean, covariance)
            return False
        return True
