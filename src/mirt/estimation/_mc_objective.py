"""Bounded shared item kernels for person-specific Monte Carlo samples."""

from collections.abc import Callable, Iterator

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON
from mirt.estimation._affine_objective import prepare_affine_objective
from mirt.estimation._dichotomous_objective import prepare_dichotomous_objective
from mirt.estimation._polytomous_objective import prepare_polytomous_objective
from mirt.models.base import BaseItemModel

_MAX_MC_OBJECTIVE_ENTRIES = 32_768
_Objective = Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]]


def _item_kernel(
    model: BaseItemModel,
    item: int,
    theta: NDArray[np.float64],
    responses: NDArray[np.int_],
    weights: NDArray[np.float64],
    bounds: list[tuple[float, float]],
) -> _Objective | None:
    if model.is_polytomous:
        counts = np.zeros((len(theta), model.n_categories[item]))
        counts[np.arange(len(theta)), responses] = weights
        return prepare_polytomous_objective(
            model, item, theta, counts, PROB_EPSILON, max_probability=1.0
        )
    correct = weights * responses
    objective = prepare_dichotomous_objective(
        model, item, theta, weights, correct, PROB_EPSILON, bounds
    )
    if objective is None:
        objective = prepare_affine_objective(
            model, item, theta, weights, correct, PROB_EPSILON, bounds
        )
    return objective


def prepare_mc_objective(
    model: BaseItemModel,
    item: int,
    responses: NDArray[np.int_],
    theta_samples: NDArray[np.float64],
    weights: NDArray[np.float64],
    n_samples: int,
    bounds: list[tuple[float, float]],
) -> _Objective | None:
    """Cache a small item kernel or stream bounded kernels for larger draws.

    Only observed persons are selected. Preparation and validation happen
    outside optimizer trials; larger draws regenerate one bounded block at a
    time instead of retaining a full observed-person sample copy.
    """
    probe = _item_kernel(
        model,
        item,
        np.zeros((1, model.n_factors)),
        np.zeros(1, dtype=np.intp),
        np.zeros(1),
        bounds,
    )
    if probe is None:
        return None
    n_persons = len(responses)
    if theta_samples.shape != (n_persons, n_samples, model.n_factors):
        raise ValueError("theta_samples has an incompatible shape")
    if weights.shape != (n_persons, n_samples):
        raise ValueError("weights has an incompatible shape")
    rows = np.flatnonzero(responses >= 0)
    categories = model.n_categories[item] if model.is_polytomous else 2
    if model.is_polytomous and np.any(responses[rows] >= categories):
        raise ValueError("model returned invalid item category probabilities")
    max_points = max(
        1, _MAX_MC_OBJECTIVE_ENTRIES // max(model.n_factors, categories, 2)
    )
    sample_chunk = min(n_samples, max_points)
    row_chunk = max(1, max_points // sample_chunk)

    def blocks() -> Iterator[
        tuple[NDArray[np.float64], NDArray[np.int_], NDArray[np.float64]]
    ]:
        for first in range(0, len(rows), row_chunk):
            selected = rows[first : first + row_chunk]
            for start in range(0, n_samples, sample_chunk):
                stop = min(start + sample_chunk, n_samples)
                theta = theta_samples[selected, start:stop].reshape(-1, model.n_factors)
                decisions = np.repeat(responses[selected], stop - start)
                observed = weights[selected, start:stop].reshape(-1)
                yield theta, decisions, observed
                del theta, decisions, observed

    cached: _Objective | None = None
    small = len(rows) * n_samples <= max_points
    for theta, decisions, observed in blocks():
        if not np.all(np.isfinite(observed)) or np.any(observed < 0.0):
            raise ValueError("weights must be finite and non-negative")
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta_samples must contain only finite observed values")
        if small:
            cached = _item_kernel(model, item, theta, decisions, observed, bounds)
        del theta, decisions, observed
    if cached is not None:
        return cached

    def objective(params: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        loss = 0.0
        gradient = np.zeros_like(params)
        for theta, decisions, observed in blocks():
            kernel = _item_kernel(model, item, theta, decisions, observed, bounds)
            if kernel is None:
                raise RuntimeError(
                    "Prepared Monte Carlo item no longer supports gradients"
                )
            block_loss, block_gradient = kernel(params)
            loss += block_loss
            gradient += block_gradient
            del kernel, theta, decisions, observed
        return loss, gradient

    return objective
