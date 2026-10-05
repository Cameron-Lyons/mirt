"""EM M-step for item parameters shared by every item.

Rating-scale families give all items common category thresholds, and the
graded rating scale model also a common slope. After the itemwise M-step, the
expected complete-data log-likelihood ``sum_j sum_q sum_c r_jqc log P_jc(q)``
is maximized over the free shared coordinates with the item parameters held
fixed. Each EM iteration is then a conditional maximization, which keeps the
marginal likelihood nondecreasing (Meng and Rubin, 1993).

References
----------
Meng, X.-L., & Rubin, D. B. (1993). Maximum likelihood estimation via the
    ECM algorithm: A general framework. Biometrika, 80(2), 267-278.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint, minimize
from scipy.special import xlogy

from mirt._core import sigmoid
from mirt.estimation.base import _free_shared_parameters, _parameter_bounds

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

_Objective = Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]]
# Smallest gap kept between movable graded thresholds, as for GRM items.
_THRESHOLD_GAP = 1e-6


class SharedParameters:
    """Free shared coordinates of a model packed into one bounded vector.

    Coordinates follow model parameter order and C order within each array.
    """

    def __init__(self, model: BaseItemModel) -> None:
        masks = model.free_parameter_masks
        self.masks = {
            name: np.asarray(masks[name], dtype=np.bool_)
            for name in _free_shared_parameters(model)
        }
        self.bounds: list[tuple[float, float]] = []
        for name, mask in self.masks.items():
            self.bounds.extend(
                [_parameter_bounds(model, name)] * int(np.count_nonzero(mask))
            )
        self.size = len(self.bounds)

    def get(self, model: BaseItemModel) -> NDArray[np.float64]:
        """Return the current free shared coordinates."""
        params = model._parameters
        chunks = [
            model._canonical_parameter_values(name, params[name])[mask]
            for name, mask in self.masks.items()
        ]
        return np.concatenate(chunks) if chunks else np.empty(0)

    def values(
        self, model: BaseItemModel, vector: NDArray[np.float64]
    ) -> dict[str, NDArray[np.float64]]:
        """Return full canonical shared arrays holding ``vector``."""
        result = {}
        offset = 0
        for name, mask in self.masks.items():
            values = model._canonical_parameter_values(name, model._parameters[name])
            count = int(np.count_nonzero(mask))
            values[mask] = vector[offset : offset + count]
            result[name] = model._canonical_parameter_values(name, values)
            offset += count
        return result

    def free(self, gradients: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Select the free coordinates of full-shape shared gradients."""
        chunks = [gradients[name][mask] for name, mask in self.masks.items()]
        return np.concatenate(chunks) if chunks else np.empty(0)


def binary_category_counts(
    correct: NDArray[np.float64], observed: NDArray[np.float64]
) -> list[NDArray[np.float64]]:
    """Return each binary item's (incorrect, correct) expected counts."""
    return [
        np.column_stack((n_k - r_k, r_k))
        for r_k, n_k in zip(correct, observed, strict=True)
    ]


def _rating_scale_objective(
    model: BaseItemModel,
    layout: SharedParameters,
    theta: NDArray[np.float64],
    counts: NDArray[np.float64],
    epsilon: float,
) -> _Objective:
    """Exact objective of RSM thresholds, a common partial-credit step."""
    from mirt.estimation._polytomous_objective import _softmax_loss_and_residual
    from mirt.models.polytomous import _partial_credit_probabilities

    location = theta[:, None, None] - model._parameters["difficulty"][None, :, None]

    def objective(vector: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        thresholds = layout.values(model, vector)["thresholds"]
        probabilities = _partial_credit_probabilities(location - thresholds)
        loss, residual = _softmax_loss_and_residual(
            probabilities, counts, epsilon, 1.0 - epsilon
        )
        # Threshold v lowers the logit of every category above v.
        tails = np.cumsum(residual[..., :0:-1], axis=-1)[..., ::-1]
        gradient = {"thresholds": -tails.sum(axis=(0, 1))}
        return loss, layout.free(gradient)

    return objective


def _graded_rating_scale_objective(
    model: BaseItemModel,
    layout: SharedParameters,
    theta: NDArray[np.float64],
    counts: NDArray[np.float64],
    epsilon: float,
) -> _Objective:
    """Exact objective of the common GRSM slope and thresholds."""
    from mirt.estimation._polytomous_objective import _clipped_loss_and_counts
    from mirt.models.polytomous import _graded_probabilities

    location = theta[:, None, None] - model._parameters["difficulty"][None, :, None]

    def objective(vector: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        values = {**model._parameters, **layout.values(model, vector)}
        slope = float(values["discrimination"][0])
        centered = location - values["thresholds"]
        logits = slope * centered
        probabilities = _graded_probabilities(logits)
        loss, effective = _clipped_loss_and_counts(
            probabilities, counts, epsilon, 1.0 - epsilon
        )
        score = np.divide(
            effective,
            probabilities,
            out=np.zeros_like(probabilities),
            where=effective != 0.0,
        )
        cumulative = sigmoid(logits)
        common = (score[..., :-1] - score[..., 1:]) * cumulative * (1.0 - cumulative)
        gradient = {
            "discrimination": np.array([np.sum(common * centered)]),
            "thresholds": -slope * common.sum(axis=(0, 1)),
        }
        return loss, layout.free(gradient)

    return objective


def _numerical_objective(
    model: BaseItemModel,
    layout: SharedParameters,
    nodes: NDArray[np.float64],
    counts: Sequence[NDArray[np.float64]],
    epsilon: float,
) -> Callable[[NDArray[np.float64]], float]:
    """Objective of any item model from its all-item probabilities.

    Trial values are written to the model's storage; the caller restores it.
    """

    def objective(vector: NDArray[np.float64]) -> float:
        model._parameters.update(layout.values(model, vector))
        probabilities = np.asarray(model.probability(nodes), dtype=np.float64)
        if probabilities.ndim == 2:
            probabilities = np.stack((1.0 - probabilities, probabilities), axis=-1)
        loss = 0.0
        for item, item_counts in enumerate(counts):
            item_probabilities = probabilities[:, item, : item_counts.shape[1]]
            loss -= float(
                np.sum(
                    xlogy(
                        item_counts, np.clip(item_probabilities, epsilon, 1 - epsilon)
                    )
                )
            )
        return loss

    return objective


def _ordered_threshold_constraint(
    model: BaseItemModel, layout: SharedParameters
) -> LinearConstraint | None:
    """Keep GRSM thresholds ordered in the free shared layout."""
    from mirt.models.polytomous import GradedRatingScaleModel

    mask = layout.masks.get("thresholds")
    if not isinstance(model, GradedRatingScaleModel) or mask is None:
        return None
    offset = 0
    for name, other in layout.masks.items():
        if name == "thresholds":
            break
        offset += int(np.count_nonzero(other))
    positions = {
        int(index): offset + column for column, index in enumerate(np.flatnonzero(mask))
    }
    values = model._parameters["thresholds"]
    rows = []
    lower = []
    for first in range(values.size - 1):
        row = np.zeros(layout.size)
        fixed_difference = 0.0
        for index, sign in ((first, -1.0), (first + 1, 1.0)):
            if index in positions:
                row[positions[index]] = sign
            else:
                fixed_difference += sign * values[index]
        if np.any(row):
            rows.append(row)
            lower.append(_THRESHOLD_GAP - fixed_difference)
    if not rows:
        return None
    return LinearConstraint(np.asarray(rows), np.asarray(lower), np.inf)


def optimize_shared_parameters(
    model: BaseItemModel,
    nodes: NDArray[np.float64],
    counts: Sequence[NDArray[np.float64]],
    *,
    epsilon: float,
    max_iter: int,
    ftol: float,
) -> None:
    """Maximize the expected complete-data log-likelihood over shared coordinates.

    Parameters
    ----------
    model : BaseItemModel
        Model updated in place. Item parameters stay fixed.
    nodes : ndarray of shape (n_points, n_factors)
        Quadrature nodes.
    counts : sequence of ndarray
        Expected category counts of each item, shape ``(n_points, C_j)``. A
        dichotomous item has columns for incorrect and correct responses.
    epsilon : float
        Probability clipping bound of the item objectives.
    max_iter : int
        Iteration limit of the optimizer.
    ftol : float
        Relative function-change tolerance of the optimizer.
    """
    from mirt.models.polytomous import (
        GradedRatingScaleModel,
        RatingScaleModel,
        _uses_authored_rating_scale_hooks,
    )

    layout = SharedParameters(model)
    if not layout.size:
        return
    exact = {
        RatingScaleModel: _rating_scale_objective,
        GradedRatingScaleModel: _graded_rating_scale_objective,
    }.get(type(model))
    analytic = exact is not None and _uses_authored_rating_scale_hooks(model)
    objective: _Objective | Callable[[NDArray[np.float64]], float]
    if exact is not None and analytic:
        # Rating-scale items share one category count.
        stacked = np.stack(counts, axis=1)
        objective = exact(model, layout, nodes[:, 0], stacked, epsilon)
    else:
        objective = _numerical_objective(model, layout, nodes, counts, epsilon)

    def value(vector: NDArray[np.float64]) -> float:
        result = objective(vector)
        return float(result[0] if isinstance(result, tuple) else result)

    start = layout.get(model)
    constraint = _ordered_threshold_constraint(model, layout)
    original = {name: model._parameters[name] for name in layout.masks}
    try:
        candidate = minimize(
            objective,
            x0=start,
            method="SLSQP" if constraint is not None else "L-BFGS-B",
            jac=analytic,
            bounds=layout.bounds,
            options={"maxiter": max_iter, "ftol": ftol},
            constraints=() if constraint is None else (constraint,),
        ).x
        # A conditional maximization step must not decrease the objective.
        accepted = (
            np.all(np.isfinite(candidate))
            and (
                constraint is None
                or np.all(constraint.A @ candidate >= constraint.lb - 1e-8)
            )
            and value(candidate) <= value(start)
        )
    finally:
        model._parameters.update(original)
    if accepted:
        model.set_parameters(**layout.values(model, candidate))
