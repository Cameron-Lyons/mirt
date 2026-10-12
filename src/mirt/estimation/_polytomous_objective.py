"""Prepared clipped category likelihoods and gradients for built-in items."""

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import xlogy

from mirt._core import sigmoid
from mirt._model_defaults import uses_builtin_model_hooks
from mirt.estimation._graded_order import THRESHOLD_GAP
from mirt.estimation._item_priors import subtract_coordinate_priors
from mirt.estimation._logistic_newton import ItemObjective
from mirt.estimation.priors import Prior
from mirt.models.base import BaseItemModel
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    NominalResponseModel,
    PartialCreditModel,
    _partial_credit_probabilities,
    _stable_softmax,
)

_Objective = Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]]


def _clipped_loss_and_counts(
    probabilities: NDArray[np.float64],
    counts: NDArray[np.float64],
    epsilon: float,
    max_probability: float,
) -> tuple[float, NDArray[np.float64]]:
    """Keep only counts whose category curve has a nonzero clip derivative."""
    active = (probabilities > epsilon) & (probabilities < max_probability)
    loss = -float(
        np.sum(xlogy(counts, np.clip(probabilities, epsilon, max_probability)))
    )
    return loss, np.where(active, counts, 0.0)


def _softmax_loss_and_residual(
    probabilities: NDArray[np.float64],
    counts: NDArray[np.float64],
    epsilon: float,
    max_probability: float,
) -> tuple[float, NDArray[np.float64]]:
    loss, effective = _clipped_loss_and_counts(
        probabilities, counts, epsilon, max_probability
    )
    residual = probabilities * effective.sum(axis=-1, keepdims=True) - effective
    return loss, residual


def prepare_polytomous_objective(
    model: BaseItemModel,
    item_idx: int,
    theta: NDArray[np.float64],
    counts: NDArray[np.float64],
    epsilon: float,
    *,
    max_probability: float | None = None,
) -> _Objective | None:
    """Prepare a pure objective in the core estimator's free-parameter layout.

    Fixed PCM slopes, NRM reference categories, and padding do not enter the
    trial vector. Custom probability, parameter setters, or layouts retain
    their model-based numerical objective. ``max_probability`` preserves an
    estimator's upper clipping convention; the default remains ``1-epsilon``.
    """
    upper = 1.0 - epsilon if max_probability is None else max_probability
    layouts = {
        GradedResponseModel: ("discrimination", "thresholds"),
        GeneralizedPartialCredit: ("discrimination", "steps"),
        PartialCreditModel: ("discrimination", "steps"),
        NominalResponseModel: ("slopes", "intercepts"),
    }
    if (
        tuple(model._parameters) != layouts.get(type(model))
        or not uses_builtin_model_hooks(model)
        or model._free_parameter_restrictions
    ):
        return None

    n_categories = model.n_categories[item_idx]
    n_factors = model.n_factors
    if type(model) is NominalResponseModel:
        n_slopes = (n_categories - 1) * n_factors

        def nominal_objective(
            params: NDArray[np.float64],
        ) -> tuple[float, NDArray[np.float64]]:
            slopes = params[:n_slopes].reshape(n_categories - 1, n_factors)
            logits = np.empty_like(counts)
            logits[:, 0] = 0.0
            logits[:, 1:] = theta @ slopes.T + params[n_slopes:]
            loss, residual = _softmax_loss_and_residual(
                _stable_softmax(logits), counts, epsilon, upper
            )
            gradient = np.empty_like(params)
            gradient[:n_slopes] = (residual[:, 1:].T @ theta).ravel()
            gradient[n_slopes:] = residual[:, 1:].sum(axis=0)
            return loss, gradient

        return nominal_objective

    # A PCM slope is fixed unless the family estimates its common value.
    fixed_slope = (
        type(model) is PartialCreditModel and not model.estimate_discrimination
    )
    n_slopes = 0 if fixed_slope else n_factors
    unidimensional = n_factors == 1
    points = theta[:, 0] if unidimensional else theta

    if type(model) is GradedResponseModel:

        def graded_objective(
            params: NDArray[np.float64],
        ) -> tuple[float, NDArray[np.float64]]:
            a = params[:n_slopes]
            thresholds = params[n_slopes:]
            centered = points[:, None] - thresholds if unidimensional else None
            logits = (
                a[0] * centered
                if unidimensional
                else (theta @ a)[:, None] - a.sum() * thresholds
            )
            cumulative = sigmoid(logits)
            probabilities = np.empty_like(counts)
            probabilities[:, 0] = 1.0 - cumulative[:, 0]
            probabilities[:, 1:-1] = cumulative[:, :-1] - cumulative[:, 1:]
            probabilities[:, -1] = cumulative[:, -1]
            loss, effective = _clipped_loss_and_counts(
                probabilities, counts, epsilon, upper
            )
            score = np.divide(
                effective,
                probabilities,
                out=np.zeros_like(probabilities),
                where=effective != 0.0,
            )
            common = (score[:, :-1] - score[:, 1:]) * cumulative * (1.0 - cumulative)
            gradient = np.empty_like(params)
            if unidimensional:
                gradient[0] = np.sum(common * centered)
            else:
                gradient[:n_slopes] = theta.T @ common.sum(axis=1) - np.sum(
                    common * thresholds
                )
            gradient[n_slopes:] = -a.sum() * common.sum(axis=0)
            return loss, gradient

        return graded_objective

    # A fixed slope is the stored common value: one for the Rasch PCM, and
    # the current estimate while a common-slope step holds it.
    fixed_a = np.array([model._parameters["discrimination"][item_idx]], dtype=float)

    def partial_credit_objective(
        params: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        a = fixed_a if fixed_slope else params[:n_slopes]
        steps = params[n_slopes:]
        scale = a.sum()
        centered = points[:, None] - steps if unidimensional else None
        increments = (
            a[0] * centered if unidimensional else (theta @ a)[:, None] - scale * steps
        )
        loss, residual = _softmax_loss_and_residual(
            _partial_credit_probabilities(increments), counts, epsilon, upper
        )
        tails = np.cumsum(residual[:, :0:-1], axis=1)[:, ::-1]
        gradient = np.empty_like(params)
        if n_slopes:
            if unidimensional:
                gradient[0] = np.sum(tails * centered)
            else:
                gradient[:n_slopes] = theta.T @ tails.sum(axis=1) - np.sum(
                    tails * steps
                )
        gradient[n_slopes:] = -scale * tails.sum(axis=0)
        return loss, gradient

    return partial_credit_objective


def polytomous_items_objective(
    family: Literal["graded", "partial_credit", "nominal"],
    theta: NDArray[np.float64],
    counts: NDArray[np.float64],
    epsilon: float,
    *,
    fixed_slopes: NDArray[np.float64] | None = None,
    priors: Sequence[tuple[slice, Prior]] = (),
) -> ItemObjective:
    """Prepare the clipped category objectives of several items for Fisher scoring.

    The items share a category count. Each objective is that of
    :func:`prepare_polytomous_objective` minus the coordinate log-priors
    ``priors``, in the same free-coordinate layout:

    - ``"graded"``: the slope and the thresholds of a unidimensional graded
      item. Thresholds closer than ``THRESHOLD_GAP`` are infeasible, with an
      infinite objective.
    - ``"partial_credit"``: the slope and the steps of a unidimensional
      generalized partial credit item, or with ``fixed_slopes`` the steps
      alone.
    - ``"nominal"``: the slopes of the non-reference categories, row-major
      over the factors, then their intercepts.

    The curvature is the expected (Fisher) information of the category
    counts, plus the priors' positive negative curvatures.

    Parameters
    ----------
    family : {"graded", "partial_credit", "nominal"}
        Item family, as above.
    theta : ndarray of shape (n_points, n_factors)
        Quadrature nodes.
    counts : ndarray of shape (n_items, n_points, n_categories)
        Expected category counts of the items.
    epsilon : float
        Probability clipping.
    fixed_slopes : ndarray of shape (n_items,), optional
        Slopes that partial credit items hold.
    priors : sequence of (slice, Prior)
        Priors of coordinate columns, as for
        :func:`~mirt.estimation._item_priors.subtract_coordinate_priors`.

    Returns
    -------
    callable
        An objective for :func:`~mirt.estimation._logistic_newton.newton_items`.
    """
    upper = 1.0 - epsilon
    theta = np.asarray(theta, dtype=np.float64)
    points = theta[:, 0]
    n_categories = counts.shape[2]
    totals = counts.sum(axis=2)
    # Category k counts the steps j < k.
    below = np.tri(n_categories, n_categories - 1, k=-1)

    def clipped(
        rows: NDArray[np.intp], probabilities: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        item_counts = counts[rows]
        active = (probabilities > epsilon) & (probabilities < upper)
        loss = -np.sum(
            xlogy(item_counts, np.clip(probabilities, epsilon, upper)), axis=(1, 2)
        )
        return loss, np.where(active, item_counts, 0.0), active

    def information(
        rows: NDArray[np.intp],
        weights: NDArray[np.float64],
        jacobian: NDArray[np.float64],
        centered: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Return sum over nodes and categories of weight * J J' per item."""
        left = jacobian if centered is None else centered
        n_rows, width = len(rows), jacobian.shape[-1]
        weighted = (weights[..., None] * left).reshape(n_rows, -1, width)
        return weighted.transpose(0, 2, 1) @ left.reshape(n_rows, -1, width)

    def softmax_terms(
        rows: NDArray[np.intp],
        x: NDArray[np.float64],
        logits: NDArray[np.float64],
        jacobian: NDArray[np.float64] | None,
    ) -> tuple[
        NDArray[np.float64], NDArray[np.float64] | None, NDArray[np.float64] | None
    ]:
        probabilities = _stable_softmax(logits)
        loss, effective, _ = clipped(rows, probabilities)
        if jacobian is None:
            subtract_coordinate_priors(priors, x, loss)
            return loss, None, None
        residual = probabilities * effective.sum(axis=2, keepdims=True) - effective
        gradient = np.einsum("iqk,iqkw->iw", residual, jacobian)
        mean = np.einsum("iqk,iqkw->iqw", probabilities, jacobian)
        centered = jacobian - mean[:, :, None, :]
        fisher = information(
            rows, totals[rows][:, :, None] * probabilities, jacobian, centered
        )
        subtract_coordinate_priors(priors, x, loss, gradient, fisher)
        return loss, gradient, fisher

    def graded(
        rows: NDArray[np.intp], x: NDArray[np.float64], derivatives: bool
    ) -> tuple[
        NDArray[np.float64], NDArray[np.float64] | None, NDArray[np.float64] | None
    ]:
        a, thresholds = x[:, :1, None], x[:, 1:]
        centered = points[None, :, None] - thresholds[:, None, :]
        cumulative = sigmoid(a * centered)
        probabilities = np.empty(counts[rows].shape)
        probabilities[..., 0] = 1.0 - cumulative[..., 0]
        probabilities[..., 1:-1] = cumulative[..., :-1] - cumulative[..., 1:]
        probabilities[..., -1] = cumulative[..., -1]
        loss, effective, active = clipped(rows, probabilities)
        loss[np.any(np.diff(thresholds, axis=1) < THRESHOLD_GAP, axis=1)] = np.inf
        if not derivatives:
            subtract_coordinate_priors(priors, x, loss)
            return loss, None, None
        score = np.divide(
            effective,
            probabilities,
            out=np.zeros_like(probabilities),
            where=effective != 0.0,
        )
        density = cumulative * (1.0 - cumulative)
        common = (score[..., :-1] - score[..., 1:]) * density
        gradient = np.empty(x.shape)
        gradient[:, 0] = np.sum(common * centered, axis=(1, 2))
        gradient[:, 1:] = -a[:, 0] * common.sum(axis=1)
        # Derivatives of the cumulative curves P*_0 = 1, ..., P*_K = 0, whose
        # differences are the category probabilities.
        steps = np.arange(n_categories - 1)
        star = np.zeros((*probabilities.shape[:2], n_categories + 1, x.shape[1]))
        star[:, :, 1:-1, 0] = density * centered
        star[:, :, steps + 1, steps + 1] = -a * density
        jacobian = star[:, :, :-1] - star[:, :, 1:]
        weights = np.divide(
            totals[rows][:, :, None],
            probabilities,
            out=np.zeros_like(probabilities),
            where=active,
        )
        fisher = information(rows, weights, jacobian)
        subtract_coordinate_priors(priors, x, loss, gradient, fisher)
        return loss, gradient, fisher

    def partial_credit(
        rows: NDArray[np.intp], x: NDArray[np.float64], derivatives: bool
    ) -> tuple[
        NDArray[np.float64], NDArray[np.float64] | None, NDArray[np.float64] | None
    ]:
        if fixed_slopes is None:
            a, steps = x[:, :1, None], x[:, 1:]
        else:
            a, steps = fixed_slopes[rows][:, None, None], x
        centered = points[None, :, None] - steps[:, None, :]
        logits = np.zeros((len(rows), len(points), n_categories))
        np.cumsum(a * centered, axis=2, out=logits[..., 1:])
        jacobian = None
        if derivatives:
            location = np.broadcast_to(
                -a[..., None] * below, (*logits.shape, n_categories - 1)
            )
            if fixed_slopes is None:
                slope = np.zeros(logits.shape)
                np.cumsum(centered, axis=2, out=slope[..., 1:])
                jacobian = np.concatenate((slope[..., None], location), axis=3)
            else:
                jacobian = np.ascontiguousarray(location)
        return softmax_terms(rows, x, logits, jacobian)

    n_factors = theta.shape[1]
    n_slopes = (n_categories - 1) * n_factors

    def nominal(
        rows: NDArray[np.intp], x: NDArray[np.float64], derivatives: bool
    ) -> tuple[
        NDArray[np.float64], NDArray[np.float64] | None, NDArray[np.float64] | None
    ]:
        slopes = x[:, :n_slopes].reshape(len(rows), n_categories - 1, n_factors)
        logits = np.zeros((len(rows), len(points), n_categories))
        logits[..., 1:] = theta @ slopes.transpose(0, 2, 1) + x[:, None, n_slopes:]
        jacobian = None
        if derivatives:
            # Category k > 0 has logit theta @ slopes[k] + intercepts[k].
            jacobian = np.zeros((*logits.shape, x.shape[1]))
            for category in range(1, n_categories):
                start = (category - 1) * n_factors
                jacobian[:, :, category, start : start + n_factors] = theta
                jacobian[:, :, category, n_slopes + category - 1] = 1.0
        return softmax_terms(rows, x, logits, jacobian)

    return {"graded": graded, "partial_credit": partial_credit, "nominal": nominal}[
        family
    ]
