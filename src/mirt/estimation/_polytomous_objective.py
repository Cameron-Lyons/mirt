"""Prepared clipped category likelihoods and gradients for built-in items."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.special import xlogy

from mirt._core import sigmoid
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
) -> tuple[float, NDArray[np.float64]]:
    """Keep only counts whose category curve has a nonzero clip derivative."""
    active = (probabilities > epsilon) & (probabilities < 1.0 - epsilon)
    loss = -float(np.sum(xlogy(counts, np.clip(probabilities, epsilon, 1.0 - epsilon))))
    return loss, np.where(active, counts, 0.0)


def _softmax_loss_and_residual(
    probabilities: NDArray[np.float64],
    counts: NDArray[np.float64],
    epsilon: float,
) -> tuple[float, NDArray[np.float64]]:
    loss, effective = _clipped_loss_and_counts(probabilities, counts, epsilon)
    residual = probabilities * effective.sum(axis=1, keepdims=True) - effective
    return loss, residual


def prepare_polytomous_objective(
    model: BaseItemModel,
    item_idx: int,
    theta: NDArray[np.float64],
    counts: NDArray[np.float64],
    epsilon: float,
) -> _Objective | None:
    """Prepare a pure objective in the core estimator's free-parameter layout.

    Fixed PCM slopes, NRM reference categories, and padding do not enter the
    trial vector. Custom probability, parameter setters, or layouts retain
    their model-based numerical objective.
    """
    layouts = {
        GradedResponseModel: ("discrimination", "thresholds"),
        GeneralizedPartialCredit: ("discrimination", "steps"),
        PartialCreditModel: ("discrimination", "steps"),
        NominalResponseModel: ("slopes", "intercepts"),
    }
    if tuple(model._parameters) != layouts.get(type(model)) or any(
        name in vars(model)
        for name in (
            "probability",
            "_category_probabilities",
            "_ensure_theta_2d",
            "set_parameters",
            "set_item_parameter",
            "_canonical_parameter_values",
            "free_parameter_masks",
        )
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
                _stable_softmax(logits), counts, epsilon
            )
            gradient = np.empty_like(params)
            gradient[:n_slopes] = (residual[:, 1:].T @ theta).ravel()
            gradient[n_slopes:] = residual[:, 1:].sum(axis=0)
            return loss, gradient

        return nominal_objective

    fixed_slope = type(model) is PartialCreditModel
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
            loss, effective = _clipped_loss_and_counts(probabilities, counts, epsilon)
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

    def partial_credit_objective(
        params: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        a = np.ones(1) if fixed_slope else params[:n_slopes]
        steps = params[n_slopes:]
        scale = a[0] if unidimensional else np.linalg.norm(a)
        centered = (
            points[:, None] - steps if unidimensional else (theta @ a)[:, None] - steps
        )
        loss, residual = _softmax_loss_and_residual(
            _partial_credit_probabilities(scale * centered), counts, epsilon
        )
        tails = np.cumsum(residual[:, :0:-1], axis=1)[:, ::-1]
        gradient = np.empty_like(params)
        if n_slopes:
            centered_score = np.sum(tails * centered)
            if unidimensional:
                gradient[0] = centered_score
            else:
                direction = a / scale if scale > 0.0 else np.zeros_like(a)
                gradient[:n_slopes] = (
                    scale * (theta.T @ tails.sum(axis=1)) + direction * centered_score
                )
        gradient[n_slopes:] = -scale * tails.sum(axis=0)
        return loss, gradient

    return partial_credit_objective
