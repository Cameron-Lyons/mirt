"""Optional diagonal complete-data uncertainty on Monte Carlo draws.

Posterior weights and draws stay fixed. This approximation excludes missing
information, covariance between parameters, and Monte Carlo sampling error.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON
from mirt.estimation._affine_objective import prepare_affine_objective
from mirt.estimation._dichotomous_objective import prepare_dichotomous_objective
from mirt.estimation._em_context import EMFitContext
from mirt.estimation._mc_objective import prepare_mc_objective
from mirt.estimation._polytomous_information import polytomous_item_curvature
from mirt.estimation.se_methods import _diagonal_item_standard_errors
from mirt.exceptions import MirtValidationError

if TYPE_CHECKING:
    from mirt.estimation.mcem import MCEMEstimator
    from mirt.models.base import BaseItemModel


def _gradient_curvature(evaluate: Callable[[float], float], step: float) -> float:
    """Differentiate a loss gradient using a valid central or one-sided stencil."""
    original_step = step
    contracted = False
    for _ in range(20):
        try:
            left, right = evaluate(-step), evaluate(step)
        except MirtValidationError:
            contracted = True
            step *= 0.5
            continue
        if contracted:
            # Once inside a nearby boundary, keep the stencil much smaller
            # than its distance to that boundary for accurate local curvature.
            step *= 0.001
            contracted = False
            continue
        return (right - left) / (2.0 * step)

    # An exact boundary cannot admit a central stencil at any step size.
    step = original_step
    center = None
    for _ in range(20):
        for offsets, coefficients in (
            ((-1, 1), (-1, 1)),
            ((0, 1, 2), (-3, 4, -1)),
            ((0, -1, -2), (3, -4, 1)),
        ):
            try:
                values = []
                for offset in offsets:
                    if offset == 0:
                        if center is None:
                            center = evaluate(0.0)
                        values.append(center)
                    else:
                        values.append(evaluate(offset * step))
            except MirtValidationError:
                continue
            return float(np.dot(coefficients, values) / (2.0 * step))
        step *= 0.5
    raise RuntimeError("Unable to construct a valid gradient-curvature stencil")


def mc_standard_errors(
    estimator: MCEMEstimator,
    model: BaseItemModel,
    responses: NDArray[np.int_],
    samples: NDArray[np.float64],
    weights: NDArray[np.float64],
    step: float,
) -> dict[str, NDArray[np.float64]]:
    """Estimate conditional diagonal item curvature while preserving the model."""
    n_persons = len(responses)
    if responses.shape != (n_persons, model.n_items):
        raise ValueError("responses has an incompatible shape")
    if samples.shape != (n_persons, estimator.n_samples, model.n_factors):
        raise ValueError("theta_samples has an incompatible shape")
    if weights.shape != (n_persons, estimator.n_samples):
        raise ValueError("weights has an incompatible shape")
    row_block = max(1, 32_768 // estimator.n_samples)
    for start in range(0, n_persons, row_block):
        block = weights[start : start + row_block]
        if not np.all(np.isfinite(block)) or np.any(block < 0.0):
            raise ValueError("weights must be finite and non-negative")

    parameters = model.parameters
    masks = model.free_parameter_masks
    result = {name: np.zeros_like(values) for name, values in parameters.items()}
    for name in result:
        result[name][masks[name]] = np.nan
    if not any(np.any(mask) for mask in masks.values()):
        return result
    shared = n_persons > 0 and samples.strides[0] == 0
    if shared:
        point_block = max(1, 32_768 // model.n_factors)
        for start in range(0, estimator.n_samples, point_block):
            if not np.all(np.isfinite(samples[0, start : start + point_block])):
                raise ValueError(
                    "theta_samples must contain only finite observed values"
                )
    context = EMFitContext(responses) if shared else None
    correct = counts = None
    if context is not None and not model.is_polytomous:
        correct, counts = context.expected_counts(weights)

    for item in range(model.n_items):
        observed = responses[:, item] >= 0
        if not np.any(observed):
            continue
        if model.is_polytomous and np.any(
            responses[:, item] >= model.n_categories[item]
        ):
            raise ValueError("model returned invalid item category probabilities")
        originals = {
            name: values[item].copy()
            for name, values in parameters.items()
            if values.ndim > 0 and values.shape[0] == model.n_items
        }
        if not any(np.any(masks[name][item]) for name in originals):
            continue

        def restore():
            for name, value in originals.items():
                model.set_item_parameter(item, name, value)

        kernel = None
        default_model = (
            estimator._uses_default_item_methods()
            and estimator._uses_default_information_model(model)
        )
        if default_model and model.is_polytomous:
            curvature = _polytomous_curvature(
                model, item, responses[:, item], samples, weights, context
            )
            for name, values in curvature.items():
                errors = np.sqrt(
                    np.divide(
                        1.0,
                        values,
                        out=np.full_like(values, np.nan),
                        where=np.isfinite(values) & (values > 0.0),
                    )
                )
                result[name][item] = np.where(masks[name][item], errors, 0.0)
            continue
        if default_model:
            current, bounds = estimator._get_item_params_and_bounds(model, item)
            # Include all possible stencils in the kernel's stable-logit box.
            bounds = [
                (min(low, value - 2 * step), max(high, value + 2 * step))
                for value, (low, high) in zip(current, bounds, strict=True)
            ]
            if context is None:
                kernel = prepare_mc_objective(
                    model,
                    item,
                    responses[:, item],
                    samples,
                    weights,
                    estimator.n_samples,
                    bounds,
                )
            else:
                assert correct is not None and counts is not None
                kernel = prepare_dichotomous_objective(
                    model,
                    item,
                    samples[0],
                    counts[item],
                    correct[item],
                    PROB_EPSILON,
                    bounds,
                )
                if kernel is None:
                    kernel = prepare_affine_objective(
                        model,
                        item,
                        samples[0],
                        counts[item],
                        correct[item],
                        PROB_EPSILON,
                        bounds,
                    )
        if kernel is not None:
            coordinate = 0
            for name, original in originals.items():
                item_mask = np.asarray(masks[name][item]).reshape(-1)
                item_result = result[name][item : item + 1].reshape(-1)
                for index in np.flatnonzero(item_mask):

                    def gradient_at_offset(offset):
                        candidate = current.copy()
                        candidate[coordinate] += offset
                        try:
                            estimator._set_item_params(model, item, candidate)
                            installed, _ = estimator._get_item_params_and_bounds(
                                model, item
                            )
                            if not np.array_equal(installed, candidate):
                                raise MirtValidationError(
                                    "Curvature trial changed parameter identification"
                                )
                            return float(kernel(candidate)[1][coordinate])
                        finally:
                            restore()

                    curvature = _gradient_curvature(gradient_at_offset, step)
                    if np.isfinite(curvature) and curvature > 0.0:
                        item_result[index] = np.sqrt(1.0 / curvature)
                    coordinate += 1
            continue

        for name, values in _numerical_item_errors(
            estimator,
            model,
            item,
            responses,
            samples,
            weights,
            originals,
            masks,
            step,
            restore,
        ).items():
            result[name][item] = values
    return result


def _polytomous_curvature(model, item, decisions, samples, weights, context):
    """Reduce exact category curvature on a shared grid or bounded draws."""
    categories = model.n_categories[item]
    max_points = max(1, 32_768 // max(model.n_factors, categories, 2))
    result = {
        name: np.zeros_like(values[item]) for name, values in model.parameters.items()
    }

    def accumulate(theta, counts):
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta_samples must contain only finite observed values")
        for name, values in polytomous_item_curvature(
            model, item, theta, counts, PROB_EPSILON
        ).items():
            result[name] += values

    if context is not None:
        counts = context.expected_category_counts(item, categories, weights)
        for start in range(0, samples.shape[1], max_points):
            accumulate(
                samples[0, start : start + max_points],
                counts[start : start + max_points],
            )
        return result

    rows = np.flatnonzero(decisions >= 0)
    sample_chunk = min(samples.shape[1], max_points)
    row_chunk = max(1, max_points // sample_chunk)
    for first in range(0, len(rows), row_chunk):
        selected = rows[first : first + row_chunk]
        for start in range(0, samples.shape[1], sample_chunk):
            stop = min(start + sample_chunk, samples.shape[1])
            theta = samples[selected, start:stop].reshape(-1, model.n_factors)
            counts = np.zeros((len(theta), categories))
            observed = np.repeat(decisions[selected], stop - start)
            counts[np.arange(len(theta)), observed] = weights[
                selected, start:stop
            ].ravel()
            accumulate(theta, counts)
            del theta, counts, observed
    return result


def _numerical_item_errors(
    estimator, model, item, responses, samples, weights, originals, masks, step, restore
):
    """Keep custom objective inputs and temporary callbacks scoped to one item."""
    observed = responses[:, item] >= 0
    item_samples, item_weights = samples[observed], weights[observed]
    decisions = responses[observed, item]
    result = {}
    for name, original in originals.items():
        if not np.any(masks[name][item]):
            continue

        def likelihood(value):
            try:
                model.set_item_parameter(item, name, value)
                return estimator._item_expected_log_likelihood(
                    model, item, decisions, item_samples, item_weights
                )
            finally:
                restore()

        value = float(original) if original.ndim == 0 else original
        errors = _diagonal_item_standard_errors(
            likelihood,
            value,
            step,
            scheme="central",
            free_mask=masks[name][item],
        )
        errors = np.asarray(errors)
        undefined = np.asarray(masks[name][item]) & (
            ~np.isfinite(errors) | (errors <= 0.0)
        )
        result[name] = np.where(undefined, np.nan, errors)
    return result
