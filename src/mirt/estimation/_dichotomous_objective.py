"""Prepared likelihoods and gradients for built-in 1PL–4PL items.

Nodes whose probability lies beyond ``[epsilon, 1 - epsilon]`` keep the
exact likelihood where it is representable and the asymptote they approach
is fixed, so that items whose expected responses saturate keep a gradient.
Other nodes are clipped.
"""

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, xlog1py, xlogy

from mirt._logistic import _logistic_probability
from mirt._model_defaults import uses_builtin_model_hooks
from mirt.estimation._item_priors import subtract_coordinate_priors
from mirt.estimation._logistic_newton import ItemObjective
from mirt.estimation.priors import Prior
from mirt.models.base import BaseItemModel

_Objective = Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]]


def logistic_tails(
    logits: NDArray[np.float64],
    observed: NDArray[np.float64],
    correct: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return exact log-likelihood terms and logit scores at finite logits.

    The log-sigmoid stays exact where the success probability rounds to zero
    or one. The scores are the derivatives of the negative terms.
    """
    terms = -(
        correct * np.logaddexp(0.0, -logits)
        + (observed - correct) * np.logaddexp(0.0, logits)
    )
    return terms, observed * expit(logits) - correct


def guessing_tails(
    success: NDArray[np.float64],
    failure: NDArray[np.float64],
    observed: NDArray[np.float64],
    correct: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Return exact terms and probability scores of nodes with asymptotes.

    ``success`` and ``failure`` are the probabilities of a correct and an
    incorrect response, each computed from its own asymptote so that a small
    one keeps its relative precision. The scores are the derivatives of the
    negative terms in the success probability. Entries whose probabilities
    underflow or whose scores overflow are not usable.
    """
    incorrect = observed - correct
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        terms = xlogy(correct, success) + xlogy(incorrect, failure)
        scores = incorrect / failure - correct / success
    usable = (success > 0.0) & (failure > 0.0) & np.isfinite(scores)
    return terms, scores, usable


def prepare_dichotomous_objective(
    model: BaseItemModel,
    item_idx: int,
    theta: NDArray[np.float64],
    observed: NDArray[np.float64],
    correct: NDArray[np.float64],
    epsilon: float,
    bounds: list[tuple[float, float]] | None = None,
) -> _Objective | None:
    """Prepare a pure objective, optionally restricted to an optimizer box.

    When bounds are supplied, trial parameters must lie within them. Ordinary
    quadrature grids can then skip repeated overflow checks. Without bounds,
    use the public curves' stable logit kernels for every evaluation.
    """
    from mirt.models.dichotomous import (
        FourParameterLogistic,
        OneParameterLogistic,
        ThreeParameterLogistic,
        TwoParameterLogistic,
    )

    layouts = {
        OneParameterLogistic: ("discrimination", "difficulty"),
        TwoParameterLogistic: ("discrimination", "difficulty"),
        ThreeParameterLogistic: ("discrimination", "difficulty", "guessing"),
        FourParameterLogistic: ("discrimination", "difficulty", "guessing", "upper"),
    }
    if (
        tuple(model._parameters) != layouts.get(type(model))
        or not uses_builtin_model_hooks(model)
        or model._free_parameter_restrictions
    ):
        return None

    # A 1PL slope is fixed unless the family estimates its common value.
    fixed_slope = (
        type(model) is OneParameterLogistic and not model.estimate_discrimination
    )
    fixed_a = (
        float(model._parameters["discrimination"][item_idx]) if fixed_slope else None
    )
    return prepare_logistic_objective(
        theta,
        observed,
        correct,
        epsilon,
        fixed_a=fixed_a,
        has_guessing="guessing" in model._parameters,
        has_upper="upper" in model._parameters,
        bounds=bounds,
    )


def prepare_logistic_objective(
    theta: NDArray[np.float64],
    observed: NDArray[np.float64],
    correct: NDArray[np.float64],
    epsilon: float,
    *,
    fixed_a: float | None = None,
    has_guessing: bool = False,
    has_upper: bool = False,
    bounds: list[tuple[float, float]] | None = None,
) -> _Objective | None:
    """Prepare the shared logistic kernel without a model adapter.

    Nodes whose probability lies beyond ``[epsilon, 1 - epsilon]`` keep the
    exact likelihood and gradient when the asymptote they approach is fixed:
    zero and one for 1PL and 2PL items at finite logits (see
    :func:`logistic_tails`), and one for 3PL items (see
    :func:`guessing_tails`). Near a free guessing or 4PL upper asymptote
    they keep the clipped probability, whose gradient vanishes.
    """
    from mirt.models.dichotomous import (
        _multidimensional_logits,
        _unidimensional_logits,
    )

    # NumPy 1.x can retain float32 during arithmetic with scalar parameters.
    # Match the public curves' precision; MC callers pass bounded theta blocks.
    theta = np.asarray(theta, dtype=np.float64)
    unidimensional = theta.shape[1] == 1
    fixed_slope = fixed_a is not None
    fixed_a = 1.0 if fixed_a is None else fixed_a
    n_slopes = 0 if fixed_slope else theta.shape[1]
    points = theta[:, 0] if unidimensional else theta
    incorrect = observed - correct
    ordinary = False
    if bounds is not None:
        if len(bounds) != n_slopes + 1 + has_guessing + has_upper:
            return None
        slope_limit = (
            abs(fixed_a)
            if fixed_slope
            else sum(max(abs(low), abs(high)) for low, high in bounds[:n_slopes])
        )
        location_limit = max(abs(value) for value in bounds[n_slopes])
        max_logit_bound = (
            float(np.abs(points).max(initial=0.0)) + float(location_limit)
        ) * float(slope_limit)
        # This also keeps every exponential normal and finite on the fast path.
        ordinary = max_logit_bound <= 700.0

    def objective(params: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        a = (
            fixed_a
            if fixed_slope
            else params[0]
            if unidimensional
            else params[:n_slopes]
        )
        b = params[n_slopes]
        slope_sum = a if unidimensional else a.sum()
        if ordinary:
            z = a * (points - b) if unidimensional else points @ a - slope_sum * b
        elif unidimensional:
            z = _unidimensional_logits(points, a, b)
        else:
            z = _multidimensional_logits(points, a, b)

        if has_guessing:
            c = params[n_slopes + 1]
            d = params[-1] if has_upper else 1.0
            if ordinary:
                tail = np.exp(-np.abs(z))
            else:
                with np.errstate(under="ignore"):
                    tail = np.exp(-np.abs(z))
            tail /= 1.0 + tail
            positive = z >= 0.0
            failure = np.where(positive, tail, 1.0 - tail)
            p = np.where(positive, d - (d - c) * tail, c + (d - c) * tail)
        elif ordinary:
            # Saturated nodes below need the logits.
            p = np.negative(z)
            np.exp(p, out=p)
            p += 1.0
            np.reciprocal(p, out=p)
        else:
            p = _logistic_probability(z.copy(), None, None)
        active = (p > epsilon) & (p < 1.0 - epsilon)
        # The clipped objective is flat beyond each probability bound, which
        # stalls the optimizer on items whose responses lie there. Near a free
        # asymptote, however, the exact derivative in it exceeds 1/epsilon,
        # which the optimizer cannot step against, so those nodes keep the
        # clip.
        exact = np.flatnonzero(~active)
        if has_guessing:
            exact = exact[:0] if has_upper else exact[p[exact] >= 1.0 - epsilon]
            unclipped = p[exact]
        np.clip(p, epsilon, 1.0 - epsilon, out=p)
        terms = xlogy(correct, p) + xlog1py(incorrect, -p)
        residual = observed * p - correct
        if has_guessing:
            score = np.divide(
                residual, p * (1.0 - p), out=np.zeros_like(p), where=active
            )
        else:
            score = residual
            score[~active] = 0.0
        if exact.size:
            if has_guessing:
                values, scores, usable = guessing_tails(
                    unclipped,
                    (1.0 - c) * failure[exact],
                    observed[exact],
                    correct[exact],
                )
                rows = exact[usable]
                terms[rows], score[rows] = values[usable], scores[usable]
            else:
                rows = exact[np.isfinite(z[exact])]
                terms[rows], score[rows] = logistic_tails(
                    z[rows], observed[rows], correct[rows]
                )
        loss = -float(np.sum(terms))
        common = score * (d - c) * tail * (1.0 - tail) if has_guessing else score
        gradient = np.empty_like(params)
        if n_slopes:
            gradient[:n_slopes] = (points - b).T @ common
        gradient[n_slopes] = -slope_sum * common.sum()
        if has_guessing:
            gradient[n_slopes + 1] = failure @ score
            if has_upper:
                success = np.where(positive, 1.0 - tail, tail)
                gradient[-1] = success @ score
        return loss, gradient

    return objective


def guessing_items_objective(
    theta: NDArray[np.float64],
    correct: NDArray[np.float64],
    observed: NDArray[np.float64],
    epsilon: float,
    priors: Sequence[tuple[slice, Prior]] = (),
) -> ItemObjective:
    """Prepare the 3PL objectives of several items for Fisher scoring.

    The objective of item ``j`` at coordinates ``(a, b, c)`` is the negative
    expected log-likelihood of :func:`prepare_logistic_objective` with a
    guessing parameter on the unidimensional nodes ``theta``, minus the
    coordinate log-priors ``priors``. Its curvature is the expected (Fisher)
    information plus the priors' positive negative curvatures.

    Parameters
    ----------
    theta : ndarray of shape (n_points,)
        Quadrature nodes.
    correct, observed : ndarray of shape (n_items, n_points)
        Expected correct and observed counts of the items.
    epsilon : float
        Probability clipping, as in :func:`prepare_logistic_objective`.
    priors : sequence of (slice, Prior)
        Priors of coordinate columns, as for
        :func:`~mirt.estimation._item_priors.subtract_coordinate_priors`.

    Returns
    -------
    callable
        An objective for :func:`~mirt.estimation._logistic_newton.newton_items`.
    """
    points = np.asarray(theta, dtype=np.float64).reshape(-1)

    def objective(
        rows: NDArray[np.intp], x: NDArray[np.float64], derivatives: bool
    ) -> tuple[
        NDArray[np.float64], NDArray[np.float64] | None, NDArray[np.float64] | None
    ]:
        a, b, c = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        centered = points - b
        z = a * centered
        with np.errstate(under="ignore", over="ignore"):
            tail = np.exp(-np.abs(z))
        tail /= 1.0 + tail
        positive = z >= 0.0
        failure = np.where(positive, tail, 1.0 - tail)
        p = np.where(positive, 1.0 - (1.0 - c) * tail, c + (1.0 - c) * tail)
        active = (p > epsilon) & (p < 1.0 - epsilon)
        hits, counts = correct[rows], observed[rows]
        # As in the itemwise objective, nodes near the fixed upper asymptote
        # keep the exact likelihood where it and its information are
        # representable; those near the free guessing one keep the clip.
        saturated = np.nonzero(p >= 1.0 - epsilon)
        unclipped = p[saturated]
        failed = (1.0 - c[saturated[0], 0]) * failure[saturated]
        values, scores, usable = guessing_tails(
            unclipped, failed, counts[saturated], hits[saturated]
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            weights = counts[saturated] / (unclipped * failed)
        usable &= np.isfinite(weights)
        exact = tuple(index[usable] for index in saturated)
        np.clip(p, epsilon, 1.0 - epsilon, out=p)
        terms = xlogy(hits, p) + xlog1py(counts - hits, -p)
        terms[exact] = values[usable]
        loss = -np.sum(terms, axis=1)
        if not derivatives:
            subtract_coordinate_priors(priors, x, loss)
            return loss, None, None
        variance = p * (1.0 - p)
        score = np.divide(
            counts * p - hits, variance, out=np.zeros_like(p), where=active
        )
        weight = np.divide(counts, variance, out=np.zeros_like(p), where=active)
        score[exact], weight[exact] = scores[usable], weights[usable]
        slope = (1.0 - c) * tail * (1.0 - tail)
        # Derivatives of p in (a, b, c); 1 - sigmoid(z) is the failure share.
        jacobian = np.stack((slope * centered, -a * slope, failure), axis=-1)
        gradient = np.einsum("iq,iqk->ik", score, jacobian)
        information = np.einsum("iq,iqk,iql->ikl", weight, jacobian, jacobian)
        subtract_coordinate_priors(priors, x, loss, gradient, information)
        return loss, gradient, information

    return objective
