"""Lock-step batch simulation for unidimensional CAT engines.

Independent simulated examinees advance together. Each iteration selects,
answers, and rescores one more item for every examinee still being tested,
using one vectorized evaluation per item position instead of one Python
session per examinee.

Only configurations whose per-session semantics are reproduced exactly are
eligible (see :func:`supports_lockstep`): unmodified engine, MFI selection,
EAP scoring, no exposure or content control, SE and maximum-length stopping,
and an unmodified built-in unidimensional model. Given identical responses,
item paths, estimates, standard errors, and histories equal those of
``CATEngine.run_simulation`` up to floating-point summation order.

Random responses use the engine's generator. The sequential loop consumes one
uniform per administered item, examinee after examinee, so each examinee's
first draw follows the previous examinee's last one. Lock step instead
reserves ``L`` consecutive uniforms for every examinee, where ``L`` is the
maximum test length (``max_items`` capped at the pool size), and uses the
``t``-th of them for the ``t``-th response. Both paths therefore produce
identical results when every examinee answers ``L`` items, as in fixed-length
tests. When tests end early, later examinees receive other, equally
distributed, responses than in the sequential loop, and the generator
advances by ``L`` draws per examinee.
"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from mirt._model_defaults import (
    original_model_hook,
    uses_builtin_model_hooks,
    uses_original_model_hook,
)
from mirt.cat._native import uses_native_defaults
from mirt.cat.content import NoContentConstraint
from mirt.cat.exposure import NoExposureControl
from mirt.cat.results import CATResult, CATState
from mirt.cat.selection import MaxFisherInformation
from mirt.constants import PROB_EPSILON

# Examinees advanced together, and bounds on (examinee x item) work arrays and
# on (examinee x position) draws and histories.
_BLOCK_ROWS = 4096
_MAX_WORK_VALUES = 1_048_576
_MAX_HISTORY_VALUES = 262_144


class _Fallback(Exception):
    """Sequential sessions would leave the shared lock-step path."""


def _supported_model_types() -> tuple[type, ...]:
    from mirt.models import (
        FourParameterLogistic,
        GeneralizedPartialCredit,
        GradedResponseModel,
        OneParameterLogistic,
        PartialCreditModel,
        ThreeParameterLogistic,
        TwoParameterLogistic,
    )

    return (
        OneParameterLogistic,
        TwoParameterLogistic,
        ThreeParameterLogistic,
        FourParameterLogistic,
        GradedResponseModel,
        GeneralizedPartialCredit,
        PartialCreditModel,
    )


def _stopping_parameters(engine: Any) -> tuple[float, int, int] | None:
    """Return the SE threshold, maximum length, and minimum length."""
    parameters: tuple[float, int, int] | None = engine._native_stopping_parameters(
        require_standard_error=False
    )
    return parameters


def supports_lockstep(engine: Any) -> bool:
    """Return whether lock-step simulation reproduces the engine's sessions."""
    from mirt.cat.engine import CATEngine

    if not uses_native_defaults(engine, CATEngine) or not uses_native_defaults(
        engine._selection, MaxFisherInformation
    ):
        return False
    if not uses_native_defaults(
        engine._exposure, NoExposureControl
    ) or not uses_native_defaults(engine._content, NoContentConstraint):
        return False
    if engine.scoring_method != "EAP":
        return False
    n_quadpts = engine.n_quadpts
    if (
        isinstance(n_quadpts, (bool, np.bool_))
        or not isinstance(n_quadpts, (int, np.integer))
        or n_quadpts < 5
    ):
        return False
    initial_theta = engine.initial_theta
    if (
        isinstance(initial_theta, (bool, np.bool_))
        or not isinstance(initial_theta, Real)
        or not np.isfinite(float(initial_theta))
    ):
        return False

    model = engine.model
    if type(model) not in _supported_model_types() or model.n_factors != 1:
        return False
    if not uses_builtin_model_hooks(
        model, likelihood=True
    ) or not uses_original_model_hook(model, "information"):
        return False
    return _stopping_parameters(engine) is not None


@dataclass(frozen=True)
class _Plan:
    """Quadrature, item likelihood table, and stopping controls for a batch."""

    nodes: NDArray[np.float64]
    log_weights: NDArray[np.float64]
    center: float
    # log P(X_j = k | node q), shape (n_categories, n_items, n_quadpts).
    log_likelihood: NDArray[np.float64]
    binary_scoring: bool
    se_threshold: float
    max_items: int
    min_items: int
    block_rows: int


def _uses_binary_eap(model: Any) -> bool:
    """Mirror the dispatch of ``mirt.cat._eap.score_binary_eap``."""
    from mirt.models.base import DichotomousItemModel

    if model.is_polytomous:
        return False
    return all(
        original_model_hook(type(model), name)
        is original_model_hook(DichotomousItemModel, name)
        for name in ("log_likelihood", "log_likelihood_batch")
    )


def _plan(engine: Any) -> _Plan | None:
    """Precompute per-item response log-likelihoods at the quadrature nodes.

    Binary models use the clipped curves of the engine's bounded EAP update.
    Other models take each item's contribution from ``log_likelihood_batch``,
    which ``fscores`` sums over administered items. Returns None when the
    table is not usable, leaving the sequential path to report the problem.
    """
    from mirt.scoring._common import build_quadrature

    stopping = _stopping_parameters(engine)
    if stopping is None:
        return None
    se_threshold, max_items, min_items = stopping

    model = engine.model
    n_items = model.n_items
    points, weights = build_quadrature(
        n_quadpts=int(engine.n_quadpts),
        n_factors=1,
        prior_mean=None,
        prior_cov=None,
    )
    nodes = points[:, 0]
    binary_scoring = _uses_binary_eap(model)

    if binary_scoring:
        table = np.empty((2, n_items, len(nodes)))
        for item_idx in range(n_items):
            probabilities = np.clip(
                model.probability(points, item_idx=item_idx),
                PROB_EPSILON,
                1.0 - PROB_EPSILON,
            ).reshape(-1)
            table[1, item_idx] = np.log(probabilities)
            table[0, item_idx] = np.log1p(-probabilities)
    else:
        categories = (
            np.asarray(model.n_categories)
            if model.is_polytomous
            else np.full(n_items, 2)
        )
        table = np.full((int(categories.max()), n_items, len(nodes)), -np.inf)
        rows_per_call = max(1, _MAX_WORK_VALUES // n_items)
        for category in range(table.shape[0]):
            items = np.flatnonzero(categories > category)
            for start in range(0, items.size, rows_per_call):
                chunk = items[start : start + rows_per_call]
                responses = np.full((chunk.size, n_items), -1, dtype=np.int_)
                responses[np.arange(chunk.size), chunk] = category
                table[category, chunk] = model.log_likelihood_batch(responses, points)

    if np.any(np.isnan(table)) or np.any(table == np.inf):
        return None
    return _Plan(
        nodes=nodes,
        log_weights=np.log(weights + 1e-300),
        center=float(weights @ nodes),
        log_likelihood=table,
        binary_scoring=binary_scoring,
        se_threshold=se_threshold,
        max_items=max_items,
        min_items=min_items,
        block_rows=max(
            1,
            min(
                _BLOCK_ROWS,
                _MAX_WORK_VALUES // n_items,
                _MAX_HISTORY_VALUES // max_items,
            ),
        ),
    )


def _item_information(model: Any, theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return item information with shape (len(theta), n_items)."""
    theta_2d = theta[:, None]
    if not model.is_polytomous:
        return np.asarray(model.information(theta_2d), dtype=np.float64).reshape(
            len(theta), model.n_items
        )
    information = np.empty((len(theta), model.n_items))
    for item_idx in range(model.n_items):
        values = np.asarray(model.information(theta_2d, item_idx=item_idx))
        information[:, item_idx] = values.reshape(len(theta), -1).sum(axis=1)
    return information


def _select_items(
    model: Any,
    theta: NDArray[np.float64],
    used: NDArray[np.bool_],
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Choose the most informative unused item, preferring the lowest index."""
    chosen = np.empty(len(theta), dtype=np.intp)
    information = np.empty(len(theta))
    rows_per_chunk = max(1, _MAX_WORK_VALUES // model.n_items)
    for start in range(0, len(theta), rows_per_chunk):
        stop = start + rows_per_chunk
        values = _item_information(model, theta[start:stop])
        # MFI ranks NaN differently from argmax; leave such pools to it.
        if not np.all(np.isfinite(values)):
            raise _Fallback
        best = np.where(used[start:stop], -np.inf, values).argmax(axis=1)
        chosen[start:stop] = best
        information[start:stop] = values[np.arange(len(best)), best]
    return chosen, information


def _simulated_responses(
    model: Any,
    true_theta: NDArray[np.float64],
    items: NDArray[np.intp],
    uniforms: NDArray[np.float64],
) -> NDArray[np.int_]:
    """Map one uniform per examinee to a response, as the engine does.

    Dichotomous responses are ``u < P(X=1)``. Categories use the inverse
    normalized CDF, which is how ``Generator.choice`` maps its single draw.
    """
    probabilities = np.asarray(
        model.probability_pairs(true_theta[:, None], items), dtype=np.float64
    )
    if not model.is_polytomous:
        return (uniforms < probabilities.reshape(-1)).astype(np.int_)
    cumulative = np.cumsum(probabilities, axis=1)
    cumulative /= cumulative[:, -1:]
    return np.sum(cumulative <= uniforms[:, None], axis=1).astype(np.int_)


def _posterior_moments(
    plan: _Plan,
    log_posterior: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return EAP means and standard deviations of unnormalized posteriors."""
    normalizer = logsumexp(log_posterior, axis=1)
    if not np.all(np.isfinite(normalizer)):
        # The engine would fall back to its crude estimates for this examinee.
        raise _Fallback
    posterior = np.exp(log_posterior - normalizer[:, None])
    if plan.binary_scoring:
        theta = posterior @ plan.nodes
        centered = plan.nodes[None, :] - theta[:, None]
        variance = np.sum(centered * (posterior * centered), axis=1)
    else:
        centered_nodes = plan.nodes - plan.center
        centered_mean = posterior @ centered_nodes
        theta = centered_mean + plan.center
        variance = posterior @ centered_nodes**2 - centered_mean**2
        np.maximum(variance, 0.0, out=variance)
    return theta, np.sqrt(variance)


@dataclass
class _Block:
    """Outcomes for a contiguous block of simulated examinees."""

    theta: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    n_items: NDArray[np.int_]
    items: NDArray[np.intp]
    responses: NDArray[np.int_]
    theta_history: NDArray[np.float64]
    se_history: NDArray[np.float64]
    info_history: NDArray[np.float64]


def _run_block(
    engine: Any,
    plan: _Plan,
    true_theta: NDArray[np.float64],
) -> _Block:
    """Administer items to every examinee in the block until each one stops."""
    model = engine.model
    n_rows, length = len(true_theta), plan.max_items
    # Row r answers its t-th item with uniforms[r, t]; see the module notes.
    uniforms = engine.rng.random((n_rows, length))
    theta = np.full(n_rows, float(engine.initial_theta))
    standard_error = np.full(n_rows, np.inf)
    log_posterior = np.tile(plan.log_weights, (n_rows, 1))
    used = np.zeros((n_rows, model.n_items), dtype=np.bool_)
    block = _Block(
        theta=theta,
        standard_error=standard_error,
        n_items=np.zeros(n_rows, dtype=np.int_),
        items=np.full((n_rows, length), -1, dtype=np.intp),
        responses=np.full((n_rows, length), -1, dtype=np.int_),
        theta_history=np.full((n_rows, length), np.nan),
        se_history=np.full((n_rows, length), np.nan),
        info_history=np.full((n_rows, length), np.nan),
    )

    active = np.arange(n_rows)
    for step in range(length):
        if active.size == 0:
            break
        chosen, information = _select_items(model, theta[active], used[active])
        responses = _simulated_responses(
            model, true_theta[active], chosen, uniforms[active, step]
        )
        used[active, chosen] = True
        log_posterior[active] += plan.log_likelihood[responses, chosen]
        updated_theta, updated_se = _posterior_moments(plan, log_posterior[active])

        theta[active] = updated_theta
        standard_error[active] = updated_se
        block.items[active, step] = chosen
        block.responses[active, step] = responses
        block.info_history[active, step] = information
        block.theta_history[active, step] = updated_theta
        block.se_history[active, step] = updated_se
        block.n_items[active] = step + 1

        # The stopping controls cap max_items at the pool size and keep
        # min_items <= max_items, so this also ends tests at pool exhaustion.
        administered = step + 1
        stop = (administered >= plan.min_items) & (
            (updated_se <= plan.se_threshold) | (administered >= plan.max_items)
        )
        active = active[~stop]
    return block


def _iter_blocks(
    engine: Any,
    plan: _Plan,
    true_thetas: NDArray[np.float64],
    n_replications: int,
) -> Iterator[tuple[NDArray[np.intp], _Block]]:
    """Yield theta indices and outcomes in theta-major replication order."""
    n_rows = len(true_thetas) * n_replications
    for start in range(0, n_rows, plan.block_rows):
        theta_index = np.arange(start, min(start + plan.block_rows, n_rows))
        theta_index //= n_replications
        yield theta_index, _run_block(engine, plan, true_thetas[theta_index])


def _block_results(engine: Any, block: _Block) -> list[CATResult]:
    """Convert block arrays into session results with authored stop reasons."""
    # Evaluate an isolated copy so the engine's interactive stopping state is
    # left intact, as the native path does.
    stopping = deepcopy(engine._stopping)
    results = []
    for row in range(len(block.theta)):
        count = int(block.n_items[row])
        items = block.items[row, :count].tolist()
        responses = block.responses[row, :count]
        theta = float(block.theta[row])
        standard_error = float(block.standard_error[row])
        stopping.reset()
        state = CATState(
            theta=theta,
            standard_error=standard_error,
            items_administered=items,
            responses=responses.tolist(),
            n_items=count,
        )
        stopping_reason = (
            stopping.get_reason()
            if stopping.should_stop(state)
            else "Item pool exhausted"
        )
        results.append(
            CATResult(
                theta=theta,
                standard_error=standard_error,
                items_administered=items,
                responses=responses,
                n_items_administered=count,
                stopping_reason=stopping_reason,
                theta_history=block.theta_history[row, :count].tolist(),
                se_history=block.se_history[row, :count].tolist(),
                item_info_history=block.info_history[row, :count].tolist(),
            )
        )
    return results


def simulate(
    engine: Any,
    true_thetas: NDArray[np.float64],
    n_replications: int,
) -> list[CATResult] | None:
    """Simulate every theta and replication in lock step.

    Returns results in the sequential order (all replications of the first
    theta, then the next theta), or None when a sequential session would
    leave the shared path, for example to recover from a non-finite
    posterior. The engine's random generator is then restored, so the caller
    can run the sequential loop instead.
    """
    plan = _plan(engine)
    if plan is None:
        return None
    state = engine.rng.bit_generator.state
    try:
        results = []
        for _, block in _iter_blocks(engine, plan, true_thetas, n_replications):
            results.extend(_block_results(engine, block))
    except _Fallback:
        engine.rng.bit_generator.state = state
        return None
    return results


def conditional_error_moments(
    engine: Any,
    true_thetas: NDArray[np.float64],
    n_replications: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None:
    """Return conditional bias, MSE, and mean test length per theta.

    Blocks are merged into running per-theta moments with Chan's update, so
    working storage does not grow with the number of replications. Returns
    None, with the random generator restored, under the same conditions as
    :func:`simulate`. Responses match :func:`simulate` for the same state.
    """
    plan = _plan(engine)
    if plan is None:
        return None
    n_thetas = len(true_thetas)
    counts = np.zeros(n_thetas)
    mean_error = np.zeros(n_thetas)
    centered_sum_squares = np.zeros(n_thetas)
    total_items = np.zeros(n_thetas)
    state = engine.rng.bit_generator.state
    try:
        for theta_index, block in _iter_blocks(
            engine, plan, true_thetas, n_replications
        ):
            errors = block.theta - true_thetas[theta_index]
            starts = np.flatnonzero(np.diff(theta_index, prepend=-1))
            index = theta_index[starts]
            size = np.diff(np.append(starts, len(errors)))
            block_mean = np.add.reduceat(errors, starts) / size
            deviations = errors - np.repeat(block_mean, size)
            block_squares = np.add.reduceat(deviations**2, starts)

            previous = counts[index]
            combined = previous + size
            delta = block_mean - mean_error[index]
            mean_error[index] += delta * size / combined
            centered_sum_squares[index] += (
                block_squares + delta**2 * previous * size / combined
            )
            counts[index] = combined
            total_items[index] += np.add.reduceat(block.n_items, starts)
    except _Fallback:
        engine.rng.bit_generator.state = state
        return None

    mse = centered_sum_squares / n_replications + mean_error**2
    return mean_error, mse, total_items / n_replications
