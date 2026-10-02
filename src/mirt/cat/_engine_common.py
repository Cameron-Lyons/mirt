"""Shared helper utilities for CAT and MCAT engines."""

from __future__ import annotations

from typing import Any

import numpy as np

from mirt.cat.content import ContentConstraint, NoContentConstraint
from mirt.cat.exposure import (
    ExposureControl,
    NoExposureControl,
    create_exposure_control,
)


def configure_exposure_control(
    exposure_control: ExposureControl | str | None,
    *,
    seed: int | None,
) -> ExposureControl:
    """Normalize exposure control and open the engine's initial session."""
    if exposure_control is None:
        control = NoExposureControl()
    elif isinstance(exposure_control, str):
        control = create_exposure_control(exposure_control, seed=seed)
    else:
        control = exposure_control
    control.reset()
    return control


def configure_content_constraint(
    content_constraint: ContentConstraint | None,
) -> ContentConstraint:
    """Normalize content-constraint configuration."""
    if content_constraint is None:
        return NoContentConstraint()
    return content_constraint


def validate_replications(n_replications: int) -> int:
    """Validate a replication count before any simulation changes session state."""
    if (
        isinstance(n_replications, (bool, np.bool_))
        or not isinstance(n_replications, (int, np.integer))
        or n_replications < 1
    ):
        raise ValueError("n_replications must be a positive integer")
    return int(n_replications)


def validate_simulation_values(values: Any, *, name: str) -> np.ndarray:
    """Return finite real ability values without silently casting complex inputs."""
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def initialize_common_engine(
    engine: Any,
    *,
    model: Any,
    scoring_method: str,
    n_quadpts: int,
    theta_bounds: tuple[float, float],
    seed: int | None,
    engine_name: str,
) -> None:
    """Initialize attributes shared by CAT and MCAT engines."""
    if not model.is_fitted:
        raise ValueError(f"Model must be fitted before use in {engine_name}")

    engine.model = model
    engine.scoring_method = scoring_method
    engine.n_quadpts = n_quadpts
    engine.theta_bounds = theta_bounds
    engine.seed = seed
    engine.rng = np.random.default_rng(seed)
    engine._exposure_session_used = False


def reset_session_state(
    engine: Any,
    *,
    n_items: int,
    history_attrs: tuple[str, ...],
) -> None:
    """Reset shared adaptive-session state for CAT/MCAT engines."""
    engine._items_administered = []
    engine._responses = []
    engine._available_items = set(range(n_items))
    for attr in history_attrs:
        setattr(engine, attr, [])
    engine._is_complete = False
    engine._stopping_reason = ""
    if hasattr(engine, "_pending_item"):
        delattr(engine, "_pending_item")

    # Construction already opens the first exposure session. Resetting before
    # any item was selected reuses that unused session, including the initial
    # reset performed by run_simulation. This prevents phantom examinees.
    if engine._exposure_session_used:
        engine._exposure.reset()
    engine._exposure_session_used = False
    engine._content.reset()

    if hasattr(engine._stopping, "reset"):
        engine._stopping.reset()


def get_pending_item(engine: Any) -> int:
    """Select an item once and retain it until a response is recorded."""
    if not hasattr(engine, "_pending_item"):
        # A selection attempt may change exposure-control eligibility state,
        # even when a downstream constraint rejects the item pool.
        engine._exposure_session_used = True
        engine._pending_item = engine._select_next_item()

    return int(engine._pending_item)


def consume_pending_item(engine: Any, response: int) -> tuple[int, int]:
    """Validate a response before consuming its selected item.

    Integer-valued numeric scalars are accepted, including NumPy scalars.
    A rejected response leaves the pending item available for a corrected
    answer and does not count an item as administered or exposed.
    """
    response_code = _normalize_response_code(response)

    if response_code < 0:
        raise ValueError("response must be a non-negative category code")
    if (
        not engine.model.is_polytomous
        and response_code > 1
        and not hasattr(engine, "_pending_item")
    ):
        raise ValueError("dichotomous response must be 0 or 1")

    item_idx = get_pending_item(engine)
    response_code = validate_item_response(engine, item_idx, response_code)

    delattr(engine, "_pending_item")
    return item_idx, response_code


def _normalize_response_code(response: Any) -> int:
    """Normalize a numeric scalar without truncating fractional category codes."""
    try:
        value = np.asarray(response)
    except (TypeError, ValueError) as exc:
        raise ValueError("response must be a finite integer category") from exc
    if value.ndim != 0 or value.dtype.kind not in "biuf":
        raise ValueError("response must be a finite integer category")
    if value.dtype.kind == "f" and (not np.isfinite(value) or value != np.floor(value)):
        raise ValueError("response must be a finite integer category")
    return int(value)


def validate_item_response(engine: Any, item_idx: int, response: Any) -> int:
    """Validate a response before consuming the disclosed item or updating state."""
    category = _normalize_response_code(response)
    n_categories = (
        int(engine.model.n_categories[item_idx]) if engine.model.is_polytomous else 2
    )
    if not 0 <= category < n_categories:
        raise ValueError(
            f"response for item {item_idx} must be between 0 and {n_categories - 1}"
        )
    return category


def finalize_administered_item(engine: Any, state: Any) -> None:
    """Update completion state after administering one item."""
    if engine._stopping.should_stop(state):
        engine._is_complete = True
        engine._stopping_reason = engine._stopping.get_reason()

    if not engine._available_items and not engine._is_complete:
        engine._is_complete = True
        engine._stopping_reason = "Item pool exhausted"

    if engine._is_complete and hasattr(engine, "_pending_item"):
        delattr(engine, "_pending_item")


def run_simulation_loop(
    engine: Any,
    true_theta: Any,
    *,
    response_generator: Any | None = None,
    reset: bool = True,
) -> Any:
    """Run the shared adaptive simulation loop."""
    if reset:
        engine.reset()

    while not engine._is_complete:
        item_idx = engine.select_next_item()
        engine._pending_item = item_idx

        if response_generator is not None:
            response = response_generator(item_idx, true_theta)
        else:
            response = engine._generate_response(item_idx, true_theta)

        engine.administer_item(response)

    return engine.get_result()


def simulate_error_moments(
    engine: Any,
    true_theta: Any,
    *,
    n_replications: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Accumulate conditional errors without retaining replicated estimates.

    Welford's recurrence computes each factor's mean and centered second
    moment, keeping the MSE stable when estimates have a large common offset.
    Working storage depends on the number of factors rather than replications.
    """
    theta = np.asarray(true_theta, dtype=np.float64)
    mean_error = np.zeros_like(theta)
    centered_sum_squares = np.zeros_like(theta)
    mean_items = 0.0
    for count in range(1, n_replications + 1):
        result = engine.run_simulation(true_theta)
        error = np.asarray(result.theta, dtype=np.float64) - theta
        delta = error - mean_error
        mean_error += delta / count
        centered_sum_squares += delta * (error - mean_error)
        mean_items += (result.n_items_administered - mean_items) / count

    mse = centered_sum_squares / n_replications + mean_error**2
    return mean_error, mse, mean_items


def record_item_administration(
    engine: Any,
    *,
    item_idx: int,
    response: int,
    theta_arr: Any,
) -> None:
    """Record item-level state updates after an administered response."""
    item_info = float(engine.model.information(theta_arr, item_idx=item_idx).sum())
    engine._info_history.append(item_info)

    engine._items_administered.append(item_idx)
    engine._responses.append(response)
    engine._available_items.discard(item_idx)
    engine._exposure.update(item_idx)


def build_administered_response_matrix(engine: Any) -> np.ndarray:
    """Build a sparse response matrix from administered CAT/MCAT items."""
    responses = np.full((1, engine.model.n_items), -1, dtype=np.int_)
    for item_idx, resp in zip(
        engine._items_administered, engine._responses, strict=True
    ):
        responses[0, item_idx] = resp
    return responses


def score_administered_responses(
    engine: Any,
    *,
    bounds: tuple[float, float] | None = None,
) -> Any:
    """Run `fscores` for the current administered responses."""
    from mirt.scoring import fscores

    scoring_kwargs = {
        "method": engine.scoring_method,
        "n_quadpts": engine.n_quadpts,
    }
    if bounds is not None:
        scoring_kwargs["bounds"] = bounds

    responses = build_administered_response_matrix(engine)
    return fscores(engine.model, responses, **scoring_kwargs)
