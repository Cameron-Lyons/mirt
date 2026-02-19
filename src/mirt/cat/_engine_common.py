"""Shared helper utilities for CAT and MCAT engines."""

from __future__ import annotations

from typing import Any

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
    """Normalize exposure-control configuration."""
    if exposure_control is None:
        return NoExposureControl()
    if isinstance(exposure_control, str):
        return create_exposure_control(exposure_control, seed=seed)
    return exposure_control


def configure_content_constraint(
    content_constraint: ContentConstraint | None,
) -> ContentConstraint:
    """Normalize content-constraint configuration."""
    if content_constraint is None:
        return NoContentConstraint()
    return content_constraint


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

    engine._exposure.reset()
    engine._content.reset()

    if hasattr(engine._stopping, "reset"):
        engine._stopping.reset()


def consume_pending_item(engine: Any) -> int:
    """Pop the pending item if present, else select one."""
    if not hasattr(engine, "_pending_item"):
        engine._pending_item = engine._select_next_item()

    item_idx = int(engine._pending_item)
    delattr(engine, "_pending_item")
    return item_idx


def finalize_administered_item(engine: Any, state: Any) -> None:
    """Update completion state after administering one item."""
    if engine._stopping.should_stop(state):
        engine._is_complete = True
        engine._stopping_reason = engine._stopping.get_reason()

    if not engine._available_items and not engine._is_complete:
        engine._is_complete = True
        engine._stopping_reason = "Item pool exhausted"


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
