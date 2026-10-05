"""Consistent process workers for independent model-fitting tasks."""

from __future__ import annotations

import os
import pickle
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from multiprocessing.context import BaseContext
from typing import Any

import numpy as np

from mirt._backend_state import get_backend_preference, set_backend_preference
from mirt.exceptions import MirtValidationError


def resolve_n_jobs(n_jobs: int, n_tasks: int | None = None) -> int:
    """Validate a worker count and resolve ``-1`` to every available CPU.

    Parameters
    ----------
    n_jobs : int
        Requested workers: ``-1`` or a positive integer.
    n_tasks : int, optional
        Number of independent tasks. When given, the result is capped at it
        (but never below one), because extra workers would stay idle.

    Returns
    -------
    int
        Positive number of workers to start.
    """
    if (
        isinstance(n_jobs, (bool, np.bool_))
        or not isinstance(n_jobs, (int, np.integer))
        or n_jobs == 0
        or n_jobs < -1
    ):
        raise MirtValidationError(
            "n_jobs must be -1 or a positive integer",
            parameter="n_jobs",
            value=n_jobs,
        )
    resolved = (os.cpu_count() or 1) if n_jobs == -1 else int(n_jobs)
    if n_tasks is not None:
        resolved = max(1, min(resolved, int(n_tasks)))
    return resolved


def ensure_picklable(*values: Any, n_jobs: int) -> None:
    """Fail early, with advice, when process workers cannot receive a task."""
    try:
        pickle.dumps(values)
    except (AttributeError, pickle.PickleError, TypeError) as exc:
        raise MirtValidationError(
            "parallel inputs must be picklable; use n_jobs=1 for locally "
            "defined models or callables",
            parameter="n_jobs",
            value=n_jobs,
        ) from exc


def _initialize_worker(
    backend: str,
    initializer: Callable[..., object] | None,
    initargs: tuple[Any, ...],
) -> None:
    """Apply the parent's backend before any caller-specific worker setup."""
    set_backend_preference(backend)
    if initializer is not None:
        initializer(*initargs)


def _process_pool(
    max_workers: int,
    mp_context: BaseContext | None = None,
    *,
    initializer: Callable[..., object] | None = None,
    initargs: tuple[Any, ...] = (),
) -> ProcessPoolExecutor:
    """Create workers without inheriting native-library threads or global state.

    Spawn works consistently across platforms and does not need a forkserver
    socket. Forking after the native backend starts its thread pool can
    deadlock the child. Callers may supply their own context without changing
    the global multiprocessing start method. Fresh workers use the parent's
    backend preference, including an explicit request to disable native
    acceleration, and then run ``initializer(*initargs)`` when one is given.
    """
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=get_context("spawn") if mp_context is None else mp_context,
        initializer=_initialize_worker,
        initargs=(get_backend_preference(), initializer, initargs),
    )
