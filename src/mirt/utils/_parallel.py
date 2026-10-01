"""Consistent process workers for independent model-fitting tasks."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from multiprocessing.context import BaseContext

from mirt._backend_state import get_backend_preference, set_backend_preference


def _process_pool(
    max_workers: int,
    mp_context: BaseContext | None = None,
) -> ProcessPoolExecutor:
    """Create workers without inheriting native-library threads or global state.

    Spawn works consistently across platforms and does not need a forkserver
    socket. Callers may supply their own context without changing the global
    multiprocessing start method. Fresh workers use the parent's backend
    preference, including an explicit request to disable native acceleration.
    """
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=get_context("spawn") if mp_context is None else mp_context,
        initializer=set_backend_preference,
        initargs=(get_backend_preference(),),
    )
