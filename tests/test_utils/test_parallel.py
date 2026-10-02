"""Real-process checks for model-fitting worker configuration."""

import multiprocessing
import os

import pytest

import mirt
from mirt.utils._parallel import _process_pool


def _worker_state():
    return os.getpid(), mirt.get_backend(), multiprocessing.get_start_method()


@pytest.mark.parametrize("backend", ["auto", "numpy"])
@pytest.mark.parametrize("start_method", [None, "spawn", "fork"])
def test_workers_preserve_backend_and_use_requested_context(backend, start_method):
    if start_method not in (None, *multiprocessing.get_all_start_methods()):
        pytest.skip(f"{start_method} is unavailable on this platform")
    context = (
        None if start_method is None else multiprocessing.get_context(start_method)
    )
    previous_backend = mirt.get_backend()
    previous_start_method = multiprocessing.get_start_method()
    mirt.set_backend(backend)
    try:
        with _process_pool(1, context) as executor:
            worker_pid, worker_backend, worker_start_method = executor.submit(
                _worker_state
            ).result(timeout=30)

        assert worker_pid != os.getpid()
        assert worker_backend == backend
        assert worker_start_method == (start_method or "spawn")
        assert mirt.get_backend() == backend
        assert (
            multiprocessing.get_start_method(allow_none=True) == previous_start_method
        )
    finally:
        mirt.set_backend(previous_backend)
