"""Real-process checks for model-fitting worker configuration."""

import multiprocessing
import os

import numpy as np
import pytest

import mirt
from mirt.exceptions import MirtValidationError
from mirt.utils._parallel import _process_pool, ensure_picklable, resolve_n_jobs


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


_INITIALIZED_BACKEND = None


def _record_backend_at_initialization(marker):
    global _INITIALIZED_BACKEND
    _INITIALIZED_BACKEND = (marker, mirt.get_backend())


def _initialized_state():
    return _INITIALIZED_BACKEND


def test_caller_initializer_runs_after_the_backend_is_applied():
    previous_backend = mirt.get_backend()
    mirt.set_backend("numpy")
    try:
        with _process_pool(
            1,
            initializer=_record_backend_at_initialization,
            initargs=("ready",),
        ) as executor:
            state = executor.submit(_initialized_state).result(timeout=30)
    finally:
        mirt.set_backend(previous_backend)

    assert state == ("ready", "numpy")


@pytest.mark.parametrize(
    ("n_jobs", "n_tasks", "expected"),
    [(1, None, 1), (3, None, 3), (np.int64(2), None, 2), (8, 3, 3), (4, 0, 1)],
)
def test_resolve_n_jobs_caps_workers_at_the_task_count(n_jobs, n_tasks, expected):
    resolved = resolve_n_jobs(n_jobs, n_tasks)
    assert resolved == expected
    assert type(resolved) is int


def test_resolve_n_jobs_uses_every_cpu_for_minus_one(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 6)
    assert resolve_n_jobs(-1) == 6
    assert resolve_n_jobs(-1, 4) == 4
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert resolve_n_jobs(-1) == 1


@pytest.mark.parametrize("n_jobs", [0, -2, True, np.bool_(True), 1.5, "2", None])
def test_resolve_n_jobs_rejects_invalid_counts(n_jobs):
    with pytest.raises(MirtValidationError, match="n_jobs"):
        resolve_n_jobs(n_jobs)


def test_ensure_picklable_explains_local_callables():
    ensure_picklable(_worker_state, [1, 2], n_jobs=2)
    with pytest.raises(MirtValidationError, match="picklable"):
        ensure_picklable(lambda value: value, n_jobs=2)
