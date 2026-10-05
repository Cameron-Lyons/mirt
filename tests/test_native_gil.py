"""Long-running native estimation kernels must let other Python threads run."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable

import numpy as np
import pytest

from mirt._rust_backend import RUST_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension unavailable")


def _responses() -> np.ndarray:
    rng = np.random.default_rng(7)
    theta = rng.standard_normal(1500)
    difficulty = np.linspace(-1.5, 1.5, 12)
    probability = 1.0 / (1.0 + np.exp(-(theta[:, None] - difficulty)))
    return (rng.random(probability.shape) < probability).astype(np.int32)


def _spinner_rate_ratio(call: Callable[[], object]) -> float:
    """Rate of a pure-Python loop while ``call`` runs, relative to its idle rate.

    A call that keeps the GIL only lets the spinning thread run around its
    boundaries (a few switch intervals, plus scheduling delays on a loaded host),
    so the ratio falls towards zero as the call gets longer.
    """
    ticks = 0
    started = threading.Event()
    stop = threading.Event()

    def spin() -> None:
        nonlocal ticks
        started.set()
        while not stop.is_set():
            ticks += 1

    previous_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-4)
    thread = threading.Thread(target=spin, daemon=True)
    try:
        thread.start()
        started.wait()
        before, start = ticks, time.perf_counter()
        time.sleep(0.05)
        idle_rate = (ticks - before) / (time.perf_counter() - start)
        before, start = ticks, time.perf_counter()
        call()
        busy_rate = (ticks - before) / (time.perf_counter() - start)
        return busy_rate / idle_rate
    finally:
        stop.set()
        thread.join()
        sys.setswitchinterval(previous_interval)


@pytest.mark.parametrize("kernel", ["gibbs", "mhrm", "bootstrap", "fixed_calib"])
def test_native_estimation_kernels_release_the_gil(kernel: str) -> None:
    from mirt import mirt_rs

    responses = _responses()
    grid = np.linspace(-4.0, 4.0, 61)
    weights = np.exp(-0.5 * grid**2)
    weights /= weights.sum()
    calls = {
        "gibbs": lambda: mirt_rs.gibbs_sample_2pl(responses, 200, 20, 1, 3),
        "mhrm": lambda: mirt_rs.mhrm_fit_2pl(responses, 400, 40, 0.5, 3),
        "bootstrap": lambda: mirt_rs.bootstrap_fit_2pl(
            responses, 16, 21, 60, 1e-6, 3, None, None
        ),
        "fixed_calib": lambda: mirt_rs.fixed_calib_em(
            responses,
            list(range(6)),
            list(range(6, 12)),
            np.ones(6),
            np.linspace(-1.5, 1.5, 12)[:6],
            grid,
            weights,
            100,
            1e-300,
        ),
    }

    # Each call runs for hundreds of milliseconds. With the GIL released the
    # spinner keeps a large share of its idle rate even while sharing cores with
    # the native workers; holding it leaves about 1% of that rate.
    assert _spinner_rate_ratio(calls[kernel]) > 0.1


def test_multigroup_e_step_releases_the_gil() -> None:
    from mirt import mirt_rs

    rng = np.random.default_rng(3)
    n_items = 120
    categories = np.full(n_items, 4, dtype=np.int32)
    responses = [
        rng.integers(-1, 4, (20_000, n_items), dtype=np.int32) for _ in range(2)
    ]
    grid = np.linspace(-4.0, 4.0, 61)
    weights = np.exp(-0.5 * grid**2)
    weights /= weights.sum()
    thresholds = np.tile(np.linspace(-1.0, 1.0, 3), (n_items, 1))

    def call() -> object:
        return mirt_rs.multigroup_e_step_grm(
            responses,
            grid,
            weights,
            [np.ones(n_items), np.ones(n_items)],
            [thresholds, thresholds],
            [categories, categories],
            np.zeros(2),
            np.ones(2),
        )

    assert _spinner_rate_ratio(call) > 0.1
