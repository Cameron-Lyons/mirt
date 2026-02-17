from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np
import pytest

import mirt


@pytest.mark.performance
def test_import_time_regression_guard() -> None:
    threshold_seconds = float(os.getenv("MIRT_IMPORT_THRESHOLD_SECONDS", "2.5"))
    command = (
        "import time; "
        "s = time.perf_counter(); "
        "import mirt; "
        "print(time.perf_counter() - s)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )
    import_seconds = float(completed.stdout.strip())
    assert import_seconds <= threshold_seconds


@pytest.mark.performance
def test_fit_time_regression_guard() -> None:
    threshold_seconds = float(os.getenv("MIRT_FIT_THRESHOLD_SECONDS", "12.0"))
    responses = mirt.simdata(model="2PL", n_persons=250, n_items=12, seed=123)

    start = time.perf_counter()
    result = mirt.fit_mirt(
        responses, model="2PL", n_quadpts=15, max_iter=120, tol=1e-3, use_rust=True
    )
    elapsed = time.perf_counter() - start

    assert np.isfinite(result.log_likelihood)
    assert elapsed <= threshold_seconds
