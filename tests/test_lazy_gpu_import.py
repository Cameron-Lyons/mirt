"""Regression coverage for optional GPU runtime loading."""

from __future__ import annotations

import subprocess
import sys


def _run_isolated(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        text=True,
    )


def test_estimator_imports_do_not_load_torch():
    result = _run_isolated(
        "import sys; "
        "from mirt.estimation.em import EMEstimator; "
        "from mirt.estimation.gvem import GVEMEstimator; "
        "assert EMEstimator and GVEMEstimator; "
        "assert 'torch' not in sys.modules"
    )

    assert result.returncode == 0, result.stderr


def test_availability_probe_does_not_load_torch():
    result = _run_isolated(
        "import sys; "
        "from mirt._gpu_backend import is_torch_available; "
        "assert isinstance(is_torch_available(), bool); "
        "assert 'torch' not in sys.modules"
    )

    assert result.returncode == 0, result.stderr


def test_explicit_cpu_estimators_do_not_probe_torch():
    result = _run_isolated(
        "import sys; "
        "from mirt.estimation.em import EMEstimator; "
        "from mirt.estimation.gvem import GVEMEstimator; "
        "assert EMEstimator(use_gpu=False)._should_use_gpu is False; "
        "assert GVEMEstimator(use_gpu=False)._should_use_gpu is False; "
        "assert 'torch' not in sys.modules"
    )

    assert result.returncode == 0, result.stderr


def test_legacy_gpu_flag_loads_runtime_on_demand():
    result = _run_isolated(
        "import sys; "
        "import mirt._gpu_backend as backend; "
        "assert 'torch' not in sys.modules; "
        "assert isinstance(backend.GPU_AVAILABLE, bool); "
        "assert ('torch' in sys.modules) == backend.is_torch_available()"
    )

    assert result.returncode == 0, result.stderr
