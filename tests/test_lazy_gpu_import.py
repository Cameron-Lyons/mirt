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
        "gpu_build = backend._torch_build_supports_gpu() is not False; "
        "assert ('torch' in sys.modules) == (backend.is_torch_available() and gpu_build)"
    )

    assert result.returncode == 0, result.stderr


def test_default_fits_do_not_import_torch_without_gpu_support():
    # Regression: the first default fit imported PyTorch (0.5 s, 190 MB) to
    # learn that a CPU-only build has no GPU, and imported scipy.stats.
    result = _run_isolated(
        "import sys; import mirt; "
        "from mirt._gpu_backend import _torch_build_supports_gpu; "
        "data = mirt.simdata(n_persons=100, n_items=5, seed=1); "
        "mirt.fit_mirt(data, model='2PL', max_iter=5); "
        "assert 'scipy.stats' not in sys.modules, 'scipy.stats'; "
        "mirt.fit_mirt(data, model='3PL', max_iter=5); "
        "gpu_build = _torch_build_supports_gpu() is not False; "
        "assert 'torch' not in sys.modules or gpu_build, 'torch'"
    )

    assert result.returncode == 0, result.stderr


def test_cpu_backend_preferences_never_import_torch():
    result = _run_isolated(
        "import sys; import mirt; "
        "data = mirt.simdata(n_persons=100, n_items=5, seed=1); "
        "backends = ['numpy'] + (['rust'] if mirt.is_rust_available() else []); "
        "[(mirt.set_backend(name), mirt.fit_mirt(data, model='3PL', max_iter=5)) "
        "for name in backends]; "
        "assert 'torch' not in sys.modules"
    )

    assert result.returncode == 0, result.stderr


def test_torch_build_probe_reads_the_version_file(tmp_path, monkeypatch):
    from importlib.machinery import ModuleSpec

    import mirt._gpu_backend as backend

    package = tmp_path / "torch"
    package.mkdir()
    spec = ModuleSpec("torch", None, is_package=True)
    spec.submodule_search_locations = [str(package)]
    monkeypatch.setattr(backend, "find_spec", lambda name: spec)
    cases = [
        ("cuda: Optional[str] = None\nhip: Optional[str] = None\n", False),
        ("cuda = None\nhip = '6.2.4'\n", True),
        ("cuda: Optional[str] = '12.4'\nhip = None\n", True),
        ("cuda = _detect()\n", None),
        ("not python (", None),
    ]
    try:
        for source, expected in cases:
            (package / "version.py").write_text(source)
            backend._torch_build_supports_gpu.cache_clear()
            assert backend._torch_build_supports_gpu() is expected, source
    finally:
        backend._torch_build_supports_gpu.cache_clear()
