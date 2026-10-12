"""Regression tests for the lightweight public Rust backend namespace."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

EXPECTED_EXPORTS = (
    "RUST_AVAILABLE",
    "is_rust_available",
    "rust_enabled",
    "mhrm_fit_2pl",
    "bootstrap_fit_2pl",
    "em_iteration_3pl",
    "simulate_grm",
    "simulate_gpcm",
    "simulate_dichotomous",
    "cat_simulate_batch_full",
    "cat_conditional_mse",
    "compute_log_likelihoods_grm",
    "compute_log_likelihoods_gpcm",
    "compute_alpha_if_deleted",
    "multigroup_e_step_grm",
    "multigroup_e_step_gpcm",
    "multigroup_e_step_2pl",
    "multigroup_e_step_3pl",
    "multigroup_e_step_nrm",
    "coordinate_descent_mstep_regularized",
    "rt_joint_log_likelihood",
    "rt_accept_person_proposals",
    "bkt_forward",
    "bkt_backward",
    "bkt_forward_backward_batch",
    "bkt_viterbi",
    "observed_score_distribution_2pl",
)


def _run_probe(source: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_public_backend_export_order_is_unchanged() -> None:
    import mirt.backends.rust as rust_backend

    assert rust_backend.__all__ == list(EXPECTED_EXPORTS)


def test_every_backend_export_has_a_package_caller() -> None:
    import mirt.backends.rust as rust_backend

    package = Path(rust_backend.__file__).resolve().parents[2]
    backend = Path(rust_backend.__file__).resolve().parent
    # Count names used in code, including lazy-import registry strings, but
    # not mentions in comments or docstrings.
    referenced: set[str] = set()
    for path in package.rglob("*.py"):
        if backend in path.resolve().parents:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Name):
                referenced.add(node.id)
            elif isinstance(node, ast.Attribute):
                referenced.add(node.attr)
            elif isinstance(node, ast.alias):
                referenced.add(node.name)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                referenced.add(node.value)
    unused = [name for name in rust_backend.__all__ if name not in referenced]

    assert not unused, (
        "mirt.backends.rust exports wrappers that no package module outside "
        f"the backend calls; route a caller to them or delete them: {unused}"
    )


def test_compatibility_shim_defers_and_caches_symbols() -> None:
    result = _run_probe(
        """
        import json
        import sys
        import mirt._rust_backend as legacy

        unloaded = "numpy" not in sys.modules
        available = legacy.RUST_AVAILABLE
        modules = sorted(name for name in sys.modules
                         if name.startswith("mirt.backends.rust."))
        cached = "RUST_AVAILABLE" in legacy.__dict__
        from mirt.backends.rust._helpers import RUST_AVAILABLE
        try:
            legacy.no_such_function
        except AttributeError:
            unknown_rejected = True
        else:
            unknown_rejected = False
        print(json.dumps({
            "unloaded": unloaded,
            "modules": modules,
            "cached": cached,
            "same": available is RUST_AVAILABLE,
            "unknown_rejected": unknown_rejected,
            "exports_visible": all(name in dir(legacy) for name in legacy.__all__),
        }))
        """
    )
    assert result == {
        "unloaded": True,
        "modules": ["mirt.backends.rust._helpers"],
        "cached": True,
        "same": True,
        "unknown_rejected": True,
        "exports_visible": True,
    }


def test_compatibility_star_import_retains_all_exports() -> None:
    import mirt.backends.rust as backend

    namespace = {}
    exec("from mirt._rust_backend import *", namespace)
    for name in EXPECTED_EXPORTS:
        assert namespace[name] is getattr(backend, name)
    assert namespace["_MAX_VECTOR_CHUNK_ENTRIES"] == 1_000_000


def test_plain_backend_import_defers_dependencies_and_modules() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt.backends.rust as rust_backend

        print(json.dumps({
            "numpy_loaded": "numpy" in sys.modules,
            "native_loaded": "mirt.mirt_rs" in sys.modules,
            "backend_submodules": sorted(
                name for name in sys.modules
                if name.startswith("mirt.backends.rust.")
            ),
            "export_count": len(rust_backend.__all__),
            "exports_visible": all(
                name in dir(rust_backend) for name in rust_backend.__all__
            ),
        }))
        """
    )

    assert result == {
        "numpy_loaded": False,
        "native_loaded": False,
        "backend_submodules": [],
        "export_count": len(EXPECTED_EXPORTS),
        "exports_visible": True,
    }


def test_backend_symbol_resolution_is_cached_and_scoped() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt.backends.rust as rust_backend

        deferred_before_access = "cat_conditional_mse" not in rust_backend.__dict__
        symbol = rust_backend.cat_conditional_mse

        from mirt.backends.rust.cat import cat_conditional_mse

        feature_modules = (
            "calibration",
            "cat",
            "diagnostics",
            "dynamic",
            "eapsum",
            "equating",
            "estep",
            "estimation",
            "gvem",
            "likelihood",
            "mstep",
            "multigroup",
            "plausible",
            "polytomous",
            "regularized",
            "response_time",
            "scoring",
            "simulation",
        )

        print(json.dumps({
            "deferred_before_access": deferred_before_access,
            "cached": rust_backend.__dict__["cat_conditional_mse"] is symbol,
            "same_symbol": symbol is cat_conditional_mse,
            "loaded_features": [
                name for name in feature_modules
                if f"mirt.backends.rust.{name}" in sys.modules
            ],
            "backend_submodules": sorted(
                name for name in sys.modules
                if name.startswith("mirt.backends.rust.")
            ),
        }))
        """
    )

    assert result == {
        "deferred_before_access": True,
        "cached": True,
        "same_symbol": True,
        "loaded_features": ["cat"],
        "backend_submodules": [
            "mirt.backends.rust._helpers",
            "mirt.backends.rust.cat",
        ],
    }


def test_star_import_resolves_every_public_backend_export() -> None:
    result = _run_probe(
        """
        import json

        import mirt.backends.rust as rust_backend

        namespace = {}
        exec("from mirt.backends.rust import *", namespace)
        exported = {name for name in namespace if not name.startswith("__")}

        print(json.dumps({
            "matches_all": exported == set(rust_backend.__all__),
            "export_count": len(exported),
            "all_cached": all(
                name in rust_backend.__dict__ for name in rust_backend.__all__
            ),
        }))
        """
    )

    assert result == {
        "matches_all": True,
        "export_count": len(EXPECTED_EXPORTS),
        "all_cached": True,
    }


def test_submodule_import_fallback_and_unknown_attribute_error() -> None:
    result = _run_probe(
        """
        import json

        import mirt.backends.rust as rust_backend
        from mirt.backends.rust import simulation

        try:
            rust_backend.NotABackendFunction
        except AttributeError as error:
            message = str(error)
        else:
            message = None

        print(json.dumps({
            "submodule_name": simulation.__name__,
            "error": message,
        }))
        """
    )

    assert result == {
        "submodule_name": "mirt.backends.rust.simulation",
        "error": "module 'mirt.backends.rust' has no attribute 'NotABackendFunction'",
    }
