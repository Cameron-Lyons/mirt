"""The native extension must export exactly the kernels the package calls.

The extension is a ``cdylib`` whose ``#[pyfunction]`` items are always
"used" by their module registration, so rustc cannot report kernels that no
Python code calls. These tests keep the registered functions and the
``mirt_rs.<name>`` references in the package in sync.
"""

from __future__ import annotations

import ast
from functools import cache
from pathlib import Path

import pytest

import mirt
from mirt._rust_backend import RUST_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension unavailable")

PACKAGE_DIR = Path(mirt.__file__).resolve().parent


@cache
def _native_references() -> dict[str, frozenset[str]]:
    """Map each ``mirt_rs.<name>`` attribute used in the package to its files."""
    references: dict[str, set[str]] = {}
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "mirt_rs"
            ):
                relative = path.relative_to(PACKAGE_DIR).as_posix()
                references.setdefault(node.attr, set()).add(relative)
    return {name: frozenset(files) for name, files in references.items()}


def _native_exports() -> set[str]:
    from mirt import mirt_rs

    return {
        name
        for name in dir(mirt_rs)
        if not name.startswith("_") and callable(getattr(mirt_rs, name))
    }


def test_reference_scan_finds_the_backend_wrappers() -> None:
    references = _native_references()

    assert "backends/rust/estep.py" in references["e_step_complete"]
    assert "estimation/_em_context.py" in references["EMThreadPool"]


def test_every_native_export_is_called_by_the_package() -> None:
    unused = sorted(_native_exports() - set(_native_references()))

    assert not unused, (
        "mirt_rs exports functions that no package module calls; wire them "
        f"through mirt.backends.rust or delete them from rust_src: {unused}"
    )


def test_every_native_reference_is_exported() -> None:
    exports = _native_exports()
    missing = {
        name: sorted(files)
        for name, files in _native_references().items()
        if name not in exports
    }

    assert not missing, f"package code references missing mirt_rs kernels: {missing}"
