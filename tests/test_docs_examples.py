"""Regression tests for documentation examples and symbol references."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import pkgutil
import re
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import mirt

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _restore_lazy_namespaces():
    """Undo lazy exports these checks resolve, so lazy-import tests stay valid."""
    # Import the (lazy) subpackages first so their pristine namespaces are
    # recorded; exports these checks resolve are then removed afterwards.
    for info in pkgutil.iter_modules(mirt.__path__, "mirt."):
        if info.ispkg:
            importlib.import_module(info.name)
    before = {
        name: set(vars(module))
        for name, module in list(sys.modules.items())
        if name == "mirt" or name.startswith("mirt.")
    }
    yield
    for name, keys in before.items():
        module = sys.modules.get(name)
        if module is None:
            continue
        namespace = vars(module)
        for key in set(namespace) - keys:
            if not isinstance(namespace[key], ModuleType):
                del namespace[key]


def _autosummary_entries(text: str) -> list[str]:
    """Return the object names listed in the autosummary blocks of ``text``."""
    entries: list[str] = []
    in_block = False
    for line in text.splitlines():
        if line.startswith(".. autosummary::"):
            in_block = True
            continue
        if not in_block:
            continue
        stripped = line.strip()
        if line and not line.startswith(" "):
            in_block = False
        elif stripped and not stripped.startswith(":"):
            entries.append(stripped)
    return entries


def test_toctree_lists_every_guide() -> None:
    """Every guide page must be reachable from the documentation index."""
    index = (REPO_ROOT / "docs" / "index.rst").read_text(encoding="utf-8")
    listed = set(re.findall(r"^\s+(guides/\w+)\s*$", index, re.MULTILINE))
    guides = {
        f"guides/{path.stem}" for path in (REPO_ROOT / "docs" / "guides").glob("*.rst")
    }
    assert guides <= listed, f"guides missing from the toctree: {guides - listed}"


def test_api_reference_entries_resolve() -> None:
    """Every autosummary entry in the API reference must import from mirt."""
    text = (REPO_ROOT / "docs" / "api" / "index.rst").read_text(encoding="utf-8")
    entries = _autosummary_entries(text)
    assert "fit_mirt" in entries and "cat.ShadowTestSelection" in entries

    missing = []
    for entry in entries:
        *modules, name = entry.split(".")
        module = importlib.import_module(".".join(["mirt", *modules]))
        if not hasattr(module, name):
            missing.append(entry)
    assert not missing, f"API reference entries that do not resolve: {missing}"


def _python_blocks(path: Path) -> list[str]:
    """Return the Python code blocks of a Markdown or reStructuredText file."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".md":
        return re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    blocks: list[str] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        directive = re.match(r"(\s*)\.\. code-block:: python\s*$", lines[index])
        index += 1
        if directive is None:
            continue
        indent = len(directive.group(1))
        body = []
        while index < len(lines) and (
            not lines[index].strip()
            or len(lines[index]) - len(lines[index].lstrip()) > indent
        ):
            body.append(lines[index])
            index += 1
        code = textwrap.dedent("\n".join(body)).splitlines()
        blocks.append("\n".join(line for line in code if not line.startswith(":")))
    return blocks


def test_documentation_code_parses_and_mirt_names_resolve() -> None:
    """README and guide code must parse and name existing mirt objects.

    A ``from`` import must not name a submodule: ``from mirt.utils import
    residuals`` bound the ``mirt.utils.residuals`` module instead of the
    function once ``Q3`` had been imported from the same package.
    """
    docs = REPO_ROOT / "docs"
    paths = [REPO_ROOT / "README.md"]
    paths += [
        path
        for path in sorted(docs.rglob("*.rst"))
        if not {"generated", "_build"} & set(path.relative_to(docs).parts)
    ]
    problems = []
    n_blocks = 0
    for path in paths:
        for number, code in enumerate(_python_blocks(path)):
            n_blocks += 1
            where = f"{path.relative_to(REPO_ROOT)} code block {number}"
            tree = ast.parse(code, filename=where)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] == "mirt":
                            importlib.import_module(alias.name)
                elif isinstance(node, ast.ImportFrom) and node.module is not None:
                    if node.module.split(".")[0] != "mirt":
                        continue
                    module = importlib.import_module(node.module)
                    for alias in node.names:
                        submodule = f"{node.module}.{alias.name}"
                        if not hasattr(module, alias.name) or (
                            hasattr(module, "__path__")
                            and importlib.util.find_spec(submodule) is not None
                        ):
                            problems.append(
                                f"{where}: from {node.module} import {alias.name}"
                            )
                elif (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "mirt"
                    and not hasattr(mirt, node.attr)
                ):
                    problems.append(f"{where}: mirt.{node.attr}")
    assert n_blocks > 100
    assert not problems, "documentation code names missing objects:\n" + "\n".join(
        problems
    )


def test_docs_do_not_reference_removed_model_aliases() -> None:
    """Documentation should only reference currently supported model symbols."""
    docs_files = (
        Path("docs/index.rst"),
        Path("docs/quickstart.rst"),
        Path("docs/api/index.rst"),
    )
    removed_symbols = ("Model1PL", "Model2PL", "Model3PL")

    for doc_file in docs_files:
        text = doc_file.read_text(encoding="utf-8")
        for symbol in removed_symbols:
            assert symbol not in text, f"{symbol} found in {doc_file}"


def test_api_reference_symbols_exist() -> None:
    """Symbols listed in docs/api/index.rst should resolve from top-level mirt."""
    documented_symbols = (
        "fit_mirt",
        "fscores",
        "itemfit",
        "personfit",
        "dif",
        "load_dataset",
        "list_datasets",
        "OneParameterLogistic",
        "TwoParameterLogistic",
        "ThreeParameterLogistic",
        "FourParameterLogistic",
        "GradedResponseModel",
        "GeneralizedPartialCredit",
        "PartialCreditModel",
        "NominalResponseModel",
        "EMEstimator",
        "MHRMEstimator",
        "GibbsSampler",
        "BLEstimator",
        "compute_fit_indices",
        "compare_models",
        "anova_irt",
        "compute_dtf",
        "compute_drf",
        "sibtest",
    )
    missing = [symbol for symbol in documented_symbols if not hasattr(mirt, symbol)]
    assert not missing, f"Undocumented or missing symbols: {missing}"


def test_quickstart_fit_and_score_smoke() -> None:
    """Core quickstart flow should run end-to-end on bundled sample data."""
    responses = mirt.load_dataset("LSAT7")["data"][:100]

    result = mirt.fit_mirt(
        responses,
        model="2PL",
        n_quadpts=11,
        max_iter=30,
        tol=1e-3,
    )
    scores = mirt.fscores(result, responses, method="EAP", n_quadpts=21)

    assert result.model.model_name == "2PL"
    assert scores.theta.shape[0] == responses.shape[0]
    assert np.all(np.isfinite(scores.theta))
