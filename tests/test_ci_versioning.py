"""Release automation must keep the versions used by installers synchronized."""

from __future__ import annotations

import runpy
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/bump_version.py"
bump_version = runpy.run_path(str(SCRIPT))["bump_version"]


@pytest.fixture
def release_project(tmp_path):
    root = tmp_path / "checkout with spaces"
    (root / "src/mirt").mkdir(parents=True)
    files = {
        "src/mirt/_version.py": (
            '"""Runtime package version."""\n'
            "__version__ = '12.8.9'  # Public version\n"
        ),
        "pyproject.toml": '[project]\nname = "mirt"\ndynamic = ["version"]\n',
        "Cargo.toml": (
            '[package]\nname = "mirt_rs"\n'
            'version = "12.8.9"  # Distribution metadata\n'
            '[dependencies]\nexternal = "12.8.9"\n'
        ),
        "Cargo.lock": (
            '# Cargo lockfile\nversion = 4\n\n[[package]]\nname = "external"\n'
            'version = "12.8.9"\nchecksum = "preserve-this"\n'
            '\n[[package]]\nname = "mirt_rs"\nversion = "12.8.9"\n'
            'dependencies = ["external"]\n'
        ),
        "uv.lock": (
            'version = 1\n\n[[package]]\nname = "mirt"\nversion = "12.8.9"\n'
            'source = { editable = "." }\n'
            '\n[[package]]\nname = "external"\nversion = "12.8.9"\n'
            'source = { registry = "https://pypi.org/simple" }\n'
        ),
    }
    for name, contents in files.items():
        (root / name).write_text(contents, encoding="utf-8")
    return root


def snapshot(root):
    return {
        path.relative_to(root): path.read_text(encoding="utf-8")
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    ("part", "expected"),
    [("major", "13.0.0"), ("minor", "12.9.0"), ("patch", "12.8.10")],
)
def test_bump_keeps_runtime_and_installer_versions_equal(
    release_project, part, expected
):
    before = snapshot(release_project)
    assert bump_version(release_project, part) == ("12.8.9", expected)

    runtime = runpy.run_path(str(release_project / "src/mirt/_version.py"))
    assert runtime["__version__"] == expected
    manifest = tomllib.loads((release_project / "Cargo.toml").read_text())
    assert manifest["package"]["version"] == expected
    assert manifest["dependencies"]["external"] == "12.8.9"
    for filename, name in [("Cargo.lock", "mirt_rs"), ("uv.lock", "mirt")]:
        lock = tomllib.loads((release_project / filename).read_text())
        packages = {package["name"]: package for package in lock["package"]}
        assert packages[name]["version"] == expected
        assert packages["external"]["version"] == "12.8.9"

    after = snapshot(release_project)
    assert after[Path("pyproject.toml")] == before[Path("pyproject.toml")]
    assert "# Public version" in after[Path("src/mirt/_version.py")]
    assert "# Distribution metadata" in after[Path("Cargo.toml")]
    assert 'checksum = "preserve-this"' in after[Path("Cargo.lock")]
    assert 'source = { editable = "." }' in after[Path("uv.lock")]


@pytest.mark.parametrize("filename", ["Cargo.toml", "Cargo.lock", "uv.lock"])
def test_mismatch_refuses_to_partially_update_the_release(release_project, filename):
    path = release_project / filename
    text = path.read_text().replace('version = "12.8.9"', 'version = "99.0.0"')
    path.write_text(text)
    before = snapshot(release_project)

    with pytest.raises(ValueError, match="Version mismatch"):
        bump_version(release_project, "patch")
    assert snapshot(release_project) == before


@pytest.mark.parametrize(
    ("filename", "package"), [("Cargo.lock", "mirt_rs"), ("uv.lock", "mirt")]
)
def test_missing_lock_entry_refuses_changes(release_project, filename, package):
    path = release_project / filename
    path.write_text(path.read_text().replace(f'name = "{package}"', 'name = "other"'))
    before = snapshot(release_project)

    with pytest.raises(ValueError, match="Expected exactly one"):
        bump_version(release_project, "patch")
    assert snapshot(release_project) == before


@pytest.mark.parametrize("version", ["12.08.9", "12.8", "12.8.9rc1"])
def test_invalid_semver_refuses_changes(release_project, version):
    path = release_project / "src/mirt/_version.py"
    path.write_text(path.read_text().replace("12.8.9", version))
    before = snapshot(release_project)

    with pytest.raises(ValueError, match="MAJOR.MINOR.PATCH"):
        bump_version(release_project, "patch")
    assert snapshot(release_project) == before


def test_cli_emits_github_outputs_and_rejects_unsupported_bump(release_project):
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "minor", "--root", str(release_project)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout == "old=12.8.9\nnew=12.9.0\n"
    before = snapshot(release_project)
    rejected = subprocess.run(
        [sys.executable, str(SCRIPT), "unsupported", "--root", str(release_project)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2
    assert snapshot(release_project) == before


def test_dynamic_uv_project_preserves_the_unversioned_lock_entry(release_project):
    path = release_project / "uv.lock"
    original = path.read_text().replace(
        'name = "mirt"\nversion = "12.8.9"\n', 'name = "mirt"\n'
    )
    path.write_text(original)
    assert bump_version(release_project, "patch") == ("12.8.9", "12.8.10")
    assert path.read_text() == original
