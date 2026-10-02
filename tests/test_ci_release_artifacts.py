"""The publish gate must inspect artifact contents before attesting or uploading."""

from __future__ import annotations

import io
import runpy
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github/scripts/check_release_artifacts.py"
)
validate_release = runpy.run_path(str(SCRIPT))["validate_release"]


def metadata(version="1.2.3", name="mirt"):
    return f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode()


def wheel(directory, platform, contents=None, *, duplicate=False):
    path = directory / f"mirt-1.2.3-cp311-abi3-{platform}.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "mirt-1.2.3.dist-info/METADATA",
            metadata() if contents is None else contents,
        )
        archive.writestr("mirt/__init__.py", "__version__ = '1.2.3'")
        if duplicate:
            archive.writestr("other.dist-info/METADATA", metadata())
    return path


def sdist(directory, contents=None, *, symlink=False):
    path = directory / "mirt-1.2.3.tar.gz"
    payload = metadata() if contents is None else contents
    with tarfile.open(path, "w:gz") as archive:
        entry = tarfile.TarInfo("mirt-1.2.3/PKG-INFO")
        if symlink:
            entry.type = tarfile.SYMTYPE
            entry.linkname = "/etc/passwd"
            archive.addfile(entry)
        else:
            entry.size = len(payload)
            archive.addfile(entry, io.BytesIO(payload))
    return path


@pytest.fixture
def release(tmp_path):
    wheel(tmp_path, "manylinux2014_x86_64")
    wheel(tmp_path, "win_amd64")
    wheel(tmp_path, "macosx_11_0_arm64")
    sdist(tmp_path)
    return tmp_path


def test_real_archive_formats_accept_a_consistent_multiplatform_release(release):
    paths = validate_release(release, "v1.2.3")
    assert len(paths) == 4
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(release), "--tag", "v1.2.3"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert completed.stdout == "Validated 4 release artifacts for v1.2.3.\n"


@pytest.mark.parametrize("archive", ["wheel", "sdist"])
def test_a_single_stale_artifact_blocks_the_entire_release(release, archive):
    if archive == "wheel":
        wheel(release, "win_amd64", metadata("1.2.2"))
    else:
        sdist(release, metadata("1.2.2"))
    with pytest.raises(ValueError, match="does not match"):
        validate_release(release, "v1.2.3")


def test_untagged_builds_check_the_version_in_the_project_manifest(release):
    project = release.with_name(release.name + "-project")
    project.mkdir()
    manifest = project / "Cargo.toml"
    manifest.write_text('[package]\nname = "mirt_rs"\nversion = "1.2.3"\n')
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(release), "--root", str(project)],
        check=True,
        text=True,
        capture_output=True,
    )
    assert completed.stdout == "Validated 4 release artifacts for v1.2.3.\n"

    manifest.write_text('[package]\nname = "mirt_rs"\nversion = "1.2.4"\n')
    failed = subprocess.run(
        [sys.executable, str(SCRIPT), str(release), "--root", str(project)],
        check=False,
        text=True,
        capture_output=True,
    )
    assert failed.returncode == 2
    assert "does not match 'v1.2.4'" in failed.stderr


def test_untagged_cargo_prerelease_normalizes_to_python_metadata(release):
    for platform in ("manylinux2014_x86_64", "win_amd64", "macosx_11_0_arm64"):
        wheel(release, platform, metadata("1.2.3rc1"))
    sdist(release, metadata("1.2.3rc1"))
    project = release.with_name(release.name + "-project")
    project.mkdir()
    (project / "Cargo.toml").write_text(
        '[package]\nname = "mirt_rs"\nversion = "1.2.3-rc.1"\n'
    )
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(release), "--root", str(project)],
        check=True,
        text=True,
        capture_output=True,
    )
    assert completed.stdout == "Validated 4 release artifacts for v1.2.3-rc.1.\n"


@pytest.mark.parametrize(
    "tag",
    [
        "v1.2.4",
        "1.2.3",
        "v01.2.3",
        "v1.02.3",
        "v1.2.03",
        "v1.2",
        "v1.2.3.0",
        "v1.2.3+local",
        "v1.2.3rc1+local",
        "v1.2.3-unknown.1",
        "v1.2.3!4",
    ],
)
def test_wrong_or_malformed_tag_blocks_publication(release, tag):
    with pytest.raises(ValueError):
        validate_release(release, tag)


@pytest.mark.parametrize(
    ("tag", "version"),
    [
        ("v1.2.3rc1", "1.2.3rc1"),
        ("v1.2.3-rc.1", "1.2.3rc1"),
        ("v1.2.3-alpha.2", "1.2.3a2"),
        ("v1.2.3-beta.2", "1.2.3b2"),
        ("v1.2.3.post1", "1.2.3.post1"),
        ("v1.2.3-post.1", "1.2.3.post1"),
        ("v1.2.3.dev1", "1.2.3.dev1"),
        ("v1.2.3-dev.1", "1.2.3.dev1"),
        ("v1.2.3-rc.1", "1.2.3RC01"),
    ],
)
def test_pep440_and_cargo_prerelease_tags_match_normalized_metadata(
    release, tag, version
):
    for platform in ("manylinux2014_x86_64", "win_amd64", "macosx_11_0_arm64"):
        wheel(release, platform, metadata(version))
    sdist(release, metadata(version))
    assert len(validate_release(release, tag)) == 4


@pytest.mark.parametrize("archive", ["wheel", "sdist"])
@pytest.mark.parametrize(
    ("tag", "matching", "stale"),
    [
        ("v1.2.3-rc.2", "1.2.3rc2", "1.2.3rc1"),
        ("v1.2.3rc2", "1.2.3rc2", "1.2.3"),
        ("v1.2.3", "1.2.3", "1.2.3rc2"),
        ("v1.2.3.post2", "1.2.3.post2", "1.2.3.post1"),
        ("v1.2.3.dev2", "1.2.3.dev2", "1.2.3.dev1"),
    ],
)
def test_stale_prerelease_or_final_metadata_blocks_publication(
    release, archive, tag, matching, stale
):
    for platform in ("manylinux2014_x86_64", "win_amd64", "macosx_11_0_arm64"):
        wheel(release, platform, metadata(matching))
    sdist(release, metadata(matching))
    if archive == "wheel":
        wheel(release, "win_amd64", metadata(stale))
    else:
        sdist(release, metadata(stale))
    with pytest.raises(ValueError, match="does not match"):
        validate_release(release, tag)


def test_local_version_artifact_cannot_match_a_public_release(release):
    wheel(release, "win_amd64", metadata("1.2.3+local"))
    with pytest.raises(ValueError, match="does not match"):
        validate_release(release, "v1.2.3")


@pytest.mark.parametrize(
    "contents",
    [
        metadata(name="other"),
        metadata() + b"Version: 1.2.3\n",
        metadata() + b"Name: mirt\n",
        b"Metadata-Version: 2.1\nName: mirt\n",
    ],
)
def test_ambiguous_or_unrelated_metadata_is_rejected(release, contents):
    wheel(release, "win_amd64", contents)
    with pytest.raises(ValueError):
        validate_release(release, "v1.2.3")


@pytest.mark.parametrize("missing", ["wheel", "sdist"])
def test_partial_release_is_rejected(release, missing):
    pattern = "*.whl" if missing == "wheel" else "*.tar.gz"
    for path in release.glob(pattern):
        path.unlink()
    with pytest.raises(ValueError, match="at least one wheel and exactly one sdist"):
        validate_release(release, "v1.2.3")


def test_duplicate_wheel_metadata_is_rejected(release):
    wheel(release, "win_amd64", duplicate=True)
    with pytest.raises(ValueError, match="exactly one wheel METADATA"):
        validate_release(release, "v1.2.3")


def test_sdist_metadata_link_is_never_followed(release):
    sdist(release, symlink=True)
    with pytest.raises(ValueError, match="exactly one sdist PKG-INFO file"):
        validate_release(release, "v1.2.3")


def test_unexpected_publish_input_is_rejected(release):
    (release / "debug.log").write_text("build details")
    with pytest.raises(ValueError, match="unsupported files"):
        validate_release(release, "v1.2.3")


def test_corrupt_archive_returns_a_clean_cli_failure(release):
    path = release / "mirt-1.2.3-cp311-abi3-win_amd64.whl"
    path.write_bytes(b"not a ZIP archive")
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(release), "--tag", "v1.2.3"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert "File is not a zip file" in completed.stderr
    assert "Traceback" not in completed.stderr
