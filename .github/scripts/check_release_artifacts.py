"""Reject incomplete releases or package metadata that disagrees with the tag."""

from __future__ import annotations

import argparse
import re
import tarfile
import tomllib
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path

from packaging.utils import parse_sdist_filename, parse_wheel_filename
from packaging.version import Version


def _release_version(tag: str) -> Version:
    component = r"(?:0|[1-9][0-9]*)"
    if not re.fullmatch(
        rf"v{component}\.{component}\.{component}(?:[A-Za-z._+-].*)?", tag
    ):
        raise ValueError(
            "Release tag must start with vMAJOR.MINOR.PATCH, "
            "optionally followed by a pre, post, or dev release suffix"
        )
    version = Version(tag[1:])
    if len(version.release) != 3:
        raise ValueError("Release tag must contain exactly three release components")
    if version.local is not None:
        raise ValueError("PyPI releases cannot contain a local version suffix")
    return version


def _require_archive_identity(path: Path, directory: str, version: Version) -> None:
    name, archive_version = parse_sdist_filename(directory + ".tar.gz")
    if name != "mirt" or archive_version != version:
        raise ValueError(
            f"{path.name}: metadata directory {directory!r} "
            f"does not match package mirt {str(version)!r}"
        )


def _metadata(path: Path, version: Version) -> bytes:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            entries = [
                name
                for name in archive.namelist()
                if re.fullmatch(r"[^/]+\.dist-info/METADATA", name)
            ]
            if len(entries) != 1:
                raise ValueError(f"{path.name}: expected exactly one wheel METADATA")
            directory = entries[0].split("/", 1)[0].removesuffix(".dist-info")
            _require_archive_identity(path, directory, version)
            return archive.read(entries[0])

    with tarfile.open(path, "r:gz") as archive:
        entries = [
            entry
            for entry in archive.getmembers()
            if re.fullmatch(r"[^/]+/PKG-INFO", entry.name)
        ]
        if len(entries) != 1 or not entries[0].isfile():
            raise ValueError(f"{path.name}: expected exactly one sdist PKG-INFO file")
        _require_archive_identity(path, entries[0].name.split("/", 1)[0], version)
        stream = archive.extractfile(entries[0])
        if stream is None:
            raise ValueError(f"{path.name}: could not read sdist PKG-INFO")
        with stream:
            return stream.read()


def validate_release(directory: Path, tag: str) -> list[Path]:
    """Check actual archive metadata for every wheel and the source distribution."""
    release_version = _release_version(tag)
    if not directory.is_dir():
        raise ValueError(f"Release directory does not exist: {directory}")
    paths = sorted(path for path in directory.iterdir() if path.is_file())
    wheels = [path for path in paths if path.suffix == ".whl"]
    sdists = [path for path in paths if path.name.endswith(".tar.gz")]
    if not wheels or len(sdists) != 1:
        raise ValueError("Release requires at least one wheel and exactly one sdist")
    if len(paths) != len(wheels) + len(sdists):
        raise ValueError("Release directory contains unsupported files")

    for path in paths:
        if path.suffix == ".whl":
            name, version, _, _ = parse_wheel_filename(path.name)
        else:
            name, version = parse_sdist_filename(path.name)
        if name != "mirt" or version != release_version:
            raise ValueError(
                f"{path.name}: artifact filename does not match {tag!r} for package mirt"
            )
        metadata = BytesParser(policy=policy.default).parsebytes(
            _metadata(path, release_version)
        )
        names = metadata.get_all("Name", [])
        versions = metadata.get_all("Version", [])
        if names != ["mirt"]:
            raise ValueError(f"{path.name}: expected one package Name: mirt")
        if len(versions) != 1 or Version(versions[0]) != release_version:
            raise ValueError(
                f"{path.name}: package version {versions!r} does not match {tag!r}"
            )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tag", default="")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        tag = args.tag
        if not tag:
            with (args.root / "Cargo.toml").open("rb") as manifest:
                tag = "v" + tomllib.load(manifest)["package"]["version"]
        paths = validate_release(args.directory, tag)
    except (
        KeyError,
        OSError,
        ValueError,
        tarfile.TarError,
        zipfile.BadZipFile,
    ) as error:
        parser.error(str(error))
    print(f"Validated {len(paths)} release artifacts for {tag}.")


if __name__ == "__main__":
    main()
