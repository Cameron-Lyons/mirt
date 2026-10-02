"""Keep runtime, distribution, and lockfile versions synchronized."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path


def _package_version(contents: str, package_name: str, *, manifest: bool) -> str:
    data = tomllib.loads(contents)
    packages = [data["package"]] if manifest else data["package"]
    matches = [package for package in packages if package["name"] == package_name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {package_name!r} package")
    return matches[0]["version"]


def _replace_package_version(
    contents: str, package_name: str, new_version: str, *, manifest: bool
) -> str:
    table = "[package]" if manifest else "[[package]]"
    pattern = re.compile(rf"(?ms)^{re.escape(table)}[^\S\n]*\n.*?(?=^\[|\Z)")
    replacements = 0

    def replace_table(match: re.Match[str]) -> str:
        nonlocal replacements
        block = match.group(0)
        name = re.search(r"(?m)^name\s*=\s*(['\"])(.*?)\1", block)
        if name is None or name.group(2) != package_name:
            return block
        updated, count = re.subn(
            r"(?m)^(version\s*=\s*)(['\"])(.*?)\2",
            lambda version: (
                f"{version.group(1)}{version.group(2)}{new_version}{version.group(2)}"
            ),
            block,
        )
        if count != 1:
            raise ValueError(f"Expected one version field for {package_name!r}")
        replacements += 1
        return updated

    updated = pattern.sub(replace_table, contents)
    if replacements != 1:
        raise ValueError(f"Could not update {package_name!r} package version")
    return updated


def bump_version(root: Path, part: str) -> tuple[str, str]:
    """Validate every version before making any changes, then bump them together."""
    version_path = root / "src/mirt/_version.py"
    cargo_path = root / "Cargo.toml"
    cargo_lock_path = root / "Cargo.lock"
    uv_lock_path = root / "uv.lock"
    paths = (version_path, cargo_path, cargo_lock_path, uv_lock_path)
    contents = {path: path.read_text(encoding="utf-8") for path in paths}

    runtime_versions = re.findall(
        r"(?m)^__version__\s*=\s*(['\"])(.*?)\1", contents[version_path]
    )
    if len(runtime_versions) != 1:
        raise ValueError("Expected one runtime __version__ assignment")
    old_version = runtime_versions[0][1]
    if not re.fullmatch(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)", old_version):
        raise ValueError("Version must have the form MAJOR.MINOR.PATCH")

    cargo_package = tomllib.loads(contents[cargo_path])["package"]["name"]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    python_package = project["project"]["name"]
    specifications = (
        (cargo_path, cargo_package, True),
        (cargo_lock_path, cargo_package, False),
    )
    uv_packages = [
        package
        for package in tomllib.loads(contents[uv_lock_path])["package"]
        if package["name"] == python_package
    ]
    if len(uv_packages) != 1:
        raise ValueError(f"Expected exactly one {python_package!r} package")
    # uv omits a project version when maturin supplies dynamic metadata.
    if "version" in uv_packages[0]:
        specifications += ((uv_lock_path, python_package, False),)
    for path, name, manifest in specifications:
        version = _package_version(contents[path], name, manifest=manifest)
        if version != old_version:
            raise ValueError(
                f"Version mismatch in {path.name}: {version!r} != {old_version!r}"
            )

    parts = [int(value) for value in old_version.split(".")]
    try:
        index = ("major", "minor", "patch").index(part)
    except ValueError as error:
        raise ValueError("Bump part must be major, minor, or patch") from error
    parts[index] += 1
    parts[index + 1 :] = [0] * (2 - index)
    new_version = ".".join(str(value) for value in parts)

    updates = {
        version_path: re.sub(
            r"(?m)^(__version__\s*=\s*)(['\"])(.*?)\2",
            lambda version: (
                f"{version.group(1)}{version.group(2)}{new_version}{version.group(2)}"
            ),
            contents[version_path],
        )
    }
    for path, name, manifest in specifications:
        updates[path] = _replace_package_version(
            contents[path], name, new_version, manifest=manifest
        )

    for path, updated in updates.items():
        path.write_text(updated, encoding="utf-8")
    return old_version, new_version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=("major", "minor", "patch"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        old_version, new_version = bump_version(args.root, args.part)
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))
    print(f"old={old_version}\nnew={new_version}")


if __name__ == "__main__":
    main()
