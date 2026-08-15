#!/usr/bin/env python3
"""Verify every workspace package inherits the exact declared MSRV."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

EXPECTED = "1.97.1"
ROOT_MANIFEST = Path("Cargo.toml")


def main() -> int:
    root = tomllib.loads(ROOT_MANIFEST.read_text(encoding="utf-8"))
    declared = root.get("workspace", {}).get("package", {}).get("rust-version")
    if declared != EXPECTED:
        print(
            f"{ROOT_MANIFEST}: workspace.package.rust-version must be {EXPECTED!r}, "
            f"found {declared!r}",
            file=sys.stderr,
        )
        return 1

    metadata = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        check=True,
        capture_output=True,
        text=True,
    )
    packages = json.loads(metadata.stdout)["packages"]
    mismatched = [
        f"{package['name']}={package['rust_version']!r}"
        for package in packages
        if package.get("rust_version") != EXPECTED
    ]
    if mismatched:
        print(
            "workspace members must inherit the exact MSRV "
            f"{EXPECTED!r}: {', '.join(mismatched)}",
            file=sys.stderr,
        )
        return 1

    print(f"workspace MSRV is exactly {EXPECTED} for {len(packages)} packages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
