#!/usr/bin/env python3
"""Run one named Rust fixture, refusing Cargo's successful zero-test result."""
import argparse
import re
import subprocess
import sys


class FixtureError(Exception):
    pass


def execute(package, name, binary=None, nocapture=False):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", package) or not re.fullmatch(r"[A-Za-z0-9_:]+", name):
        raise FixtureError("invalid package or fully qualified fixture name")
    if binary is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", binary):
        raise FixtureError("invalid binary target")
    base = ["cargo", "test", "--locked", "-p", package]
    base += ["--bin", binary] if binary is not None else ["--lib"]
    base += [name, "--", "--exact"]
    listed = subprocess.run(base + ["--list"], text=True, capture_output=True, check=False)
    if listed.returncode != 0:
        sys.stdout.write(listed.stdout)
        sys.stderr.write(listed.stderr)
        raise FixtureError("fixture listing failed")
    names = [line[:-6] for line in listed.stdout.splitlines() if line.endswith(": test")]
    if names != [name]:
        raise FixtureError("exact fixture must resolve to one test; check its module path")
    result = subprocess.run(base + (["--nocapture"] if nocapture else []),
                            text=True, capture_output=True, check=False)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    summaries = re.findall(r"^test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored;",
                           result.stdout, re.MULTILINE)
    if result.returncode != 0 or summaries != [("ok", "1", "0", "0")]:
        raise FixtureError("exact fixture must execute one passing, nonignored test")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--package", required=True)
    parser.add_argument("--test-name", required=True)
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument("--lib", action="store_true", help="Library unit fixture (default)")
    targets.add_argument("--bin", dest="binary")
    parser.add_argument("--nocapture", action="store_true")
    args = parser.parse_args(argv)
    try:
        execute(args.package, args.test_name, args.binary, args.nocapture)
        return 0
    except (FixtureError, OSError) as error:
        print("Exact Rust fixture failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
