#!/usr/bin/env python3
"""Select migration changes, then execute every selected fixture exactly once."""
import argparse
import importlib.util
from pathlib import Path
import re
import subprocess
import sys

SPEC = importlib.util.spec_from_file_location("exact_fixture", Path(__file__).with_name("run-exact-rust-test.py"))
EXACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXACT)


def touches_migrations(paths):
    # test_support lives in migrations/mod.rs; keep the whole directory covered.
    return any(path.startswith("crates/db/src/migrations/") or path in {
        "crates/db/src/schema.rs", "scripts/run-migration-gate.py",
        ".github/workflows/rust-ci.yml",
    } for path in paths)


def migration_names():
    listed = subprocess.run(["cargo", "test", "--locked", "-p", "graphrag-db", "--lib",
                             "migrations::tests::", "--", "--list"],
                            text=True, capture_output=True, check=True)
    names = [line[:-6] for line in listed.stdout.splitlines() if line.endswith(": test")]
    if not names or any(not re.fullmatch(r"migrations::tests::[A-Za-z0-9_]+", name) for name in names):
        raise EXACT.FixtureError("migration discovery must find only migrations::tests::* fixtures")
    return names


def run():
    for name in migration_names():
        EXACT.execute("graphrag-db", name)
    EXACT.execute("graphrag-db", "repository::fulltext_projection_tests::lexical_migration_backfills_all_rows_and_portable_restore_rebuilds_exact_indexes")
    for name in ["schema_fifteen_v1_backup_remains_restorable_without_capture_table",
                 "empty_backup_round_trip_is_valid_and_restores_a_fresh_schema",
                 "schema23_reviewed_enrichment_backup_reopens_and_recovers_exact_leased_checkpoint"]:
        # The ignored subprocess worker is invoked by schema23, never directly.
        EXACT.execute("graphrag-cli", "backup::tests::" + name, "graphrag")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run()
    else:
        # Manual dispatch, an absent push base, or initial branch creation runs all.
        unbounded = not args.base or set(args.base) == {"0"}
        paths = [] if unbounded else subprocess.check_output(
            ["git", "diff", "--name-only", "-z", args.base, args.head], text=True).split("\0")
        selected = unbounded or touches_migrations(paths)
        print("selected=" + str(selected).lower())
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (EXACT.FixtureError, OSError, subprocess.CalledProcessError) as error:
        print("Migration gate failed: " + str(error), file=sys.stderr)
        sys.exit(1)
