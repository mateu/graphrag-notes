"""Exercise fail-closed behavior without querying a registry or compiling Rust."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "check-audit-exemptions.sh"


class AuditExemptionTests(unittest.TestCase):
    def check_tree(self, tree, status=0):
        with tempfile.TemporaryDirectory() as directory:
            cargo = Path(directory) / "cargo"
            cargo.write_text(
                '#!/bin/sh\n'
                'test "$*" = "tree --locked --target x86_64-unknown-linux-gnu '
                '-e normal,build --prefix none" || exit 99\n'
                'printf "%s\\n" "$AUDIT_TEST_TREE"\n'
                'exit "$AUDIT_TEST_STATUS"\n'
            )
            cargo.chmod(0o755)
            return subprocess.run(
                ["bash", str(SCRIPT)], capture_output=True, text=True,
                env={**os.environ, "PATH": directory + os.pathsep + os.environ["PATH"],
                     "AUDIT_TEST_TREE": tree, "AUDIT_TEST_STATUS": str(status)},
            )

    def test_unreachable_exemptions_pass(self):
        result = self.check_tree("graphrag-cli v0.1.0\naws-lc-rs v1.18.0")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_each_reachable_exemption_fails_including_deduplicated_entries(self):
        for crate, version in (("rkyv", "0.7.46"), ("rsa", "0.5.5"), ("crypto-bigint", "0.5.5")):
            with self.subTest(crate=crate):
                result = self.check_tree(f"graphrag-cli v0.1.0\n{crate} v{version} (*)")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"{crate}" + (" v0.7" if crate == "rkyv" else "") + " is active", result.stderr)

        # SurrealDB 3.3 reaches the non-affected rkyv 0.8 series natively.
        self.assertEqual(self.check_tree("rkyv v0.8.18").returncode, 0)

    def test_failed_dependency_resolution_cannot_pass_as_an_empty_tree(self):
        result = self.check_tree("", status=101)
        self.assertEqual(result.returncode, 101)
        self.assertNotIn("remain unreachable", result.stdout)


if __name__ == "__main__":
    unittest.main()
