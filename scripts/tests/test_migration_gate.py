import importlib.util
from pathlib import Path
import subprocess
import os
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("migration_gate", Path(__file__).resolve().parents[1] / "run-migration-gate.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class MigrationGateTests(unittest.TestCase):
    def test_paths_include_registry_helper_schema_and_gate_itself(self):
        for path in ["crates/db/src/migrations/mod.rs", "crates/db/src/migrations/v026.surql",
                     "crates/db/src/schema.rs", "scripts/run-migration-gate.py", ".github/workflows/rust-ci.yml"]:
            with self.subTest(path=path):
                self.assertTrue(gate.touches_migrations([path]))
        self.assertFalse(gate.touches_migrations(["docs/ci-quality-gates.md", "crates/db/src/repository/graph.rs"]))

    def test_discovery_is_fail_closed_and_includes_future_migrations(self):
        for output in ["0 tests, 0 benchmarks\n", "other::test: test\n"]:
            with mock.patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0, output, "")):
                with self.assertRaises(gate.EXACT.FixtureError):
                    gate.migration_names()
        with mock.patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "migrations::tests::future: test\n", "")):
            self.assertEqual(gate.migration_names(), ["migrations::tests::future"])

    def test_every_fixture_uses_the_exact_nonignored_runner(self):
        with mock.patch.object(gate, "migration_names", return_value=["migrations::tests::new_schema"]), mock.patch.object(gate.EXACT, "execute") as execute:
            gate.run()
        self.assertEqual(execute.call_count, 5)
        self.assertEqual(execute.call_args_list[0].args, ("graphrag-db", "migrations::tests::new_schema"))
        self.assertTrue(any("lexical_migration" in call.args[1] for call in execute.call_args_list))
        self.assertTrue(any("schema23_reviewed" in call.args[1] for call in execute.call_args_list))
        self.assertFalse(any("subprocess_worker" in call.args[1] for call in execute.call_args_list))

    def test_rename_out_of_schema_path_cannot_bypass_gate(self):
        with tempfile.TemporaryDirectory() as root:
            def git(*args):
                return subprocess.check_output(["git", "-C", root, *args], text=True).strip()
            git("init", "-q")
            schema = Path(root) / "crates/db/src/schema.rs"
            schema.parent.mkdir(parents=True)
            schema.write_text("// historical schema\n" * 20)
            git("add", ".")
            git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "base")
            base = git("rev-parse", "HEAD")
            destination = schema.with_suffix("") / "mod.rs"
            destination.parent.mkdir()
            schema.rename(destination)
            git("add", "-A")
            git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "rename")
            head = git("rev-parse", "HEAD")
            self.assertEqual(git("diff", "--find-renames", "--name-only", base, head), "crates/db/src/schema/mod.rs")
            original_cwd = os.getcwd()
            try:
                os.chdir(root)
                with mock.patch("sys.argv", ["run-migration-gate.py", "--base", base, "--head", head]), mock.patch("builtins.print") as output:
                    self.assertEqual(gate.main(), 0)
                output.assert_called_once_with("selected=true")
            finally:
                os.chdir(original_cwd)

    def test_initial_push_does_not_diff_zero_sha(self):
        with mock.patch("sys.argv", ["run-migration-gate.py", "--base", "0" * 40]), mock.patch.object(subprocess, "check_output") as diff, mock.patch("builtins.print") as output:
            self.assertEqual(gate.main(), 0)
        diff.assert_not_called()
        output.assert_called_once_with("selected=true")


if __name__ == "__main__":
    unittest.main()
