import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("exact_fixture", Path(__file__).resolve().parents[1] / "run-exact-rust-test.py")
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)
NAME = "ingestion::librarian::tests::fictional_fixture"


def response(stdout, status=0):
    return subprocess.CompletedProcess([], status, stdout, "")


class ExactFixtureTests(unittest.TestCase):
    def test_unknown_or_wrong_module_fails_before_test_execution(self):
        for listing in ["0 tests, 0 benchmarks\n", "other::fixture: test\n1 test, 0 benchmarks\n"]:
            with self.subTest(listing=listing), mock.patch.object(subprocess, "run", return_value=response(listing)) as run:
                with self.assertRaises(fixture.FixtureError):
                    fixture.execute("graphrag-agents", NAME)
                self.assertEqual(run.call_count, 1)

    def test_runtime_zero_ignored_failed_or_multiple_tests_are_refused(self):
        for summary, code in [("ok. 0 passed; 0 failed; 0 ignored;", 0),
                              ("ok. 0 passed; 0 failed; 1 ignored;", 0),
                              ("FAILED. 0 passed; 1 failed; 0 ignored;", 101),
                              ("ok. 2 passed; 0 failed; 0 ignored;", 0)]:
            with self.subTest(summary=summary), mock.patch.object(subprocess, "run", side_effect=[
                    response(NAME + ": test\n"), response("test result: " + summary + "\n", code)]), mock.patch("sys.stdout"):
                with self.assertRaises(fixture.FixtureError):
                    fixture.execute("graphrag-agents", NAME)

    def test_one_exact_passing_fixture_preserves_literal_target_and_options(self):
        with mock.patch.object(subprocess, "run", side_effect=[
                response(NAME + ": test\n"), response("test result: ok. 1 passed; 0 failed; 0 ignored;\n")]) as run, mock.patch("sys.stdout"):
            fixture.execute("graphrag-cli", NAME, "graphrag", True)
            base = ["cargo", "test", "--locked", "-p", "graphrag-cli", "--bin", "graphrag", NAME, "--", "--exact"]
            self.assertEqual(run.call_args_list[0].args[0], base + ["--list"])
            self.assertEqual(run.call_args_list[1].args[0], base + ["--nocapture"])
            self.assertNotIn("shell", run.call_args.kwargs)

    def test_nonzero_listing_and_invalid_identity_fail(self):
        with mock.patch.object(subprocess, "run", return_value=response("", 101)), mock.patch("sys.stderr"):
            with self.assertRaises(fixture.FixtureError):
                fixture.execute("graphrag-db", NAME)
        with mock.patch.object(subprocess, "run") as run:
            with self.assertRaises(fixture.FixtureError):
                fixture.execute("graphrag-db", "--bad $(echo nope)")
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
