"""Public harness contracts, using private executable fixtures and no models."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "validate-daily-workflow.py"
SPEC = importlib.util.spec_from_file_location("daily_workflow_validation", SCRIPT)
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)


class DailyWorkflowContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="graphrag-harness-contract-")
        self.directory = Path(self.temporary.name)
        self.binary = self.directory / "fixture-cli"
        self.binary.write_text(
            "#!" + sys.executable + "\n"
            "import json,os,sys,time\n"
            "arguments=sys.argv[1:]\n"
            "if '--fixture-fail' in arguments:\n"
            " print('partial output',flush=True)\n"
            " print('actionable fixture failure',file=sys.stderr,flush=True)\n"
            " sys.exit(7)\n"
            "if '--fixture-timeout' in arguments:\n"
            " print('started before timeout',flush=True)\n"
            " print('timeout evidence',file=sys.stderr,flush=True)\n"
            " time.sleep(10)\n"
            "print(json.dumps({'arguments':arguments,'stdin':sys.stdin.read(),"
            "'home':os.environ.get('HOME'),'xdg':os.environ.get('XDG_CONFIG_HOME'),"
            "'secret':os.environ.get('DAILY_WORKFLOW_TEST_SECRET'),'cwd':os.getcwd()}))\n"
        )
        self.binary.chmod(0o700)
        self.workflows = []

    def tearDown(self):
        for workflow in self.workflows:
            shutil.rmtree(workflow.directory)
        self.temporary.cleanup()

    def workflow(self, binary=None, timeout=5):
        options = SimpleNamespace(
            command_timeout=timeout,
            deadline_seconds=30,
            ollama_url="http://127.0.0.1:1",
        )
        workflow = HARNESS.Workflow(binary or self.binary, "contract fixture", options, None)
        self.workflows.append(workflow)
        return workflow

    def test_workspace_inputs_and_forwarded_ids_are_separate_from_processes(self):
        workflow = self.workflow()
        with workflow.task("counted-actions"):
            workspace = workflow.run(
                "workspace", ["workspace"], input="keyword atlas\nselect 1\nquit\n"
            )
            note_id = "note:actual-fixture-id"
            workflow.run(
                "inspect", ["inspect", note_id], ids=[(note_id, "fixture-result")]
            )
            workflow.run("capture", ["capture", "--stdin"], input="three\nnote\nlines\n")
        self.assertEqual(json.loads(workspace.stdout)["stdin"], "keyword atlas\nselect 1\nquit\n")
        result = workflow.result()
        self.assertEqual(result["cli_subprocess_count"], 3)
        self.assertEqual(result["workspace_input_count"], 3)
        self.assertEqual(result["automated_canonical_id_transfer_count"], 1)
        self.assertEqual(result["canonical_id_transfers"][0]["id"], note_id)
        self.assertFalse(result["canonical_id_transfers"][0]["observed_human_copy"])
        self.assertIsNone(result["observed_human_manual_id_copies"])
        self.assertIsNone(result["observed_human_elapsed_seconds"])
        self.assertEqual(result["steps"][0]["cli_subprocess_count"], 3)
        self.assertEqual(result["steps"][0]["workspace_input_count"], 3)

    def test_subprocess_gets_private_paths_and_does_not_inherit_secrets(self):
        workflow = self.workflow()
        with mock.patch.dict(os.environ, {"DAILY_WORKFLOW_TEST_SECRET": "must-not-inherit"}):
            output = workflow.run("isolation", ["--fixture-env"])
        received = json.loads(output.stdout)
        self.assertEqual(received["home"], str(workflow.directory / "home"))
        self.assertEqual(received["xdg"], str(workflow.directory / "xdg"))
        self.assertEqual(Path(received["cwd"]), workflow.directory)
        self.assertIsNone(received["secret"])
        self.assertEqual(received["arguments"][:4], [
            "--config", str(workflow.config), "--db-path", str(workflow.db)
        ])
        self.assertEqual(stat.S_IMODE(workflow.directory.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(workflow.config.stat().st_mode), 0o600)

    def test_failed_action_retains_evidence_and_actual_execution_count(self):
        workflow = self.workflow()
        with self.assertRaisesRegex(RuntimeError, "exited 7"):
            with workflow.task("failed-action"):
                workflow.run("failure", ["--fixture-fail"])
        saved = json.loads((workflow.directory / "metrics.json").read_text())
        self.assertFalse(saved["steps"][0]["success"])
        self.assertIn("exited 7", saved["steps"][0]["error"])
        self.assertEqual(saved["cli_subprocess_count"], 1)
        command = saved["commands"][0]
        self.assertEqual(command["exit_code"], 7)
        self.assertIn("partial output", Path(command["stdout_path"]).read_text())
        self.assertIn("actionable fixture failure", Path(command["stderr_path"]).read_text())

    def test_spawn_failure_is_an_attempt_not_a_cli_execution(self):
        workflow = self.workflow(binary=self.directory / "absent-cli")
        with self.assertRaisesRegex(RuntimeError, "Cannot execute"):
            with workflow.task("spawn-failure"):
                workflow.run("not-started", ["workspace"], input="stats\nquit\n")
        result = workflow.result()
        self.assertEqual(result["command_attempt_count"], 1)
        self.assertEqual(result["cli_subprocess_count"], 0)
        self.assertEqual(result["workspace_input_count"], 0)
        self.assertFalse(result["commands"][0]["executed"])
        self.assertIsNone(result["binary_sha256"])

    def test_timeout_is_bounded_and_preserves_partial_output(self):
        workflow = self.workflow(timeout=0.5)
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            with workflow.task("timeout"):
                workflow.run("slow", ["--fixture-timeout"])
        self.assertLess(time.monotonic() - started, 5)
        result = workflow.result()
        self.assertEqual(result["cli_subprocess_count"], 1)
        command = result["commands"][0]
        self.assertIsNone(command["exit_code"])
        self.assertEqual(command["error"], "command timeout")
        self.assertIn("started before timeout", Path(command["stdout_path"]).read_text())
        self.assertIn("timeout evidence", Path(command["stderr_path"]).read_text())

    def test_report_refuses_existing_files_and_dangling_symlinks_before_runtime(self):
        existing = self.directory / "existing.json"
        existing.write_text("retained evidence")
        symlink = self.directory / "dangling.json"
        symlink.symlink_to(self.directory / "absent-target")
        for report in (existing, symlink):
            with self.subTest(report=report.name):
                with mock.patch.object(HARNESS, "OfflineProvider") as provider:
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as error:
                            HARNESS.main(["--binary", str(self.binary), "--report", str(report)])
                self.assertEqual(error.exception.code, 2)
                provider.assert_not_called()
        self.assertEqual(existing.read_text(), "retained evidence")
        self.assertTrue(symlink.is_symlink())
        self.assertFalse(symlink.resolve().exists())

    def test_invalid_deadlines_are_rejected_before_runtime(self):
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value):
                with mock.patch.object(HARNESS, "OfflineProvider") as provider:
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as error:
                            HARNESS.main(["--binary", str(self.binary), "--command-timeout", value])
                self.assertEqual(error.exception.code, 2)
                provider.assert_not_called()

    def test_rc10_impostor_is_rejected_before_onboarding_or_corpus_mutation(self):
        mutation_marker = self.directory / "unexpected-mutation"
        self.binary.write_text(
            "#!" + sys.executable + "\n"
            "import pathlib,sys\n"
            "if '--version' in sys.argv:\n"
            " print('graphrag 0.1.0-rc.10')\n"
            "elif '--help' in sys.argv:\n"
            " print('fixture help')\n"
            "else:\n"
            " pathlib.Path(" + repr(str(mutation_marker)) + ").touch()\n"
        )
        workflow = self.workflow()
        with self.assertRaisesRegex(RuntimeError, "actual published 0.1.0-rc.1 baseline"):
            workflow.execute(candidate=False)
        self.assertEqual(
            [command["label"] for command in workflow.commands],
            ["binary-version", "binary-help"],
        )
        self.assertEqual(workflow.result()["cli_subprocess_count"], 2)
        self.assertFalse(mutation_marker.exists())
        self.assertFalse(workflow.db.exists())
        self.assertFalse(workflow.steps[0]["success"])

    def test_interrupted_sync_uses_one_cumulative_command_timeout(self):
        self.binary.write_text(
            "#!" + sys.executable + "\n"
            "import json,signal,sys,time\n"
            "def cancel(*_):\n"
            " print('Cancellation requested; sync --resume processing_job:fixture',file=sys.stderr,flush=True)\n"
            " time.sleep(0.65)\n"
            " print(json.dumps({'data':{'cancelled':True,'job_id':'processing_job:fixture',"
            "'files':[{'status':'pending'}]}}),flush=True)\n"
            " sys.exit(5)\n"
            "signal.signal(signal.SIGINT,cancel)\n"
            "print('Ingesting markdown from: synthetic.md',file=sys.stderr,flush=True)\n"
            "while True: time.sleep(0.01)\n"
        )
        workflow = self.workflow(timeout=1)
        entered = threading.Event()
        release = threading.Event()
        timer = threading.Timer(0.65, entered.set)
        workflow.provider = SimpleNamespace(
            arm=timer.start, entered=entered, release=release
        )
        started = time.monotonic()
        try:
            with self.assertRaisesRegex(RuntimeError, "timed out|deadline expired"):
                workflow.interrupt_sync()
        finally:
            timer.cancel()
            timer.join(2)
        self.assertLess(time.monotonic() - started, 1.6)
        result = workflow.result()
        self.assertEqual(result["cli_subprocess_count"], 1)
        command = result["commands"][0]
        self.assertIsNotNone(command["error"])
        evidence = Path(command["stderr_path"]).read_text()
        self.assertIn("Ingesting markdown from:", evidence)
        self.assertIn("Cancellation requested;", evidence)
        self.assertTrue(release.is_set())


if __name__ == "__main__":
    unittest.main()
