"""Exercise the probe against a fake executable, never a real service."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROBE = Path(__file__).resolve().parents[1] / "post-deploy-probe.py"
PRIVATE = "PRIVATE-query-content"
IDENTIFIER = "note:PRIVATE-record-id"
TOKEN = "PRIVATE-bearer-token"
FAKE = r'''#!/usr/bin/env python3
import json, os, sys, time
args = sys.argv[1:]
command = args[4]
step = ("keyword_graph_off" if "keyword" in args else "hybrid_graph_on") if command == "search" else ("inspect_top_hit" if command == "inspect" else command)
with open(os.environ["FAKE_LOG"], "a") as log:
    log.write(json.dumps({"args": args, "token": os.environ.get("PROBE_TOKEN")}) + "\n")
scenario = os.environ.get("FAKE_SCENARIO", "")
if scenario == step + ":timeout":
    print("PRIVATE-query-content", flush=True)
    time.sleep(5)
if scenario == step + ":exit":
    print("PRIVATE-query-content PRIVATE-bearer-token note:PRIVATE-record-id", file=sys.stderr)
    sys.exit(2)
if command == "doctor":
    check = {"state": "ready", "summary": "Readiness evidence available."}
    code = 1 if scenario == "doctor:warning" else 0
    report = {"schema_version": 1, "exit_code": code, "read_only": True,
              "transport": check, "authorization": check,
              "application": {name: check for name in ("storage", "embeddings", "extraction", "sources", "backup")},
              "jobs": {"readiness": check}}
    print(json.dumps(report) if scenario != "doctor:malformed" else "not JSON")
    sys.exit(code)
data = {"records": [{"id": "note:PRIVATE-record-id", "hit_type": "note", "content": "PRIVATE-query-content"}], "graph_status": "ok"}
if command == "inspect":
    data = {"id": "note:PRIVATE-record-id", "hit_type": "note", "content": "PRIVATE-query-content"}
if scenario == step + ":empty": data["records"] = []
if scenario == step + ":non_note": data["records"][0]["hit_type"] = "message"
if scenario == step + ":wrong_id": data["id"] = "note:different"
if scenario.startswith(step + ":status_"):
    data["graph_status"] = scenario.split(":status_", 1)[1]
if scenario == step + ":missing_status": data.pop("graph_status", None)
value = {"schema_version": 1, "command": "get_record" if command == "inspect" else "search_notes", "success": True, "errors": [], "warnings": [], "data": {"schema_version": 1, "error": None, "data": data}}
if scenario == step + ":schema": value["schema_version"] = 2
if scenario == step + ":error": value["data"]["error"] = {"message": "PRIVATE-query-content"}
if scenario == step + ":failure": value["success"] = False
if scenario == step + ":missing_error": value["data"].pop("error")
if scenario == step + ":malformed":
    print("not JSON PRIVATE-query-content")
else:
    print(json.dumps(value))
'''


class PostDeployProbeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.cli = root / "mock-graphrag"
        self.cli.write_text(FAKE)
        self.cli.chmod(0o700)
        self.log = root / "calls.jsonl"

    def run_probe(self, scenario="", extra=(), token=TOKEN, query=PRIVATE):
        env = dict(os.environ, PROBE_TOKEN=token, FAKE_LOG=str(self.log), FAKE_SCENARIO=scenario)
        result = subprocess.run([sys.executable, str(PROBE), "--cli", str(self.cli),
                                 "--server", "http://127.0.0.1:3000/mcp",
                                 "--credential-env", "PROBE_TOKEN", "--query=" + query,
                                 *extra], env=env, capture_output=True, text=True, timeout=10)
        for private in (PRIVATE, IDENTIFIER, TOKEN):
            self.assertNotIn(private, result.stdout + result.stderr)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        return result, calls

    def test_success_only_remote_read_commands_and_environment_credentials(self):
        result, calls = self.run_probe()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 4)
        self.assertIn("PASS doctor keyword_graph_off inspect_top_hit hybrid_graph_on\n", result.stdout)
        for call in calls:
            self.assertEqual(call["args"][:4], ["--server", "http://127.0.0.1:3000/mcp", "--credential-env", "PROBE_TOKEN"])
            self.assertEqual(call["token"], TOKEN)
            self.assertNotIn(TOKEN, call["args"])
        self.assertEqual([call["args"][4] for call in calls], ["doctor", "search", "inspect", "search"])
        self.assertIn("keyword", calls[1]["args"])
        self.assertIn("off", calls[1]["args"])
        self.assertIn("notes", calls[1]["args"])
        self.assertEqual(calls[2]["args"][5], IDENTIFIER)
        self.assertIn("hybrid", calls[3]["args"])
        self.assertIn("on", calls[3]["args"])

    def test_doctor_warning_passes_and_summaries_are_visible(self):
        result, calls = self.run_probe("doctor:warning")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 4)
        self.assertIn("doctor: exit 1", result.stdout)
        self.assertIn("doctor storage: ready: Readiness evidence available.", result.stdout)

    def test_doctor_invalid_response_fails_closed(self):
        result, calls = self.run_probe("doctor:malformed")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(calls), 1)
        self.assertIn("FAIL doctor: invalid doctor response", result.stderr)

    def test_every_command_failure_stops_at_named_step(self):
        for count, step in enumerate(("doctor", "keyword_graph_off", "inspect_top_hit", "hybrid_graph_on"), 1):
            with self.subTest(step=step):
                self.log.unlink(missing_ok=True)
                result, calls = self.run_probe(step + ":exit")
                self.assertEqual(result.returncode, 1)
                self.assertIn("FAIL " + step + ":", result.stderr)
                self.assertEqual(len(calls), count)

    def test_invalid_results_fail_closed(self):
        cases = {"keyword_graph_off": ("empty", "non_note", "schema", "error", "failure", "malformed", "missing_error"),
                 "inspect_top_hit": ("wrong_id", "malformed", "error"),
                 "hybrid_graph_on": ("missing_status", "status_off", "status_skipped", "status_timed_out", "status_unknown", "empty", "schema")}
        for step, failures in cases.items():
            for failure in failures:
                with self.subTest(step=step, failure=failure):
                    result, _ = self.run_probe(step + ":" + failure)
                    self.assertEqual(result.returncode, 1)
                    self.assertIn("FAIL " + step + ":", result.stderr)

    def test_hybrid_client_timeout_is_enforced_and_output_hidden(self):
        result, calls = self.run_probe("hybrid_graph_on:timeout", extra=("--timeout", "0.5"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL hybrid_graph_on: client timeout", result.stderr)
        self.assertEqual(len(calls), 4)

    def test_setup_rejects_missing_credentials_and_bad_timeout_without_cli(self):
        for kwargs in ({"token": ""}, {"token": "bad\nheader"}, {"extra": ("--timeout", "nan")},
                       {"extra": ("--timeout", "0")}, {"query": " "},
                       {"extra": ("--server", "http://remote.example/mcp")},
                       {"extra": ("--server", "https://example.test/mcp?token=secret")},
                       {"extra": ("--server", "https://user:secret@example.test/mcp")}):
            with self.subTest(kwargs=kwargs):
                result, calls = self.run_probe(**kwargs)
                self.assertEqual(result.returncode, 1)
                self.assertIn("FAIL setup:", result.stderr)
                self.assertEqual(calls, [])

    def test_missing_executable_names_doctor_without_echoing_path(self):
        result, _ = self.run_probe(extra=("--cli", "/nonexistent/" + PRIVATE))
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL doctor: CLI could not start", result.stderr)

    def test_query_that_looks_like_option_is_passed_as_literal(self):
        result, calls = self.run_probe(query="--PRIVATE-query-content")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(calls[1]["args"][-2:], ["--", "--PRIVATE-query-content"])


if __name__ == "__main__":
    unittest.main()
