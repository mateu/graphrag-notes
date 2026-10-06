"""Deterministic capture-boundary faults; only fictional notes and credentials."""
import concurrent.futures
import copy
import http.server
import importlib.util
import json
import os
from pathlib import Path
import re
import select
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "mcp-capture-journal.py"
SPEC = importlib.util.spec_from_file_location("capture_journal", SCRIPT)
J = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(J)
HEADERS = {"Authorization": "Bearer fictional-first-token", "MCP-Protocol-Version": "2025-11-25", "Content-Type": "application/json"}
ARGS = {"request_id": "fictional-fixed-request", "content": "Atlas\nFictional meeting at ten.\n",
        "title": "Atlas", "tags": [" tag ", "duplicate", "duplicate"],
        "provenance": {"uri": None, "label": "fictional isolated fixture", "metadata": {}}}


def request(args=None, request_id=7):
    return {"jsonrpc": "2.0", "id": request_id, "method": "tools/call",
            "params": {"name": "capture_note", "arguments": copy.deepcopy(args or ARGS)}}


def response(data, request_id="journal-check"):
    value = {"jsonrpc": "2.0", "id": request_id, "result": {"content": [], "isError": False,
             "structuredContent": {"schema_version": 1, "data": data, "error": None}}}
    return 200, {"Content-Type": "application/json"}, J.encode(value)


class Crash(BaseException):
    pass


class FakeService:
    server = "http://127.0.0.1:3300/mcp"

    def __init__(self, journal):
        self.journal = journal
        self.calls = []
        self.receipts = {}
        self.lock = threading.Lock()
        self.loss_after_commit = False
        self.failure_before_commit = False
        self.crash_after_commit = False
        self.readback_fail = False
        self.bad_receipt = False
        self.principal = "fictional-clawd"
        self.capture_seen_states = []

    def call(self, name, args, headers, meta=None):
        self.calls.append(name)
        if name == "service_status":
            return response({"schema_version": 1, "read_only": True, "inference_probed": False, "instance_id": self.principal})
        if name == "get_record":
            if self.readback_fail:
                raise J.JournalError("fictional_readback_down")
            record = next(copy.deepcopy(v["record"]) for v in self.receipts.values() if v["record"]["id"] == args["id"])
            if args != {"id": record["id"], "revision": record["revision"], "neighbors": 0}:
                raise AssertionError("Readback must be pinned and have zero neighbors")
            return response({k: record[k] for k in ("id", "revision", "title", "content", "provenance")})
        raise AssertionError("Unexpected fixture tool")

    def send(self, body, headers, method="POST"):
        value = json.loads(body)
        args = value["params"]["arguments"]
        key = J.journal_id(self.server, self.principal, args["request_id"])
        pending = self.journal.read(key)
        self.capture_seen_states.append(pending["state"])
        if pending["arguments"] != args or pending["state"] != "uncertain":
            raise AssertionError("Exact draft must be durable before capture")
        self.calls.append("capture_note")
        if self.failure_before_commit:
            self.failure_before_commit = False
            raise J.JournalError("fictional_failure_before_commit")
        with self.lock:
            replayed = key in self.receipts
            if not replayed:
                record = {"id": "note:" + key, "revision": "a" * 64, "content": args["content"],
                          "title": args.get("title") or "Generated title", "tags": args.get("tags", []),
                          "created_at": "2026-10-05T00:00:00Z", "updated_at": "2026-10-05T00:00:00Z",
                          "provenance": {"instance_id": self.principal, "source": args.get("provenance")}}
                self.receipts[key] = {"request_id": args["request_id"], "replayed": False, "record": record}
            receipt = copy.deepcopy(self.receipts[key])
            receipt["replayed"] = replayed
        if self.loss_after_commit:
            self.loss_after_commit = False
            raise J.JournalError("fictional_response_lost_after_commit")
        if self.crash_after_commit:
            self.crash_after_commit = False
            raise Crash()
        if self.bad_receipt:
            receipt["record"]["tags"] = ["changed"]
        return response(receipt, value["id"])


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="fictional-capture-journal-")
        self.directory = Path(self.temporary.name) / "journal"
        self.journal = J.Journal(self.directory)
        self.service = FakeService(self.journal)
        self.boundary = J.CaptureBoundary(self.journal, self.service, "fictional-clawd")
        self.entry_id = J.journal_id(self.service.server, "fictional-clawd", ARGS["request_id"])

    def tearDown(self):
        self.journal.close()
        self.temporary.cleanup()

    def reopen(self):
        self.journal.close()
        self.journal = J.Journal(self.directory)
        self.service.journal = self.journal
        self.boundary = J.CaptureBoundary(self.journal, self.service, "fictional-clawd")

    def test_committed_response_loss_reset_and_credential_rotation_recover_one_note(self):
        self.service.loss_after_commit = True
        with self.assertRaises(J.JournalError):
            self.boundary.submit(request(), HEADERS)
        self.assertEqual(self.journal.read(self.entry_id)["state"], "uncertain")
        self.reopen()  # Chat reset/client restart does not carry any draft in memory.
        rotated = dict(HEADERS, Authorization="Bearer fictional-rotated-token")
        result = self.boundary.recover(self.entry_id, rotated)
        self.assertTrue(result["independently_verified"])
        self.assertEqual(len(self.service.receipts), 1)
        self.assertTrue(self.journal.read(self.entry_id)["receipt"]["replayed"])
        self.assertEqual(self.service.capture_seen_states, ["uncertain", "uncertain"])
        private_bytes = b"".join(p.read_bytes() for p in self.directory.iterdir())
        self.assertNotIn(b"fictional-first-token", private_bytes)
        self.assertNotIn(b"fictional-rotated-token", private_bytes)

    def test_client_crash_and_precommit_failure_both_keep_exact_draft(self):
        for fault, exception in (("crash_after_commit", Crash), ("failure_before_commit", J.JournalError)):
            with self.subTest(fault=fault):
                args = copy.deepcopy(ARGS)
                args["request_id"] = fault
                entry_id = J.journal_id(self.service.server, "fictional-clawd", fault)
                setattr(self.service, fault, True)
                with self.assertRaises(exception):
                    self.boundary.submit(request(args), HEADERS)
                self.reopen()
                self.assertEqual(self.journal.read(entry_id)["arguments"], args)
                self.assertTrue(self.boundary.recover(entry_id, HEADERS)["independently_verified"])
        self.assertEqual(len(self.service.receipts), 2)

    def test_storage_failure_never_submits_capture(self):
        with mock.patch.object(self.journal, "write", side_effect=OSError("fictional disk full")):
            with self.assertRaises(OSError):
                self.boundary.submit(request(), HEADERS)
        self.assertNotIn("capture_note", self.service.calls)
        # fsync failure during the uncertain-state transition also prevents send.
        original = self.journal.write
        def fail_uncertain(entry):
            if entry["state"] == "uncertain":
                raise OSError("fictional fsync failed")
            original(entry)
        with mock.patch.object(self.journal, "write", side_effect=fail_uncertain):
            with self.assertRaises(OSError):
                self.boundary.submit(request(), HEADERS)
        self.assertEqual(self.journal.read(self.entry_id)["state"], "pending")
        self.assertNotIn("capture_note", self.service.calls)

    def test_receipt_known_failed_readback_not_verified_or_cleanup_eligible(self):
        self.service.readback_fail = True
        original_response = self.boundary.submit(request(), HEADERS)
        self.assertEqual(json.loads(original_response[2])["id"], 7)
        self.assertEqual(self.journal.read(self.entry_id)["state"], "receipt_known")
        self.assertFalse(self.journal.summaries()[0]["independently_verified"])
        with self.assertRaises(J.JournalError):
            self.journal.cleanup(self.entry_id)
        self.service.readback_fail = False
        self.assertTrue(self.boundary.recover(self.entry_id, HEADERS)["independently_verified"])
        self.journal.cleanup(self.entry_id)
        self.assertIsNone(self.journal.read(self.entry_id))

    def test_all_binding_and_payload_mismatches_halt_before_submission(self):
        self.boundary.submit(request(), HEADERS)
        captures = self.service.calls.count("capture_note")
        altered = copy.deepcopy(ARGS)
        altered["content"] += "Different payload."
        with self.assertRaises(J.JournalError):
            self.boundary.submit(request(altered), HEADERS)
        self.service.principal = "different-authenticated-principal"
        with self.assertRaises(J.JournalError):
            self.boundary.recover(self.entry_id, HEADERS)
        self.service.principal = "fictional-clawd"
        self.service.server = "https://different.example.test/mcp"
        with self.assertRaises(J.JournalError):
            self.boundary.recover(self.entry_id, HEADERS)
        self.assertEqual(self.service.calls.count("capture_note"), captures)
        self.service.server = "http://127.0.0.1:3300/mcp"
        entry = self.journal.read(self.entry_id)
        entry["arguments"]["request_id"] = "changed-request-id"
        self.journal.write(entry)
        with self.assertRaises(J.JournalError):
            self.boundary.recover(self.entry_id, HEADERS)
        self.assertEqual(self.service.calls.count("capture_note"), captures)

    def test_mismatched_receipt_never_claims_verification(self):
        self.service.bad_receipt = True
        with self.assertRaises(J.JournalError):
            self.boundary.submit(request(), HEADERS)
        entry = self.journal.read(self.entry_id)
        self.assertEqual(entry["state"], "uncertain")
        self.assertIsNone(entry["receipt"])
        self.service.bad_receipt = False
        self.assertTrue(self.boundary.recover(self.entry_id, HEADERS)["independently_verified"])
        # A changed original receipt cannot overwrite the retained authoritative one.
        self.service.receipts[self.entry_id]["record"]["revision"] = "b" * 64
        with self.assertRaisesRegex(J.JournalError, "original_receipt_changed"):
            self.boundary.recover(self.entry_id, HEADERS)
        self.assertEqual(self.journal.read(self.entry_id)["receipt"]["record"]["revision"], "a" * 64)

    def test_concurrent_duplicate_and_distinct_saves_keep_stable_identities(self):
        args = [copy.deepcopy(ARGS) for _ in range(12)]
        for index in range(4, 12):
            args[index]["request_id"] = "fictional-concurrent-" + str(index)
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda value: self.boundary.submit(request(value), HEADERS), args))
        self.assertEqual(len(results), 12)
        self.assertEqual(len(self.service.receipts), 9)
        self.assertEqual(len(self.journal.summaries()), 9)
        self.assertTrue(all(e["independently_verified"] for e in self.journal.summaries()))
        self.assertTrue(all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in self.directory.iterdir()))
        self.assertEqual(stat.S_IMODE(self.directory.stat().st_mode), 0o700)

    def test_exposed_symlink_fifo_and_corrupted_entries_fail_without_following(self):
        self.boundary.submit(request(), HEADERS)
        path = self.directory / (self.entry_id + ".json")
        saved = path.read_bytes()
        path.chmod(0o644)
        with self.assertRaises(J.JournalError):
            self.journal.read(self.entry_id)
        path.unlink()
        external = Path(self.temporary.name) / "external"
        external.write_bytes(saved)
        external.chmod(0o600)
        path.symlink_to(external)
        with self.assertRaises(OSError):
            self.journal.read(self.entry_id)
        path.unlink()
        os.mkfifo(path, 0o600)
        with self.assertRaises(J.JournalError):
            self.journal.read(self.entry_id)
        path.unlink()
        path.write_bytes(b"invalid fixture JSON")
        path.chmod(0o600)
        with self.assertRaises(J.JournalError):
            self.journal.read(self.entry_id)
        self.assertEqual(external.read_bytes(), saved)

    def test_endpoint_and_protocol_identity_checks(self):
        self.assertEqual(J.endpoint("http://LOCALHOST:80/"), "http://localhost/mcp")
        for server in ("http://192.0.2.1/mcp", "https://user:secret@example.test/mcp", "https://example.test/mcp?token=secret"):
            with self.assertRaises(J.JournalError):
                J.endpoint(server)
        wrong_id = response({"ok": True}, "unrelated")
        with self.assertRaises(J.JournalError):
            J.tool_data(wrong_id)
        for malformed in (b'{"request_id":"original","request_id":"changed"}', b'{"value":NaN}'):
            with self.assertRaises(J.JournalError):
                J.decode(malformed)

    def test_loopback_proxy_preserves_catalog_and_capture_response_and_rejects_browser_origin(self):
        # Fake transport here still runs through the actual local HTTP proxy.
        catalog = {"jsonrpc": "2.0", "id": 3, "result": {"tools": [{"name": "capture_note"}], "cacheScope": "private"}}
        original = self.service.send
        def passthrough(body, headers, method="POST"):
            if json.loads(body).get("method") == "tools/list":
                return 200, {"Content-Type": "application/json"}, J.encode(catalog)
            return original(body, headers, method)
        self.service.send = passthrough
        server = J.proxy_server(self.boundary, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = "http://127.0.0.1:" + str(server.server_port) + "/mcp"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def post(value, headers=None):
            with opener.open(urllib.request.Request(url, J.encode(value), headers=headers or HEADERS), timeout=3) as result:
                return json.load(result)
        try:
            self.assertEqual(post({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}), catalog)
            self.assertEqual(self.journal.summaries(), [])
            saved = post(request())
            self.assertEqual(saved["id"], 7)
            self.assertFalse(saved["result"]["structuredContent"]["data"]["replayed"])
            self.assertTrue(self.journal.summaries()[0]["independently_verified"])
            denied = post(request(), dict(HEADERS, Origin="https://fictional-browser.example.test"))
            self.assertIn("error", denied)
            denied_batch = post([request()])
            self.assertIn("error", denied_batch)
            self.assertEqual(self.service.calls.count("capture_note"), 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_real_upstream_http_dropped_committed_response_and_rotated_auth(self):
        fixture = self.service
        seen_credentials = []
        committed = threading.Event()
        release_response = threading.Event()
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                value = json.loads(raw)
                token = self.headers.get("Authorization")
                seen_credentials.append(token)
                if token not in {HEADERS["Authorization"], "Bearer fictional-rotated-token"}:
                    self.send_response(403)
                    self.end_headers()
                    return
                params = value["params"]
                try:
                    if params["name"] == "capture_note":
                        status, headers, body = fixture.send(raw, dict(self.headers))
                        if params["arguments"]["request_id"] == "fictional-process-crash" and value["id"] == 7:
                            committed.set()
                            release_response.wait(timeout=5)
                    else:
                        status, headers, body = fixture.call(params["name"], params["arguments"], dict(self.headers))
                except J.JournalError:
                    self.close_connection = True  # Simulate server commit then dropped TCP response.
                    return
                self.send_response(status)
                self.send_header("Content-Type", headers["Content-Type"])
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except BrokenPipeError:
                    pass
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        fixture.server = "http://127.0.0.1:" + str(server.server_port) + "/mcp"
        upstream = J.Upstream(fixture.server, timeout=2)
        self.boundary = J.CaptureBoundary(self.journal, upstream, "fictional-clawd")
        entry_id = J.journal_id(fixture.server, "fictional-clawd", ARGS["request_id"])
        try:
            # Ambient proxies never receive bearer headers.
            with mock.patch.dict(os.environ, {"HTTP_PROXY": "http://192.0.2.1:1", "ALL_PROXY": "http://192.0.2.1:1"}):
                fixture.loss_after_commit = True
                with self.assertRaises(J.JournalError):
                    self.boundary.submit(request(), HEADERS)
                self.assertEqual(self.journal.read(entry_id)["state"], "uncertain")
                self.journal.close()
                self.journal = J.Journal(self.directory)
                fixture.journal = self.journal
                self.boundary = J.CaptureBoundary(self.journal, J.Upstream(fixture.server, timeout=2), "fictional-clawd")
                with self.assertRaises(J.JournalError):
                    self.boundary.recover(entry_id, dict(HEADERS, Authorization="Bearer fictional-revoked-token"))
                result = self.boundary.recover(entry_id, dict(HEADERS, Authorization="Bearer fictional-rotated-token"))
            self.assertTrue(result["independently_verified"])
            self.assertEqual(len(fixture.receipts), 1)
            self.assertEqual(fixture.calls.count("capture_note"), 2)
            self.assertIn("Bearer fictional-rotated-token", seen_credentials)
            self.assertNotIn(b"Bearer", (self.directory / (entry_id + ".json")).read_bytes())
            # Kill a separate actual proxy process while the server has committed
            # but withheld its response. A new CLI process recovers after reset.
            environment = {"PATH": os.environ.get("PATH", ""), "HOME": self.temporary.name}
            base = [sys.executable, str(SCRIPT), "--journal-dir", str(self.directory)]
            child = subprocess.Popen(base + ["proxy", "--server", fixture.server,
                                     "--instance-id", "fictional-clawd", "--port", "0"],
                                     env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertTrue(select.select([child.stdout], [], [], 3)[0], "Proxy did not become ready")
                proxy_url = re.search(r"http://127\.0\.0\.1:\d+/mcp", child.stdout.readline()).group(0)
                args = copy.deepcopy(ARGS)
                args["request_id"] = "fictional-process-crash"
                def capture_in_child():
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    with opener.open(urllib.request.Request(proxy_url, J.encode(request(args)), headers=HEADERS), timeout=5) as result:
                        return result.read()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    client = pool.submit(capture_in_child)
                    self.assertTrue(committed.wait(timeout=3))
                    child.kill()
                    child.wait(timeout=3)
                    release_response.set()
                    with self.assertRaises(Exception):
                        client.result(timeout=3)
                crash_id = J.journal_id(fixture.server, "fictional-clawd", args["request_id"])
                self.assertEqual(self.journal.read(crash_id)["state"], "uncertain")
                environment["GRAPHRAG_TOKEN"] = "fictional-rotated-token"
                recovered = subprocess.run(base + ["recover", crash_id, "--server", fixture.server,
                                           "--instance-id", "fictional-clawd"], env=environment,
                                           capture_output=True, text=True, timeout=8)
                self.assertEqual(recovered.returncode, 0, recovered.stderr)
                self.assertTrue(json.loads(recovered.stdout)["independently_verified"])
                self.assertEqual(len(fixture.receipts), 2)
                self.assertNotIn("fictional-rotated-token", recovered.stdout + recovered.stderr)
            finally:
                release_response.set()
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=3)
                child.stdout.close()
                child.stderr.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
