import importlib.util
import json
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import tempfile
from unittest import mock
from pathlib import Path
import threading
import time
import unittest


SPEC = importlib.util.spec_from_file_location("search_benchmark", Path(__file__).resolve().parents[1] / "benchmark-search.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def case():
    return {"name": "fictional-atlas", "split": "calibration", "category": "entity",
            "answerability": "answerable", "eval": {"schema_version": 2, "query": "Atlas launch",
            "k": 5, "limit": 5, "relevance": [{"id": "note:atlas", "grade": 3}]}}


def record():
    return {"id": "note:atlas", "revision": "fictional-revision", "title": "Atlas",
            "content": "Fictional Atlas launch tomorrow.", "provenance": {"instance_id": "fictional-owner"}}


class Client:
    def initialize(self):
        return 2

    def call(self, name, args):
        if name == "search_notes":
            return {"records": [record()]}, 12, "a" * 64
        return record(), 1, "b" * 64


class SearchBenchmarkTests(unittest.TestCase):
    def test_real_http_lifecycle_requires_notification_before_tools(self):
        class Handler(BaseHTTPRequestHandler):
            methods = []
            initialized = False
            def log_message(self, *args):
                pass
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                Handler.methods.append(request["method"])
                if request["method"] == "notifications/initialized":
                    self.server.notification_has_id = "id" in request
                    Handler.initialized = True
                    self.send_response(202)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if request["method"] == "initialize":
                    result = {"protocolVersion": "2025-11-25"}
                elif not Handler.initialized:
                    self.send_error(400)
                    return
                else:
                    result = {"structuredContent": {"schema_version": 1, "error": None, "data": {"records": []}}}
                body = json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = benchmark.McpClient(f"http://127.0.0.1:{server.server_port}/mcp", "fictional", 2)
            with self.assertRaises(benchmark.BenchmarkError):
                client.call("search_notes", {})
            self.assertGreaterEqual(client.initialize(), 0)
            self.assertEqual(client.call("search_notes", {})[0], {"records": []})
            self.assertFalse(server.notification_has_id)
            self.assertEqual(Handler.methods[-3:], ["initialize", "notifications/initialized", "tools/call"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_truncated_http_body_is_counted_without_aborting_scheduled_work(self):
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            variant = "chunked"
            def log_message(self, *args):
                pass
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                if self.variant == "chunked":
                    self.send_header("Transfer-Encoding", "chunked")
                else:
                    self.send_header("Content-Length", "16")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(b"10\r\n{}" if self.variant == "chunked" else b"{}")
                self.close_connection = True
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        class Truncated(benchmark.McpClient):
            def initialize(self):
                return 0
        try:
            factory = lambda: Truncated(f"http://127.0.0.1:{server.server_port}/mcp", "fictional", 2)
            for variant in ("chunked", "content-length"):
                Handler.variant = variant
                with self.subTest(variant=variant):
                    rows = benchmark.run({"cases": [case()]}, factory, 2, [1, 4])
                    self.assertEqual(len(rows), 24)
                    self.assertTrue(all(row["status"] == "failed" and row["error"] == "transport_or_timeout" for row in rows))
                    self.assertTrue(all(len(row["rpc_id_sha256"]) == 64 for row in rows))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_first_observed_percentiles_do_not_receive_warmed_budget_verdicts(self):
        cases = []
        for number in range(20):
            item = case()
            item["name"] += str(number)
            cases.append(item)
        rows = benchmark.run({"cases": cases}, Client, 1, [1])
        result = benchmark.summary({"suite_sha256": "fixture", "samples": rows})
        first = [group for group in result["groups"] if group["phase"] == "first_observed"]
        warm = [group for group in result["groups"] if group["phase"] == "warmed"]
        self.assertTrue(all(group["search_rpc"]["p95_ms"] == 12 for group in first))
        self.assertTrue(all(group["within_target"] is None and group["target_ms"] is None for group in first))
        self.assertTrue(all(group["within_target"] is True and group["target_ms"] in (250, 1000) for group in warm))

    def test_every_warm_policy_uses_all_four_persistent_lanes_without_mixed_batches(self):
        seen, active = {}, {}
        peaks = {}
        first_wave = {policy: threading.Barrier(4) for policy in benchmark.evaluation.POLICIES}
        lock = threading.Lock()
        original = benchmark.sample
        def observed(client, item, policy, phase, number, concurrency, connect_ms):
            first = False
            with lock:
                self.assertTrue(all(value == policy for value in active.values()))
                active[id(client)] = policy
                if phase == "warmed":
                    first = id(client) not in seen.setdefault(policy, set())
                    seen.setdefault(policy, set()).add(id(client))
                    peaks[policy] = max(peaks.get(policy, 0), len(active))
            try:
                if first:
                    first_wave[policy].wait(timeout=2)
                time.sleep(0.001)
                return original(client, item, policy, phase, number, concurrency, connect_ms)
            finally:
                with lock:
                    active.pop(id(client))
        with mock.patch.object(benchmark, "sample", side_effect=observed):
            rows = benchmark.run({"cases": [case()]}, Client, 8, [4])
        self.assertEqual(len(rows), 36)
        self.assertTrue(all(len(lanes) == 4 for lanes in seen.values()))
        self.assertTrue(all(peak == 4 for peak in peaks.values()))
        self.assertEqual(len(set.union(*seen.values())), 4)

    def test_nullable_eval_options_use_resolved_mcp_arguments_and_metrics(self):
        item = case()
        item["eval"].update(scope=None, limit=None, k=None, relevance=[{"id": "note:atlas", "grade": None}])
        class Checked(Client):
            def call(self, name, args):
                if name == "search_notes":
                    self.arguments = args
                return super().call(name, args)
        client = Checked()
        row = benchmark.sample(client, item, "hybrid-off", "warmed", 0, 1, 2)
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["metrics"]["k"], 5)
        self.assertEqual(client.arguments["scope"], "notes")
        self.assertEqual(client.arguments["limit"], 5)

    def test_raw_and_summary_path_alias_is_rejected_before_measurement(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            root.chmod(0o700)
            alias = root / "alias"
            alias.symlink_to(root, target_is_directory=True)
            suite = root / "suite.json"
            suite.write_text(json.dumps({"schema_version": 1, "metadata": {}, "cases": [case()]}))
            with mock.patch.object(benchmark, "run") as run, \
                    mock.patch.dict("os.environ", {"FIXTURE_TOKEN": "fictional"}), mock.patch("sys.stderr"):
                code = benchmark.main(["--suite", str(suite), "--endpoint", "http://127.0.0.1:1/mcp",
                    "--credential-env", "FIXTURE_TOKEN", "--output", str(root / "result.json"),
                    "--summary", str(alias / "result.json")])
            self.assertEqual(code, 2)
            self.assertFalse((root / "result.json").exists())
            run.assert_not_called()

    def test_http_rejects_duplicate_fields_nonfinite_json_and_boolean_schema(self):
        class Handler(BaseHTTPRequestHandler):
            variant = "valid"
            def log_message(self, *args):
                pass
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                value = {"jsonrpc": "2.0", "id": request["id"], "result": {"structuredContent": {
                    "schema_version": 1, "error": None, "data": {"records": []}}}}
                body = json.dumps(value)
                if self.variant == "duplicate":
                    body = body.replace('"jsonrpc": "2.0"', '"jsonrpc": "bad", "jsonrpc": "2.0"')
                elif self.variant == "nonfinite":
                    body = body.replace('"records": []', '"records": [], "extra": NaN')
                elif self.variant == "boolean":
                    body = body.replace('"schema_version": 1', '"schema_version": true')
                self.send_response(200)
                self.send_header("Content-Length", str(len(body.encode())))
                self.end_headers()
                self.wfile.write(body.encode())
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = benchmark.McpClient(f"http://127.0.0.1:{server.server_port}/mcp", "fictional", 2)
            self.assertEqual(client.call("search_notes", {})[0], {"records": []})
            for variant in ("duplicate", "nonfinite", "boolean"):
                Handler.variant = variant
                with self.subTest(variant=variant), self.assertRaises(benchmark.BenchmarkError) as error:
                    client.call("search_notes", {})
                self.assertEqual(error.exception.code, "protocol")
                self.assertEqual(len(error.exception.key), 64)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_failed_tool_and_invalid_record_samples_retain_exact_rpc_hash(self):
        key = benchmark.rpc_hash('fictional-failed-request')
        client = benchmark.McpClient('http://127.0.0.1:1/mcp', 'fictional', 2)
        result = {'structuredContent': {'schema_version': 1, 'error': {'code': 'provider_unavailable'}, 'data': None}}
        with mock.patch.object(client, 'request', return_value=(result, 4, key)):
            row = benchmark.sample(client, case(), 'hybrid-off', 'warmed', 0, 1, 2)
        self.assertEqual(row['status'], 'failed')
        self.assertEqual(row['error'], 'provider_unavailable')
        self.assertEqual(row['rpc_id_sha256'], key)
        for envelope in ({'structuredContent': {}}, {'structuredContent': {'schema_version': 1, 'error': None, 'data': {'records': 'invalid'}}}):
            with self.subTest(envelope=envelope), mock.patch.object(client, 'request', return_value=(envelope, 4, key)):
                row = benchmark.sample(client, case(), 'hybrid-off', 'warmed', 0, 1, 2)
                self.assertEqual(row['status'], 'failed')
                self.assertEqual(row['rpc_id_sha256'], key)

    def test_report_uses_suite_and_runner_loaded_before_requests(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            root.chmod(0o700)
            suite, output, aggregate = (root / value for value in ("suite.json", "report.json", "summary.json"))
            original = json.dumps({"schema_version": 1, "metadata": {}, "cases": [case()]}).encode()
            suite.write_bytes(original)
            def replace_inputs(loaded, factory, rounds, concurrency):
                self.assertEqual(loaded["cases"][0]["name"], "fictional-atlas")
                suite.write_text("replaced suite")
                return []
            runner = root / "runner.py"
            runner.write_text("replacement code")
            with mock.patch.object(benchmark, "run", side_effect=replace_inputs), \
                    mock.patch.object(benchmark, "__file__", str(runner)), \
                    mock.patch.dict("os.environ", {"FIXTURE_TOKEN": "fictional"}), mock.patch("sys.stdout"):
                code = benchmark.main(["--suite", str(suite), "--endpoint", "http://127.0.0.1:1/mcp",
                    "--credential-env", "FIXTURE_TOKEN", "--output", str(output), "--summary", str(aggregate)])
            report = json.loads(output.read_text())
            self.assertEqual(code, 0)
            self.assertEqual(report["suite_sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(report["runner_sha256"], benchmark.RUNNER_SHA256)
            self.assertEqual(report["evaluation_runner_sha256"], benchmark.evaluation.RUNNER_SHA256)

    def test_endpoint_refuses_insecure_remote_userinfo_and_redirect_locations(self):
        for value in ("http://example.test/mcp", "http://user:secret@localhost/mcp",
                      "http://localhost/mcp?token=secret", "http://localhost/other", "http:///mcp"):
            with self.subTest(value=value), self.assertRaises(benchmark.BenchmarkError):
                benchmark.endpoint(value)
        self.assertEqual(benchmark.endpoint("http://127.0.0.1:3000/mcp"), "http://127.0.0.1:3000/mcp")
        self.assertIsNone(benchmark.NoRedirect().redirect_request(None))

    def test_failed_initialization_counts_all_scheduled_comparisons(self):
        class Failed(Client):
            def initialize(self):
                raise benchmark.BenchmarkError("unauthorized")
        rows = benchmark.run({"cases": [case()]}, Failed, 2, [1, 4])
        self.assertEqual(len(rows), 24)
        self.assertTrue(all(row["status"] == "failed" and row["error"] == "unauthorized" for row in rows))

    def test_stale_readback_is_a_failure_and_preserves_timed_search(self):
        class Stale(Client):
            def call(self, name, args):
                value, elapsed, key = super().call(name, args)
                if name == "get_record":
                    value["revision"] = "changed"
                return value, elapsed, key
        row = benchmark.sample(Stale(), case(), "hybrid-off", "warmed", 0, 1, 2)
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["search_rpc_ms"], 12)
        self.assertNotIn("metrics", row)

    def test_readbacks_are_outside_search_timing_and_retained_for_quality(self):
        row = benchmark.sample(Client(), case(), "hybrid-auto", "warmed", 0, 1, 2)
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["search_rpc_ms"], 12)
        self.assertEqual(row["metrics"]["reciprocal_rank_of_judged_positives"], 1)
        self.assertEqual(row["readbacks"][0]["revision"], "fictional-revision")

    def test_trace_correlation_uses_exact_rpc_hash_and_only_known_phases(self):
        key = benchmark.rpc_hash("fictional-rpc")
        text = (f'\x1b[32mDEBUG\x1b[0m mcp_search{{rpc_id_sha256={key}}}: phase="query_embedding" elapsed_ms=3.5\n'
                f'DEBUG mcp_search{{rpc_id_sha256={key}}}: phase="application_search" elapsed_ms=5\n'
                f'DEBUG phase="query_embedding" elapsed_ms=999\n'
                f'DEBUG mcp_search{{rpc_id_sha256={key}}}: phase="private_field" elapsed_ms=99\n')
        self.assertEqual(benchmark.parse_phases(text), {key: {"query_embedding": [3.5], "application_search": [5]}})

    def test_percentiles_require_enough_samples_and_use_nearest_rank_p95(self):
        self.assertIsNone(benchmark.distribution([1] * 19)["p95_ms"])
        result = benchmark.distribution(list(range(1, 21)))
        self.assertEqual(result["p50_ms"], 10.5)
        self.assertEqual(result["p95_ms"], 19)

    def test_summary_drops_private_metadata_queries_records_and_labels(self):
        rows = benchmark.run({"cases": [case()]}, Client, 20, [1])
        report = {"suite_sha256": "fictional-sha", "metadata": {"secret": "PRIVATE"}, "samples": rows}
        summary = benchmark.summary(report)
        text = json.dumps(summary)
        for private in ("PRIVATE", "fictional-atlas", "note:atlas", "Fictional Atlas", "fictional-owner", "fictional-revision"):
            self.assertNotIn(private, text)
        self.assertEqual(summary["graph_pairs"][0]["paired_search_rpc_delta"]["samples"], 20)
        self.assertEqual(summary["graph_pairs"][0]["judged_positive_rr_regressions"], 0)

    def test_concurrent_lane_clients_are_not_shared_between_threads(self):
        seen = {}
        lock = threading.Lock()
        class BoundClient(Client):
            def call(self, name, args):
                identity = threading.get_ident()
                with lock:
                    prior = seen.setdefault(id(self), identity)
                    if prior != identity:
                        raise AssertionError("client shared across threads")
                time.sleep(0.001)
                return super().call(name, args)
        rows = benchmark.run({"cases": [case()]}, BoundClient, 4, [4])
        self.assertEqual(len(seen), 4)
        self.assertTrue(all(row["status"] == "ok" for row in rows))

    def test_failed_pair_prevents_budget_claim_even_with_twenty_fast_successes(self):
        rows = benchmark.run({"cases": [case()]}, Client, 21, [1])
        row = next(r for r in rows if r["phase"] == "warmed" and r["policy"] == "hybrid-off")
        row.update(status="failed", error="transport_or_timeout")
        summary = benchmark.summary({"suite_sha256": "fixture", "samples": rows})
        pair = summary["graph_pairs"][0]
        self.assertEqual(pair["paired_search_rpc_delta"]["samples"], 20)
        self.assertEqual(pair["failed_pairs"], 1)
        self.assertIsNone(pair["within_target"])


if __name__ == "__main__":
    unittest.main()
