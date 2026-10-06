#!/usr/bin/env python3
"""Opt-in read-only search measurements; private cases and traces stay local."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

RUNNER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

SPEC = importlib.util.spec_from_file_location("retrieval_evaluation", Path(__file__).with_name("evaluate-retrieval.py"))
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)
PHASES = {"application_search", "query_embedding", "note_vector", "note_fulltext",
          "graph_entity_matching", "graph_mentions", "graph_edge_expansion",
          "graph_note_hydration", "graph_provenance"}
SAFE_ERRORS = {"unauthorized", "forbidden", "provider_unavailable", "compatibility", "busy",
               "cancelled", "invalid_input", "service_unreachable", "internal"}
ANSI = re.compile(r"\x1b\[[0-9;]*m")


class BenchmarkError(Exception):
    def __init__(self, code, key=None):
        self.code, self.key = code, key


def endpoint(value):
    parsed = urllib.parse.urlsplit(value)
    if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path != "/mcp":
        raise BenchmarkError("invalid_endpoint")
    try:
        loopback = parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname).is_loopback
    except (ValueError, TypeError):
        loopback = False
    if parsed.scheme not in ("http", "https") or not parsed.hostname or (parsed.scheme == "http" and not loopback):
        raise BenchmarkError("invalid_endpoint")
    try:
        parsed.port
    except ValueError as error:
        raise BenchmarkError("invalid_endpoint") from error
    return value


def rpc_hash(identifier):
    return hashlib.sha256(json.dumps(identifier, separators=(",", ":")).encode()).hexdigest()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class McpClient:
    def __init__(self, url, token, timeout, deadline=None):
        self.url, self.token, self.timeout = endpoint(url), token, timeout
        self.deadline = deadline
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, method, params):
        identifier = str(uuid.uuid4())
        key = rpc_hash(identifier)
        remaining = self.deadline - time.monotonic() if self.deadline is not None else self.timeout
        if remaining <= 0:
            raise BenchmarkError("suite_deadline", key)
        body = json.dumps({"jsonrpc": "2.0", "id": identifier, "method": method, "params": params},
                          separators=(",", ":")).encode()
        request = urllib.request.Request(self.url, data=body, headers={
            "Authorization": "Bearer " + self.token, "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream", "MCP-Protocol-Version": "2025-11-25"})
        started = time.monotonic()
        try:
            with self.opener.open(request, timeout=min(self.timeout, remaining)) as response:
                raw = response.read(2 * 1024 * 1024 + 1)
            elapsed = (time.monotonic() - started) * 1000
        except urllib.error.HTTPError as error:
            code = "unauthorized" if error.code == 401 else "forbidden" if error.code == 403 else "busy" if error.code == 429 else "http_error"
            error.close()
            raise BenchmarkError(code, key) from None
        except (urllib.error.URLError, OSError, TimeoutError):
            raise BenchmarkError("transport_or_timeout", key) from None
        if len(raw) > 2 * 1024 * 1024:
            raise BenchmarkError("response_too_large")
        try:
            value = json.loads(raw)
            if not isinstance(value, dict) or value.get("jsonrpc") != "2.0" or value.get("id") != identifier or "error" in value or "result" not in value:
                raise BenchmarkError("protocol")
        except (ValueError, UnicodeError):
            raise BenchmarkError("protocol") from None
        return value["result"], elapsed, key

    def initialize(self):
        result, elapsed, _ = self.request("initialize", {
            "protocolVersion": "2025-11-25", "capabilities": {},
            "clientInfo": {"name": "private-search-benchmark", "version": "1"}})
        if not isinstance(result, dict) or result.get("protocolVersion") != "2025-11-25":
            raise BenchmarkError("protocol")
        return elapsed

    def call(self, name, arguments):
        result, elapsed, key = self.request("tools/call", {"name": name, "arguments": arguments})
        if not isinstance(result, dict):
            raise BenchmarkError("protocol")
        envelope = result.get("structuredContent")
        if not isinstance(envelope, dict) or envelope.get("schema_version") != 1 or "error" not in envelope or "data" not in envelope:
            raise BenchmarkError("protocol")
        if envelope["error"] is not None:
            error = envelope["error"]
            code = error.get("code") if isinstance(error, dict) else None
            raise BenchmarkError(code if code in SAFE_ERRORS else "remote_error")
        if result.get("isError") or not isinstance(envelope["data"], dict):
            raise BenchmarkError("protocol")
        return envelope["data"], elapsed, key


def sample(client, item, policy, phase, round_number, concurrency, connect_ms):
    started = time.monotonic()
    row = {"name": item["name"], "category": item["category"], "policy": policy, "phase": phase,
           "round": round_number, "concurrency": concurrency, "initialize_ms": connect_ms}
    try:
        case = item["eval"]
        mode, graph = evaluation.POLICIES[policy]
        args = {"query": case["query"], "mode": mode, "graph": graph, "scope": case.get("scope", "notes"),
                "limit": case.get("limit", 5), "since_days": case.get("since_days"), "source_uri": case.get("source_uri")}
        data, elapsed, key = client.call("search_notes", args)
        records = data.get("records")
        evaluation.validate_records(records, args["limit"])
        row.update(search_rpc_ms=elapsed, rpc_id_sha256=key, records=records)
        readbacks = []
        for record in records:
            inspected, _, _ = client.call("get_record", {"id": record["id"], "revision": record["revision"], "neighbors": 0})
            readbacks.append(evaluation.verify_readback(record, inspected))
        row.update(status="ok", metrics=evaluation.metrics(item, records), readbacks=readbacks)
    except (BenchmarkError, evaluation.EvaluationError) as error:
        row.update(status="failed", error=error.code if isinstance(error, BenchmarkError) else "retrieval_or_readback")
        if isinstance(error, BenchmarkError) and error.key and "rpc_id_sha256" not in row:
            row["rpc_id_sha256"] = error.key
    row["sample_with_readbacks_ms"] = (time.monotonic() - started) * 1000
    return row


def run(suite, factory, rounds, concurrency_values):
    rows = []
    # One fresh client per active lane. Initialization is outside each timed
    # search; each lane performs search plus guarded readback before its next job.
    for concurrency in concurrency_values:
        clients, initialization = [], []
        for _ in range(concurrency):
            client = factory()
            try:
                initialized = client.initialize()
            except BenchmarkError as error:
                # Account for every scheduled comparison if a lane cannot connect.
                initialized = None
                client = error
            clients.append(client)
            initialization.append(initialized)
        jobs = [(item, policy, phase, number) for phase, count in (("first_observed", 1), ("warmed", rounds))
                for number in range(count) for item in suite["cases"] for policy in evaluation.POLICIES]
        def lane(index):
            output = []
            for item, policy, phase, number in jobs[index::concurrency]:
                client = clients[index]
                if isinstance(client, BenchmarkError):
                    output.append({"name": item["name"], "category": item["category"], "policy": policy,
                                   "phase": phase, "round": number, "concurrency": concurrency,
                                   "status": "failed", "error": client.code})
                else:
                    output.append(sample(client, item, policy, phase, number, concurrency, initialization[index]))
            return output
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            outputs = list(pool.map(lane, range(concurrency)))
        # Fixed identity/order aids comparisons even when completion order differs.
        combined = [row for output in outputs for row in output]
        combined.sort(key=lambda row: (row["phase"] != "first_observed", row["round"], row["name"], row["policy"]))
        rows.extend(combined)
    return rows


def parse_phases(text):
    output = {}
    for line in text.splitlines():
        line = ANSI.sub("", line)
        key = re.search(r"rpc_id_sha256=([0-9a-f]{64})", line)
        phase = re.search(r'phase="([a-z_]+)"', line)
        elapsed = re.search(r"elapsed_ms=([0-9.eE+-]+)", line)
        if key and phase and elapsed and phase[1] in PHASES:
            value = float(elapsed[1])
            if math.isfinite(value) and value >= 0:
                output.setdefault(key[1], {}).setdefault(phase[1], []).append(value)
    return output


def distribution(values, minimum=20):
    if not values:
        return {"samples": 0, "p50_ms": None, "p95_ms": None}
    ordered = sorted(values)
    return {"samples": len(values), "p50_ms": statistics.median(values) if len(values) >= minimum else None,
            "p95_ms": ordered[math.ceil(0.95 * len(values)) - 1] if len(values) >= minimum else None,
            "min_ms": ordered[0], "max_ms": ordered[-1]}


def summary(report):
    result = {"schema_version": 1, "suite_sha256": report["suite_sha256"], "groups": [],
              "limits": ["Stateless HTTP client timings exclude MCP initialization and guarded readback.",
                         "First observations are not proof of unloaded model/cold OS cache.",
                         "Percentiles require at least 20 successful samples; failures remain counted.",
                         "Phase timings use the same hashed RPC identity; nested phases are not additive.",
                         "Concurrent lanes include guarded readback between searches; this is bounded client load.",
                         "Conversational model time and direct OpenClaw dispatch require separate observations."]}
    for concurrency in sorted({row["concurrency"] for row in report["samples"]}):
        for policy in evaluation.POLICIES:
            for phase in ("first_observed", "warmed"):
                rows = [r for r in report["samples"] if (r["concurrency"], r["policy"], r["phase"]) == (concurrency, policy, phase)]
                good = [r for r in rows if r["status"] == "ok"]
                observed = [r for r in good if len(r.get("backend_phases", {}).get("application_search", [])) == 1]
                group = {"concurrency": concurrency, "policy": policy, "phase": phase,
                         "scheduled": len(rows), "successful": len(good), "failed": len(rows) - len(good),
                         "failure_categories": {code: sum(r.get("error") == code for r in rows)
                                                for code in sorted({r["error"] for r in rows if r["status"] != "ok"})},
                         "search_rpc": distribution([r["search_rpc_ms"] for r in good]),
                         "same_rpc_backend_observations": len(observed), "backend_phases": {}, "categories": {}}
                for category in sorted({r["category"] for r in rows}):
                    selected = [r for r in rows if r["category"] == category]
                    passed = [r for r in selected if r["status"] == "ok"]
                    group["categories"][category] = {
                        "scheduled": len(selected), "failed": len(selected) - len(passed),
                        "search_rpc": distribution([r["search_rpc_ms"] for r in passed])}
                for name in sorted(PHASES):
                    # Repeated phases within a request are summed for that phase only.
                    values = [sum(r["backend_phases"][name]) for r in good if name in r.get("backend_phases", {})]
                    if values:
                        group["backend_phases"][name] = distribution(values)
                budget = 250 if policy == "keyword-off" else 1000
                group["target_ms"] = budget
                p95 = group["search_rpc"]["p95_ms"]
                group["within_target"] = p95 <= budget if p95 is not None and not group["failed"] else None
                result["groups"].append(group)
    pairs = []
    indexed = {(r["concurrency"], r["phase"], r["round"], r["name"], r["policy"]): r for r in report["samples"]}
    for concurrency in sorted({r["concurrency"] for r in report["samples"]}):
        for policy in ("hybrid-auto", "hybrid-on"):
            deltas, regressions, comparisons, scheduled, failed = [], 0, 0, 0, 0
            for row in report["samples"]:
                if row["concurrency"] != concurrency or row["phase"] != "warmed" or row["policy"] != policy:
                    continue
                scheduled += 1
                base = indexed.get((concurrency, "warmed", row["round"], row["name"], "hybrid-off"))
                if row["status"] != "ok" or not base or base["status"] != "ok":
                    failed += 1
                    continue
                deltas.append(row["search_rpc_ms"] - base["search_rpc_ms"])
                a, b = base["metrics"]["reciprocal_rank_of_judged_positives"], row["metrics"]["reciprocal_rank_of_judged_positives"]
                if a is not None and b is not None:
                    comparisons += 1
                    regressions += b < a
            dist = distribution(deltas)
            pairs.append({"concurrency": concurrency, "policy": policy, "paired_search_rpc_delta": dist,
                          "scheduled_pairs": scheduled, "failed_pairs": failed,
                          "target_delta_ms": 250, "within_target": dist["p95_ms"] <= 250 if dist["p95_ms"] is not None and not failed else None,
                          "judged_positive_rr_comparisons": comparisons, "judged_positive_rr_regressions": regressions})
    result["graph_pairs"] = pairs
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--credential-env", default="GRAPHRAG_TOKEN")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--concurrency", default="1,4")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--deadline-seconds", type=int, default=900)
    parser.add_argument("--backend-log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        os.umask(0o077)
        suite_bytes = args.suite.read_bytes()
        suite_sha256 = hashlib.sha256(suite_bytes).hexdigest()
        suite = evaluation.load_suite(args.suite, captured_bytes=suite_bytes)
        concurrency = [int(v) for v in args.concurrency.split(",")]
        evaluation.require(concurrency and len(set(concurrency)) == len(concurrency) and all(1 <= v <= 8 for v in concurrency), "invalid concurrency")
        evaluation.require(1 <= args.rounds <= 200 and 0 < args.timeout <= 300 and math.isfinite(args.timeout), "invalid bounds")
        evaluation.require(1 <= args.deadline_seconds <= 14400 and len(suite["cases"]) <= 60, "invalid suite bounds")
        evaluation.require(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.credential_env), "invalid credential environment")
        token = os.environ.get(args.credential_env, "")
        evaluation.require(token and not any(ch.isspace() for ch in token), "missing credential")
        endpoint(args.endpoint)
        evaluation.require(not args.output.exists() and not args.summary.exists(), "report exists")
        evaluation.private_directory(args.output.parent)
        evaluation.private_directory(args.summary.parent)
        deadline = time.monotonic() + args.deadline_seconds
        samples = run(suite, lambda: McpClient(args.endpoint, token, args.timeout, deadline), args.rounds, concurrency)
        if args.backend_log:
            evaluation.require(args.backend_log.stat().st_size <= 64 * 1024 * 1024, "trace exceeds bound")
            phases = parse_phases(args.backend_log.read_text())
            for row in samples:
                row["backend_phases"] = phases.get(row.get("rpc_id_sha256"), {})
        report = {"schema_version": 1, "metadata": suite["metadata"], "samples": samples,
                  "suite_sha256": suite_sha256, "runner_sha256": RUNNER_SHA256,
                  "evaluation_runner_sha256": evaluation.RUNNER_SHA256}
        evaluation.write_new(args.output, report)
        evaluation.write_new(args.summary, summary(report))
        failed = sum(r["status"] != "ok" for r in samples)
        print(json.dumps({"samples": len(samples), "failed": failed}))
        return 1 if failed else 0
    except (BenchmarkError, evaluation.EvaluationError, OSError, ValueError, TypeError):
        print("Benchmark setup failed; check private permissions, schema, endpoint and bounds.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
