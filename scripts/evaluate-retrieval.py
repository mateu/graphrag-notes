#!/usr/bin/env python3
"""Compare read-only retrieval policies against private graded judgments.

Uses the existing remote CLI and version-two eval relevance fields. Real cases,
rankings, readback content, and identifiers stay in the private report. Only
explicitly whitelisted aggregate fields enter the separately requested summary.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

RUNNER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

POLICIES = {"keyword-off": ("keyword", "off"), "hybrid-off": ("hybrid", "off"),
            "hybrid-auto": ("hybrid", "auto"), "hybrid-on": ("hybrid", "on")}
CATEGORIES = {"title", "direct", "paraphrase", "entity", "relationship", "filters", "negative"}


class EvaluationError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise EvaluationError(message)


def private_directory(path):
    path = Path(path)
    require(not path.is_symlink(), "private directory must not be a symlink")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    mode = path.stat()
    require(mode.st_uid == os.getuid() and stat.S_IMODE(mode.st_mode) & 0o077 == 0,
            "evaluation directory must be owner-private")
    return path


def write_new(path, value):
    path = Path(path)
    private_directory(path.parent)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_destinations(output, summary):
    require(summary is None or Path(output).resolve() != Path(summary).resolve(),
            "private report and sanitized summary destinations must differ")
    require(not Path(output).exists() and (summary is None or not Path(summary).exists()),
            "output already exists; use new report paths")
    private_directory(Path(output).parent)
    if summary is not None:
        private_directory(Path(summary).parent)


def strict_json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON field")
            value[key] = item
        return value
    def constant(value):
        raise EvaluationError("non-finite JSON constant")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def load_suite(path, *, captured_bytes=None):
    suite = strict_json(Path(path).read_bytes() if captured_bytes is None else captured_bytes)
    require(isinstance(suite, dict) and type(suite.get("schema_version")) is int and suite["schema_version"] == 1,
            "unsupported retrieval suite version")
    require(isinstance(suite.get("metadata"), dict), "suite metadata missing")
    require(isinstance(suite.get("cases"), list) and suite["cases"], "suite has no cases")
    names = set()
    for item in suite["cases"]:
        require(isinstance(item, dict), "invalid case")
        require(set(item) <= {"name", "split", "category", "formulation", "answerability", "judgment", "eval"},
                "unknown case field")
        name = item.get("name")
        require(isinstance(name, str) and name and name not in names, "case name invalid or duplicate")
        names.add(name)
        require(item.get("split") in {"calibration", "holdout"}, "invalid case split")
        require(item.get("category") in CATEGORIES, "invalid case category")
        require(item.get("formulation", "question") in {"question", "terms", "title"}, "invalid formulation")
        require(item.get("answerability") in {"answerable", "unanswerable", "unjudged"},
                "invalid answerability")
        case = item.get("eval")
        require(isinstance(case, dict) and case.get("schema_version") == 2, "eval case must use v2")
        require(set(case) <= {"schema_version", "query", "scope", "limit", "k", "since_days",
                              "source_uri", "relevance", "expected_ids"}, "unsupported eval field")
        require(isinstance(case.get("query"), str) and case["query"].strip(), "query missing")
        require(case.get("scope", "notes") in {"notes", "messages", "all"}, "invalid scope")
        limit = case.get("limit", 5)
        k = case.get("k", limit)
        require(type(k) is int and type(limit) is int and 1 <= k <= limit <= 200, "invalid result bounds")
        require(case.get("since_days") is None or (type(case["since_days"]) is int and
                0 <= case["since_days"] <= 365000), "invalid recency filter")
        require(case.get("source_uri") is None or isinstance(case["source_uri"], str), "invalid source filter")
        grades = judgments(case)
        require(item["answerability"] != "answerable" or any(g > 0 for g in grades.values()),
                "answerable case has no positive judgment")
        require(item["answerability"] != "unanswerable" or not any(g > 0 for g in grades.values()),
                "unanswerable case has positive judgments")
    return suite


def judgments(case):
    grades = {}
    require(isinstance(case.get("expected_ids", []), list), "expected IDs must be a list")
    for identifier in case.get("expected_ids", []):
        identifier = normalized_id(identifier)
        grades[identifier] = 1
    require(isinstance(case.get("relevance", []), list), "relevance must be a list")
    explicit = {}
    for item in case.get("relevance", []):
        require(isinstance(item, dict) and set(item) <= {"id", "grade"}, "invalid relevance entry")
        identifier, grade = item.get("id"), item.get("grade", 1)
        identifier = normalized_id(identifier)
        require(type(grade) is int and 0 <= grade <= 63, "invalid relevance grade")
        require(identifier not in explicit or explicit[identifier] == grade, "conflicting relevance judgments")
        explicit[identifier] = grade
        grades[identifier] = max(grades.get(identifier, 0), grade)
    return grades


def normalized_id(value):
    require(isinstance(value, str) and value.strip(), "invalid judgment/result ID")
    return value.strip().lower()


def metrics(item, records):
    case = item["eval"]
    k = case.get("k", case.get("limit", 5))
    grades = judgments(case) if item["answerability"] != "unjudged" else {}
    ranked = records[:k]
    ids = [normalized_id(record["id"]) for record in ranked]
    require(len(ids) == len(set(ids)), "duplicate ranked records")
    positive = {identifier for identifier, grade in grades.items() if grade > 0}
    known = [grades.get(identifier) for identifier in ids]
    relevant = sum(grade is not None and grade > 0 for grade in known)
    fully_judged = all(grade is not None for grade in known) and item["answerability"] != "unjudged"
    # An explicitly reviewed no-answer case judges all returned hits irrelevant.
    negative = item["answerability"] == "unanswerable"
    if negative:
        known = [0] * len(ids)
        fully_judged = True
    result = {"retrieved": len(ids), "k": k, "known_positive_total": len(positive),
              "unjudged_hits": sum(grade is None for grade in known),
              "fully_judged_top_k": fully_judged,
              "top_k_useful_lower_bound": relevant / k,
              "top_k_useful_upper_bound": (relevant + sum(g is None for g in known)) / k,
              "recall_of_judged_positives": relevant / len(positive) if positive else None,
              "reciprocal_rank_of_judged_positives": next((1 / (i + 1) for i, g in enumerate(known) if g and g > 0), 0.0)
                                  if positive else None,
              "precision_at_k": relevant / k if fully_judged and positive else None,
              "false_positive_count": len(records) if negative else
                                      sum(grades.get(normalized_id(record["id"])) == 0 for record in records),
              "negative_empty": len(ids) == 0 if negative else None,
              "ndcg_at_k": None}
    if fully_judged and positive:
        dcg = sum((2 ** (grade or 0) - 1) / math.log2(i + 2) for i, grade in enumerate(known))
        ideal = sorted((grade for grade in grades.values() if grade > 0), reverse=True)[:k]
        idcg = sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(ideal))
        result["ndcg_at_k"] = dcg / idcg
    return result


def cli_data(value, command):
    require(isinstance(value, dict) and type(value.get("schema_version")) is int and value["schema_version"] == 1 and
            value.get("command") == command and value.get("success") is True and
            value.get("errors") == [], "invalid CLI success envelope")
    service = value.get("data")
    require(isinstance(service, dict) and type(service.get("schema_version")) is int and service["schema_version"] == 1 and
            "error" in service and service["error"] is None and "data" in service,
            "invalid service success envelope")
    return service["data"]


class RemoteCli:
    def __init__(self, binary, endpoint, credential_env, timeout):
        self.base = [str(binary), "--server", endpoint, "--credential-env", credential_env]
        self.timeout = timeout

    def call(self, arguments, command):
        try:
            response = subprocess.run(self.base + [arguments[0], "--format", "json"] + arguments[1:],
                                      text=True, capture_output=True, timeout=self.timeout, check=False)
        except subprocess.TimeoutExpired as error:
            raise EvaluationError("remote CLI request exceeded evaluation timeout") from error
        require(response.returncode == 0, "remote CLI request failed; no mode fallback was attempted")
        try:
            return cli_data(strict_json(response.stdout), command)
        except (ValueError, TypeError) as error:
            raise EvaluationError("invalid remote CLI JSON") from error

    def search(self, item, policy):
        case, (mode, graph) = item["eval"], POLICIES[policy]
        args = ["search", "--mode", mode, "--graph", graph, "--scope", case.get("scope", "notes"),
                "--limit", str(case.get("limit", 5))]
        for key in ("since_days", "source_uri"):
            if case.get(key) is not None:
                args += ["--" + key.replace("_", "-") + "=" + str(case[key])]
        # Query as one argv token after '--': shell syntax and leading dashes are data.
        args += ["--", case["query"]]
        data = self.call(args, "search_notes")
        require(isinstance(data, dict) and isinstance(data.get("records"), list), "missing search records")
        return data["records"]

    def inspect(self, record):
        return self.call(["inspect", "--neighbors", "0", "--revision", record["revision"],
                          record["id"]], "get_record")


def validate_records(records, limit):
    require(isinstance(records, list) and len(records) <= limit, "search result limit exceeded")
    ids = set()
    for record in records:
        require(isinstance(record, dict), "invalid record")
        identifier = normalized_id(record.get("id"))
        require(identifier not in ids, "invalid or duplicate record ID")
        require(isinstance(record.get("revision"), str) and record["revision"], "missing result revision")
        require(isinstance(record.get("provenance"), dict), "missing result provenance")
        require(isinstance(record.get("content"), str), "invalid result content")
        ids.add(identifier)


def verify_readback(record, inspected):
    require(isinstance(inspected, dict), "invalid inspected record")
    for key in ("id", "revision", "title", "provenance"):
        require(inspected.get(key) == record.get(key), "inspection differs from revision-pinned search result")
    require(isinstance(inspected.get("content"), str), "inspection has no full content")
    # Search previews may be shortened, so never assert full content equals the snippet.
    return {"id": inspected["id"], "revision": inspected["revision"],
            "content_sha256": hashlib.sha256(inspected["content"].encode()).hexdigest()}


def evaluate(suite, search, inspect):
    report = {"schema_version": 1, "metadata": suite["metadata"], "cases": []}
    for item in suite["cases"]:
        for policy in POLICIES:
            started = time.monotonic()
            try:
                records = search(item, policy)
                elapsed = (time.monotonic() - started) * 1000
                validate_records(records, item["eval"].get("limit", 5))
                readbacks = [verify_readback(record, inspect(record)) for record in records]
                scored = metrics(item, records)
                scored["duplicate_full_content_count"] = len(readbacks) - len({r["content_sha256"] for r in readbacks})
                row = {"name": item["name"], "split": item["split"], "category": item["category"],
                       "formulation": item.get("formulation", "question"),
                       "policy": policy, "status": "ok", "query": item["eval"]["query"],
                       "metrics": scored, "records": records, "readbacks": readbacks,
                       "search_elapsed_ms": elapsed}
            except EvaluationError:
                # Keep categorized failures and continue comparisons, never silently substitute modes.
                row = {"name": item["name"], "split": item["split"], "category": item["category"],
                       "formulation": item.get("formulation", "question"),
                       "policy": policy, "status": "failed", "error": "retrieval_or_readback_failed"}
            report["cases"].append(row)
    return report


def aggregate(report):
    metrics_names = {"top_k_useful_lower_bound", "top_k_useful_upper_bound", "precision_at_k",
                     "recall_of_judged_positives", "reciprocal_rank_of_judged_positives", "ndcg_at_k", "negative_empty",
                     "duplicate_full_content_count", "false_positive_count"}
    summary = {"schema_version": 1, "suite_sha256": report.get("suite_sha256"),
               "limits": ["Recall counts judged positives, not every relevant record in the corpus.",
                          "Reciprocal rank is of judged positives and is a lower bound while hits remain unjudged.",
                          "Unknown hits are reported separately; precision/nDCG require judged top-k.",
                          "Search timing is diagnostic, not an end-to-end latency benchmark."], "groups": []}
    for policy in POLICIES:
        for split in ("calibration", "holdout"):
            rows = [r for r in report["cases"] if r["policy"] == policy and r["split"] == split]
            good = [r for r in rows if r["status"] == "ok"]
            group = {"policy": policy, "split": split, "cases": len(rows), "successful": len(good),
                     "formulations": {kind: sum(r.get("formulation", "question") == kind for r in rows)
                                      for kind in ("question", "terms", "title")},
                     "failed": len(rows) - len(good), "unjudged_hits": sum(r["metrics"]["unjudged_hits"] for r in good),
                     "fully_judged_cases": sum(r["metrics"]["fully_judged_top_k"] for r in good),
                     "metrics": {}, "formulation_metrics": {}}
            negatives = [r for r in good if r["metrics"]["negative_empty"] is not None]
            group["reviewed_no_answer"] = {
                "cases": len(negatives),
                "empty_results": sum(r["metrics"]["negative_empty"] for r in negatives),
                "returned_false_positives": sum(r["metrics"]["false_positive_count"] for r in negatives)}
            for name in sorted(metrics_names):
                values = [r["metrics"][name] for r in good if r["metrics"][name] is not None]
                group["metrics"][name] = {"mean": sum(values) / len(values) if values else None,
                                          "cases_scored": len(values)}
            for kind in ("question", "terms", "title"):
                selected = [r for r in good if r.get("formulation", "question") == kind]
                group["formulation_metrics"][kind] = {}
                for name in ("recall_of_judged_positives", "reciprocal_rank_of_judged_positives"):
                    values = [r["metrics"][name] for r in selected if r["metrics"][name] is not None]
                    group["formulation_metrics"][kind][name] = {
                        "mean": sum(values) / len(values) if values else None, "cases_scored": len(values)}
            summary["groups"].append(group)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--credential-env", default="GRAPHRAG_TOKEN")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--recorded", type=Path, help="Private precomputed fixture rankings/readbacks; no network")
    args = parser.parse_args(argv)
    try:
        os.umask(0o077)
        suite_bytes = args.suite.read_bytes()
        suite_sha256 = hashlib.sha256(suite_bytes).hexdigest()
        suite = load_suite(args.suite, captured_bytes=suite_bytes)
        require(args.timeout > 0 and math.isfinite(args.timeout), "invalid timeout")
        validate_destinations(args.output, args.summary)
        if args.recorded:
            require(args.binary is None and args.endpoint is None, "recorded and live modes conflict")
            fixture = strict_json(args.recorded.read_text())
            def search(item, policy):
                try:
                    return fixture["rankings"][item["name"]][policy]
                except (KeyError, TypeError) as error:
                    raise EvaluationError("recorded ranking missing") from error
            def inspect(record):
                try:
                    return fixture["inspections"][record["id"]]
                except (KeyError, TypeError) as error:
                    raise EvaluationError("recorded inspection missing") from error
        else:
            require(args.binary is not None and args.endpoint, "live evaluation requires binary and endpoint")
            require(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.credential_env) is not None,
                    "invalid credential environment name")
            require(bool(os.environ.get(args.credential_env)), "credential environment variable missing")
            client = RemoteCli(args.binary, args.endpoint, args.credential_env, args.timeout)
            search, inspect = client.search, client.inspect
        report = evaluate(suite, search, inspect)
        report["suite_sha256"] = suite_sha256
        report["runner_sha256"] = RUNNER_SHA256
        write_new(args.output, report)
        if args.summary:
            write_new(args.summary, aggregate(report))
        failures = sum(row["status"] != "ok" for row in report["cases"])
        print(json.dumps({"policy_cases": len(report["cases"]), "failures": failures}))
        return 1 if failures else 0
    except (EvaluationError, ValueError, OSError, KeyError, TypeError):
        print("Evaluation failed; check private input/output permissions, schema, and connection configuration.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
