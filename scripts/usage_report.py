#!/usr/bin/env python3
"""Summarize graphrag_usage records from a service log.

The service writes one `graphrag_usage` INFO line per MCP tool call (see
crates/service/src/usage.rs). This report groups them by principal and tool
to answer: who calls what, over exactly which interval, how often calls fail,
how often searches come back empty, how fast calls are, and how often
build_context returned graph-annotated chunks.

A principal identifies a credential, not a workload: one credential can serve
scheduled sync, evaluation runs and interactive use. The report therefore
shows the call shape of retrieval calls and assigns no intent labels.

Usage:
  scripts/usage_report.py service.stderr.log [more.log ...]
                          [--since 2026-11-02 | --since 2026-11-02T15:00:00Z]
                          [--until 2026-11-09T15:00:00Z] [--last 7d]
                          [--principal NAME] [--json]

Bounds are a half-open UTC interval [since, until). A bare date means
00:00:00 UTC of that day. `--last 7d` (or `36h`) ends at `--until`, or now.
"""

import argparse
import datetime as dt
import json
import re
import statistics
import sys
from collections import Counter, defaultdict

SCHEMA_VERSION = 2
# Below this many samples a p95 is effectively the maximum, so the report
# prints min/median/max instead and marks the group as low-sample.
MIN_PERCENTILE_SAMPLES = 20
# A sync client checks get_source before a create-only upload. A `not_found`
# followed by the same principal's successful upload_source within this many
# seconds is that expected existence probe, not a failure.
PROBE_CORRELATION_SECONDS = 5.0

UTC = dt.timezone.utc
ANSI = re.compile(r"\x1b\[[0-9;]*m")
# Any line carrying this marker is a usage record; one that then fails to
# parse is counted as skipped instead of silently ignored.
MARKER = "graphrag_usage principal="
# Production `graphrag serve` omits targets and prefixes active spans, e.g.
# "<ts>  INFO serve_inner: graphrag_usage principal=...". With targets enabled
# the same record reads "INFO graphrag_usage: graphrag_usage principal=...".
LINE = re.compile(r"^(?P<ts>\S+)\s+INFO\s+(?:\S+: )*graphrag_usage (?P<fields>principal=.*)$")
FIELD = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\S+)')
TIMESTAMP = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})[Tt](\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?(?:([Zz])|([+-])(\d{2}):?(\d{2}))$"
)
DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
DURATION = re.compile(r"^([1-9]\d*)([dh])$")
PHASES = {
    "embedding_ms",
    "lexical_ms",
    "vector_fusion_ms",
    "graph_query_candidates_ms",
    "graph_entity_matching_ms",
    "graph_mentions_ms",
    "graph_seed_evidence_ms",
    "graph_edge_expansion_ms",
    "graph_note_hydration_ms",
    "graph_provenance_ms",
    "context_packing_ms",
}
NUMERIC = PHASES | {
    "max_chunks_per_source", "elapsed_ms", "limit", "query_chars", "query_terms", "hits", "top_score",
    "graph_considered", "graph_selected", "dropped_for_relevance", "total_tokens",
}
# Workload class by tool name only. It says what kind of call this is, never
# why the client made it. Tools not listed here are reported under "other".
WORKLOADS = {
    "search_notes": "retrieval",
    "build_context": "retrieval",
    "get_note": "record_read",
    "get_record": "record_read",
    "upload_source": "ingest_sync",
    "get_source": "ingest_sync",
    "capture_note": "ingest_sync",
    "capture_metadata_schema": "ingest_sync",
    "get_job": "ingest_sync",
    "list_jobs": "ingest_sync",
    "cancel_job": "ingest_sync",
    "resume_job": "ingest_sync",
    "get_source_enrichment_job": "ingest_sync",
    "retry_source_enrichment_job": "ingest_sync",
    "cancel_source_enrichment_job": "ingest_sync",
    "scan_relationship_candidates": "graph_review",
    "propose_endpoint_relationship": "graph_review",
    "list_proposals": "graph_review",
    "get_proposal": "graph_review",
    "decide_proposal": "graph_review",
    "service_status": "status",
}
WORKLOAD_ORDER = ["retrieval", "record_read", "ingest_sync", "graph_review", "status", "other"]
RETRIEVAL_TOOLS = ("search_notes", "build_context")


def parse_timestamp(text):
    """Parse an RFC 3339 timestamp with `Z` or a numeric offset into an aware datetime.

    Hand-rolled because Python 3.9's `fromisoformat` rejects `Z` and
    fractional seconds that are not exactly three or six digits.
    """
    match = TIMESTAMP.match(text)
    if not match:
        raise ValueError(f"not an RFC 3339 timestamp with Z or an offset: {text!r}")
    year, month, day, hour, minute, second, fraction, _zulu, sign, off_h, off_m = match.groups()
    zone = UTC
    if sign:
        offset = dt.timedelta(hours=int(off_h), minutes=int(off_m))
        zone = dt.timezone(-offset if sign == "-" else offset)
    micro = int((fraction or "0")[:6].ljust(6, "0"))
    return dt.datetime(
        int(year), int(month), int(day), int(hour), int(minute), int(second), micro, tzinfo=zone
    ).astimezone(UTC)


def parse_bound(text):
    """A `--since`/`--until` value: a UTC date (its 00:00:00) or a full timestamp."""
    match = DATE.match(text)
    try:
        if match:
            return dt.datetime(*(int(part) for part in match.groups()), tzinfo=UTC)
        return parse_timestamp(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected YYYY-MM-DD or an RFC 3339 timestamp with Z or an offset, got {text!r} ({error})"
        )


def parse_duration(text):
    match = DURATION.match(text)
    if not match:
        raise argparse.ArgumentTypeError(f"expected a positive duration such as 7d or 36h, got {text!r}")
    count, unit = int(match[1]), match[2]
    return dt.timedelta(days=count) if unit == "d" else dt.timedelta(hours=count)


def iso(moment):
    if moment is None:
        return None
    precision = "microseconds" if moment.microsecond else "seconds"
    return moment.astimezone(UTC).isoformat(timespec=precision).replace("+00:00", "Z")


def parse_line(line):
    """Return one usage record as a dict, or None for any other log line.

    `ts` keeps the logged text; `at` is the same instant as an aware datetime.
    A usage line whose timestamp cannot be parsed returns None.
    """
    match = LINE.match(ANSI.sub("", line).rstrip("\n"))
    if not match:
        return None
    try:
        record = {"ts": match["ts"], "at": parse_timestamp(match["ts"])}
    except ValueError:
        return None
    for key, raw in FIELD.findall(match["fields"]):
        try:
            value = json.loads(raw) if raw.startswith('"') else raw
        except ValueError:
            return None
        if key in NUMERIC:
            try:
                value = float(value)
            except ValueError:
                continue
        record[key] = value
    return record if "tool" in record and "principal" in record else None


def read_records(lines):
    """Parse log lines into (records, skipped), where skipped counts usage lines that did not parse."""
    records, skipped = [], 0
    for line in lines:
        record = parse_line(line)
        if record:
            records.append(record)
        elif MARKER in ANSI.sub("", line):
            skipped += 1
    return records, skipped


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


def distribution(values):
    """Sample count plus p50/p95, or min/median/max when the sample is too small for a p95."""
    count = len(values)
    if count < MIN_PERCENTILE_SAMPLES:
        return {
            "n": count,
            "low_sample": True,
            "min": min(values) if values else None,
            "median": statistics.median(values) if values else None,
            "max": max(values) if values else None,
        }
    return {"n": count, "low_sample": False, "p50": percentile(values, 0.5), "p95": percentile(values, 0.95)}


def classify_outcomes(records):
    """Set `outcome_class` on every record: ok, expected_probe, role_denied or error.

    Usage records carry no source IDs, so an expected probe is a timing
    correlation only: each successful upload_source vouches for at most one
    earlier `get_source` `not_found` from the same principal.
    """
    window = dt.timedelta(seconds=PROBE_CORRELATION_SECONDS)
    pending = defaultdict(list)
    for record in sorted(records, key=lambda r: r["at"]):
        outcome = record.get("outcome")
        if outcome == "ok":
            record["outcome_class"] = "ok"
            if record["tool"] == "upload_source":
                probes = pending[record["principal"]]
                while probes and record["at"] - probes[0]["at"] > window:
                    probes.pop(0)
                if probes:
                    probes.pop(0)["outcome_class"] = "expected_probe"
        elif outcome == "forbidden":
            record["outcome_class"] = "role_denied"
        else:
            record["outcome_class"] = "error"
            if outcome == "not_found" and record["tool"] == "get_source":
                pending[record["principal"]].append(record)
    return records


def call_shape(record):
    """Allowlisted request options that distinguish kinds of retrieval call."""
    shape = "/".join(str(record.get(key, "-")) for key in ("mode", "graph", "scope"))
    shape += f" limit={int(record['limit'])}" if "limit" in record else " limit=-"
    if "max_chunks_per_source" in record:
        shape += f" cap={int(record['max_chunks_per_source'])}"
    return shape


def result_stats(ok):
    """Empty-result and hit statistics for successful calls that report hits."""
    hits = [r["hits"] for r in ok if "hits" in r]
    if not hits:
        return {}
    empty = sum(1 for h in hits if h == 0)
    return {"empty_results": empty, "empty_rate": round(empty / len(hits), 3), "hits_median": statistics.median(hits)}


def summarize(records):
    """Per (principal, tool) groups, ordered by workload class."""
    if any("outcome_class" not in record for record in records):
        classify_outcomes(records)
    groups = defaultdict(list)
    for record in records:
        groups[(record["principal"], record["tool"])].append(record)
    summary = []
    for (principal, tool), rows in groups.items():
        ok = [r for r in rows if r["outcome_class"] == "ok"]
        entry = {
            "workload": WORKLOADS.get(tool, "other"),
            "principal": principal,
            "tool": tool,
            "calls": len(rows),
            "ok": len(ok),
            "expected_probe": sum(1 for r in rows if r["outcome_class"] == "expected_probe"),
            "role_denied": sum(1 for r in rows if r["outcome_class"] == "role_denied"),
            "errors": dict(Counter(r.get("outcome", "?") for r in rows if r["outcome_class"] == "error")),
            "days": dict(sorted(Counter(r["at"].strftime("%Y-%m-%d") for r in rows).items())),
            "latency_ms": distribution([r["elapsed_ms"] for r in rows if "elapsed_ms" in r]),
        }
        entry["graph_status"] = dict(Counter(r["graph_status"] for r in rows if "graph_status" in r))
        entry["phase_ms"] = {
            phase: distribution(values)
            for phase in sorted(PHASES)
            if (values := [r[phase] for r in rows if phase in r])
        }
        entry.update(result_stats(ok))
        if tool in RETRIEVAL_TOOLS:
            shapes = defaultdict(list)
            for record in ok:
                shapes[call_shape(record)].append(record)
            entry["shapes"] = []
            for shape, shaped in sorted(shapes.items()):
                scores = [r["top_score"] for r in shaped if "top_score" in r]
                item = {
                    "shape": shape,
                    "calls": len(shaped),
                    "latency_ms": distribution([r["elapsed_ms"] for r in shaped if "elapsed_ms" in r]),
                    # One shape has one mode, so scores here share a scale.
                    "top_score_median": round(statistics.median(scores), 4) if scores else None,
                }
                item.update(result_stats(shaped))
                entry["shapes"].append(item)
        caps = [r["max_chunks_per_source"] for r in ok if "max_chunks_per_source" in r]
        if caps:
            entry["max_chunks_per_source"] = dict(Counter(str(int(cap)) for cap in caps))
        selected = [r["graph_selected"] for r in ok if "graph_selected" in r]
        if selected:
            entry["graph_annotated"] = sum(1 for s in selected if s > 0)
            entry["graph_annotated_of"] = len(selected)
            entry["graph_annotated_rate"] = round(entry["graph_annotated"] / len(selected), 3)
        summary.append(entry)
    summary.sort(key=lambda e: (WORKLOAD_ORDER.index(e["workload"]), e["principal"], e["tool"]))
    return summary


def build_report(records, skipped=0, since=None, until=None, principal=None):
    """Filter to the half-open interval [since, until) and describe what was actually covered."""
    # Classify before filtering, so a probe and its upload on either side of a
    # bound are still correlated.
    classify_outcomes(records)
    available = sorted(r["at"] for r in records)
    included = [
        r for r in records
        if (since is None or r["at"] >= since) and (until is None or r["at"] < until)
        and (principal is None or r["principal"] == principal)
    ]
    times = sorted(r["at"] for r in included)
    window = {
        "requested_since": iso(since),
        "requested_until": iso(until),
        "principal": principal,
        "records": len(included),
        "first_record": iso(times[0]) if times else None,
        "last_record": iso(times[-1]) if times else None,
        "log_first_record": iso(available[0]) if available else None,
        "log_last_record": iso(available[-1]) if available else None,
        "skipped_usage_lines": skipped,
        "starts_before_log": bool(since and available and since < available[0]),
    }
    return {"schema_version": SCHEMA_VERSION, "window": window, "groups": summarize(included)}


def render_distribution(stats):
    if not stats["n"]:
        return "n=0"
    if stats["low_sample"]:
        return (
            f"n={stats['n']} (low sample) min {stats['min']} ms, "
            f"median {stats['median']} ms, max {stats['max']} ms"
        )
    return f"n={stats['n']} p50 {stats['p50']} ms, p95 {stats['p95']} ms"


def render_results(stats):
    return f"empty {stats['empty_results']} ({stats['empty_rate']:.0%}), median hits {stats['hits_median']}"


def render(report):
    window = report["window"]
    lines = [
        "Interval [since, until), UTC: "
        f"{window['requested_since'] or 'start of log'} to {window['requested_until'] or 'end of log'}"
        + (f"; principal {window['principal']}" if window["principal"] else ""),
        f"Included {window['records']} records"
        + (f", first {window['first_record']}, last {window['last_record']}" if window["records"] else ""),
        f"Log holds usage records from {window['log_first_record']} to {window['log_last_record']}; "
        f"skipped {window['skipped_usage_lines']} unparseable usage lines",
    ]
    if window["starts_before_log"]:
        lines.append(
            "NOTE: the requested start precedes the first usage record in these logs; "
            "earlier activity is not visible here and is not zero usage."
        )
    if not report["groups"]:
        lines.append("No graphrag_usage records in this interval.")
        return "\n".join(lines)
    lines.append(
        "A principal identifies a credential, not a workload; call shape is evidence of intent, not proof. "
        "Top scores are comparable only within one mode and are not confidences."
    )
    workload = None
    for e in report["groups"]:
        if e["workload"] != workload:
            workload = e["workload"]
            lines.extend(["", f"[{workload}]"])
        head = f"{e['principal']} · {e['tool']}: {e['calls']} calls, {e['ok']} ok"
        if e["expected_probe"]:
            head += f", {e['expected_probe']} expected pre-upload probes (timing correlation)"
        if e["role_denied"]:
            head += f", {e['role_denied']} role denials"
        if e["errors"]:
            head += f", errors {e['errors']}"
        lines.append(head)
        lines.append(f"  latency {render_distribution(e['latency_ms'])}; by day {e['days']}")
        if "empty_rate" in e:
            lines.append(f"  {render_results(e)}")
        for shape in e.get("shapes", []):
            detail = f"  shape {shape['shape']}: {shape['calls']} ok, latency {render_distribution(shape['latency_ms'])}"
            if "empty_rate" in shape:
                detail += f", {render_results(shape)}"
            if shape["top_score_median"] is not None:
                detail += f", median top score {shape['top_score_median']}"
            lines.append(detail)
        if "max_chunks_per_source" in e:
            lines.append(f"  max_chunks_per_source {e['max_chunks_per_source']}")
        if "graph_annotated_rate" in e:
            lines.append(
                f"  graph-annotated contexts {e['graph_annotated']} of {e['graph_annotated_of']} "
                f"({e['graph_annotated_rate']:.0%}); includes zero-hop annotation of already-selected "
                "chunks and does not show added support"
            )
    return "\n".join(lines)


def main(argv=None, now=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+")
    parser.add_argument(
        "--since", type=parse_bound, help="inclusive start: YYYY-MM-DD (00:00 UTC) or RFC 3339 timestamp"
    )
    parser.add_argument(
        "--until", type=parse_bound, help="exclusive end: YYYY-MM-DD (00:00 UTC) or RFC 3339 timestamp"
    )
    parser.add_argument(
        "--last", type=parse_duration, metavar="N{d,h}",
        help="rolling interval ending at --until, or now (for example 7d)",
    )
    parser.add_argument("--principal", help="only this principal")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)
    since, until = args.since, args.until
    if args.last:
        if since:
            parser.error("--last cannot be combined with --since")
        until = until or now or dt.datetime.now(UTC)
        since = until - args.last
    if since and until and since >= until:
        parser.error("--since must be earlier than --until")
    records, skipped = [], 0
    for path in args.logs:
        with open(path, errors="replace") as handle:
            parsed, bad = read_records(handle)
        records.extend(parsed)
        skipped += bad
    report = build_report(records, skipped, since, until, args.principal)
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
