#!/usr/bin/env python3
"""Summarize graphrag_usage records from a service log.

The service writes one `graphrag_usage` INFO line per MCP tool call (see
crates/service/src/usage.rs). This report groups them by principal and tool
to answer: who calls what, how often calls fail, how often searches come back
empty, how strong the top hit is, how fast calls are, and whether the graph
contributes anything to build_context.

Usage:
  scripts/usage_report.py service.stderr.log [more.log ...] [--since 2026-11-01]
                          [--principal NAME] [--json]
"""

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(r"^(?P<ts>\S+)\s+INFO\s+graphrag_usage: tool call (?P<fields>.*)$")
FIELD = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\S+)')
NUMERIC = {
    "elapsed_ms", "limit", "query_chars", "query_terms", "hits", "top_score",
    "graph_considered", "graph_selected", "dropped_for_relevance", "total_tokens",
}


def parse_line(line):
    """Return one usage record as a dict, or None for any other log line."""
    match = LINE.match(ANSI.sub("", line).rstrip("\n"))
    if not match:
        return None
    record = {"ts": match["ts"]}
    for key, raw in FIELD.findall(match["fields"]):
        value = json.loads(raw) if raw.startswith('"') else raw
        if key in NUMERIC:
            try:
                value = float(value)
            except ValueError:
                continue
        record[key] = value
    return record if "tool" in record and "principal" in record else None


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["principal"], record["tool"])].append(record)
    summary = []
    for (principal, tool), rows in sorted(groups.items()):
        ok = [r for r in rows if r.get("outcome") == "ok"]
        latency = [r["elapsed_ms"] for r in rows if "elapsed_ms" in r]
        entry = {
            "principal": principal,
            "tool": tool,
            "calls": len(rows),
            "ok": len(ok),
            "errors": dict(Counter(r.get("outcome", "?") for r in rows if r.get("outcome") != "ok")),
            "days": dict(sorted(Counter(r["ts"][:10] for r in rows).items())),
            "latency_ms_p50": percentile(latency, 0.5),
            "latency_ms_p95": percentile(latency, 0.95),
        }
        hits = [r["hits"] for r in ok if "hits" in r]
        if hits:
            scores = [r["top_score"] for r in ok if "top_score" in r]
            entry.update({
                "empty_results": sum(1 for h in hits if h == 0),
                "empty_rate": round(sum(1 for h in hits if h == 0) / len(hits), 3),
                "hits_median": statistics.median(hits),
                "top_score_median": round(statistics.median(scores), 4) if scores else None,
                "options": dict(Counter(
                    "/".join(str(r.get(k, "-")) for k in ("mode", "graph", "scope")) for r in ok
                )),
            })
        selected = [r["graph_selected"] for r in ok if "graph_selected" in r]
        if selected:
            entry["graph_contributed"] = sum(1 for s in selected if s > 0)
            entry["graph_contributed_rate"] = round(entry["graph_contributed"] / len(selected), 3)
        summary.append(entry)
    return summary


def render(summary):
    if not summary:
        return "No graphrag_usage records found."
    lines = []
    for e in summary:
        head = f"{e['principal']} · {e['tool']}: {e['calls']} calls, {e['ok']} ok"
        if e["errors"]:
            head += f", errors {e['errors']}"
        lines.append(head)
        lines.append(f"  latency p50 {e['latency_ms_p50']} ms, p95 {e['latency_ms_p95']} ms; by day {e['days']}")
        if "empty_rate" in e:
            lines.append(
                f"  empty {e['empty_results']} ({e['empty_rate']:.0%}), median hits {e['hits_median']}, "
                f"median top score {e['top_score_median']}; mode/graph/scope {e['options']}"
            )
        if "graph_contributed_rate" in e:
            lines.append(f"  graph contributed to {e['graph_contributed']} ({e['graph_contributed_rate']:.0%}) contexts")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--since", help="only records on or after this ISO date")
    parser.add_argument("--principal", help="only this principal")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)
    records = []
    for path in args.logs:
        with open(path, errors="replace") as handle:
            records.extend(r for r in map(parse_line, handle) if r)
    if args.since:
        records = [r for r in records if r["ts"][:10] >= args.since]
    if args.principal:
        records = [r for r in records if r["principal"] == args.principal]
    summary = summarize(records)
    print(json.dumps(summary, indent=2) if args.json else render(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
