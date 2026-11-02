#!/usr/bin/env python3
"""Compare exact serialized matcher results (including metadata, order and cap).

Usage: issue166-compare.py baseline.jsonl optimized.jsonl
The Rust issue166_benchmark example emits one JSON record per query. Timings
are deliberately not compared; any result/order/metadata change is a failure.
"""
import json
import sys


def load(path):
    with open(path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.startswith("{")]
    by_query = {row["query"]: row for row in rows}
    if not rows or len(rows) != len(by_query):
        raise ValueError("empty or duplicate-query benchmark")
    return by_query


def main():
    before, after = map(load, sys.argv[1:])
    assert before.keys() == after.keys(), "query inventory changed"
    for query, row in before.items():
        assert row["results"] == after[query]["results"], query
        print(
            f'{query}: identical {len(row["results"])} results; '
            f'p50={after[query]["p50_ms"]:.3f} ms '
            f'p95={after[query]["p95_ms"]:.3f} ms'
        )
    failed = [q for q, r in after.items() if r["p95_ms"] >= 100]
    if failed:
        print("LATENCY TARGET NOT MET: " + ", ".join(failed), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
