# Retrieval fusion fixture comparison

This report is a committed, reproducible comparison target for issue #14. It
uses the sanitized #6 fixture inputs in `cases-v2.jsonl` and the corresponding
committed ranked output in `ranked-results.json`; neither file contains user
data.

| Metric (micro, k from each case) | Legacy weighted | RRF default | Delta |
| --- | ---: | ---: | ---: |
| Recall@k | 1.000 | 1.000 | 0.000 |
| MRR | 1.000 | 1.000 | 0.000 |
| nDCG@k | 1.000 | 1.000 | 0.000 |
| Retrieval latency | not recorded by fixture | not recorded by fixture | N/A |

The committed fixture captures only pre-ranked output, not a running SurrealDB
instance or inference service, so it cannot honestly measure end-to-end
latency. The CLI records latency when replaying the same cases:

```bash
graphrag --config config.toml eval-augment tests/fixtures/eval/cases-v2.jsonl \
  --format json > new-rrf-report.json
```

Run the command once with `fusion_strategy = "weighted"` to create a legacy
baseline, then with `fusion_strategy = "rrf"` and compare using
`--baseline legacy-report.json`. The zero fixture delta is intentional: this
change establishes deterministic, scale-independent fusion without claiming a
relevance gain on a small committed fixture. Production release notes must add
the measured latency delta from that replay before asserting an improvement.

Tie-break contract: descending fused score, then the best component rank, hit
type (`note`, `message`, `conversation-summary`), and canonical record ID.
