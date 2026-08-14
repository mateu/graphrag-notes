# Retrieval fusion fixture comparison

This report is a committed, reproducible fusion comparison target for issue
#14. `retrieval-fusion-input.json` defines one deterministic candidate set and
its graded relevance. The two separate output files below are independently
replayed by `fusion::tests::committed_weighted_and_rrf_outputs_are_reproducible_from_the_same_input`.
That test recomputes rankings and all displayed relevance metrics, so a ranking
regression cannot leave this report's fixtures passing unchanged.

| Metric (k=3) | Weighted output | RRF output | Delta |
| --- | ---: | ---: | ---: |
| Recall@k | 1.000 | 1.000 | 0.000 |
| MRR | 0.500 | 1.000 | +0.500 |
| nDCG@k | 0.541 | 1.000 | +0.459 |
| End-to-end retrieval latency | not part of this deterministic unit fixture | not part of this deterministic unit fixture | N/A |

The committed fixture deliberately isolates rank fusion; it is not a claim of
end-to-end corpus relevance or latency. The #6 CLI fixture records latency
when replayed against a running database and inference service:

```bash
graphrag --config config.toml eval-augment tests/fixtures/eval/cases-v2.jsonl \
  --format json > new-rrf-report.json
```

Run the command once with `fusion_strategy = "weighted"` to create a legacy
baseline, then with `fusion_strategy = "rrf"` and compare using
`--baseline legacy-report.json`. Production release notes must add the measured
latency delta from that replay before asserting an end-to-end improvement.

Tie-break contract: descending fused score, then the best component rank, hit
type (`note`, `message`, `conversation-summary`), and canonical record ID.
