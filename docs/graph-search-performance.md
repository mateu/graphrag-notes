# Graph search performance

Graph retrieval fetches the note documents referenced by entity mentions and
accepted edges, then hydrates its bounded candidate IDs directly. It retains
per-entity seed pages and per-frontier/per-table edge limits. Source, age,
current-generation visibility, direction, confidence, and visited-path filters
apply before those limits. Deleted endpoints and record ranges are excluded;
compound record keys are checked against stored documents.

Graph remains explicit in the direct OpenClaw command: `/notes --graph QUERY`
selects hybrid retrieval with graph `auto`. The response identifies the selected
policy, and an unavailable or timed-out request reports its failure. Ordinary
keyword and hybrid commands keep graph off.

## Interactive budget and build profile

The initial operating budget is warmed keyword responses within 250 ms,
graph expansion adding at most 250 ms to the same hybrid query, and warmed
hybrid/graph responses within one second on the shared-service setup. These are
targets for the measured workload, not enforced deadlines or guarantees for
arbitrary corpus sizes. First model loading, client startup, contention, and
remote transport must be measured separately.

Use an optimized release binary for a long-running service. A source build is:

```sh
cargo build --release --locked -p graphrag-cli
./target/release/graphrag --help
```

Debug builds are useful for development but should not establish production
latency expectations. Hybrid retrieval still uses the existing exact cosine
search and embedding identity. Explicit-distance KNN is a brute-force search
in [SurrealDB's vector documentation](https://surrealdb.com/docs/learn/data-models/vector-search/vector-indexes).
An approximate-index diagnostic changed ordered candidates on the restored
corpus, so this change keeps the existing vector query.

## Reproduce scaling and inspect phases

The provider-free fixture uses fictional stable records, distinct deterministic
vectors, and graph off/auto/on. Its six cases cover no/sparse/dense mentions,
empty/nonempty accepted edge tables, narrow/broad entity queries, and source/age
filters. It checks result uniqueness, visibility, candidate bounds, accepted-only
paths, and stable ranked evidence. Timings are reported without flaky pass/fail
thresholds.

```sh
GRAPHRAG_GRAPH_LATENCY_ROUNDS=10 \
GRAPHRAG_GRAPH_LATENCY_REPORT=/tmp/graph-latency-new.json \
  cargo test -p graphrag-agents --test graph_latency_fixture --locked -- --nocapture
```

The report path must be new. Default populations are 256 and 2,048 notes;
`GRAPHRAG_GRAPH_LATENCY_LARGE=1` adds 10,000. An explicit comma-separated
`GRAPHRAG_GRAPH_LATENCY_SCALES=64,256` selects smaller diagnostic populations.
Reports separate first-per-policy calls from warmed samples. Small-sample p95
values are emitted only with at least ten observations. Independent repository
probes must not be added and presented as an end-to-end trace.

Debug tracing adds elapsed milliseconds and counts for query embedding, note
vector/full-text work, entity matching, mentions, edge expansion, note hydration,
and provenance. Record `EXPLAIN`/`EXPLAIN FULL` against an isolated restored
corpus before optimizing queries. Operator output counts are not automatically
equivalent to storage records read. Keep query text, note bodies, IDs, source
URIs, and credentials out of published profiles. Never open the live database
while its service owns it.

## Observed query improvement

Three warmed in-process observations on a restored 9,751-note corpus with
365 entities and 458 mentions used the same debug profile and query arguments.
The medians below exclude provider generation, database opening, MCP/SSH,
and OpenClaw dispatch. They are phase probes, not latency percentiles.

| Phase | Before | After |
| --- | ---: | ---: |
| Entity mention lookup | 5,304 ms | 6.26 ms |
| Graph note hydration | 2,308 ms | 0.89 ms |
| Empty accepted-edge lookup | 5.56 ms | 3.25 ms |

Actual plans showed full-note scans in the old mention membership subquery and
hydration. The replacement checks referenced documents and selects bound IDs.
A shared full-corpus `LET` snapshot remained expensive in the diagnostic; it
was not selected. Keyword plans used the existing full-text indexes, while the
unchanged exact vector phase remained about three seconds in the debug build.

The before/after fictional fixture compared 12 cases at 64/256 notes, each with
three graph policies. All 36 full ranked result/evidence snapshots were identical.
One warmed observation of the dense, source/age-filtered 256-note case with
accepted edges fell from 21,823 ms to 261 ms; its hybrid/off call stayed near
83/80 ms. These are individual observations, not percentiles.

This page describes the source changes for #84. Older published assets are not
replaced by source changes; deployment must use a verified build containing them.
