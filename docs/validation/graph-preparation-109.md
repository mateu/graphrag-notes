# Graph preparation operating evidence (#109)

The [sanitized measurement data](graph-preparation-109.json) contains every policy/load/category denominator, failures, first observations, warm distributions, matched backend phases, paired graph deltas and frozen relevance aggregates. Private queries, IDs, corpus/configuration fingerprints, returned content and raw traces remain outside the repository.

## Change and correctness

Graph-enabled hybrid note search retains the bounded lexical prefix already retrieved for fusion and uses it for seed preparation. The former second full-text query is removed. Fusion still receives its original candidate limit; graph preparation keeps its separate lexical supplement beyond the requested top-k and its hard 200 candidate/mention bounds. Graph-off SQL and ranking stay unchanged.

Provider-free fixtures compare complete old/new ranked records, scores, evidence and graph summaries for 32/256 notes and sparse/dense mentions, including source/time/current-generation visibility, aliases, ties, exact titles and seeds outside top-k. All 201 agent and 181 database unit tests passed, as did 19 benchmark-tool tests. Existing #93 regression fixtures are included. The opt-in mention-plan fixture uses 260 fictional mentions; pinned SurrealDB 3.2.4 already selected its compound mentions(out,in) index for the tested bound out predicate, and an extra single-out index changed no plan. No index/schema change was adopted. This observation does not establish every correlated alias query plan.

On the unchanged private #92 suite, all 200 policy cases passed and exactly matched the baseline in ordered records, pinned readbacks and metrics. The 50 cases retain 34 calibration/16 holdout cases (32/14 answerable and 2/2 no-answer), all formulation counts, unjudged hits and metric denominators. Keyword no-answer results remain empty; hybrid/graph still return false positives on those four questions. Known-positive RR/recall remain lower bounds with unjudged hits; no full-corpus recall claim or silently updated ranking baseline is made.

## Controlled release runs

Apple Silicon macOS had 18 logical CPUs and 48 GiB RAM. Other issue builds and inference experiments were paused. Each role restored the same private typed-pilot archive: 9,751 notes, 482 entities, 582 mentions and zero accepted real inter-note edges. Existing bge-m3:latest (1,024 dimensions), phi4-mini:latest and processing concurrency 2 were preserved on an isolated loopback provider. The benchmark process owned the provider lifecycle and checked its health. An earlier attempted after-run had an exited provider: its 50 keyword cases passed and 150 semantic cases failed immediately; it produced no performance samples and is excluded from latency comparisons. The failed attempt remains private.

Before: immutable published rc.3 binary SHA-256 `e17c039740355261d974b933a7052bcf3131dd735664db8b8c13b3c364f1a4e7`, compiled inputs `885844fc6168d896fb119e079c81021791e09c25`. Rust sources, Cargo manifests/lock and build script are byte-identical to main `51daf4cc32a130cdc953208ff2bb67e7e6d21674`. After: immutable binary SHA-256 `a9784daccf04b4d8b5f8207887afe00cb7eb0e0ffffc32b1af31392f247636e0`, compiled inputs `2a1627a06675f310ef80ca2b4e3b27f56b5b34d0`. Documentation commits after that checkpoint do not change compiled runtime inputs. Both builds use locked Rust 1.97.1/static OpenSSL/macOS 15 release settings. The original baseline report labelled equivalent main as its compiled revision; a private lineage correction preserves the original report and records the actual compilation source.

Each of four serialized roles passed 672 search/readback samples (32 first observations plus 640 warm samples), for 2,688 performance samples and zero failures. Each category/policy/load has 20 successful warm repetitions. One/four lanes use homogeneous policy batches, with guarded readbacks between searches. First observations are retained separately and imply neither a cold model nor a cold OS cache. Tracing is separated from ordinary logging. All 672 before/after sample pairs in each comparison exactly matched ordered records, readbacks and metrics.

### Ordinary logging: warm RPC p50 / p95 ms

| Policy | One lane before → after | Four lanes before → after | Target p95 |
| --- | ---: | ---: | ---: |
| keyword-off | 2.5 / 234.3 → 2.9 / 238.2 | 3.1 / 304.3 → 3.1 / 291.3 | 250 ms |
| hybrid-off | 451.5 / 683.6 → 447.9 / 678.3 | 533.4 / 809.8 → 520.7 / 790.6 | 1,000 ms |
| hybrid-auto | 610.1 / 988.6 → 603.6 / 762.8 | 795.9 / 1271.9 → 743.4 / 932.4 | 1,000 ms |
| hybrid-on | 613.6 / 993.0 → 603.4 / 759.6 | 771.7 / 1263.0 → 742.4 / 907.3 | 1,000 ms |

### Same-RPC traced preparation and graph overhead

Phase observations match their actual RPC identity. Nested phases are not additive. The paired graph delta matches the same query/round across sequential policy batches; it is not a subtraction of independent percentiles. Each delta group has 80 successful pairs and 60 known-positive RR comparisons with zero regressions.

| Graph policy/load | Query preparation p95 before → after | RPC p95 before → after | Paired graph delta p95 before → after |
| --- | ---: | ---: | ---: |
| hybrid-auto, 1 lane(s) | 240.85 → 0.19 ms | 1015.5 → 798.8 ms | 341.7 → 183.1 ms |
| hybrid-on, 1 lane(s) | 234.54 → 0.18 ms | 1016.5 → 800.5 ms | 345.1 → 180.9 ms |
| hybrid-auto, 4 lane(s) | 283.74 → 0.25 ms | 1215.7 → 960.8 ms | 443.0 → 259.6 ms |
| hybrid-on, 4 lane(s) | 280.19 → 0.23 ms | 1188.4 → 948.6 ms | 396.9 → 236.0 ms |

## Remaining operating limits

The original targets remain 250 ms keyword RPC, 1,000 ms hybrid/graph RPC and 250 ms paired graph overhead. All per-category distributions and misses are retained in the JSON. Broad full-text scoring still dominates the keyword tail; exact vector scans and graph entity matching remain material costs. Traced four-lane graph auto has 259.6 ms paired overhead and misses its target; this is retained rather than rounded into a pass. The equivalent graph-on traced pair is 236.0 ms. With ordinary logging, four-lane auto overhead is 329.2 ms and also misses; graph-on is 240.5 ms. The ordinary-logging broad keyword p95 is 244.1 / 294.0 ms at one/four lanes, so the four-lane category still misses its 250 ms target. All after-run hybrid/graph category p95 values are below 1,000 ms for this workload. Traced and ordinary-logging budgets are separate observations, not a causal estimate of tracing overhead.

This sparse real graph and eight-note typed pilot do not establish latency on a broadly extracted corpus. Dense fictional fixtures establish bounded correctness only. Paired relevance/performance acceptance must be repeated after integrating #106/#107. Normal OpenClaw handler/connection/dispatch measurements are separate in #108.
