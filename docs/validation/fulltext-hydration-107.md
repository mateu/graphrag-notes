# Note full-text scoring and hydration evidence (#107)

The [sanitized data](fulltext-hydration-107.json) retains all policy/load/category counts, failures, first observations, warm distributions, actual matched backend phases, paired graph deltas and unchanged relevance aggregates. Private questions, records, configuration/corpus fingerprints, raw traces and provenance remain local.

## Change and correctness

Schema 21 materializes a derived `note_search` table containing lexical fields, canonical note identity and source/time visibility fields. Its keys mirror the native primary IDs, including their typed ordering. The same title/content analyzers, BM25 weights and exact-title precedence apply to the same indexed population, including unembedded notes and old source generations. One SQL statement selects and sorts the filtered lexical candidates before the original limit, then hydrates only those canonical primary records in the same statement snapshot. Source/current-generation predicates, score conversion, deterministic ties and graph lexical candidate limits remain unchanged.

Synchronous create/update/delete events maintain the derived rows. Migration backfills existing notes; portable export excludes the derived table and canonical primary restores rebuild it through those events. The final migration visits exclusive native-key ranges with at most 128 IDs per page, committing each materialization page before recording migration completion. Creates and changes to the six mirrored fields update the derived row; embedding, tags and reindex bookkeeping alone do not rewrite its indexes. This bounds the runner’s ID batch; it does not impose a byte limit on individual note content or claim a bound on all engine/index memory. The original primary full-text indexes remain. Message and conversation full-text paths are unchanged.

Provider-free fixtures compare full ordered payloads and native f32 score bits against the legacy SQL for limits 0/1/5/50/200, broad/narrow/empty and sparse/dense matches, exact titles, numeric/string/UUID/object ID ties, source/time/current-generation filtering, edits/deletes/source promotion/URI renames, failed transaction rollback, schema-20 empty/existing database upgrades and actual typed-ID portable restore. The original measured source passed all 187 database tests (three opt-in diagnostics excluded); all-target database Clippy passed with warnings denied. The comparison harness passed two complete fictional cases and rejected 11 incomplete/changed cases.

Earlier primary-table projection rewrites preserved results but regressed broad screening timings and were rejected. An optimized fictional thin-table screen compared 378 calls on each of memory/256-note and fresh RocksDB/1,024-note databases with exact ordered payload/score-bit agreement. Its direct thin candidate improved broad limit-5 RocksDB p95 from 121.33 to 29.10 ms; an alternative one-materialization query had a 74.04 ms tail and was rejected. These fictional one-lane screens justified the candidate; the real-corpus comparison below establishes its operating evidence.

## Controlled release comparison

Apple Silicon macOS had 18 logical CPUs and 48 GiB RAM. Other issue builds and inference experiments were paused. Each role restored the identical immutable typed-pilot snapshot: 9,751 notes, 482 entities, 582 mentions, eight typed pilot notes and no accepted real inter-note edges. Existing bge-m3:latest (1,024 dimensions), phi4-mini:latest and processing concurrency 2 were preserved. The benchmark owned a separate loopback provider, verified its model digests/details before and after every role, and stopped all owned services and provider on completion. The manual pre-service doctor check performs one embedding-dimension compatibility probe per role. Traced roles also run relevance before latency; first observations establish neither a cold model nor a cold filesystem cache.

Before: binary SHA-256 `a9784daccf04b4d8b5f8207887afe00cb7eb0e0ffffc32b1af31392f247636e0`, compiled inputs `2a1627a06675f310ef80ca2b4e3b27f56b5b34d0`, schema 19. After: binary SHA-256 `af95cb003b8e7a16d1981433397b244265e811f89d2fe3669d74bcf10b97f3a8`, compiled inputs `5b7999e6fe80c01a846b029a5be4925d6d3eb7b3`, schema 21. The after runtime includes schema-20 extraction-policy support; no migration admissions or extraction generation occurs in these search runs. Both are immutable locked Rust 1.97.1/static OpenSSL/macOS-15 optimized builds.

The final PR integrates the merged graph preparation, maintained OpenClaw client and extraction-policy migration. Relative to the measured after source, the Rust tree additionally carries the rejected vector experiment’s test module and development dependency; the final review also replaces the linear migration ID list with native-key batches and skips unrelated lexical update events. The production retrieval query and policy-migration implementation remain unchanged. Provider-free regression fixtures cover multiple backfill pages, typed-key boundaries and the actual number of derived writes for each mirrored and unmirrored update. These measurements retain their original source/binary identities. The final integrated database source passes 193 tests with zero failures (eight opt-in diagnostics excluded), including the new write-count fixture and multi-page migration/restore. All-target database Clippy passes with warnings denied. The integrated release requires its own native build, installation and deployment evidence.

Four serialized roles each passed 672 searches with revision-pinned readbacks (32 first observations and 640 warm samples): 2,688 samples, zero failures. Each category/policy/load has 20 warm observations; one/four persistent lanes run homogeneous policy batches with guarded readbacks between searches. All 672 traced and 672 ordinary before/after pairs exactly match full ordered records, pinned readbacks and metrics. Every one of the unchanged 200 relevance cases also matches exactly. The frozen 50-question suite retains 34 calibration/16 holdout cases, 32/14 answerable and 2/2 reviewed no-answer cases, all formulation/judgment counts, unjudged hits and denominators. Known-positive recall/RR are lower bounds; precision/nDCG remain withheld when unjudged hits prevent a supported claim. Hybrid/graph no-answer false positives remain visible.

### Ordinary logging: warmed RPC p50 / p95 ms

| Policy | One lane before → after | Four lanes before → after | Target p95 |
| --- | ---: | ---: | ---: |
| keyword-off | 2.96 / 231.39 → 2.27 / 58.23 | 3.02 / 287.92 → 3.55 / 101.62 | 250 ms |
| hybrid-off | 445.00 / 676.86 → 438.56 / 501.52 | 521.55 / 789.55 → 513.27 / 588.78 | 1,000 ms |
| hybrid-auto | 606.71 / 759.13 → 586.77 / 609.90 | 736.08 / 922.48 → 717.91 / 759.15 | 1,000 ms |
| hybrid-on | 604.59 / 770.90 → 587.54 / 606.64 | 722.31 / 913.79 → 720.87 / 768.90 | 1,000 ms |

Broad/direct keyword p50/p95 improves from 227.63/235.44 to 57.50/59.55 ms at one lane and 276.63/293.10 to 99.41/108.34 ms at four lanes. The JSON preserves the exact measured medians and all distributions. All after-run keyword/hybrid/graph category p95 values meet their RPC targets on this snapshot.

Keyword traces emit `application_search` only. Full-text phase attribution comes from the actual `note_fulltext` phase inside the broad/direct hybrid-off RPC, grouped separately at each load. Phase observations can nest and must not be added or inferred by subtracting independent percentiles.

| Actual broad hybrid-off full-text phase | Before p50 / p95 | After p50 / p95 | Warm observations per role |
| --- | ---: | ---: | ---: |
| One lane | 226.76 / 235.86 ms | 74.58 / 76.45 ms | 20 |
| Four lanes | 266.04 / 281.84 ms | 91.60 / 95.01 ms | 20 |

## Mixed selective tails and remaining limits

Ordinary one-lane keyword selective p95 remains 1.57–2.75 ms after the change. Four-lane entity/title tails rose from 2.84/4.82 to 8.83/10.70 ms, while the matched traced counterparts changed from 3.46/5.11 to 3.43/4.51 ms. These small absolute but mixed tails are retained.

A separate keyword-only confirmation restored the same snapshot/configuration for four release roles in before/after/after/before order. It preserved all four original cases, their order, query meaning and judgments, with the same one/four-lane schedule and 20 warm repetitions plus first observations per category/load/role. All 672 searches succeeded and exactly matched the original cohorts' full ranked records, pinned readbacks and metrics; all 336 balanced before/after pairs also matched. Every first, minimum, median, p95 and maximum remains in the supplemental JSON section. Direct `schema-version` checks verified the actual migration ledger before each sole service owner; no doctor, embedding/extraction calls or provider processes ran. All owned processes stopped. Unused narrower preparation plans produced no actual samples and remain private.

| Four-lane keyword category | First pair p95 before → after | Reversed pair p95 before → after |
| --- | ---: | ---: |
| Broad/direct | 326.60 → 115.13 ms | 324.33 → 115.91 ms |
| Entity | 3.65 → 3.00 ms | 3.91 → 3.25 ms |
| Source/time filters | 3.28 → 2.69 ms | 3.51 → 2.71 ms |
| Exact title | 5.29 → 4.78 ms | 5.08 → 4.64 ms |

The confirmation did not reproduce the original larger four-lane selective rises. Not every selective percentile improves: one-lane filter p95 in its first pair increases from 1.66 to 1.79 ms, and maxima near 11 ms for filters and 109 ms for a candidate broad query remain visible. This evidence supports broad-query gains with selective queries still taking a few milliseconds; it does not establish statistical significance or erase the adverse original observations. Keyword-only traffic omits the other policies' first-observation work from the main cohort, and neither run establishes cold-cache latency or continuous four-client saturation.

Paired graph overhead uses the same query/round across sequential policy batches, with 80 successful pairs per group and 60 known-positive RR comparisons without regressions. Ordinary after-run p95 is 176.20/180.96 ms at one lane and 244.76/248.20 ms at four lanes for auto/on. Traced four-lane auto is 221.00 ms; traced four-lane on is 251.99 ms and misses the unchanged 250 ms overhead target. Separate cohorts are not a causal measurement of tracing overhead.

Exact vector scanning and graph entity matching still contribute substantial work. This sparse graph does not establish speed for a widely extracted corpus or a dense real relationship graph. The extraction-policy pilot needs separate paired quality evidence after promotion, and OpenClaw connection/handler/dispatch measurements remain in #108.
