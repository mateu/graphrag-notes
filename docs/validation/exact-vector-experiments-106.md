# Exact vector execution experiments (#106)

These provider-free experiments investigate [#106](https://github.com/mateu/graphrag-notes/issues/106). They do not change production queries, schemas, model settings, ranking, concurrency or client defaults. The latency acceptance criteria remain open: none of the completed alternatives improves both the median and the retained tail across the tested workloads. A faster median alone does not justify a new derived table or execution path.

[#115](https://github.com/mateu/graphrag-notes/issues/115) tracks the specific order-sensitive scan/SDK tail and the measured alternative requirements. #106 stays open until an exact-path optimization satisfies its acceptance criteria.

The [sanitized final page-matrix data](exact-vector-experiments-106.json) retains all 1,008 measured calls, first observations, 20 warm observations per group, unfavorable maxima, same-call phases and actual bounds. It contains fictional population counts and timing data, without queries, record IDs, content, configuration or credentials. Earlier raw fictional experiments remain in the isolated evidence archive; their rejected designs and operating limits are summarized below.

## Existing execution and measured alternatives

The locked engine is SurrealDB 3.2.4. Actual `EXPLAIN FULL` plans with the existing HNSW index retained show a table scan and exact KNN top-k for the explicit `COSINE` operator. The scan reads complete stored records; narrowing the returned fields does not eliminate decoding the primary embedding arrays and note bodies. Final primary-record hydration is already bounded by the candidate limit.

The pinned engine's `src/exec/operators/scan/pipeline.rs` processes stored records before selective output projection. Its computed-field path can evaluate a computed embedding even when that field is omitted from the output, when projection dependencies cannot be resolved statically. `src/fnc/encoding.rs` decodes CBOR into a public value and then converts it to the internal value. These source observations explain which work remains in the tested paths; operator durations are nested and are not added together to infer a causal total.

| Alternative | Observation | Decision |
| --- | --- | --- |
| Narrow ID projection with bounded primary hydration | The exact scan still decodes full primary records; repeated linked-field retrieval does not reduce measured cost. | Retain the existing query. |
| Single linked-record hydration, explicit cosine expression, literal query vector, prepared source lookup | No consistent improvement over the existing exact path. | Retain the existing query. |
| Thin table retaining a native Float embedding array | Removing large bodies helps some cases but leaves vector-array decoding and execution cost. | No production adoption. |
| CBOR bytes with a bounded paged Rust heap | Much faster warmed medians in several dense fixtures, but larger retained tails in unfiltered top-50 groups. | No production adoption. |
| Single SQL KNN over a typed computed CBOR embedding | Slower than the existing query; defining the computed field also contaminated a raw-page comparator on the same table. | Reject the rewrite; retain the contamination qualification. |
| Single SQL KNN over a separate computed `TYPE any` table | Eliminates the shared-table contamination and some coercion, but still loses in every tested RocksDB p95 group. | Reject the rewrite. |

## Fictional fixtures and exactness

The experiments use dense, deterministic, finite 1,024-dimensional vectors originating as float32 values, with variable note bodies, source/time filters and current/stale source generations. Limits 5 and 50 cover unfiltered, recent and source-restricted groups. Native numeric, string, UUID and object record keys and 72 equal-distance records exercise the top-k tie boundary in the current page matrix. Separate edge-case fixtures include absent and empty embeddings, dimension mismatches, zero and signed zero, negative coordinates, float64 precision and historical nonfinite values.

Each measured search compares the complete ordered result payload and exact returned float32 distance bits with the existing engine query. Finite CBOR arrays retain float64 values without quantization. CBOR canonicalizes some NaN representations, so the experimental derived row marks nonfinite embeddings for canonical primary-record read-through. Native-engine comparisons cover positive/negative NaN payloads and infinities rather than assuming that Rust's total ordering matches the engine's KNN behavior.

The paged prototype uses one actual SDK transaction for source visibility, each vector page and final bounded hydration. Its engine-ordered scan sequence resolves distance ties. Provider-free fault fixtures on memory and fresh RocksDB databases verify acknowledged rollback on normal exit, panic and task abort, and coherent generation/payload results during concurrent source promotion. Three-scope projection fixtures cover notes, messages and conversation summaries, synchronous create/update/delete events, injected derived-write failures, bulk edits, primary-only portable restore and derived-row reconstruction. These are experiments, not an adopted migration or production execution helper.

## Retained release observations

Measurements ran on Apple Silicon macOS with 18 logical CPUs and 48 GiB RAM. Heavy builds, inference and other measurements were paused during controlled timing windows. Each group contains one first measured observation and 20 successful warm observations. The first measured observation follows an unmeasured legacy reference query; it is not a cold database, OS-cache or model observation. `EXPLAIN FULL` also executes the query, so plan calls are separate unmeasured work. All first observations, warm samples and maxima are retained, including unfavorable tails.

All timing values below are milliseconds. Warm p50 uses the median; warm p95 uses nearest rank over the 20 successful samples. These are fictional storage experiments, not real-corpus MCP or one/four-lane operating-budget acceptance.

### Paged bytes: fast medians with a retained tail

Compiled source `80fb61a8165ae4913d5438b86b372f4a3094d041`, immutable executable SHA-256 `f13260593c1f6834bfcc82690e2917f6075213f17129380eb8370a6abccb4b66`. Fresh RocksDB populations of 160 and 1,024 notes plus a memory population of 1,024 notes completed 1,890 measured searches with exact ordered results and distance bits.

| RocksDB, 1,024 notes | Existing warm p50 / p95 | Paged CBOR warm p50 / p95 |
| --- | ---: | ---: |
| Unfiltered, limit 5 | — / 43.93 | — / 12.79 |
| Unfiltered, limit 50 | 41.91 / 44.59 | 15.19 / 99.03 |
| Recent, limit 50 | — / 41.89 | — / 13.50 |
| Source, limit 50 | — / 37.38 | — / 7.25 |

The unfiltered top-50 outliers coincide with larger same-call SQL scan/DTO, transaction begin/cancel and hydration durations. CBOR decode and cosine inside the heap phase remain roughly 5.3–5.6 ms in those calls. This locates the observed extra time within that execution but does not establish an allocator, OS or engine scheduling cause. The same pattern appears on memory storage, so it is not exclusively a RocksDB effect.

### Order controls

Compiled source `79cd72be814f49bd9fc8f5314e3168cf58c8102d`, executable SHA-256 `2469163a00932cfc7b2ae7535740a86b3385f01104159b3ae7048e03dc1764d7`. A fresh CBOR-first role and a fresh alternating-order role each completed 378 exact searches. Another 256 begin/cancel or begin/`RETURN NONE`/cancel sham cycles tested lifecycle work without retrieval.

Changing execution order moves the tail: CBOR-first unfiltered top-50 warm p95 is 16.28 ms, while its recent top-5 group reaches 84.70 ms. Alternating variants retains unfiltered top-50 maxima/tails in the existing query, thin Float table and paged CBOR path; their p95 values are 70.39, 66.70 and 96.95 ms. Sham p95 is 0.034/0.055 ms. The effect is shared and order-sensitive, and paged retrieval can amplify it. The underlying cause remains unproven; no tail is excluded from the adoption decision.

### Typed single SQL bytes: rejected

Compiled source `05726f5ed3ae89449986b174beb5633d5a4d1b77`, executable SHA-256 `dba3f053bff6e55c87750223c6e5d9c424a47fe90186e5215df980f927f9f9c5`. Memory 256 and fresh RocksDB 1,024 populations completed 1,008 exact searches across four alternating variants. The typed computed-array path loses to the existing query. For RocksDB unfiltered top-50, existing warm p50/p95 is 42.05/44.11 ms versus 60.77/62.50 ms. The raw paged comparator in this particular experiment shares a table with the computed field and is unsuitable for an uncontaminated comparison with earlier raw-page measurements.

### Separate `TYPE any` single SQL bytes: rejected

Compiled source `5486696992b43cac834199bbd18ba511d7bb78c6`, executable SHA-256 `46b193bede7aae46833b23e4aeeff9441756ecd1d76b035e9a4652d4e8ac31c1`. Memory 256 and fresh RocksDB 1,024 populations completed another 1,008 exact searches. Raw bytes and computed embeddings use separate tables, removing the preceding shared-table contamination.

| RocksDB, 1,024 notes | Existing warm p95 | Separate `TYPE any` SQL warm p95 |
| --- | ---: | ---: |
| Unfiltered, limit 5 | 45.11 | 53.49 |
| Unfiltered, limit 50 | 66.35 | 86.24 |
| Recent, limit 5 | 42.11 | 50.01 |
| Recent, limit 50 | 42.99 | 54.77 |
| Source, limit 5 | 38.53 | 45.30 |
| Source, limit 50 | 39.91 | 50.47 |

Same-call engine execution accounts for most of that regression; SDK DTO extraction is at most 0.021 ms at p95. The uncontaminated raw-page path again has a faster unfiltered top-50 median (15.10 versus 43.67 ms) and a worse tail (p95 93.49 versus 66.35 ms, maximum 107.02 versus 73.03 ms). Its memory source top-50 tail is also retained. No single SQL byte variant is adopted.

## Bounded page-size experiment

The frozen diagnostic compares the existing query and raw CBOR pages of 128, 256 and 512 rows on fresh RocksDB populations of 4,096 and 1,024 notes. Six filter/limit groups and four alternating variants each receive 21 observations, for 1,008 measured searches. The immutable compilation source is `da8abaa5f225e0e4c56d410093215eee346ca868`, executable SHA-256 `51a3ad1ea8c46973697874ba7a5b4f8b5be3f111138d7f4af28de0bc4125da9a`. All measured calls pass complete ordered payload and distance-bit comparisons, with zero errors. Timing ran serially on the 4,096-note population and then the 1,024-note population during an exclusive quiet window.

The page loop enforces 1–512 rows and records actual peak returned rows and projected embedding bytes. For this finite float32-origin fixture it checks at most 8,192 vector-payload bytes per row: 1/2/4 MiB at pages 128/256/512. This excludes record metadata, SDK/engine allocations and arbitrary historical float64-CBOR representations; it is not a whole-process memory bound. No whole-corpus vector DTO or mutable process cache is introduced.

Before timing, both memory and fresh RocksDB correctness runs passed the five page sizes 1/3/128/256/512 against native edge-case queries, transaction cancellation and concurrent promotion fixtures, plus the actual mixed-key/dense-tie page matrix. Correctness-only matrix runs contain one observation per variant and make no percentile or performance claim.

### Warm p50 / p95 ms

| Population / filter / limit | Existing query | Page 128 | Page 256 | Page 512 |
| --- | ---: | ---: | ---: | ---: |
| 4,096 / unfiltered / 5 | 169.30 / 175.55 | 201.09 / 209.83 | 111.00 / 375.69 | 67.49 / 71.26 |
| 4,096 / unfiltered / 50 | 171.87 / 182.22 | 204.73 / 235.76 | 113.83 / 118.43 | 69.97 / 77.50 |
| 4,096 / recent / 5 | 164.33 / 169.05 | 175.68 / 181.84 | 98.01 / 101.07 | 58.81 / 60.95 |
| 4,096 / recent / 50 | 166.06 / 170.51 | 180.15 / 186.21 | 99.74 / 103.83 | 60.81 / 62.83 |
| 4,096 / source / 5 | 146.90 / 151.19 | 69.40 / 72.91 | 38.22 / 39.73 | 19.08 / 20.13 |
| 4,096 / source / 50 | 147.85 / 150.46 | 72.05 / 73.48 | 40.96 / 42.36 | 21.88 / 22.78 |
| 1,024 / unfiltered / 5 | 43.10 / 44.24 | 18.31 / 19.17 | 13.10 / 13.62 | 10.92 / 11.50 |
| 1,024 / unfiltered / 50 | 43.60 / 48.46 | 20.55 / 22.14 | 15.06 / 17.03 | 12.88 / 14.08 |
| 1,024 / recent / 5 | 41.41 / 42.37 | 16.00 / 16.79 | 11.48 / 12.18 | 9.91 / 10.57 |
| 1,024 / recent / 50 | 41.66 / 42.61 | 17.93 / 19.03 | 13.29 / 14.45 | 11.88 / 12.35 |
| 1,024 / source / 5 | 36.85 / 37.40 | 5.97 / 6.31 | 4.67 / 5.00 | 4.64 / 4.92 |
| 1,024 / source / 50 | 37.05 / 37.87 | 8.11 / 8.69 | 6.81 / 7.40 | 6.73 / 7.28 |

Page 512 improves the median and p95 in all 12 groups, but it creates a worse first observation and maximum in the larger unfiltered top-5 group. Its 631.41 ms maximum is a successful warm observation; nearest-rank p95 over 20 samples does not include that single worst sample. The unfavorable samples remain in the decision and JSON.

| 4,096 notes / unfiltered / limit 5 | First measured | Warm maximum | Same-call scan or engine time at the maximum |
| --- | ---: | ---: | ---: |
| Existing query | 165.31 ms | 371.05 ms | 365.47 ms native engine |
| Page 128 | 195.61 ms | 2,033.60 ms | 1,987.83 ms scan/SDK wait |
| Page 256 | 429.18 ms | 1,119.52 ms | 1,074.06 ms scan/SDK wait |
| Page 512 | 472.71 ms | 631.41 ms | 586.28 ms scan/SDK wait |

The first warm observation is the worst observation for all four variants in this group. CBOR decode and cosine in the heap phase remain 21.55–21.96 ms in the three paged maxima, while scan/SDK waits dominate; begin/cancel and bounded hydration also rise. This is a common timing effect with amplification in the paged path, not proof of a particular engine, allocator or OS cause. The smaller population also retains first-observation and maximum increases in some groups. No new production path is adopted from these results.

Peak returned rows are 128/256/512 respectively, with maximum actual projected vector payload 655,368/1,310,716/2,621,344 bytes in the larger population. The enforced fictional payload bounds pass on every call.

## Reproduction and normal CI

Five bounded provider-free correctness tests run in normal database CI: native semantic equivalence, acknowledged cancellation on normal/panic/abort exits, coherent promotion across pages/hydration, three-scope event/backfill/portable restore fault atomicity, and concurrent single-statement projection coherence. They make no timing assertion. The schemas and helpers are registered only in the test module, and `ciborium` is a development dependency. Hardware-sensitive loops and plans remain ignored by default.

To rerun those regressions:

```sh
cargo test --locked -p graphrag-db --lib repository::exact_vector_diagnostic_tests
```

To reproduce the page matrix against a fresh owned RocksDB directory:

```sh
probe_root="$(mktemp -d)"
GRAPHRAG_VECTOR_PROBE_RECORDS=4096 \
GRAPHRAG_VECTOR_PROBE_DENSE=1 \
GRAPHRAG_VECTOR_PROBE_PAGE_MATRIX=1 \
GRAPHRAG_VECTOR_PROBE_ORDER=alternating \
GRAPHRAG_VECTOR_PROBE_ROCKS_ROOT="$probe_root" \
cargo test --locked --release -p graphrag-db --lib \
  repository::exact_vector_diagnostic_tests::exact_vector_skinny_storage_diagnostic \
  -- --exact --ignored --nocapture
```

Repeat with `GRAPHRAG_VECTOR_PROBE_RECORDS=1024` in the same owned root; each population uses a separate fresh child directory and refuses to reopen an existing one. Omit the RocksDB environment variable for memory storage. `GRAPHRAG_VECTOR_PROBE_CORRECTNESS_ONLY=1` runs one observation per variant instead of 21 and cannot provide a warmed latency distribution. Save stdout and the compilation/binary identity, keep other load idle and retain all observations. `GRAPHRAG_VECTOR_PROBE_ORDER=cbor_first` and `legacy_first` provide separate ordering controls; comparison roles require fresh databases.

## Adoption requirements still open

Any production path needs cancellation and error handling without diagnostic panics, a bound covering initializing/active/closing snapshots until acknowledged SDK cancellation, transactional derived writes/backfill/restore and an explicit schema compatibility fence. The experimental backfill first collects the whole primary ID list; it is not a bounded paged migration. All three record scopes, current source generations and concurrent source promotion must remain coherent.

After a fictional candidate shows an improvement without a new tail, it still needs the unchanged private #92 relevance suite and controlled immutable release comparisons at documented one/four lanes, with at least 20 warm observations per category, same-request phases and ordinary-logging RPC totals. The warmed 1,000 ms hybrid/graph target is unchanged. These experiments do not complete those acceptance criteria or establish a real-corpus latency benefit.
