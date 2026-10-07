# Native Float mirror qualification (#106 / #115)

The test-only native-Float mirror completed all **6,048** declared calls with exact results, but failed the strict latency gate: **122 of 144** conditions passed. It is **not accepted for production**. [#106](https://github.com/mateu/graphrag-notes/issues/106) and [#115](https://github.com/mateu/graphrag-notes/issues/115) remain open.

The [sanitized full-sample artifact](native-float-qualification-106.json) preserves all 144 conditions, each first call and every one of its 20 warmed calls for both variants, individual decisions and same-call phase samples. No failed tail or sample is excluded. Historical [exact-vector experiments](exact-vector-experiments-106.md) remain unchanged.

## Controlled mechanism and exactness

Compiled source `bc0a74dc66e37485750d3cefb0e55c2d86b92a85`, optimized immutable test binary SHA-256 `5add0d585141b05d60dc997865819a45b129cddd36115f9454dcc1d6c4862aac`. Both variants use SurrealDB 3.2.4, the system allocator, Rust 1.97.1 and runtime JSON float-roundtrip parsing. Only a test module and its test-only registration are added; production queries, schemas, model settings, ranking, concurrency and defaults are unchanged.

This revisits the historical thin native-Float representation under the current precision-capable baseline and a broader fixture; it is not a new cosine algorithm. Each scope has a separately defined test mirror containing native Float vectors, canonical native record keys and filter inputs. The unchanged exact COSINE KNN operator selects its bounded pool, and one statement hydrates the authoritative primary records. Mirrors are created and populated once in bounded pages while the fixture is idle. No CBOR/computed field, cached magnitude or mutable process cache is used.

Notes, messages and conversation summaries cover full ordered payloads, native F64 distance bits, serving F32 bits and native numeric/string/UUID/object/array keys, including mixed numeric key kinds. All measured ordered comparisons and before/after inventories of the three primary tables passed, with reference parity across backends and orders. The ordinary correctness selector passed once on Memory and once on fresh RocksDB; it covers selected missing/empty/malformed/nonfinite Float cases, limits 1/5/50, source/time filters and source promotion. It does not prove general production mirror maintenance or every schemaless numeric representation.

The measured populations are 1,024 and 4,096 records per scope, with 1,024-dimensional Float vectors and 96 dense ties per scope. Bodies mix zero, 1,024 and 4,096 repetitions by row; this is not a homogeneous large-body condition. The original HNSW indexes remain installed in measured roles, while explicit COSINE retains exact KNN behavior. Separate malformed-history correctness fixtures remove those indexes so ingestion does not reject the intended edge cases before comparison.

## Schedule and retained outcomes

Eight roles cover Memory/RocksDB × two populations × native-first/mirror-first starting orders. Each role has 18 conditions: three scopes × all/recent/source filters × limits 5/50. The physical variant order alternates per sample. Calls are serial under the default Tokio current-thread test runtime; the normal CLI uses a multithread runtime, so this does not establish host-runtime or production concurrency behavior. Each condition retains one first measured call and 20 warmed calls per variant: 756 calls per role, 6,048 total, 288 first and 5,760 warmed calls. An additional 18 unmeasured native reference calls per role precede comparisons.

First means the first measured call after inventory and its per-condition native reference, not a cold process/database/cache. Warm p50 is the conventional median; nearest-rank p95 is the 19th sorted sample of 20. Strict acceptance requires a lower warm median and p95 with no larger first or warm maximum in every condition.

| Gate | Passed / 144 | Failed / 144 |
| --- | ---: | ---: |
| First candidate ≤ native | 134 | 10 |
| Warm median candidate < native | 139 | 5 |
| Warm p95 candidate < native | 141 | 3 |
| Warm maximum candidate ≤ native | 133 | 11 |
| All four gates together | 122 | 22 |

Gate decisions use the unrounded sample arrays, even when rounded display values coincide. Failure counts overlap; 22 distinct conditions fail at least one required gate. Every role exits successfully with exactness, which is separate from its latency decision.

| Backend / records per scope / starting order | Strict passes / 18 | Median change range |
| --- | ---: | ---: |
| memory / 1,024 / baseline_first | 13 | -18.84% to +1.13% |
| memory / 1,024 / mirror_first | 15 | -17.96% to +1.11% |
| memory / 4,096 / baseline_first | 18 | -19.94% to -7.40% |
| memory / 4,096 / mirror_first | 18 | -19.82% to -8.46% |
| rocksdb / 1,024 / baseline_first | 11 | -18.97% to +0.56% |
| rocksdb / 1,024 / mirror_first | 12 | -19.17% to +2.60% |
| rocksdb / 4,096 / baseline_first | 17 | -15.60% to -5.32% |
| rocksdb / 4,096 / mirror_first | 18 | -15.35% to -5.41% |

Across all conditions, candidate median change is -19.94% to +2.60% and p95 change is -21.81% to +1.98% (negative means faster). Aggregate ranges do not waive any individual first or maximum failure.

### All 22 failed conditions

All times below are milliseconds. Each tuple is **first / warm median / warm p95 / warm max**; the warm statistics describe all 20 declared warmed calls.

| Backend / population / order / scope / filter / K | Native tuple | Float-mirror tuple | Failed gates |
| --- | ---: | ---: | --- |
| memory / 1024 / baseline_first / note / all / 5 | 86.002 / 39.593 / 122.121 / 123.511 | 91.301 / 34.773 / 112.968 / 124.529 | first, max |
| memory / 1024 / baseline_first / note / all / 50 | 39.405 / 39.071 / 42.346 / 42.412 | 38.322 / 38.491 / 40.011 / 44.109 | max |
| memory / 1024 / baseline_first / message / recent / 50 | 35.256 / 34.553 / 35.861 / 36.035 | 34.833 / 34.942 / 36.041 / 36.846 | median, p95, max |
| memory / 1024 / baseline_first / message / source / 50 | 63.042 / 63.372 / 65.064 / 65.329 | 61.717 / 61.806 / 63.818 / 66.876 | max |
| memory / 1024 / baseline_first / conversation / recent / 50 | 34.052 / 34.144 / 35.590 / 37.189 | 34.479 / 33.552 / 34.484 / 34.491 | first |
| memory / 1024 / mirror_first / note / all / 5 | 95.927 / 40.053 / 119.678 / 134.820 | 84.273 / 35.063 / 122.044 / 122.507 | p95 |
| memory / 1024 / mirror_first / message / recent / 50 | 34.439 / 35.189 / 36.363 / 36.500 | 35.281 / 35.580 / 36.309 / 36.500 | first, median |
| memory / 1024 / mirror_first / message / source / 50 | 63.297 / 63.653 / 64.475 / 64.557 | 63.699 / 62.497 / 63.565 / 63.998 | first |
| rocksdb / 1024 / baseline_first / note / all / 5 | 127.841 / 44.264 / 140.490 / 155.925 | 112.631 / 37.657 / 133.200 / 168.042 | max |
| rocksdb / 1024 / baseline_first / note / recent / 50 | 40.169 / 40.365 / 42.368 / 42.704 | 39.322 / 39.586 / 41.863 / 42.842 | max |
| rocksdb / 1024 / baseline_first / note / source / 50 | 37.027 / 35.774 / 36.977 / 37.562 | 35.721 / 34.002 / 34.646 / 38.018 | max |
| rocksdb / 1024 / baseline_first / message / all / 5 | 38.256 / 37.855 / 39.932 / 44.780 | 39.096 / 32.446 / 38.409 / 40.412 | first |
| rocksdb / 1024 / baseline_first / message / recent / 50 | 36.827 / 36.513 / 41.038 / 42.212 | 37.502 / 36.717 / 37.738 / 38.973 | first, median |
| rocksdb / 1024 / baseline_first / conversation / all / 5 | 37.283 / 37.558 / 38.125 / 39.113 | 39.095 / 31.861 / 32.271 / 32.767 | first |
| rocksdb / 1024 / baseline_first / conversation / recent / 50 | 36.260 / 35.548 / 35.923 / 35.926 | 35.651 / 35.113 / 35.727 / 42.609 | max |
| rocksdb / 1024 / mirror_first / note / all / 50 | 41.617 / 41.763 / 42.406 / 43.346 | 40.819 / 40.973 / 42.172 / 45.998 | max |
| rocksdb / 1024 / mirror_first / message / all / 5 | 38.356 / 37.775 / 39.266 / 39.993 | 40.279 / 32.862 / 33.233 / 33.698 | first |
| rocksdb / 1024 / mirror_first / message / recent / 50 | 38.365 / 37.555 / 38.414 / 43.255 | 38.110 / 37.916 / 38.501 / 38.794 | median, p95 |
| rocksdb / 1024 / mirror_first / conversation / all / 5 | 38.349 / 38.168 / 38.943 / 39.381 | 39.786 / 32.438 / 33.058 / 33.158 | first |
| rocksdb / 1024 / mirror_first / conversation / all / 50 | 38.843 / 38.343 / 39.497 / 39.879 | 38.057 / 37.118 / 37.821 / 41.052 | max |
| rocksdb / 1024 / mirror_first / conversation / recent / 50 | 36.026 / 36.480 / 39.996 / 40.740 | 36.063 / 37.429 / 39.230 / 39.322 | first, median |
| rocksdb / 4096 / baseline_first / conversation / all / 50 | 144.255 / 142.306 / 144.908 / 146.193 | 129.653 / 130.055 / 132.346 / 149.538 | max |

## Phase and background limits

Every observed call retains SDK-await, raw-value take and nullable engine-execution elapsed durations in the JSON. Engine time is nested in SDK await; await/take are bounded by the same call total with a 0.05 ms tolerance. These are elapsed diagnostics, not exclusive CPU; raw Vec<Value> extraction is timed, while public typed-DTO conversion and exactness comparisons are outside the measured query path. Do not add nested timings or subtract unrelated percentiles to attribute a cause.

Over all 756 calls per role (both variants, 18 heterogeneous conditions, first and warm), the median same-call engine-execution/total ratio ranges from 99.65% to 99.92%. The median raw-value take across all 6,048 calls is 0.000541ms. These ratios locate elapsed time inside the measured engine statement; they do not establish exclusive CPU or the cause of an individual tail. Some early shared tails also contain a larger SDK-await/engine boundary.

Controlled roles are provider-free, and owned build/provider/host operations were paused during timing. Normal desktop/shared services remained. Three short Node mocked suites (about 62/77/63 ms) and three Python pure/temp suites (about 15/6/9 ms), plus small syntax/source/AST/file preparation and root logging/metadata work, overlapped. They were declared and paused. Approximate timestamps derive adjacent writes/tool cells rather than sampled process starts. No sample is removed or causally attributed to this background; the run is not described as a perfectly idle machine.

## Separate query-norm SQL screen: correctness rejection

Source `1601f771a71f6384f23e38f1598b0dfd6b8aac93` passed 27 finite Float paired queries across the three scopes, then failed record/payload selection at K=1 for negative NaN and for a finite query after adding an integer vector. Both variants returned rows with equal serving F32 bits in each failed pair. In total 37 pairs were observed; later edge/Decimal cases were not reached. No query-norm timing was performed.

This SQL rewrite changes candidate-pool selection. Native KNN admits a bounded pool before final output ordering; the rewrite sorts computed cosine distances and limits the inner result. Its bound LIMIT does not satisfy the pinned streaming planner's literal-only stable TopK optimization, and ordinary native full sorting lacks an insertion-sequence tie breaker. The outer id ordering cannot restore discarded records. This is a source-supported mechanism, not complete attribution of either specific failure: actual EXPLAIN and native F64 ranking keys were not retained, and equal serving F32 bits do not prove equal native ranking keys. The screen stays rejected, with all failed children retained.

## Reproduction and scope

The nonignored correctness fixture is registered only under cfg(test) and makes no performance assertion. It uses Memory unless an explicitly fresh owned RocksDB root is supplied. The hardware-sensitive release selector is ignored by default. No native build or test was executed by the report generator.

```sh
cargo test --locked --release -p graphrag-db --lib \
  repository::exact_vector_native_float_qualification::native_float_all_scopes_preserve_exact_payload_keys_and_distance_bits \
  -- --exact --nocapture --test-threads=1
```

For one measured role, use an explicitly owned fresh root and select a population and starting order:

```sh
probe_root="$(mktemp -d)"
GRAPHRAG_NATIVE_FLOAT_ROCKS_ROOT="$probe_root" \
GRAPHRAG_NATIVE_FLOAT_RECORDS=4096 \
GRAPHRAG_NATIVE_FLOAT_ORDER=baseline_first \
cargo test --locked --release -p graphrag-db --lib \
  repository::exact_vector_native_float_qualification::native_float_all_scopes_release_qualification \
  -- --exact --ignored --nocapture --test-threads=1
```

Use a separate fresh root for each role; existing child DBs are refused. Repeat with 1,024/4,096 and baseline_first/mirror_first on both backends; omit the RocksDB variable for Memory. GRAPHRAG_NATIVE_FLOAT_CORRECTNESS_ONLY=1 runs a single observation per variant and cannot establish warm latency. Preserve compiler/binary identity, full stdout/attempt ledgers, exit/cleanup receipts and all unfavorable observations. The fixture asserts exactness; the latency decision must be recomputed from every first+20 warm samples.

No runtime optimization is qualified. The static fixture has no transactional create/update/delete, event ordering, cancellation, production backfill/restore or migration lifecycle for maintaining the mirror. Even a future fictional mechanism win needs those coherence fences and the unchanged real-corpus relevance/one-four-lane RPC operating gates. This diagnostic does not close #106/#115 or alter the documented accepted release/Fam migration sequence.
