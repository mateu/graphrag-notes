# Fixed-runtime exact-vector diagnostic (#115 / #106)

The test-only native-Float mirror preserved exact results in all **12,096** declared calls. It passed **123/144** strict conditions under an explicit current-thread runtime and **134/144** under an explicit two-worker runtime. Both arms retain latency failures and **neither qualifies a production optimization**. [#115](https://github.com/mateu/graphrag-notes/issues/115) and [#106](https://github.com/mateu/graphrag-notes/issues/106) remain open.

This is a controlled runtime follow-up to the [historical native-Float qualification](native-float-qualification-106.md), whose **122/144** result remains unchanged. The [full-sample artifact](native-float-runtime-control-115.json) preserves all 144 conditions per new arm, all first+20 warmed arrays, all same-call phase timings and 144 matched conditions for both variants. Every unfavorable observation remains included.

## Controlled source and correctness

Compiled source `59edb35669e6b6c06520977449a35b04266006ae`, optimized immutable test binary SHA-256 `f93567ddc7cdacfce318a04ec7ff2e05fab90bc5d1470985cb6b60190c0a7312`. Both measured arms use the same binary, SurrealDB 3.2.4, system allocation, Rust 1.97.1, Tokio 1.53.1 and JSON float-roundtrip parsing. That measured source factors the original ordinary/measurement bodies into shared async helpers and adds explicit wrappers, retaining the original shared body bytes, queries, arithmetic, data, indexes, filters and timing boundaries. Production code, manifests, dependencies, concurrency and defaults are unchanged.

After measurement, review found that selecting both runtime wrappers with one fresh RocksDB root reused the same database directory. The fixture now accepts a startup-only database label/suffix: current-thread keeps the original names and two-worker adds `-two-worker`, for both correctness and measurement wrappers. This changes the helper signatures and database-opening statements; the final helper bodies are not byte-identical to the measured source. SQL, data, keys, arithmetic, assertions, query timing and sample schedules stay unchanged. The compiled source/binary and full-sample JSON above remain the original measured artifacts; the directory-isolation fix has no new timing result. Native combined-wrapper validation is pending.

Ordinary correctness passed once for each Memory/RocksDB × current-thread/two-worker combination, four separate children with one selected test and no failures. The fixture covers notes, messages and conversation summaries; complete ordered payloads, native F64 and serving F32 distance bits; native numeric/string/UUID/object/array keys and numeric key kinds; selected invalid/nonfinite Float cases, limits 1/5/50, source/time filters and source promotion. The unchanged static mirror is populated once and then used with exact native COSINE pool selection and one-statement primary hydration. It does not implement production mirror maintenance.

Each wrapper asserts and emits its observed Tokio runtime flavor and scheduler worker count before DB creation and after its shared body, outside timed queries. The current-thread arm observes one scheduler worker; the multithread arm observes exactly two. These counts do not describe total native, blocking, Rayon or OS threads or prove that a particular request ran on both workers. The normal CLI uses a multithread runtime without this fixed worker count, so this is not a reproduction of its live default.

The measured artifact was Mach-O arm64 on macOS 27.0.1 (build 26A434), with 48 GiB RAM and 18 physical / 18 logical CPU cores. These are reproducibility facts; they do not establish scheduling or maintenance attribution.

## Matched schedule and acceptance rules

Sixteen serial child roles pair runtimes within the original eight Memory/RocksDB × 1,024/4,096 records-per-scope × baseline_first/mirror_first slots. Even slots run current-thread first; odd slots run two-worker first. Both runtime arms preserve the original eight-slot order. Each role uses a fresh DB/root and the same sealed binary, clean environment and `--test-threads=1`. The fixture retains 1,024 dimensions, 96 dense ties per scope, heterogeneous body repetitions 0/1,024/4,096 and the original indexes. No extra database probe query, sleep, scheduler option, native option, warmup, index removal or production concurrency change is added.

Every role has 18 conditions: three scopes × all/recent/source filters × K=5/50. Variant order alternates per sample. Each condition retains one first measured call and 20 warm calls per variant: 756 calls per role, 6,048 per runtime arm, 12,096 total, 576 first and 11,520 warm calls. One unmeasured baseline reference precedes each category’s 21 × 2 measurements, 18 references per role. First means its first measured call after inventory and that reference; it is not a cold-start or cold-cache measurement.

Within each arm, the mirror must have a lower warm median and p95 with no larger first or warm maximum in **every** condition. Warm median is conventional; nearest-rank p95 is the nineteenth sorted sample of twenty. All decisions use unrounded arrays.

| Mirror versus primary gate | Current-thread passed / 144 | Two-worker passed / 144 |
| --- | ---: | ---: |
| First mirror ≤ primary | 130 | 138 |
| Warm median mirror < primary | 139 | 140 |
| Warm p95 mirror < primary | 141 | 139 |
| Warm maximum mirror ≤ primary | 135 | 136 |
| All four together | 123 | 134 |

Failure counts overlap: current-thread has 14 first, 5 median, 3 p95 and 9 maximum failures across 21 distinct conditions; two-worker has 6 first, 4 median, 5 p95 and 8 maximum failures across 10 conditions. The additional strict passes under two workers do not waive any failed tail.

| Runtime / backend / population / starting order | Mirror strict passes / 18 | Mirror warm median change range |
| --- | ---: | ---: |
| current_thread / memory / 1024 / baseline_first | 12 | -17.48% to +0.77% |
| current_thread / memory / 1024 / mirror_first | 16 | -17.52% to +2.40% |
| current_thread / memory / 4096 / baseline_first | 18 | -18.42% to -7.83% |
| current_thread / memory / 4096 / mirror_first | 18 | -18.95% to -7.81% |
| current_thread / rocksdb / 1024 / baseline_first | 14 | -20.04% to +1.29% |
| current_thread / rocksdb / 1024 / mirror_first | 11 | -19.41% to +0.57% |
| current_thread / rocksdb / 4096 / baseline_first | 18 | -15.52% to -4.68% |
| current_thread / rocksdb / 4096 / mirror_first | 16 | -16.86% to -5.52% |
| multi_thread / memory / 1024 / baseline_first | 16 | -17.86% to +1.09% |
| multi_thread / memory / 1024 / mirror_first | 17 | -18.41% to +1.78% |
| multi_thread / memory / 4096 / baseline_first | 17 | -20.64% to -8.35% |
| multi_thread / memory / 4096 / mirror_first | 18 | -20.47% to -7.12% |
| multi_thread / rocksdb / 1024 / baseline_first | 16 | -17.36% to +2.29% |
| multi_thread / rocksdb / 1024 / mirror_first | 16 | -17.11% to +1.25% |
| multi_thread / rocksdb / 4096 / baseline_first | 17 | -21.06% to -5.80% |
| multi_thread / rocksdb / 4096 / mirror_first | 17 | -18.87% to -5.38% |

Negative change means faster. Across the 144 mirror conditions, current-thread median change ranges from -20.04% to +2.40% and p95 from -23.84% to +1.17%; two-worker median change ranges from -21.06% to +2.29% and p95 from -27.59% to +7.05%. Ranges summarize heterogeneous conditions and do not replace any individual gate.

## Runtime comparison is a separate question

For each unchanged variant, the matched runtime gate compares two workers against the contemporaneous current-thread control with the same lower-median/lower-p95/no-larger-first/no-larger-max rules. Only **59/144** primary and **50/144** mirror conditions pass all four. These comparisons do not establish that changing the runtime is uniformly faster.

| Two-worker versus current-thread gate | Primary passed / 144 | Mirror passed / 144 |
| --- | ---: | ---: |
| First two-worker ≤ current-thread | 89 | 79 |
| Warm median two-worker < current-thread | 94 | 80 |
| Warm p95 two-worker < current-thread | 92 | 87 |
| Warm maximum two-worker ≤ current-thread | 95 | 91 |
| All four together | 59 | 50 |

Two-worker warm median change versus current-thread ranges from -21.45% to +3.47% for the primary path and -22.47% to +3.26% for the mirror. The JSON retains both runtime tuples, both full sample arrays and every individual gate for all 288 variant-specific matched comparisons.

## All 31 mirror gate failures

Tuples are first / warm median / warm p95 / warm maximum in milliseconds. Warm statistics use all 20 declared warm calls.

| Runtime / backend / population / order / scope / filter / K | Primary tuple | Mirror tuple | Failed gates |
| --- | ---: | ---: | --- |
| current_thread / memory / 1024 / baseline_first / note / all / 5 | 72.669 / 38.052 / 113.358 / 119.619 | 78.605 / 34.922 / 114.389 / 119.283 | first, p95 |
| current_thread / memory / 1024 / baseline_first / note / recent / 5 | 36.240 / 35.883 / 37.704 / 38.501 | 31.733 / 31.714 / 33.200 / 40.433 | max |
| current_thread / memory / 1024 / baseline_first / note / recent / 50 | 35.699 / 36.419 / 38.713 / 38.819 | 35.831 / 35.976 / 37.528 / 37.786 | first |
| current_thread / memory / 1024 / baseline_first / message / recent / 50 | 34.330 / 33.950 / 36.021 / 36.135 | 34.520 / 34.213 / 35.853 / 36.471 | first, max, median |
| current_thread / memory / 1024 / baseline_first / conversation / all / 50 | 36.357 / 35.107 / 37.194 / 43.173 | 35.579 / 33.954 / 35.747 / 44.471 | max |
| current_thread / memory / 1024 / baseline_first / conversation / recent / 50 | 32.038 / 33.295 / 34.747 / 37.320 | 32.653 / 32.842 / 34.386 / 34.893 | first |
| current_thread / memory / 1024 / mirror_first / message / recent / 50 | 34.621 / 34.790 / 39.301 / 39.389 | 35.189 / 35.624 / 38.721 / 40.702 | first, max, median |
| current_thread / memory / 1024 / mirror_first / conversation / recent / 50 | 33.334 / 33.609 / 34.676 / 34.942 | 33.276 / 33.720 / 34.393 / 34.457 | median |
| current_thread / rocksdb / 1024 / baseline_first / note / all / 5 | 98.346 / 40.432 / 122.967 / 124.415 | 105.607 / 35.608 / 121.537 / 122.371 | first |
| current_thread / rocksdb / 1024 / baseline_first / message / all / 5 | 36.997 / 36.694 / 37.800 / 38.367 | 37.855 / 30.927 / 31.829 / 31.909 | first |
| current_thread / rocksdb / 1024 / baseline_first / message / recent / 50 | 35.214 / 35.334 / 37.050 / 38.434 | 35.849 / 35.789 / 37.295 / 38.342 | first, median, p95 |
| current_thread / rocksdb / 1024 / baseline_first / conversation / all / 5 | 38.441 / 36.748 / 38.227 / 38.315 | 39.116 / 30.698 / 31.752 / 32.274 | first |
| current_thread / rocksdb / 1024 / mirror_first / note / all / 5 | 109.689 / 41.649 / 126.441 / 127.088 | 91.833 / 35.241 / 116.979 / 158.303 | max |
| current_thread / rocksdb / 1024 / mirror_first / note / source / 50 | 33.588 / 34.088 / 35.461 / 35.517 | 32.179 / 32.555 / 34.605 / 40.122 | max |
| current_thread / rocksdb / 1024 / mirror_first / message / all / 5 | 36.889 / 36.511 / 39.301 / 39.470 | 38.951 / 31.121 / 32.950 / 33.500 | first |
| current_thread / rocksdb / 1024 / mirror_first / message / all / 50 | 37.848 / 38.563 / 43.004 / 48.362 | 38.194 / 37.862 / 40.285 / 44.629 | first |
| current_thread / rocksdb / 1024 / mirror_first / message / recent / 50 | 34.918 / 35.241 / 36.337 / 36.715 | 35.309 / 35.441 / 36.762 / 37.229 | first, max, median, p95 |
| current_thread / rocksdb / 1024 / mirror_first / conversation / all / 5 | 38.197 / 36.530 / 37.858 / 37.971 | 38.903 / 30.658 / 31.755 / 31.804 | first |
| current_thread / rocksdb / 1024 / mirror_first / conversation / recent / 50 | 34.041 / 34.492 / 35.423 / 35.866 | 34.129 / 33.952 / 35.114 / 35.299 | first |
| current_thread / rocksdb / 4096 / mirror_first / note / recent / 50 | 148.108 / 147.510 / 150.118 / 157.204 | 133.062 / 134.307 / 138.014 / 157.600 | max |
| current_thread / rocksdb / 4096 / mirror_first / message / source / 5 | 244.081 / 246.014 / 249.591 / 250.432 | 227.026 / 229.742 / 235.981 / 261.675 | max |
| multi_thread / memory / 1024 / baseline_first / message / recent / 50 | 34.378 / 34.946 / 35.994 / 36.420 | 34.816 / 35.327 / 36.615 / 37.644 | first, max, median, p95 |
| multi_thread / memory / 1024 / baseline_first / conversation / recent / 50 | 35.195 / 33.438 / 34.635 / 34.865 | 34.182 / 33.386 / 37.076 / 43.706 | max, p95 |
| multi_thread / memory / 1024 / mirror_first / message / recent / 50 | 34.885 / 34.820 / 35.739 / 35.841 | 35.362 / 35.438 / 36.272 / 36.901 | first, max, median, p95 |
| multi_thread / memory / 4096 / baseline_first / message / source / 50 | 240.269 / 240.438 / 245.559 / 253.331 | 220.252 / 220.371 / 232.166 / 256.133 | max |
| multi_thread / rocksdb / 1024 / baseline_first / note / all / 5 | 39.071 / 39.746 / 40.528 / 40.621 | 41.712 / 34.399 / 35.150 / 35.248 | first |
| multi_thread / rocksdb / 1024 / baseline_first / message / recent / 50 | 35.231 / 35.273 / 36.578 / 37.145 | 35.809 / 36.080 / 37.208 / 37.486 | first, max, median, p95 |
| multi_thread / rocksdb / 1024 / mirror_first / note / all / 5 | 39.871 / 39.399 / 41.281 / 41.601 | 42.157 / 34.023 / 35.417 / 36.175 | first |
| multi_thread / rocksdb / 1024 / mirror_first / message / recent / 50 | 36.268 / 35.957 / 36.781 / 36.935 | 36.348 / 36.407 / 37.107 / 37.484 | first, max, median, p95 |
| multi_thread / rocksdb / 4096 / baseline_first / message / recent / 50 | 115.902 / 115.227 / 122.822 / 123.577 | 105.052 / 104.273 / 111.952 / 133.321 | max |
| multi_thread / rocksdb / 4096 / mirror_first / message / source / 5 | 244.097 / 245.779 / 254.669 / 254.908 | 231.022 / 231.269 / 241.171 / 259.230 | max |

## Same-call phases and background limits

All 12,096 calls retain SDK await, raw Vec<Value> take and nullable engine-execution elapsed timings. Across the 16 roles, each grouping all 756 heterogeneous calls in that role, the median same-call engine/total ratio ranges from 99.70% to 99.92%. The median raw-value take across all 12,096 calls is 0.000417ms. These are elapsed boundaries, not exclusive CPU or a cause of any tail. Engine time is nested in SDK await; await+take and engine≤await checks use 0.05ms tolerance. Public typed-DTO conversion and exact comparisons occur outside query timing. Do not add nested durations or subtract unrelated percentiles.

Root retained an exclusive owned timing-window declaration: builds, correctness/application children, formatter/mock preparation, host setup and GitHub commands finished before dispatch; authors/reviewers paused local work. Controller guards and file parsing occur between roles. Normal desktop/shared services remained, and there is no sampled OS scheduling or maintenance attribution. No observation is removed. The historical mock-overlap qualification stays with the historical run and is not carried into this window as an observed overlap.

## Reproduction and deferred scope

The two ordinary correctness wrappers are nonignored, while both hardware-sensitive measurement wrappers are ignored by default. Each pair uses one shared body with distinct runtime startup database names. These exact selectors run one child at a time:

```text
repository::exact_vector_native_float_qualification::native_float_all_scopes_preserve_exact_payload_keys_and_distance_bits
repository::exact_vector_native_float_qualification::native_float_all_scopes_preserve_exact_payload_keys_and_distance_bits_two_worker_runtime
repository::exact_vector_native_float_qualification::native_float_all_scopes_release_qualification
repository::exact_vector_native_float_qualification::native_float_all_scopes_release_qualification_two_worker_runtime
```

Use `cargo test --locked --release -p graphrag-db --lib SELECTOR -- --exact --nocapture --test-threads=1`, adding `--ignored` for a measurement selector. Set `GRAPHRAG_NATIVE_FLOAT_RECORDS` to 1024 or 4096 and `GRAPHRAG_NATIVE_FLOAT_ORDER` to baseline_first or mirror_first. For RocksDB, supply `GRAPHRAG_NATIVE_FLOAT_ROCKS_ROOT` as a fresh owned role root; omit it for Memory. Keep one compiled binary and the declared counterbalanced 16-role schedule. Preserve raw logs, attempt ledgers, all samples and exit/cleanup receipts; no retry or correctness-only run can substitute for a measured role.

The ordinary current-thread directory is `native-float-correctness`; two-worker uses `native-float-correctness-two-worker`. Measurement directories are `native-float-POPULATION-ORDER` and `native-float-POPULATION-ORDER-two-worker`. Both wrappers may therefore be selected together on a fresh root, without reusing an existing database. Separate child roles remain required for the matched timing schedule above.

The [rejected query-norm SQL screen](native-float-qualification-106.md#separate-query-norm-sql-screen-correctness-rejection) remains rejected and unchanged. This follow-up adds no new query-norm outcome and does not treat its source-supported pool/tie explanation as complete specific-row causality.

Neither runtime arm meets the complete component gate. The static mirror still lacks transactional write/delete/generation promotion, cancellation, backfill/restore and migration coherence qualification. Production adoption would also need unchanged real-corpus relevance and one/four-lane RPC gates. No production runtime, concurrency, default, schema, engine or Fam routing change is made or accepted by this diagnostic.
