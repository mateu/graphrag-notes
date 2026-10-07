# Named-field hydration: corrected wider qualification (#115)

D passed 275 of 288 conditions, so the corrected wider screen is rejected: CurrentThread passed 137/144 and MultiThread passed 138/144. Every D warmed median improved, but first-call and tail failures violate the selected continuation rule. No production optimization is qualified and #106/#115 remain open.

B is the production baseline, M is experimental wildcard hydration, and D is named-field hydration. M comparisons are descriptive. This test-only screen preserves the pinned SurrealDB 3.2.4 engine, arithmetic, canonical typed keys and current-generation filters; it implements no production derived-data maintenance, schema migration or runtime/default/concurrency change.

[The complete corrected JSON](named-field-hydration-wide-115.json) retains all 18,144 measured elapsed/same-call-phase samples and all 288 condition arrays/statistics/gates. It is separate from the [unqualified interrupted history](named-field-hydration-wide-interrupted-115.json). Displayed timings are rounded to three decimals; decisions use the complete unrounded values.

| Runtime | Observed conditions | Four-gate passes | Rejected conditions | First failures | Median failures | p95 failures | Maximum failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CT1 | 144 | 137 | 7 | 4 | 0 | 3 | 4 |
| MT2 | 144 | 138 | 6 | 1 | 0 | 3 | 4 |

Gate-failure counts overlap. Across 288 conditions, median passed 288/288, p95 passed 282/288, first passed 283/288 and maximum passed 280/288. The rule requires D warmed median and p95 strictly below B, and D first measured/warmed maximum no larger than B, in every 144 conditions for each runtime. Observation completeness is separate from continuation; a complete rejection retains every sample. A pass would only permit considering production coherence and real-corpus qualification.

## Protocol and first-history correction

The run used four fresh all-scope correctness prerequisites, followed by 16 serial matrix roles: Memory/RocksDB, CurrentThread (one worker)/MultiThread (two workers), 1,024/4,096 rows per scope and two fixed measured-order offsets. Each matrix role covered 18 categories: note/message/conversation × all/recent/source × K5/K50.

Each category completed one untimed B, M and D reference in that fixed order before any measured call. All three had to return exactly K finite numeric distances and the same ordered native payload, canonical key/Number kinds, native F64/serving F32 bits and successful public DTO as B. The private log retained the successful full B body once; M/D retained metadata and hashes. Failed comparisons retain full actual/reference bodies. There were 54 reference pairs per role and 864 over the full matrix, excluded from latency samples.

Measured calls used the predeclared three-arm rotation with 21 observations per variant/category: first plus 20 warmed calls. Each variant appeared seven times in every call slot. First is after all three declared references and validation. Equal untimed execution counts do not establish identical cache state: references use fixed B/M/D order, and measured predecessors are not independently counterbalanced. First is neither cold nor query first use. No additional warmup, clipping, retry or tail removal was used.

Median averages warmed sorted indices 9/10; p95 is nearest rank, warmed sorted index 18 of 20. All four D/B gates and descriptive M comparisons were fixed before dispatch. Native roles had a 300-second deadline. After completion, stdout capture/read acceptance was bounded at 32 MiB; this is not a physical writer or output cap. Available output and partial failure files remain retained on oversize/refusal. The source-derived passing stdout allowance was a planning estimate, not a promise that diagnostics or failure bodies fit.

## Exactness, runtime and lifecycle

All 20 native children passed with zero libtest failures and verified owned-group cleanup. Four correctness children covered both backends and both runtimes before any matrix timing; each checked 317 coherent KNN/public-repository cases, three native-key hydration cases, 27 payload-edge cases and six plain/FULL plan calls. Historical coherent cases had to succeed. Deliberate computed-query errors and explicit-null/missing-field DTO controls retained their separate qualified expectations. Successful-generation promotion is intentional in correctness, so the Source table is not claimed unchanged.

The declared ledger contains 4,188 correctness pairs and 18,144 measured pairs, with 864 reference pairs reported separately. There are 1,268 additional public-repository checks outside those ledgers. These counts exclude fixture setup and inventory queries. Each matrix call is compared with the immutable B reference after stopping the timer: full typed payload fingerprint, ordered canonical keys/raw Numbers, native F64/serving F32 bits, successful public DTO, finite distances and exact K. This is declared between-call work. Whole canonical/mirror inventories before/after each role and all B/M/D reference metadata must match across eight roles sharing each population.

The independent raw audit checked all 20 captured streams/exits, recomputed 648 full payload hashes across 10,212 retained rows and 522,720 finite raw Number/F64/F32 entries, and rebuilt all 864 reference and 18,144 measured-call ledgers. DTO and inventory checks are compiled source assertions corroborated by retained events; the file reviewer did not rerun Rust deserialization or open/requery stores. Runtime flavor/worker assertions bracket the body outside measured queries.

Compiled source: `801bd3d987bdd2d22f13fec986fd7889ae7276b4`. Wide module SHA256: `5dd59b55871bd0faae548fa0e67cc0f32047b596bcaa646be62868cc00bd686c`. Binary SHA256: `b3e62ca0e030568776951a50e406d5eaa747ddec614c426fa2a8ca39ca330a71`. The build used Rust 1.97.1, release tests, system allocator, optimized serde_json float_roundtrip and pinned SDK/core 3.2.4. Documentation descendants do not relabel this compiled source.

## Descriptive measurements

D/B per-condition warm-median ratios range 0.784407–0.946194 across 288 conditions, corresponding to 5.38%–21.56% reductions. These are synthetic exact-query measurements; they do not qualify user RPC latency or production load.

Each row below contains 72 conditions. Ranges are min–max of those condition summaries, rather than quantiles of pooled calls. All three variants and every underlying sample are retained in JSON.

| Runtime/backend | Variant | First range (ms) | Warm median range (ms) | Warm p95 range (ms) | Warm max range (ms) |
| --- | --- | --- | --- | --- | --- |
| CT1/Memory | B | 27.436–254.314 | 27.359–253.030 | 27.995–259.450 | 28.412–289.947 |
| CT1/Memory | M | 22.295–231.793 | 22.341–230.034 | 22.620–238.934 | 23.066–251.231 |
| CT1/Memory | D | 22.194–230.827 | 22.115–226.536 | 22.519–235.433 | 22.664–264.935 |
| CT1/RocksDB | B | 34.441–278.394 | 35.017–276.483 | 36.051–281.860 | 36.056–291.242 |
| CT1/RocksDB | M | 29.089–260.918 | 29.326–260.867 | 30.160–265.988 | 30.408–300.178 |
| CT1/RocksDB | D | 29.063–260.147 | 29.170–258.951 | 29.949–269.709 | 30.523–274.585 |
| MT2/Memory | B | 27.930–244.336 | 27.658–242.445 | 28.528–250.379 | 28.697–252.216 |
| MT2/Memory | M | 22.665–223.791 | 22.660–224.634 | 23.039–227.805 | 23.091–228.977 |
| MT2/Memory | D | 22.485–221.423 | 22.360–220.351 | 23.050–227.872 | 23.134–229.528 |
| MT2/RocksDB | B | 27.104–249.510 | 27.582–251.762 | 28.180–273.472 | 28.195–280.671 |
| MT2/RocksDB | M | 22.786–236.558 | 23.029–237.838 | 23.605–242.722 | 23.718–263.312 |
| MT2/RocksDB | D | 22.998–233.104 | 22.984–234.579 | 23.358–238.500 | 23.374–263.147 |

All 13 rejected conditions are below. Each timing cell is B→D in ms. Flags use the unrounded source samples; a rounded near-equality can still fail.

| Runtime/backend | Rows/scope | Order | Scope/filter/K | First B→D | Median B→D | p95 B→D | Max B→D | Failed gates |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| CT1/Memory | 1024 | baseline_first | note/all/K5 | 98.283→101.358 | 38.593→33.503 | 114.650→104.861 | 131.163→115.810 | first |
| CT1/Memory | 1024 | baseline_first | note/source/K50 | 33.247→28.632 | 33.614→28.910 | 36.383→36.450 | 42.869→38.984 | p95 |
| CT1/Memory | 1024 | baseline_first | message/recent/K50 | 37.411→36.027 | 34.791→32.634 | 38.939→39.927 | 39.555→40.358 | max, p95 |
| MT2/Memory | 1024 | mirror_first | message/all/K50 | 39.399→41.068 | 37.330→33.952 | 38.474→36.393 | 39.763→36.610 | first |
| CT1/Memory | 4096 | mirror_first | message/recent/K50 | 144.420→121.190 | 138.529→121.617 | 158.997→174.482 | 195.236→263.559 | max, p95 |
| CT1/RocksDB | 1024 | baseline_first | note/all/K5 | 132.505→134.190 | 47.516→41.476 | 62.373→50.028 | 127.736→136.688 | first, max |
| CT1/RocksDB | 1024 | baseline_first | message/source/K5 | 76.511→92.209 | 73.773→67.385 | 77.972→75.850 | 79.310→79.179 | first |
| MT2/RocksDB | 1024 | baseline_first | note/all/K50 | 40.777→36.732 | 42.104→37.861 | 44.259→45.064 | 44.920→47.495 | max, p95 |
| MT2/RocksDB | 1024 | baseline_first | note/recent/K5 | 38.676→33.197 | 39.795→33.827 | 42.127→35.240 | 42.856→42.975 | max |
| MT2/RocksDB | 1024 | mirror_first | note/all/K50 | 40.873→36.845 | 40.737→36.995 | 44.309→48.432 | 49.568→49.050 | p95 |
| MT2/RocksDB | 1024 | mirror_first | conversation/all/K50 | 38.006→35.093 | 37.266→33.310 | 38.383→43.861 | 41.160→48.751 | max, p95 |
| CT1/RocksDB | 1024 | mirror_first | message/all/K50 | 50.114→51.069 | 45.313→41.147 | 47.952→42.848 | 48.331→49.490 | first, max |
| MT2/RocksDB | 4096 | mirror_first | note/recent/K5 | 147.046→127.692 | 146.835→127.841 | 151.533→132.241 | 152.844→161.082 | max |

Eleven of 13 rejected conditions used 1,024 rows per scope and two used 4,096 (144 conditions per population). All six p95 failures were K50 (six of 144 K50 conditions); K5 had zero p95 failures out of 144. Seven rejections passed both warmed gates but failed first/max; two passed first/max but failed p95; four failed p95 plus tails. No median failed. These are descriptive associations, not population/filter/runtime causes.

| Partition | Conditions | Passes | Failures |
| --- | ---: | ---: | ---: |
| scope=note | 96 | 89 | 7 |
| scope=message | 96 | 91 | 5 |
| scope=conversation | 96 | 95 | 1 |
| filter=all | 96 | 89 | 7 |
| filter=recent | 96 | 92 | 4 |
| filter=source | 96 | 94 | 2 |

Largest D−B increases were 15.698 ms for first, 15.485 ms for warm p95 and 68.323 ms for warm maximum, each selected across 288 condition summaries. These maxima can belong to different calls/conditions and are not a stage decomposition.

## Same-call phase limits

For each physical role below, the ratio is the median of same-call engine_execution_ms/elapsed_ms over its non-null engine observations (up to 1,134 measured calls per role, equally including B/M/D). These intervals are inclusive elapsed observations, not exclusive CPU.

| Role | Engine observations | Median engine/elapsed (%) |
| --- | ---: | ---: |
| slot-00-memory-1024-baseline-first-current-thread | 1134 | 99.6889 |
| slot-00-memory-1024-baseline-first-two-worker | 1134 | 99.6868 |
| slot-01-memory-1024-mirror-first-two-worker | 1134 | 99.6869 |
| slot-01-memory-1024-mirror-first-current-thread | 1134 | 99.6713 |
| slot-02-memory-4096-baseline-first-current-thread | 1134 | 99.9124 |
| slot-02-memory-4096-baseline-first-two-worker | 1134 | 99.8919 |
| slot-03-memory-4096-mirror-first-two-worker | 1134 | 99.8906 |
| slot-03-memory-4096-mirror-first-current-thread | 1134 | 99.9087 |
| slot-04-rocksdb-1024-baseline-first-current-thread | 1134 | 99.6912 |
| slot-04-rocksdb-1024-baseline-first-two-worker | 1134 | 99.6748 |
| slot-05-rocksdb-1024-mirror-first-two-worker | 1134 | 99.6944 |
| slot-05-rocksdb-1024-mirror-first-current-thread | 1134 | 99.7334 |
| slot-06-rocksdb-4096-baseline-first-current-thread | 1134 | 99.9266 |
| slot-06-rocksdb-4096-baseline-first-two-worker | 1134 | 99.8916 |
| slot-07-rocksdb-4096-mirror-first-two-worker | 1134 | 99.8930 |
| slot-07-rocksdb-4096-mirror-first-current-thread | 1134 | 99.9265 |

Raw native Value take median was 0.000417 ms across all 18,144 measured calls. SDK await includes the query future; this Value take is not public DTO conversion. Full fingerprint/DTO/key/bit work is outside elapsed and can affect between-call state. Do not add/subtract inclusive phases or unrelated percentiles to infer a stage cost, CPU time or maintenance/scheduler/cache cause.

## Host and background limits

The root-reported test host is an arm64 Mac with 48 GiB RAM and 18 logical CPUs. Normal desktop/shared production services remained active. Local preparation, competing compilation, file audits and mocks were paused during the root-owned 20-role window. This does not establish quiet CPU, representative production background load or a hardware/scheduling cause.

CurrentThread (one worker) and MultiThread (two workers) are diagnostic wrappers whose runtime assertions run outside query timing. Serial fictional calls do not qualify real RPC/model/provider behavior, concurrent workers, production distributions, atomic derived-data maintenance, restore coherence or application-default runtime behavior. The engine and production defaults remain unchanged.

## Retained interrupted history

The separate interrupted JSON (SHA256 `c222e83a2d4978668ebd8bc28d3b17e7034eeac46afa95bc26f5f384f2a97fd9`) retains 4,536 completed measured samples and 72 conditions from four completed timing roles, following four passing correctness roles. Its compiled source is `541b4bef00ed5fac0dcb8ad527ad317c70e45e4f`. Only B received an untimed reference, so preparation was unequal; the run was stopped for that review finding and remains incomplete and unqualified. Interrupted current-role samples/counts are excluded and unresolved. All verified owned children were reconciled; there is no replay or automatic retry.

The descriptive completed-prefix outcomes were CT 31/36 and MT 34/36; neither is a candidate continuation decision. The prefix is never merged with the corrected matrix or used to qualify D. Original thin-mirror and runtime-control reports retain their original outcomes and provenance.

The corrected all-three-reference experiment still fails the complete continuation rule. No production coherence/maintenance implementation or adoption follows. Any further work needs a separately declared bounded diagnostic; #106 and #115 stay open.
