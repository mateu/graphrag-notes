# Message/recent/K50 admission and hydration screen (#115 / #106)

All four fictional correctness/plan roles passed at compiled source `3695cc9de6fabb89e60c5154ab79272e481a1f5a`, binary `fbd149c569045761db3a920da3cca442d2cc1e56d4b769832e187ba214727071`, using SurrealDB 3.2.4, Rust 1.97.1 and optimized release tests. This screen captures query structure and instrumented operator metrics. It has **no first-use/warmed latency benchmark samples and no performance acceptance**. #106 and #115 remain open.

The [complete public plan artifact](message-recent-k50-stage-screen-115.json) retains all 32 plain/FULL plan projections, generated selected-child paths, recognized operator names, field coverage and individual FULL integer metrics. Raw plans, SQL, records, typed keys, payloads and reference/inventory fingerprints remain private.

## Fixture and exactness

Four fresh serial roles cover Memory/RocksDB × CurrentThread1/MultiThread2. Observed runtime flavor and worker count are asserted before database creation and after the body, outside queries. Each role populates the original three scopes with 1,024 records per scope, 1,024 dimensions, heterogeneous bodies and native key kinds. The focused recent-message case has 797 eligible rows, 75 eligible identical-vector rows and K50, exercising partial native tie admission.

B is the unchanged full primary query; M is the unchanged full mirror query. P lifts M's exact inner native admission. H preserves M's canonical hydration and final projection/order while binding P's complete native values directly. No JSON, string-ID or DTO round trip constructs H's bound pool.

Each role verified full ordered B/M/H payload equality, P's complete ordered native pool, native key/Number kinds, native F64 and serving F32 distance bits, serving DTOs and the current public message query. Full primary and mirror inventories remained exact before/after. All four tests passed with zero failures, exit 0 and owned-child cleanup verified.

Per role: 18 logged requests comprise 2 eligibility reads, 8 normal B/M/P/H result queries, 4 plain EXPLAIN requests and 4 EXPLAIN FULL executions. Two additional public-contract queries and setup/inventory queries are separately declared. The four-role total is 72 logged request/completion pairs, 8 public-contract checks, 8 runtime observations, 32 normal stage results, 16 plain plans and 16 FULL executions. The selected stdout bound was 32 MiB per child; the earlier 64 MiB preparation suggestion was not executed.

## Observed plans

Every captured plain/FULL plan has one selected `SortByKey` and zero expression-embedded operator roots. Selected-child node counts are B7, M10, P6 and H7 in all roles and both modes. The M subquery flattened into its selected child tree; its scan and KNN operators have nonzero metrics. Two SQL ORDER BY clauses therefore did not produce two physical sorts in these captured plans.

H's `SourceExpr` reads the bound admission pool; it is not an expression-embedded SELECT. The general 3.2.4 limitation that child-only metric enabling may omit expression-embedded work is not an observed M/H coverage gap here. Plain plans have no numeric metrics; all 120 selected nodes across the 16 FULL plans retain their observed integer metrics in the JSON artifact.

The following are **individual instrumented poll elapsed observations**, expressed in milliseconds for readability. They are not query wall-time samples, percentiles, exclusive stage costs or latency acceptance. The JSON retains every original integer value for every selected node, including rows and batches. H has no selected scan/KNN operator.

| Role | Query | Selected nodes | TableScan poll elapsed (ms) | KnnTopK poll elapsed (ms) |
| --- | --- | --- | --- | --- |
| Memory / CurrentThread1 | B | 7 | 23.628208 | 8.966625 |
| Memory / CurrentThread1 | M | 10 | 17.640376 | 8.872334 |
| Memory / CurrentThread1 | P | 6 | 17.453459 | 8.945375 |
| Memory / CurrentThread1 | H | 7 | — | — |
| RocksDB / CurrentThread1 | B | 7 | 23.838043 | 9.033791 |
| RocksDB / CurrentThread1 | M | 10 | 18.156582 | 9.028416 |
| RocksDB / CurrentThread1 | P | 6 | 18.226750 | 9.133334 |
| RocksDB / CurrentThread1 | H | 7 | — | — |
| Memory / MultiThread2 | B | 7 | 23.180124 | 9.095084 |
| Memory / MultiThread2 | M | 10 | 18.048583 | 9.231957 |
| Memory / MultiThread2 | P | 6 | 18.327875 | 9.064376 |
| Memory / MultiThread2 | H | 7 | — | — |
| RocksDB / MultiThread2 | B | 7 | 23.392040 | 9.018835 |
| RocksDB / MultiThread2 | M | 10 | 18.724833 | 9.081999 |
| RocksDB / MultiThread2 | P | 6 | 18.454375 | 8.977418 |
| RocksDB / MultiThread2 | H | 7 | — | — |

B/M/P scans each emitted 797 rows in three batches and their KNN operators emitted 50 rows in one batch. This is an observed execution shape. It does not establish which operator or materialization causes the persistent untraced median regression.

## Retained first attempt and unchanged timing evidence

The first source attempt, `67df5bdf4c1568905c6701ce262ef69b70830f65`, failed compilation before fixture execution because the pinned SDK does not implement scalar `QueryResult` for `serde_json::Value`. Its failure remains retained. The corrected source takes supported native `Value`, retains its exact typed plan privately and converts a clone with `into_json_value` solely for JSON traversal. That conversion is best-effort; native P/H values and payload comparisons remain unchanged.

The companion only adds ignored child tests and registration; removing the registration reconstructs the frozen parent `63219e54` exactly. The current lockfile is `a6a22f7f`, from the merged main base. Production queries, runtime defaults, schema/events, lifecycle and dependencies are unchanged by this screen.

Original [native-Float qualification](native-float-qualification-106.md) and [runtime-control timing evidence](native-float-runtime-control-115.md) remain unchanged. Runtime measurements were compiled at `59edb356` with binary `f93567dd`: the current-thread mirror passed 123/144 strict conditions and the two-worker mirror 134/144, with persistent failures. The earlier 122/144 result also remains retained. This correctness/plan screen supplies no replacement latency observations.

## Reproduction and limits

The two ignored selectors are `repository::exact_vector_native_float_qualification::message_recent_stage_diagnostic::message_recent_k50_plans_current_thread` and `repository::exact_vector_native_float_qualification::message_recent_stage_diagnostic::message_recent_k50_plans_two_worker`. Run each separately in an optimized locked build. For Memory, omit `GRAPHRAG_NATIVE_FLOAT_ROCKS_ROOT`; for RocksDB, set it to a fresh owned root for that role. Keep the fixed fixture and all result/inventory assertions. No provider is required.

Plain EXPLAIN does not execute the underlying query. FULL executes/drains it and returns a plan, not independently verified search payload; separate normal queries supply the payload comparisons. Each FULL plan is one instrumented execution; plain plans do not execute the query.

FULL elapsed metrics measure inclusive operator polling, overlap child work, omit intervals between pending polls and include instrumentation effects. Do not sum nested durations, add P+H as M cost, or treat their separate statement/planning/cache contexts as a decomposition. The metrics do not measure exclusive CPU or prove OS, scheduler or maintenance causality.

The screen supports inspection of actual admission/hydration structure. It does not qualify lower first/median/p95/max latency, general mirror write coherence, real-corpus relevance/RPC behavior or production adoption. Engine 3.2.4 and all prior strict acceptance gates remain unchanged.
