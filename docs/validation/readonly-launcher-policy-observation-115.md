# Read-only scheduling observation

An isolated observation on 2026-10-07 used the original optimized SurrealDB 3.2.4 benchmark with the System allocator. One fresh Memory/current-thread fixture held 1,024 records per scope and rotated the production exact query (B), experimental wildcard mirror hydration (M), and experimental named-field mirror hydration (D). All 1,134 timed calls, 54 untimed references and 54 arm statistics are retained in the [companion JSON](readonly-launcher-policy-observation-115.json). The native test passed and both owned process groups stopped with cleanup verified.

The observer recorded 217 snapshots: 216 had a verified native identity bracket, and the final unavailable bracket remains in the report. The public fields were stable across their known records:

| Field | Observed value | Known-record denominator |
|---|---:|---:|
| Native nice | 0 | 216 |
| Native task priority | 31 | 216 |
| Thread base priority | 31 | 5,152 |
| Thread current priority | 31 | 5,152 |
| Ancestor nice | 0 | 165 |

Thread records are repeated getter observations, not unique thread lifetimes. Ancestor records are verified captures, not unique processes. Getters do not form an atomic scheduling snapshot. The JSON also retains the raw flag and maximum-priority histograms, every availability fact, and separate denominators for unknown returns.

The nominal cadence was 250 ms. Across all 216 observed intervals, minimum/median/p95/maximum were 240.482000 / 250.000500 / 256.467000 / 259.513000 ms. Display rounding affects only this sentence; the JSON retains every original integer interval.

The measured nice and priority fields identify no variation to target with a scheduling change. Requested and effective QoS, clamps and overrides remain unknown because the used public external getters do not expose them. These raw fields cannot establish complete scheduling policy, a tail cause, or a remedy.

Normal desktop and shared services stayed active. The observer consumed CPU, and the mixed workload included between-call DTO conversion, hashing and validation. Elapsed measurements cover SDK await plus native Value take; engine wall time is nested. First measured observations follow three untimed B/M/D references and are not cold or first-use measurements. Buffered stdout marker receipts do not locate query execution. The timing data is descriptive instrumented evidence and stays separate from unobserved control runs.

Native full-payload, key, Number, F64/F32, DTO and inventory facts are compiled assertions corroborated by the pinned raw-event audit. Report generation did not reopen databases or reexecute those assertions. The native stdout limit was a post-completion read acceptance bound; the collector had its own physical output cap. All retained captures fit their declared bounds.

The first report-generator version contained a category-rotation check that did not match the benchmark. Independent source review caught it before any actual projection; the corrected version was independently reviewed and materialized once. Original experiment data and the historical generator remain intact.

This observation makes no clean-latency, per-query scheduling, CPU-share, causal, serving/RPC qualification, production adoption or issue-closure claim. #106 and #115 remain open.
