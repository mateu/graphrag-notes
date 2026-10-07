# Execution-local query-magnitude cache: preliminary screen

Screen completed on 2026-10-07 UTC. The private candidate passed all seven native roles, including seven core unit tests and four original-versus-patched correctness controls. The strict performance screen was rejected: 3/18 conditions passed all four preselected gates.

B is the production native exact query over canonical tables. M uses wildcard mirror hydration, and D uses named-field mirror hydration. The candidate changes the KNN core used by all three arms; the selected performance comparison is patched B versus original B.

The screen used two fresh Memory/current-thread population-1,024 fixtures, original artifact first and patched artifact second. Each category had untimed B/M/D references before one first measured observation and 20 warm observations per arm; first is not cold or first-use. All 2,268 elapsed observations and same-call phases are retained in the [companion JSON](query-magnitude-cache-preliminary-screen-115.json).

The four correctness roles covered Memory and RocksDB with current-thread and two-worker runtimes: 4,188 paired attempts and 1,268 coherent successful public-contract queries. Their full native payload, key, Number, F64/F32 and DTO/error contracts matched the retained original artifact. Intentional error controls remained distinct from successful coherent cases.

The separate standalone core unit artifact used serde_json 1.0.149; application controls and timing used 1.0.151 with float_roundtrip. Both were optimized test artifacts using SurrealDB core 3.2.4 and the System allocator. No serving/RPC or production acceptance follows.

Warm median is the mean of sorted indices 9 and 10; p95 is sorted index 18 of 20 warm values. A condition passes only when patched median and p95 are strictly lower and first measured and warm maximum are no higher. Values below are milliseconds, rounded for display only; JSON retains unsmoothed values.

| Scope | Filter | K | First B → patched | Median B → patched | p95 B → patched | Max B → patched | First | Median | p95 | Max | All |
|---|---|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|
| note | all | 5 | 112.715375 → 101.505709 | 38.335625 → 38.324979 | 116.441375 → 118.474958 | 122.606625 → 125.882958 | pass | pass | fail | fail | fail |
| note | all | 50 | 38.227416 → 38.611375 | 38.533333 → 38.635188 | 39.691083 → 39.374667 | 39.921083 → 39.549666 | fail | fail | pass | pass | fail |
| note | recent | 5 | 37.031208 → 36.754291 | 36.270146 → 36.551312 | 39.709750 → 37.559625 | 44.994292 → 37.686000 | pass | fail | pass | pass | fail |
| note | recent | 50 | 38.128125 → 37.049916 | 37.269854 → 36.915667 | 39.990958 → 38.522958 | 41.362333 → 38.969791 | pass | pass | pass | pass | pass |
| note | source | 5 | 32.209000 → 34.061667 | 32.333750 → 33.199542 | 33.838791 → 39.168959 | 33.926584 → 39.570625 | fail | fail | fail | fail | fail |
| note | source | 50 | 33.043666 → 34.802541 | 33.075271 → 33.506417 | 34.301833 → 34.327333 | 34.489125 → 34.921250 | fail | fail | fail | fail | fail |
| message | all | 5 | 36.350583 → 34.592667 | 35.132042 → 35.371895 | 36.378667 → 36.945875 | 38.992250 → 38.109959 | pass | fail | fail | pass | fail |
| message | all | 50 | 37.071958 → 37.179541 | 36.788896 → 36.951271 | 38.176500 → 37.386250 | 38.178708 → 37.488250 | fail | fail | pass | pass | fail |
| message | recent | 5 | 32.318958 → 32.244000 | 32.357917 → 32.532437 | 33.786042 → 33.662541 | 33.925125 → 33.676750 | pass | fail | pass | pass | fail |
| message | recent | 50 | 34.453625 → 34.238667 | 34.261834 → 34.334958 | 35.635916 → 35.620458 | 36.254541 → 35.891000 | pass | fail | pass | pass | fail |
| message | source | 5 | 59.763833 → 61.720000 | 61.108166 → 61.527167 | 63.095959 → 63.135416 | 64.304917 → 63.973917 | fail | fail | fail | pass | fail |
| message | source | 50 | 62.206917 → 64.979250 | 63.051646 → 62.962271 | 65.091042 → 64.236750 | 65.231875 → 64.350792 | fail | pass | pass | pass | fail |
| conversation | all | 5 | 36.683500 → 35.089875 | 35.186855 → 35.015750 | 36.506500 → 36.111917 | 36.849833 → 38.818333 | pass | pass | pass | fail | fail |
| conversation | all | 50 | 35.164250 → 35.015083 | 35.032979 → 34.722105 | 36.561291 → 35.317666 | 37.201584 → 35.424583 | pass | pass | pass | pass | pass |
| conversation | recent | 5 | 33.083167 → 33.097666 | 33.491938 → 33.080604 | 34.921584 → 34.008167 | 35.849750 → 34.174542 | fail | pass | pass | pass | fail |
| conversation | recent | 50 | 33.732250 → 32.542750 | 33.799396 → 33.155146 | 38.985250 → 33.912958 | 45.071125 → 34.792208 | pass | pass | pass | pass | pass |
| conversation | source | 5 | 28.178208 → 29.650917 | 27.851979 → 28.097687 | 28.803291 → 34.002625 | 29.330416 → 35.073750 | fail | fail | fail | fail | fail |
| conversation | source | 50 | 27.216292 → 27.525750 | 27.740667 → 27.759604 | 29.158167 → 28.262583 | 29.187625 → 28.438750 | fail | fail | pass | pass | fail |

Gate passes out of 18: first measured 9, warm median 7, warm p95 12, warm maximum 13.

The elapsed interval covers SDK await and native Value take; engine wall time is nested within await. Typed public DTO conversion, fingerprinting and exactness assertions run after elapsed and between calls. The fixed serial artifact order, normal background activity and this synthetic fixture limit interpretation. No causal savings, exclusive CPU or query-magnitude cost share is established.

Earlier preflight and build-validation refusals remain preserved. The corrected builder distinguished custom-build profiles from optimized runtime libraries; it did not change the candidate core source. This completed run passed native correctness but failed the declared performance rule. There was no retry or threshold change after observations.

Each child had a 300-second deadline. The 32 MiB stdout limit was a post-completion capture/read acceptance bound, not a physical writer cap. All owned children stopped with cleanup verified. Source promotion was intentional; unchanged inventories covered three canonical and three mirror tables rather than the entire database.

The candidate remains unadopted. This screen does not qualify typed serving/RPC latency, production coherence, deployment, broader continuation or closure of #106/#115.
