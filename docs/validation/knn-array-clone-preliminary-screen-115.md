# KnnTopK terminal-array clone removal: preliminary screen

Screen completed on 2026-10-07 UTC. The private candidate passed all seven native roles, including three core unit tests and four original-versus-patched correctness controls. The strict performance screen was rejected: 15/18 conditions passed all four preselected gates.

B is the production native exact query over canonical tables. M uses wildcard mirror hydration, and D uses named-field mirror hydration. The candidate changes extraction in the KNN core used by all three arms; the selected performance comparison is patched B versus original B. The compact Number vector allocation/copy and original distance arithmetic remain. This is separate from the rejected query-magnitude cache experiment.

The screen used two fresh Memory/current-thread population-1,024 fixtures, original artifact first and patched artifact second. Each category had untimed B/M/D references before one first measured observation and 20 warm observations per arm; first is not cold or first-use. All 2,268 elapsed observations and same-call phases are retained in the [companion JSON](knn-array-clone-preliminary-screen-115.json).

The four correctness roles covered Memory and RocksDB with current-thread and two-worker runtimes: 4,188 paired attempts and 1,268 coherent successful public-contract queries. Their full native payload, key, Number, F64/F32 and DTO/error contracts matched the retained original artifact. Intentional error controls remained distinct from successful coherent cases.

The separate standalone core unit artifact used serde_json 1.0.149; application controls and timing used 1.0.151 with float_roundtrip. Both were optimized test artifacts using SurrealDB core 3.2.4 and the System allocator. No serving/RPC or production acceptance follows.

Warm median is the mean of sorted indices 9 and 10; p95 is sorted index 18 of 20 warm values. A condition passes only when patched median and p95 are strictly lower and first measured and warm maximum are no higher. Values below are milliseconds, rounded for display only; JSON retains unsmoothed values.

| Scope | Filter | K | First B → patched | Median B → patched | p95 B → patched | Max B → patched | First | Median | p95 | Max | All |
|---|---|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|
| note | all | 5 | 99.669875 → 101.677542 | 38.617521 → 36.472583 | 116.139375 → 123.661750 | 123.373125 → 134.096833 | fail | pass | fail | fail | fail |
| note | all | 50 | 38.634500 → 37.281208 | 38.946938 → 36.570166 | 47.017875 → 43.080917 | 51.295833 → 43.110583 | pass | pass | pass | pass | pass |
| note | recent | 5 | 38.411000 → 33.688791 | 36.727000 → 34.238188 | 37.890541 → 35.025625 | 39.052750 → 35.190000 | pass | pass | pass | pass | pass |
| note | recent | 50 | 36.396625 → 35.136792 | 38.175917 → 34.522937 | 39.988875 → 35.403125 | 41.388625 → 35.619667 | pass | pass | pass | pass | pass |
| note | source | 5 | 34.683083 → 32.398541 | 34.573229 → 32.288229 | 36.917709 → 34.166750 | 38.768000 → 37.007958 | pass | pass | pass | pass | pass |
| note | source | 50 | 32.621583 → 33.375458 | 33.834104 → 32.935625 | 35.062542 → 33.939750 | 37.705125 → 34.617375 | fail | pass | pass | pass | fail |
| message | all | 5 | 35.683958 → 31.613500 | 35.656438 → 32.678729 | 36.892167 → 33.056000 | 36.999542 → 33.697750 | pass | pass | pass | pass | pass |
| message | all | 50 | 37.330875 → 34.120667 | 37.859958 → 34.214042 | 40.578791 → 35.254250 | 41.572208 → 35.597500 | pass | pass | pass | pass | pass |
| message | recent | 5 | 32.625416 → 30.215458 | 32.859396 → 30.129250 | 33.693041 → 31.078459 | 33.797542 → 31.222708 | pass | pass | pass | pass | pass |
| message | recent | 50 | 36.419125 → 32.204708 | 34.533791 → 31.804167 | 36.045709 → 32.647542 | 36.488666 → 33.075458 | pass | pass | pass | pass | pass |
| message | source | 5 | 63.538250 → 60.631000 | 61.691896 → 60.529063 | 63.600750 → 61.298666 | 64.291292 → 61.313792 | pass | pass | pass | pass | pass |
| message | source | 50 | 63.705709 → 63.137209 | 63.347000 → 61.702687 | 65.814458 → 63.070167 | 66.448792 → 63.124750 | pass | pass | pass | pass | pass |
| conversation | all | 5 | 35.980416 → 31.275084 | 35.381729 → 32.050271 | 37.225916 → 33.402833 | 37.284875 → 33.489917 | pass | pass | pass | pass | pass |
| conversation | all | 50 | 34.778750 → 32.769375 | 35.712291 → 32.520916 | 36.764375 → 36.915959 | 37.488375 → 37.504708 | pass | pass | fail | fail | fail |
| conversation | recent | 5 | 33.214917 → 30.763250 | 33.373229 → 30.756813 | 34.668250 → 31.679750 | 35.057500 → 31.780125 | pass | pass | pass | pass | pass |
| conversation | recent | 50 | 33.415250 → 30.907333 | 33.549979 → 30.773083 | 34.208708 → 31.985041 | 34.332834 → 34.267916 | pass | pass | pass | pass | pass |
| conversation | source | 5 | 28.753500 → 26.379000 | 27.626604 → 26.655417 | 28.682833 → 27.743208 | 28.877625 → 28.435250 | pass | pass | pass | pass | pass |
| conversation | source | 50 | 27.965583 → 26.921916 | 27.922688 → 27.018146 | 28.624333 → 27.875042 | 29.047375 → 28.181834 | pass | pass | pass | pass | pass |

Gate passes out of 18: first measured 16, warm median 18, warm p95 16, warm maximum 16.

Across all 18 conditions, warm median reductions ranged from 1.8849% to 9.6300%. The strict failed conditions were note/all/K=5, note/source/K=50, conversation/all/K=50. Median improvement alone does not satisfy the selected tail limits.

The elapsed interval covers SDK await and native Value take; engine wall time is nested within await. Typed public DTO conversion, fingerprinting and exactness assertions run after elapsed and between calls. The fixed serial artifact order, normal background activity and this synthetic fixture limit interpretation. No causal savings, exclusive CPU or terminal-array clone cost share is established.

The original Clone1 compile failure remains preserved; it occurred before native tests. The corrected stable field.0.as_slice accessor kept the extraction contract and performance rule unchanged. This completed run passed native correctness but failed the declared performance rule. There was no retry or threshold change after observations.

Each child had a 300-second deadline. The 32 MiB stdout limit was a post-completion capture/read acceptance bound, not a physical writer cap. All owned children stopped with cleanup verified. Source promotion was intentional; unchanged inventories covered three canonical and three mirror tables rather than the entire database.

The candidate remains unadopted. This screen does not qualify typed serving/RPC latency, production coherence, deployment, broader continuation or closure of #106/#115.
