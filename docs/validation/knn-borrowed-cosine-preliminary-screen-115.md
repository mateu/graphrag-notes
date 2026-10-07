# KnnTopK borrowed cosine: preliminary screen

Screen completed on 2026-10-07 UTC. The private candidate passed all seven native roles, including three core unit tests and four original-versus-patched correctness controls. The strict performance screen was rejected: 13/18 conditions passed all four preselected gates.

B is the production native exact query over canonical tables. M uses wildcard mirror hydration, and D uses named-field mirror hydration. The candidate borrows native Number references for direct Object/single-field Cosine without materializing the record vector. It retains ordered generic Number arithmetic and recomputes query magnitude per row. Original skips/fallbacks and other distances remain. All three arms share the patched core; the selected performance comparison is patched B versus original B. This is separate from the rejected query-magnitude cache and array-clone experiments.

The screen used two fresh Memory/current-thread population-1,024 fixtures, original artifact first and patched artifact second. Each category had untimed B/M/D references before one first measured observation and 20 warm observations per arm; first is not cold or first-use. All 2,268 elapsed observations and same-call phases are retained in the [companion JSON](knn-borrowed-cosine-preliminary-screen-115.json).

The four correctness roles covered Memory and RocksDB with current-thread and two-worker runtimes: 4,188 paired attempts and 1,268 coherent successful public-contract queries. Their full native payload, key, Number, F64/F32 and DTO/error contracts matched the retained original artifact. Intentional error controls remained distinct from successful coherent cases. DTO and inventory facts are compiled native assertions corroborated by the fixed independent raw/hash audit; this report does not reexecute DTO conversion or reopen stores.

The separate standalone core unit artifact used serde_json 1.0.149; application controls and timing used 1.0.151 with float_roundtrip. Both were optimized test artifacts using SurrealDB core 3.2.4 and the System allocator. No serving/RPC or production acceptance follows.

Warm median is the mean of sorted indices 9 and 10; p95 is sorted index 18 of 20 warm values. A condition passes only when patched median and p95 are strictly lower and first measured and warm maximum are no higher. Values below are milliseconds, rounded for display only; JSON retains unsmoothed values.

| Scope | Filter | K | First B → patched | Median B → patched | p95 B → patched | Max B → patched | First | Median | p95 | Max | All |
|---|---|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|
| note | all | 5 | 101.236958 → 96.732583 | 39.016813 → 35.367896 | 117.479708 → 115.695917 | 124.419959 → 121.224083 | pass | pass | pass | pass | pass |
| note | all | 50 | 38.468416 → 35.180625 | 38.838271 → 35.359042 | 39.682334 → 37.045750 | 39.948375 → 39.052916 | pass | pass | pass | pass | pass |
| note | recent | 5 | 36.194750 → 34.510167 | 36.753958 → 33.920000 | 38.024208 → 34.911667 | 38.119667 → 34.940375 | pass | pass | pass | pass | pass |
| note | recent | 50 | 37.043667 → 34.318916 | 37.066958 → 34.346646 | 38.571584 → 35.297500 | 40.138916 → 35.323083 | pass | pass | pass | pass | pass |
| note | source | 5 | 33.173917 → 32.800500 | 32.824458 → 32.343042 | 34.400333 → 33.064584 | 34.556583 → 33.956459 | pass | pass | pass | pass | pass |
| note | source | 50 | 33.424042 → 32.556042 | 33.354062 → 33.019021 | 34.526584 → 34.138125 | 34.740500 → 34.237000 | pass | pass | pass | pass | pass |
| message | all | 5 | 35.422625 → 32.262584 | 36.474209 → 31.682167 | 41.179917 → 32.343375 | 44.731208 → 32.361375 | pass | pass | pass | pass | pass |
| message | all | 50 | 36.809333 → 33.642250 | 37.260771 → 33.607771 | 38.513000 → 34.704958 | 39.250375 → 34.720583 | pass | pass | pass | pass | pass |
| message | recent | 5 | 32.671042 → 30.375500 | 33.251208 → 29.934062 | 33.963125 → 36.277208 | 36.034292 → 39.632583 | pass | pass | fail | fail | fail |
| message | recent | 50 | 34.492041 → 35.557917 | 35.297625 → 31.418583 | 37.094750 → 32.706625 | 37.468500 → 32.927666 | fail | pass | pass | pass | fail |
| message | source | 5 | 61.359166 → 59.966666 | 62.360292 → 60.512187 | 71.986125 → 65.231833 | 75.454292 → 65.769500 | pass | pass | pass | pass | pass |
| message | source | 50 | 62.998834 → 60.994500 | 63.228562 → 62.002188 | 66.708834 → 73.387417 | 68.342083 → 73.570292 | pass | pass | fail | fail | fail |
| conversation | all | 5 | 34.628209 → 32.422625 | 35.673771 → 31.701375 | 36.428042 → 32.411792 | 37.078416 → 32.795541 | pass | pass | pass | pass | pass |
| conversation | all | 50 | 37.187458 → 33.219541 | 35.825854 → 32.421708 | 36.843458 → 34.241584 | 37.173250 → 34.461959 | pass | pass | pass | pass | pass |
| conversation | recent | 5 | 32.950417 → 30.766000 | 33.473917 → 30.880417 | 34.240916 → 31.216416 | 34.625250 → 31.289583 | pass | pass | pass | pass | pass |
| conversation | recent | 50 | 34.088833 → 32.207750 | 33.562375 → 31.992833 | 35.725958 → 36.555750 | 35.742000 → 37.420291 | pass | pass | fail | fail | fail |
| conversation | source | 5 | 27.713958 → 27.896875 | 27.762187 → 26.548770 | 28.649125 → 27.361750 | 28.682875 → 28.231500 | fail | pass | pass | pass | fail |
| conversation | source | 50 | 28.121125 → 26.484250 | 28.025208 → 26.900354 | 29.196458 → 27.589042 | 29.241208 → 27.637666 | pass | pass | pass | pass | pass |

Gate passes out of 18: first measured 16, warm median 18, warm p95 15, warm maximum 15.

Across all 18 conditions, warm median reductions ranged from 1.0045% to 13.1382%. The strict failed conditions were message/recent/K=5, message/recent/K=50, message/source/K=50, conversation/recent/K=50, conversation/source/K=5. Median improvement alone does not satisfy the selected tail limits.

The elapsed interval covers SDK await and native Value take; engine wall time is nested within await. Typed public DTO conversion, fingerprinting and exactness assertions run after elapsed and between calls. The fixed serial artifact order, normal background activity and this synthetic fixture limit interpretation. No causal savings, exclusive CPU or borrowed-cosine cost share is established.

Earlier private source-assembly/preparation failures and rejected candidate runs remain separate and retained. This completed BorrowCosine1 run passed native correctness but failed the declared performance rule. No candidate combination, retry, tail removal or threshold change followed observations.

Each child had a 300-second deadline. The 32 MiB stdout limit was a post-completion capture/read acceptance bound, not a physical writer cap. All owned children stopped with cleanup verified. Source promotion was intentional; unchanged inventories covered three canonical and three mirror tables rather than the entire database.

The candidate remains unadopted. This screen does not qualify typed serving/RPC latency, production coherence, deployment, broader continuation or closure of #106/#115.
