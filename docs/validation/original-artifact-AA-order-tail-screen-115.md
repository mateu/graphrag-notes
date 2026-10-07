# Same-artifact A/A order and tail controls

Four serial runs on 2026-10-07 used the same original optimized SurrealDB 3.2.4 executable and System allocator. The artifact labels A/B/B/A are bookkeeping for identical source and binary. Each run created a fresh Memory/current-thread fixture with 1,024 records per scope. Normal desktop and shared services stayed active; these runs had no scheduling observer.

The [companion JSON](original-artifact-AA-order-tail-screen-115.json) retains all 4,536 ordered timing and same-call phase samples, 216 arm statistics and 108 chronological comparison rows. It also records completion of 216 untimed references. All four native tests passed and owned process cleanup was verified. Compiled assertions checked full ordered payloads, native keys, Number/F64/F32 distance evidence, public DTOs and fixture inventories; the retained raw-event audit corroborates those assertions. Report generation did not reopen databases or reexecute them.

Each run measured three query arms: B is the production exact query over canonical tables; M is experimental wildcard mirror hydration; D is experimental named-field mirror hydration. These query-arm labels are distinct from the artifact labels. The measured rotation resets per category as `(sample_index + call_offset) % 3`, starting B/M/D. Every category has three untimed references before its first measured sample, followed by 20 warm samples per arm. First measured is not cold or first use, and fresh fixtures do not establish identical cache state.

The chronological comparisons are roles 0→1 and 2→3. For the 18 primary-B conditions in each pair, the following counts describe the original preliminary-screen relations:

| Relation in later run | Roles 0→1 | Roles 2→3 |
|---|---:|---:|
| First measured no larger | 5/18 | 7/18 |
| Warm median strictly lower | 1/18 | 1/18 |
| Warm p95 strictly lower | 5/18 | 6/18 |
| Warm maximum no larger | 8/18 | 13/18 |
| All four relations hold | 0/18 | 0/18 |

The same executable produced different first, median, p95 and maximum observations. This establishes descriptive sensitivity to these separate runs; it does not identify a cause, quantify a false-rejection probability or prove equivalence. The counts are not candidate acceptance or rejection. They do not dismiss observed candidate tails or relax continuation requirements. Every first and warm value is retained, without clipping, smoothing or pooling with the scheduling and stack observations.

Warm median is the mean of sorted warm positions 9 and 10, p95 is position 18, and maximum is position 19, all zero-based. Positive later-minus-earlier differences mean the later run was slower. A zero earlier denominator gives a null percentage. Elapsed time covers SDK await plus native Value take; engine wall time is nested. Between-call DTO conversion, hashing and validation remain part of the workload outside those elapsed measurements.

No scheduling/QoS remedy, CPU share, causal tail mechanism, serving/RPC qualification, graph-quality result or production adoption follows from these controls. #106 and #115 remain open; the no-new-tail and complete exactness/relevance/RPC requirements remain in force.
