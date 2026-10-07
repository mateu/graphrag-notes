# Named-field hydration diagnostic screen

This is a test-only screen. It does not accept a production optimization or close the investigation.

B is the current production query. M uses wildcard mirror hydration. D uses named-field mirror hydration, retaining the canonical fields required by the same outer projection.

On each backend, all 317 coherent cases returned equal ordered B/M/D native payloads, record keys, raw distance Numbers and F64/F32 bits, with successful public DTO conversion and the inherited public-contract checks. Intentional missing, malformed and computed-field controls retain their declared native/DTO-error comparisons; missing canonical rows and unused computed errors do not establish general baseline/mirror equivalence. Every timed result also matched its full untimed reference and public DTO/key/bit checks.

The [companion JSON](named-field-hydration-focused-pilot-115.json) retains all 126 elapsed samples and their same-call phases.

Compiled source: `a08b3f60654bd6ee6c51e67fe8793df8b6970275`.

| Role | Outcome | Native passed / failed | Cleanup verified |
| --- | --- | --- | --- |
| memory-correctness | passed_correctness | 1 / 0 | true |
| rocksdb-correctness | passed_correctness | 1 / 0 | true |
| memory-pilot | pilot_observations_complete | 1 / 0 | true |
| rocksdb-pilot | pilot_observations_complete | 1 / 0 | true |

Decision: `consider_wider_qualification_only`.

First means the first measured call after three declared untimed references. Each variant has 20 warm calls. Median is the mean of sorted indices 9 and 10; p95 is sorted index 18.

| Backend | Variant | First (ms) | Warm min | Median | p95 | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| memory | B | 47.661 | 33.840 | 51.820 | 56.851 | 57.729 |
| memory | M | 47.951 | 34.403 | 51.417 | 58.361 | 60.154 |
| memory | D | 45.349 | 31.365 | 48.596 | 56.048 | 57.304 |
| rocksdb | B | 52.430 | 34.262 | 52.540 | 72.081 | 72.631 |
| rocksdb | M | 51.173 | 35.219 | 53.796 | 62.819 | 70.555 |
| rocksdb | D | 50.270 | 32.003 | 52.084 | 56.793 | 60.751 |

The selected continuation rule compares D with production B on both backends: lower warm median and p95, with first measured and warm maximum no higher. M comparisons are descriptive. A pass only permits considering wider qualification.

Native await/take times and same-call engine wall phases are not CPU measurements or public DTO conversion time. Reference output, fingerprint, DTO and key/bit checks are declared between-call work.

Successful libtest outcomes are distinct from intentional query and DTO errors in payload controls. Inventory assertions cover three canonical tables and three mirrors; the fixture deliberately promotes Source.successful_generation.

All four tests passed with zero libtest failures and verified owned cleanup. The two correctness roles completed 634 successful coherent cases in total, plus 636 extra public-contract queries across correctness and pilots. There were 2,226 started/completed pairs and eight runtime bracket events. Required payload-control query and DTO errors remain explicit.

The timed screen covers only message/recent/K50, 1,024 rows per scope and 1,024-dimensional vectors, on one current-thread run per backend. Each backend used three full untimed references, then the fixed B/M/D, M/D/B, D/B/M rotation repeated seven times. All 126 raw elapsed values and corresponding same-call phases are retained in the companion JSON. Normal desktop and shared services remained active. This is not a repeated statistical study, wider matrix or multi-thread qualification.

The initial attempt remains a failed Memory correctness run. It completed 317 coherent successful-row/DTO cases, then stopped at an explicit-NULL optional-field control: B/M/D native payloads, key/Number/F64/F32 bits and DTO-error identity matched, but the fixture required DTO success. No RocksDB or pilot role was dispatched and no timing samples were produced. The separate correction requires equal DTO errors for explicit NULL in all three scopes; it preserves successful native rows, all 317 coherent DTO gates, SQL, pilot order, quantiles and the preselected continuation rule. The later attempt used a fresh compiled source, artifact and stores; the original failure was retained.

Qualification date: 2026-10-07 UTC. The result permits considering wider qualification only; production mirror atomic coherence, read-through and restore gates remain open, and issues #106/#115 are not closed by this screen.
