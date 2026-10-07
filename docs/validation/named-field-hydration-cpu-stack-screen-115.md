# Mixed exact-query CPU-stack screen for #106 / #115

Two separately captured current-thread Memory and RocksDB roles provide broad sampled-frame clues. Both native test programs and samplers exited successfully, and both owned process groups were cleaned up. The original Memory controller remains failed at its strict header-path check; separately accepted file-only recovery binds its retained evidence to the intended artifact. Memory was not rerun. The later RocksDB-only controller completed successfully.

**This is an instrumented mixed-workload screen, with no performance qualification, CPU-share estimate, cause claim or production adoption.** The untraced [wider qualification](named-field-hydration-wide-115.md) remains a separate complete rejection; its samples are not combined with these profiles. Issues #106 and #115 remain open.

The same unchanged release artifact uses pinned SDK/core SurrealDB 3.2.4, Rust 1.97.1, the system allocator and optimized debug0 code. Its exact compiled-source and artifact identities remain in the wider report and private binding records. No new compilation, query change, provider call, production change or default change was introduced for this screen.

B is the production native exact query over the canonical tables; M is the experimental native-Float mirror with wildcard hydration; D is the experimental native-Float mirror with named-field hydration. Each role uses the unchanged 1,024-row-per-scope fixture and 1,024-dimensional embeddings, scopes notes/messages/conversations, filters all/recent/source and K=5/50. There are 18 categories, 54 untimed reference pairs in fixed B/M/D order, and 1,134 instrumented calls: 21 observations per arm per category. The first measured value follows all three references; it is not cold or guaranteed to have identical cache history. The measured three-arm order rotates. Between calls, complete native/DTO/key/Number/F64/F32 comparison, fingerprinting and logging work runs.

Each role completed 2,398 marker events and two runtime observations, with one selected native test passed and none failed. The compiled assertions require exact K, finite distances, complete ordered payload/key/Number/F64/F32 and successful public DTO equality, plus before/after inventories. File-only audits corroborated captured protocol and identity records; they did not reexecute Rust DTO conversion or reopen/requery the stores.

The sampler was launched after the flushed fixture-ready marker against the exact owned PID and requested 30 seconds at a 1 ms interval. This requested interval is not a claim about achieved cadence. Native and sampler deadlines were 300 and 45 seconds respectively. Native stdout had a 32 MiB post-completion capture/read acceptance bound, not a physical-output cap; available outputs remain retained on an oversize or other failure. No automatic retries ran.

| Capture | Original controller result | Accepted evidence | Native / sampler exits | Both groups cleaned |
|---|---|---|---|---|
| Memory | Failed header-path validation, unchanged | Separate additive target/protocol recovery | 0 / 0 | Yes |
| RocksDB | One-role observations complete | Strict target and complete protocol audit | 0 / 0 | Yes |

The Memory refusal arose because the operating system redacted the reported artifact path. Recovery required the sealed artifact, sole main-image UUID and load address, owned PID/start/full argv, launch second and exact accepted redaction grammar together; a basename match alone was insufficient. The original failed source, binding and proofs remain unchanged. RocksDB used the strengthened validator with a fresh authorization and role; it did not resume or replay Memory.

| Capture | Thread trees | Observations per thread | All-thread observations | Reported collapsed table | Recovered below threshold |
|---|---:|---:|---:|---:|---:|
| Memory | 24 | 21,125 | 507,000 | 506,560 | 440 |
| RocksDB | 59 | 20,848 | 1,230,032 | 1,229,491 | 541 |

Counts are sampled thread-stack observations, not CPU time or elapsed-time fractions. For each printed call-tree node, subtracting its immediate children produces nonnegative terminal-frame residuals. Under a fixed broad symbol-family precedence, these residuals sum exactly to every thread root and to the all-thread totals. The collapsed top-of-stack table uses a count threshold of five; the below-threshold values above are recovered from the complete trees. All per-thread counts and denominators are preserved in the companion JSON.

| Terminal-frame family | Memory, all threads | Memory, selected test | RocksDB, all threads | RocksDB, selected test |
|---|---:|---:|---:|---:|
| Wait or event syscalls | 485,163 | 49 | 1,209,063 | 56 |
| Allocation and memory operations | 8,184 | 8,182 | 7,839 | 7,828 |
| Revisioned record/value decoding | 3,676 | 3,676 | 4,745 | 4,745 |
| Clone or drop | 3,555 | 3,550 | 3,070 | 3,070 |
| Generic Number arithmetic | 1,878 | 1,878 | 1,713 | 1,713 |
| KnnTopK / generic Distance::compute | 989 | 989 | 920 | 920 |
| Indexed-vector / priority-queue symbols | 559 | 559 | 465 | 465 |
| Hashing / JSON serialization | 414 | 414 | 352 | 352 |
| Other engine expression/value work | 451 | 449 | 348 | 347 |
| Other resolved frames | 1,415 | 1,377 | 1,516 | 1,351 |
| Unresolved frames | 716 | 2 | 1 | 1 |

These terminal categories are disjoint under the classifier, not semantic cost centers. Allocation, copying or cloning can occur below higher-level engine work. Indexed-vector and priority-queue symbols are broad token families and do not establish an ANN path or identify generic exact-query magnitude cost.

| Family present anywhere on selected-thread stack | Memory / 21,125 | RocksDB / 20,848 |
|---|---:|---:|
| Wait or event syscalls | 49 | 56 |
| Allocation and memory operations | 8,502 | 8,081 |
| Revisioned record/value decoding | 11,913 | 12,734 |
| Clone or drop | 3,953 | 3,393 |
| Generic Number arithmetic | 1,878 | 1,713 |
| KnnTopK / generic Distance::compute | 4,628 | 4,258 |
| Indexed-vector / priority-queue symbols | 640 | 545 |
| Hashing / JSON serialization | 494 | 428 |
| Other engine expression/value work | 19,473 | 19,446 |
| Other resolved frames | 20,936 | 20,656 |
| Unresolved frames | 191 | 195 |

**Stack-family presence is nonadditive.** A single reconstructed stack can contain decoding, allocation, generic Number arithmetic and KnnTopK simultaneously; family counts must not be added or converted into exclusive CPU shares. The two captures have different observed thread counts and sampled windows, so raw counts are not a backend cost comparison.

Memory retains 716 unresolved terminal observations, including 2 on the selected test thread; an unresolved ancestor or terminal appears in 191 selected-thread stacks. RocksDB retains 1 unresolved terminal observation and 195 selected-thread stacks with an unresolved ancestor or terminal. Resolved terminal frames can have unresolved ancestors; both counts remain explicit.

Both captures include revisioned record/value decoding, allocation/copying, cloning, KnnTopK or generic Distance::compute, and generic Number arithmetic. Neither retained report resolves a separate named cosine or magnitude symbol. This optimized debug0 artifact can inline those helpers; absence of a name is not evidence that the work is absent. Generic Number or indexed-vector dot frames do not isolate query-magnitude recomputation or quantify a scalar-cache benefit.

| Capture | Sampler receipt interval, seconds | Native receipt interval, seconds | Observed category span |
|---|---:|---:|---|
| Memory | 30.574082 | 58.186727 | 0–10 |
| RocksDB | 31.033862 | 67.844631 | 0–9 |

These durations and category spans use watcher receipt observations, not query execution timestamps. The sampler aggregates its whole mixed slice over all target-process threads. Cursor spans can disclose observed references and arms, but they cannot assign individual stacks to an arm, query, category, exclusive stage or between-call phase.

| Capture | B/M/D completed untimed references observed | B/M/D completed measured calls observed | Additional started call observed |
|---|---|---|---|
| Memory | 11/11/11 | 229/230/230 | B |
| RocksDB | 10/10/10 | 205/204/204 | M |

The complete native roles continued beyond these sampler cursor spans and each retained all 1,134 instrumented calls. The cursor counts above are a mixed-workload context summary, not per-arm CPU evidence or a partial-latency sample set.

The root-reported host is an arm64 Mac with 48 GiB RAM and 18 logical CPUs. Normal desktop/shared services remained active; authoring, audits, mocks and competing compilation were paused during each root-owned profiling window. These conditions do not establish quiet CPU, representative production load or a hardware/scheduling cause. Waiting/background threads, profiler overhead and observer receipt lag remain in scope.

The frames suggest a next targeted diagnostic that distinguishes record/value decoding and allocation from generic exact-KNN/Number work while preserving the complete result contract. They do not select a production patch, rank a query-norm cache benefit, explain the untraced first/p95/max failures, or establish any optimization acceptance. Any follow-up remains separate from this mixed-stack screen and from the rejected wider qualification.

[Paired counts, per-thread denominators and protocol limits](named-field-hydration-cpu-stack-screen-115.json)
