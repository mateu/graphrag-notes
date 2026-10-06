# Graph relevance validation (#93)

The final compiled source is `99321df2daa85fdd2333e1c762d84abef3ec39e7`,
with release binary SHA-256
`542c4b2ff0195a321df916497b385016253b7fa3b5f3ba381cbc0263239e98c6`.
The [sanitized aggregate](graph-relevance-93.json) records the frozen-suite hash,
paired quality counters, pilot integrity and final service observations.
Documentation-only commits after this source do not relabel or rebuild the binary.

## Final frozen corpus and selected pilot

The agent-curated suite has 50 cases: 34 calibration, 16 reserved holdout,
46 answerable and four reviewed no-answer cases. The frozen baseline's 200
policy cases had zero failures. Final candidate before-pilot and after-pilot
runs on a fresh schema-18 restore each completed 200 policy cases with zero
retrieval/readback failures. The unmerged schema-19 migration includes the
optional durable extraction scope; earlier provisional schema-19 copies remain
paired with their earlier artifact instead of bypassing the checksum fence.

Relative to the earlier graph implementation, hybrid/auto and hybrid/on each
improved reciprocal rank of judged positives in four calibration cases and
recall of judged positives in three, with zero known worsening. Each graph
policy changed 23 calibration and nine held-out rankings; held-out judged
reciprocal-rank and recall metrics stayed unchanged. Direct keyword/off and
hybrid/off remained unchanged. Candidate graph modes matched hybrid/off on
these limited metrics: calibration reciprocal rank 0.71354 and recall 0.67813;
held-out reciprocal rank and recall both 0.85714. This supports removal of
observed graph regressions without an observed advantage over hybrid/off.

The explicitly selected eight-note pilot added 117 entities and 124 mentions:
37 Technology, 16 Organization, 14 Project, 32 Concept, three Other, four
Person and 11 Location. Alias validation and the eight-alias bounds passed.
All 458 historical mentions were preserved. Full note content, embeddings,
sources, generations and the 810 existing portable remote upload jobs remained
unchanged. Exactly eight selected notes received the intended durable
`extraction_scope` field. The corpus had zero accepted note edges before and
after the pilot.

Repeat extraction preserved entity IDs, entity metadata and logical mention
endpoints/metadata. Atomic forced replacement recreates physical mention row
IDs; idempotency refers to stable identities and logical evidence. Both local
extraction jobs completed all eight items with zero failed items. Local jobs
are excluded from portable archives, which export remote upload jobs. The
pilot changed no suite rankings, judged reciprocal ranks or judged recall,
so it demonstrates bounded reproducibility without further measured ranking
benefit. Broad extraction needs additional measured value.

Unseen hits remain unjudged, and reciprocal rank/recall describe known judged
positives rather than complete population relevance. Most hybrid cases are
not fully judged, so aggregate precision and NDCG are not used as population
claims. Hybrid no-answer cases still return results; no abstention claim is
made. The suite was curated by Codex agents; human review was not observed.
The graph author reviewed provisional curation before this assignment, but
neither the final frozen cases nor the pilot selection was examined during
graph implementation. This is a reserved regression set, not a statistically
blind population estimate.

## Final optimized service observations

The final post-pilot release artifact completed 672 scheduled search cases
with applicable guarded readbacks and zero failed cases. One-client and
four-client runs used sequential homogeneous policy batches, with four first
observations and 80 warmed samples per policy/load group. Timings measure
search RPC only; MCP initialization, guarded readback and conversational model
time are excluded. First observations do not establish cold model/OS-cache
behavior. No same-RPC backend phase observations were available in this frozen
source, so these observations support no causal phase attribution.

| Concurrent clients | Keyword/off warmed p95 (ms) | Hybrid/off (ms) | Hybrid/auto (ms) | Hybrid/on (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 234.7 | 686.9 | 1032.3 | 1078.7 |
| 4 | 361.8 | 971.6 | 1471.3 | 1459.5 |

| Concurrent clients | Auto paired RPC-delta p95 (ms) | On paired RPC-delta p95 (ms) |
| ---: | ---: | ---: |
| 1 | 349.7 | 402.6 |
| 4 | 520.1 | 509.2 |

Each graph/load group had 80 paired observations and 60 judged-positive
reciprocal-rank comparisons, with zero judged regressions. Paired differences
are between matching graph/off searches in separate policy batches; they are
not isolated graph-stage timings. All four delta groups exceed the 250 ms
expansion budget. Concurrent local optimized and full-workspace compilation
changed host load during these sequential batches. The results support the
observed RPC distributions on this isolated snapshot and leave bounded graph
seed optimization as follow-up; they do not establish an unloaded-host or
cross-host latency guarantee. This change preserves graph policies and saved
client defaults; use explicit graph off when selecting the measured base
retrieval path.

## Fictional regression coverage

The full all-feature workspace run passed 765 tests with zero failures and
five existing ignored tests. Current-source format and all-target/all-feature
Clippy passed. Independent Codex source audit and the three new exact compiled
regressions passed; GitHub Codex review was green at the compiled source.
The only change after the full test run was a test-only conversion from typed
record-ID set keys to immutable canonical strings, followed by a focused rerun.

Fictional fixtures cover high-degree seeds, useful seeds below the requested
hybrid limit, exact direct-title retention, visibility/source/time bounds,
accepted-only paths, scoped homonyms, type/alias compatibility, cancellation
and resume. Source lineage survives earlier insertions/removals, validated
legacy adoption, zero-mention generations, forced reprocessing and portable
restore. An actual remote-upload fixture refreshes twice before enabling its
first extraction. A one-note seed cap finds the alias owner beyond the first
ID page and cannot lend its alias support to an unrelated shared-entity chunk.
Repeated edits, atomic rollback, source successors and deletion retain only
current mention evidence; manual/detached provenance stays independently scoped.

## Historical 3dc616a fictional timing checkpoint

These earlier observations compared patched-main retrieval (`287d8ea4`) with
checkpoint source `3dc616a09ce37465dd96a3298ddf3e405ab62a54` at 256 and
2,048 notes, using matching standalone-agent feature settings. Twelve fictional
cases ran graph off/auto/on with three warmed observations per policy. They
predate the durable lineage and mention-specific seed fixes. Earlier artifacts
and private acceptance archives remain paired with that source.

These are debug-build warmed medians on one Apple Silicon host under concurrent
development work, excluding external model calls and remote transport. Three
observations establish neither p95 values nor a production latency guarantee;
independent phase probes are not additive traces.

| Notes | Mentions | Case | Before auto (ms) | After auto (ms) | After off (ms) |
| ---: | ---: | --- | ---: | ---: | ---: |
| 256 | 0 | A-no-mentions-empty-edges | 104.18 | 110.64 | 91.97 |
| 256 | 12 | B-sparse-mentions-empty-edges | 116.02 | 127.27 | 90.02 |
| 256 | 256 | C-dense-mentions-empty-edges-broad | 232.16 | 295.34 | 158.06 |
| 256 | 256 | D-dense-mentions-empty-edges-source | 238.60 | 317.11 | 84.63 |
| 256 | 12 | E-sparse-mentions-accepted-edges | 146.12 | 162.82 | 91.25 |
| 256 | 256 | F-dense-mentions-accepted-edges-source-age | 278.63 | 371.06 | 85.91 |
| 2048 | 0 | A-no-mentions-empty-edges | 674.40 | 716.81 | 692.49 |
| 2048 | 12 | B-sparse-mentions-empty-edges | 704.65 | 806.00 | 731.08 |
| 2048 | 2048 | C-dense-mentions-empty-edges-broad | 1511.70 | 1130.48 | 736.47 |
| 2048 | 2048 | D-dense-mentions-empty-edges-source | 1698.19 | 1070.39 | 741.06 |
| 2048 | 12 | E-sparse-mentions-accepted-edges | 752.89 | 711.52 | 644.22 |
| 2048 | 2048 | F-dense-mentions-accepted-edges-source-age | 1644.33 | 1051.18 | 648.01 |

The historical dense 2,048-mention cases improved while some smaller cases
added work; their debug graph overhead still exceeded the 250 ms target.
The final optimized acceptance above is the current source's service evidence.
Earlier pilot timing (63.25 seconds initially and 149 ms on cached repeat)
belongs to this historical checkpoint and is not a timing claim for the final
lineage implementation.
