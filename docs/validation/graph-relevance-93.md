# Graph relevance validation (#93)

The provider-free fictional fixture compared patched-main retrieval (`287d8ea4`) with final candidate source `3dc616a09ce37465dd96a3298ddf3e405ab62a54` at 256 and 2,048 notes, using matching standalone-agent feature settings. Each of twelve cases ran graph off/auto/on with three warmed observations per policy. The fixture asserts bounds, uniqueness, current source visibility, source/time filters, deterministic ranked evidence, and accepted-only paths.

These are debug-build warmed medians on one Apple Silicon host under concurrent development work. They exclude external model calls and remote transport. Three observations do not establish p95 values or a latency guarantee; independent phase probes are not additive traces.

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

Dense 2,048-mention graph cases improved in these observations; some smaller cases added work. Larger debug-build graph overhead still exceeds the initial 250 ms expansion target, so these results support boundedness and relevance work rather than a production latency claim. Release-profile service-phase acceptance remains a separate measurement.

Fictional relevance regressions explicitly cover a useful entity seed beyond the first ID page, a seed outside the requested hybrid candidates recovered by the bounded indexed lexical supplement, actual accepted-edge explanations, exact direct-title retention, homonymous scoped entities, alias compatibility, schema upgrade/old portable restoration, forced pilot idempotence, cancellation/resume, and unchanged unrelated mentions. Repeated remote edits also retract stale aliases without identity growth; shared-source chunk reprocessing, exact successors, filters and deletion retain only current mention alias evidence.

The frozen private calibration/holdout and a small explicitly selected typed/alias pilot were evaluated independently as described below. Their raw queries, judgments, note text, IDs, and source paths remain private. Broad extraction remains outside this issue.


## Frozen corpus and selected pilot acceptance

The [sanitized aggregate](graph-relevance-93.json) records the exact compiled
source, binary and frozen-suite hashes, policy/split counters, and pilot
integrity checks. The frozen suite has 50 cases: 34 calibration, 16 reserved
holdout, 46 answerable and four reviewed no-answer cases. Baseline, candidate
before pilot, and candidate after pilot each completed 200 policy cases with
zero retrieval/readback failures.

Relative to the earlier graph implementation, hybrid/auto and hybrid/on each
improved reciprocal rank of judged positives in four calibration cases and
recall of judged positives in three, with no observed worsening of those
metrics. Each graph policy changed nine held-out rankings, while judged reciprocal-rank
and recall metrics stayed unchanged. Direct keyword/off and hybrid/off
remained unchanged. Candidate graph modes matched hybrid/off on these
limited judged metrics: calibration reciprocal rank 0.71354 and recall
0.67813; held-out reciprocal rank and recall both 0.85714. This demonstrates
removal of observed graph regressions, without an observed advantage over
hybrid/off in this suite.

The explicitly selected eight-note pilot on an isolated restored corpus added
117 entities and 124 mentions:
37 Technology, 16 Organization, 14 Project, 32 Concept, three Other, four
Person and 11 Location. Alias validation and the eight-alias bounds passed.
All 458 historical mentions were preserved; note payloads, sources,
generations, embeddings and existing portable remote upload jobs were
unchanged. The corpus had zero accepted note edges before and after.
Local extraction jobs were legitimately created/completed and are excluded
from portable archives, which export only remote upload jobs.

The first extraction took 63.25 seconds; the repeated selection took 149 ms
under the configured cache. Entity IDs and logical mention endpoints/metadata
were identical on repeat. Atomic forced replacement recreates physical mention
row IDs, so idempotency refers to logical evidence rather than literal row IDs.
Verified private backups were retained before extraction, after extraction and
after repeat. The pilot changed no rankings, judged reciprocal ranks or judged
recall in the frozen suite. It validates typed, bounded, reproducible evidence;
broad extraction needs additional measured retrieval value.

The suite is delegated-agent curated, with no observed human judgment review.
The graph author reviewed provisional corpus curation before this assignment;
the final frozen cases and pilot selection were not examined during graph
implementation. The holdout is therefore a reserved regression set, not a
statistically blind population estimate. Judgments identify partial known
positives, and unseen hits remain unjudged: overall precision/NDCG is withheld
where not fully judged. Hybrid policies still returned hits for the reviewed
no-answer cases; keyword/off remained empty. These limits prevent claims of
universal relevance gains or reliable hybrid abstention. Optimized service-phase
latency acceptance is tracked separately by #95.
