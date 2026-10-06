# Graph relevance validation (#93)

The provider-free fictional fixture compared patched-main retrieval with query-ranked graph seeds at 256 and 2,048 notes. Each of twelve cases ran graph off/auto/on with three warmed observations per policy. The fixture asserts bounds, uniqueness, current source visibility, source/time filters, deterministic ranked evidence, and accepted-only paths.

These are debug-build warmed medians on one Apple Silicon host under concurrent development work. They exclude external model calls and remote transport. Three observations do not establish p95 values or a latency guarantee; independent phase probes are not additive traces.

| Notes | Mentions | Case | Before auto (ms) | After auto (ms) | After off (ms) |
| ---: | ---: | --- | ---: | ---: | ---: |
| 256 | 0 | A-no-mentions-empty-edges | 104.18 | 117.48 | 97.70 |
| 256 | 12 | B-sparse-mentions-empty-edges | 116.02 | 136.66 | 96.31 |
| 256 | 256 | C-dense-mentions-empty-edges-broad | 232.16 | 318.43 | 84.95 |
| 256 | 256 | D-dense-mentions-empty-edges-source | 238.60 | 314.96 | 88.00 |
| 256 | 12 | E-sparse-mentions-accepted-edges | 146.12 | 157.61 | 91.10 |
| 256 | 256 | F-dense-mentions-accepted-edges-source-age | 278.63 | 350.20 | 84.71 |
| 2048 | 0 | A-no-mentions-empty-edges | 674.40 | 655.45 | 708.84 |
| 2048 | 12 | B-sparse-mentions-empty-edges | 704.65 | 676.40 | 635.17 |
| 2048 | 2048 | C-dense-mentions-empty-edges-broad | 1511.70 | 971.19 | 627.95 |
| 2048 | 2048 | D-dense-mentions-empty-edges-source | 1698.19 | 1018.21 | 646.43 |
| 2048 | 12 | E-sparse-mentions-accepted-edges | 752.89 | 712.59 | 648.57 |
| 2048 | 2048 | F-dense-mentions-accepted-edges-source-age | 1644.33 | 1119.20 | 669.69 |

Dense 2,048-mention graph cases improved in these observations; some smaller cases added work. Larger debug-build graph overhead still exceeds the initial 250 ms expansion target, so these results support boundedness and relevance work rather than a production latency claim. Release-profile service-phase acceptance remains a separate measurement.

Fictional relevance regressions explicitly cover a useful entity seed beyond the first ID page, a seed outside the requested hybrid candidates recovered by the bounded indexed lexical supplement, actual accepted-edge explanations, exact direct-title retention, homonymous scoped entities, alias compatibility, schema upgrade/old portable restoration, forced pilot idempotence, cancellation/resume, and unchanged unrelated mentions. Repeated remote edits also retract stale aliases without identity growth; shared-source chunk reprocessing, exact successors, filters and deletion retain only current mention alias evidence.

The frozen private calibration/holdout and a small explicitly selected typed/alias pilot must be accepted independently before broad extraction. Their raw queries, judgments, note text, IDs, and source paths are kept out of public artifacts.
