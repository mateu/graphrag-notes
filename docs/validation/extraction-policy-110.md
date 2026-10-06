# Retained extraction-policy migration rehearsal (#110)

This proof covers an isolated restored corpus, not a production migration or a
release latency benchmark. The runtime was an isolated development debug build compiled
from `c372dd7b90ead6df94b469f36279f7b59eb0219d`, with compiled inputs unchanged
at source `2e2ad2203aed8fab6b3c39499e4ff9512a370be1`. Its binary SHA-256 was
`725e130333bfeb990e1c7613b9310f597bafab6b873442c4187bfe876b78ca3f`.
Schema 20 was observed before and after the quality roles. The configured
`bge-m3:latest` embedding and `phi4-mini:latest` extraction model identities were
checked against provider metadata, with no model, endpoint, permission, or
collection-default change. Doctor performed two declared embedding dimension
probes per quality role; it made no entity-generation probe.

The coherent indexed-memory copy contained 790 documents and 809 registered
parts. An explicit reviewed plan selected only eight retained extraction-policy
parts with unchanged originals. The collection default remained
`extract_entities=false`; the other 801 registered parts were preserved. This
work did not change the production corpus or official client registry.

## Migration and preservation

All eight durable jobs completed and each selected source generation advanced
once. The eight retained policies became current; no automatic recovery,
cancellation, or broad extraction sweep was performed. The verified portable
before/after payloads establish:

- 40 one-to-one successor chunk lineages preserve content, source identity,
  chunk locations and ordinals, and actual serving F32 vector bits. Successor
  canonical IDs legitimately change; selected stored F64 representation is not
  claimed identical.
- All 9,740 unselected notes preserve their full fields, stored F64 vector bits,
  and serving F32 bits. All 806 unselected corpus source records and 29
  unselected mention records are unchanged. Corpus source records and registered
  collection parts have distinct denominators.
- All 374 previous entity IDs remain; the reviewed extraction produced 386 new
  entity IDs. Selected mentions changed from 438 to 426. These are structural
  counts, not an entity-accuracy score or a claim that every entity field is
  byte-identical.
- One existing unselected relationship and all audited relation-table counts
  remain unchanged. No selected chat-provenance relationship existed in this
  real rehearsal; fictional lifecycle fixtures cover that preservation path.

Two subsequent ordinary refreshes each made exactly one authenticated status
read and 809 source reads, with zero upload or job calls and zero inference.
The owned temporary service stopped after every role and endpoint cleanup was
checked. Configuration, `.env`, model metadata and the unselected registry
entries stayed unchanged.

The published rc.3 binary refused the private schema-20 database before opening
it for backup. Fictional tests additionally cover bounded staged checkpoint
backup/restore and resume, provider failure, cancellation, worker interruption,
policy and source races, exact replay, transactional promotion rollback, and
manual relationship deletion after a failed promotion. Rollback requires a
matching pre-upgrade corpus and collection-state restore; replacing only the
binary is insufficient.

## Paired retrieval quality

Both roles completed all 200 policy cases with zero execution or pinned-readback
failures, and each performed 790 full record readbacks. All 200 full record
arrays, readback arrays and metric rows are exactly equal before and after.
Canonical and lineage-normalized ranked-ID lists both have zero changes.
Both roles use the frozen 50-question suite across four policies, with 34
calibration questions and 16 held-out questions. The successor map validates all
40 chunk lineages but matches none of the frozen judged IDs, so the after suite
is byte-identical to the before suite. Questions, splits, relevance grades,
filters, formulation and judgment denominators remain unchanged.

This frozen suite exercises corpus-wide retrieval and contains no judged
positives or returned ranked results among the selected chunks. It cannot directly grade the new
extraction's accuracy. Reciprocal rank and recall refer only to the judged
positive set; unjudged hits keep precision and nDCG withheld. No-answer cases
are reported separately with empty/nonempty counts and false-positive results.
Other false-positive totals count only explicitly zero-grade results; unjudged
results remain unknown.

The following means are unchanged before and after. Reciprocal-rank and recall
denominators are 32 judged-positive calibration cases and 14 judged-positive
holdout cases per policy; the four no-answer questions are accounted for
separately. These are judged-positive metrics, not complete-corpus relevance
scores.

| Policy | Split | Judged-positive cases | Mean reciprocal rank | Mean recall |
| --- | --- | ---: | ---: | ---: |
| keyword-off | calibration | 32 | 0.244792 | 0.237500 |
| keyword-off | holdout | 14 | 0.428571 | 0.428571 |
| hybrid-off | calibration | 32 | 0.713542 | 0.678125 |
| hybrid-off | holdout | 14 | 0.857143 | 0.857143 |
| hybrid-auto | calibration | 32 | 0.713542 | 0.678125 |
| hybrid-auto | holdout | 14 | 0.857143 | 0.857143 |
| hybrid-on | calibration | 32 | 0.713542 | 0.678125 |
| hybrid-on | holdout | 14 | 0.857143 | 0.857143 |

Keyword returned no results for all four reviewed no-answer questions in both
roles. Each hybrid policy returned five results for each of those questions:
10 false-positive results in calibration and 10 in holdout, unchanged before
and after. These pre-existing no-answer failures remain visible; completing a
request is not a relevance-success claim. All known false-positive totals and
precision-withholding denominators are included in the aggregate.

The aggregate is [extraction-policy-110.json](extraction-policy-110.json).
Private source content, queries, selected identities, plan/policy/configuration
fingerprints, corpus fingerprints and raw reports are excluded.
