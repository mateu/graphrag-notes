# Gardener candidate quality and throughput (#230)

Aggregate counts only. No note text, entity values or excerpts are recorded here.

## Method

- **Corpus.** An isolated restore of the newest portable backup of the shared
  corpus (schema 28, 9,830 notes, 882 sources). The backup predates the first
  live scan, so none of the 30 reviewed pairs had a proposal or decision. The
  scan therefore re-proposes them unchanged, and no offline evaluation hook was
  needed.
- **Service.** A scratch service on a loopback port, with a scratch `garden`
  principal. No live service, database, config or credential was used.
- **Scope.** The same scope as the first live scan: orphan anchors in record-ID
  order, 210 anchors in 6 dry-run pages of 35, default policy (floor 0.80,
  8 neighbours, 2 per endpoint, generic above 10 notes).
- **Labels.** The 30 candidates of the first live batch with the two agreeing
  blind reviews: 3 kept (2 `related_to`, 1 re-typed `derived_from`) and 27
  rejected (11 same-topic lineage logs, 10 duplicates or restatements,
  6 generic overlaps).
- **Baseline check.** The deployed binary and this branch with no new options
  produced identical receipts on this scope: 508 nominations, the same
  32 candidates (all 30 labelled pairs plus 2 unlabelled) and the same
  rejection counts as the live scan. The only differences are the new
  zero-valued counters and the actor.
- **Attribution.** Each labelled pair's rejection reason comes from a dry run
  scoped to one endpoint and the other endpoint's source, from each side.
  Gates apply in a fixed order (machine-dump anchor, same document, machine
  dump, near duplicate, then entity evidence), so each pair is counted once.

## Measured policy

```toml
[gardener.candidates]
near_duplicate_ceiling = 0.95
max_shared_span = 0.5
max_machine_line_share = 0.6
max_lineage_documents = 2
min_specific_entities = 2
textual_cues = true
similarity_batch_anchors = 64
```

## The 30 labelled pairs

| Reviewer category | Pairs | Produced before | Produced after | Rejected after, by reason |
|---|---:|---:|---:|---|
| kept `related_to` | 2 | 2 | **2** | none |
| kept, re-typed `derived_from` | 1 | 1 | **1** | none |
| same topic (lineage logs) | 11 | 11 | 0 | lineage template 5, too few specific entities 4, same document 2 |
| duplicate or restatement | 10 | 10 | 0 | near duplicate 3, too few specific entities 3, lineage template 2, machine-dump anchor 2 |
| generic overlap | 6 | 6 | 0 | same document 2, near duplicate 2, lineage template 1, machine-dump anchor 1 |
| **all rejects** | **27** | **27** | **0** | 27 of 27 |

Every reviewer-agreed keep is still produced. Both `related_to` keeps now carry
the `textual_cue` signal (`relative_date` and `answered_question`). The
`derived_from` keep passes on six lineage-specific shared entities.

## Same scope, before and after

| | Before (legacy policy) | After (measured policy) |
|---|---:|---:|
| Anchors examined | 210 | 210 (17 skipped as machine dumps) |
| Nominations | 508 | 415 |
| Candidates | 32 | 8 |
| Labelled keeps among candidates | 3 | 3 |
| Labelled rejects among candidates | 27 | 0 |
| Unlabelled (not yet reviewed) candidates | 2 | 5 |

Rejections after, by reason (they sum with the 8 candidates to the 415
nominations): generic entities only 212, near duplicate 113, same document 24,
no second signal 17, lineage template 12, machine dump 10, exact duplicate
content 10, too few specific entities 9, endpoint cap 0. Before: generic
entities only 433, no second signal 20, duplicate content 15, endpoint cap 8.

Seven of the eight candidates carry a textual cue (6 `relative_date`,
1 `answered_question`). The other is the `derived_from` keep, with shared
specific entities. The five unlabelled candidates are new to review. Projected
yield on this scope is therefore between 3/8 (38%, if all five are rejected) and
8/8. The first live batch's yield was 3/30 (10%).

## Latency

Each anchor's neighbour read was one exact cosine scan of the visible corpus.
On this restore that is about 0.47 s, of which about 0.35 s is the cosine
filter itself. The joined/origin exclusions cost under 1 ms. The cost does not
depend on how many anchors share a scan, so batching removes it.

| Run on the same 210 anchors | Elapsed | Per anchor |
|---|---:|---:|
| Deployed binary, legacy policy | 110.8 s | 528 ms |
| This branch, legacy policy | 102.9 s | 490 ms |
| This branch, quality gates, no batching | 94.7 s | 451 ms |
| This branch, legacy policy, `similarity_batch_anchors = 64` | 8.6 s | 41 ms |
| This branch, quality gates and batching | 8.3 s | 40 ms |

Batching changed no result. The batched and per-anchor runs returned identical
receipts: candidates, similarity values, rejections and cursors. A
35-anchor page takes about 1.4 s, so a 100-anchor page (which the live scan
cut at its deadline after about 72 anchors) needs two corpus passes, about 3 s.

## Sensitivity and limits

30 labels is a very small set. The thresholds are principled defaults, not
fitted values. An offline replay of the same 507 nominations gives the margins
below. The replay approximates the scan's text rules: under the measured
policy it agrees with the service on 29 of the 30 labelled pairs, and lets
through one duplicate that the service rejects.

- **Kept set unchanged (3 of 3)** for a near-duplicate ceiling of 0.93, 0.95
  or off; a shared span of 0.4–0.6; a machine-line share of 0.5–0.7;
  lineage documents of 2–4; and with the ceiling, span and dump gates all off.
- **`max_lineage_documents = 1`** loses the `derived_from` keep.
- **`max_lineage_documents` of 3 or 4** lets 2 same-topic pairs through.
- **`min_specific_entities = 1`** lets 4 same-topic pairs and 2 more
  duplicates through.
- **Without textual cues**, the entity rules reject both `related_to` keeps.

On the labelled pairs, most of the precision comes from lineage specificity
and the two-entity rule. The near-duplicate and machine-dump gates overlap
with them there, but on the full scope they are the first reason for
123 rejections and 17 skipped anchors. Each kept `related_to` pair depends on
a single cue. A larger reviewed batch is needed before trusting the yield
estimate.
