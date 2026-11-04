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

## Cue precision (second reviewed batch)

The second capped live batch ran the measured policy above on 600 fresh orphan
anchors and produced 12 candidates. Two blind reviewers and the owner kept 1
and rejected 11 (10 same topic only, 1 generic overlap). Five of the rejects
were heartbeat or status entries nominated only by `relative_date`, with no
shared specific entity. A sixth heartbeat pair was carried by two shared
entities.
In those entries "yesterday" named a monitoring window or an aggregate log, not
the paired note. Three opt-in tightenings target that failure mode; see
[connection review](../connection-review.md#cue-precision-230-opt-in).

### Method

- **Corpora.** Isolated restores of two portable backups, one per label set:
  - the backup taken before the first live scan, where the 30 batch-1 pairs
    are undecided;
  - the backup taken after batch 1 and before batch 2, where the 12 batch-2
    pairs are undecided.

  Scratch services ran on loopback ports with a scratch `garden` principal.
  No live service, database, config or credential was used.
- **Baseline check.** The deployed binary with the live policy reproduced
  both live results exactly:
  - batch 2: 600 anchors, 12 candidates (the same 12 pairs), and the same
    rejection counts (generic entities only 539, near duplicate 261, same
    document 124, no second signal 53, duplicate content 49, too few specific
    entities 33, lineage template 26, machine dump 20);
  - batch 1: 210 anchors, 8 candidates, 3/3 keeps, 0/27 rejects.
- **Attribution.** Each labelled pair was run as a single-anchor dry run from
  each side, with the other endpoint's source as the only counterpart source.
- **Labels.** All 42 owner-reviewed pairs: 4 keeps (3 from batch 1, 1 from
  batch 2) and 38 rejects. Where the reviewers split on a reject reason, the
  first published table's reason is used.

### Measured policy

The measured policy above plus:

```toml
resolve_relative_dates = true
periodic_status_entries = 3
dedupe_counterparts = true
```

### The 42 labelled pairs

| Batch | Reviewer category | Pairs | Produced before | Produced after | Rejected after, by reason |
|---|---|---:|---:|---:|---|
| 1 | keeps (2 `related_to`, 1 `derived_from`) | 3 | 3 | **3** | none |
| 1 | same topic only | 11 | 0 | 0 | periodic status 8, same document 2, lineage template 1 |
| 1 | duplicate or restatement | 10 | 0 | 0 | near duplicate 3, machine-dump anchor 2, duplicate content 2, lineage template 2, too few specific entities 1 |
| 1 | generic overlap | 6 | 0 | 0 | same document 2, near duplicate 2, machine-dump anchor 1, lineage template 1 |
| 2 | keep (`related_to`) | 1 | 1 | **1** | none |
| 2 | same topic only | 10 | 10 | 4 | periodic status 6 |
| 2 | generic overlap | 1 | 1 | 1 | none |
| | **batch-2 rejects** | **11** | **11** | **5** | 6 of 11 |
| | **batch-1 rejects** | **27** | **0** | **0** | no regression |

The 6 batch-2 rejects removed are the 5 `relative_date` status pairs and the
entity-carried heartbeat pair. Each rule alone removes 5:
- 4 are removed by either rule;
- 1 only by the relative-date rule (its entry headings each occur once in their
  logs);
- 1 only by the periodic rule (no cue; two shared entities). The 5 batch-2 rejects still produced carry no textual cue:
log-warning boilerplate, an epic rollup listing child PRs, a PR heartbeat
followed by its merge note, and two pairs from consecutive sessions editing one
file. They need a different signal.

The batch-1 `relative_date` keep resolves (its "yesterday" sentence shares a
topic span with the counterpart and names no other time). The two
`answered_question` keeps and the entity-carried `derived_from` keep are not
affected.

### Same scope, before and after

| | 210 anchors (batch 1) before | after | 600 anchors (batch 2) before | after |
|---|---:|---:|---:|---:|
| Nominations | 415 | 415 | 1,117 | 1,117 |
| Candidates | 8 | **3** | 12 | **6** |
| Labelled keeps | 3 | 3 | 1 | 1 |
| Labelled rejects | 0 (+4 batch-2 rejects) | 0 | 11 | 5 |
| Unlabelled | 1 | 0 | 0 | 0 |

The 210-anchor scope's 8 "before" candidates include 4 batch-2 heartbeat
rejects, which the first scan had not reached at that cursor. All 4 are gone
after.

Rejections after, 210 anchors: generic entities only 118, near duplicate 113,
periodic status 112, same document 24, no second signal 14, duplicate content
10, machine dump 10, lineage template 6, too few specific entities 4,
duplicate counterpart 1.

Rejections after, 600 anchors: generic entities only 298, periodic status 287,
near duplicate 261, same document 124, duplicate content 49, no second signal
36, machine dump 20, too few specific entities 20, lineage template 15,
unresolved relative date 1.

The periodic rule is checked before entity evidence, so pairs it removes no
longer reach the generic-entity and lineage counters. Most of its count moves
from those counters.

### Latency

| Run | 210 anchors | 600 anchors |
|---|---:|---:|
| Deployed binary, measured policy (two runs) | 8.4–8.6 s (40–41 ms/anchor) | 18.1–18.9 s (30–32 ms/anchor) |
| This change, all three options (two runs) | 8.7–9.1 s (41–43 ms/anchor) | 19.4–20.0 s (32–33 ms/anchor) |

The periodic rule adds one indexed read per scan of the heading paths of the
sources of clock-stamped notes. The cost is about 1–2 ms per anchor, close to
run-to-run noise.

### Sensitivity and limits

- **`periodic_status_entries`**, with the other two options on:
  - **3 or 5** give the same labelled result: 4/4 keeps, 6/11 batch-2 rejects
    removed, 0/27 batch-1 regressions. The same-scope counts are 3 and 6. At 5,
    three of the six are caught by the relative-date rule instead.
  - **2 loses two batch-1 keeps.** Both keeps are entries whose clock-stamped
    heading template occurs twice in their own log. **The margin at 3 is one
    entry.** Do not set it below 3.
- **Each rule alone** keeps all 4 keeps:
  - `resolve_relative_dates` alone removes 5 of the 6;
  - `periodic_status_entries = 3` alone removes 5 of the 6;
  - `dedupe_counterparts` alone removes 1 same-scope candidate on 600 anchors
    (one of the two pairs whose targets are identical copies) and 2 on 210
    anchors. Single-pair attribution cannot show it, because each copy pairs
    on its own.
- **Fragility.** Each keep that rests on a cue still rests on a single cue.
  The batch-1 `relative_date` keep passes on one shared two-word span. The
  periodic rule depends on heading structure. Logs with timestamped entries
  but no clock time in the heading are not detected, and a one-off
  clock-stamped entry is never periodic. 42 labels are still a small set.
  Projected yield on the 600-anchor scope rises from 1/12 to 1/6, from one
  keep. A larger reviewed batch is needed before trusting it.
