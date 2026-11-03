# Graph entity matching: hubs, duplicates and latency (#216, part b)

This note gives aggregate counts only. It records no record IDs, query text,
titles, snippets, source paths or credentials.

## Protocol

- **Data:** an isolated restore of the verified pre-deploy portable backup
  taken minutes before the `5619a92` deploy: schema 28, 9,830 notes, 92,729
  entities, 114,731 mentions and 32 accepted reviewed edges.
- **Service:** each variant ran as a scratch loopback service with a scratch
  read-only credential. The live service and database were not used.
- **Runner:** the unchanged runner (`runner_sha256` `b96bcc10…0e7e`), with the
  frozen 8-question suite in both cap modes and the 12-edge probe with the cap
  off. Policies were off, on, on, off, at `max_chunks=2`,
  `max_total_tokens=1000` and `max_chunk_tokens=400`.
- **Held-out set:** the owner-approved suite v2 (22 items), with the same
  policies and budgets and the cap off.
- **Controls:**
  - The deployed `5619a92` binary on the restore reproduced the live
    post-deploy rerun in 112/112 rounds, byte for byte.
  - A local experiment build with every switch off reproduced that control in
    112/112 rounds. It was rebuilt five times and reproduced it every time.
- **Latency:** phase timings come from the scratch services' `graphrag_usage`
  lines, 56 graph-on calls per variant. They are compared within one build
  unless stated otherwise.

## Where entity matching spends its time

- **Exact tier:** the whole-query tier never matches a sentence query. It
  costs about 0.3 ms.
- **Contained-phrase tier:** this is where the time goes.
  - Every contiguous query n-gram (12–47 terms, so up to about 1,100
    n-grams) is looked up in `mention_lexeme` and `entity_lexeme`.
    `EXPLAIN FULL` shows one `IndexScan` per n-gram under a
    `UnionIndexScan`. Common words return hundreds of rows each.
  - This yields 75–3,280 distinct candidate entities per call (p50 628,
    p95 2,191), and only 4 are kept.
  - The lexeme lookup itself costs p50 27 ms and p95 40 ms.
  - **Hydration** reads every candidate's mention rows (one indexed
    `mentions WHERE out = $entity` probe each, in 128-statement batches),
    then the visibility fields of every endpoint note. It costs p50 97 ms,
    p95 349 ms and at most 451 ms.
  - **Ranking** is one SELECT over every candidate with its alias and rank
    maps. It costs p50 22 ms, p95 179 ms and at most 377 ms.
- **Mention seeding** after matching costs about 2.7 ms. Mention fan-out is
  not a latency cost.
- **No matched hubs:** every matched entity has a note frequency of 1–4. No
  matched entity is a hub in the Gardener's sense (more than 10 notes). The
  hubs are lexemes: common words that index hundreds of entities.

## Rank-limited composition

Over the 56 graph-on rounds, 190 graph candidates were rank-limited: 178 were
hop-0 entity-mention seeds and 12 were accepted-edge neighbors. Within the
window, 36 mention seeds and 50 edge neighbors were kept. 28 of the cut
mention seeds share a source with another cut mention seed.

## Variants

Every variant is budget-neutral and affects graph-on only. The latency
columns show graph entity matching and graph-on total p95, in ms.

| Variant | Graph-on output vs control | rank_limited (nocap / cap1 / probe) | Cut: mention seeds / edge neighbors | Entity matching p95 | Total p95 |
| --- | --- | --- | --- | --- | --- |
| Switches off (same build) | identical, 112/112 rounds | 56 / 56 / 78 | 178 / 12 | 465–502 | 1,550–1,723 |
| H1+H2: hub cutoff 10 plus IDF down-weighting | identical; inert, no matched entity exceeds 10 notes | 56 / 56 / 78 | 178 / 12 | 469 | 1,546 |
| H3: mention fan-out cap 1 per entity | same chunks; 12/16, 12/16 and 20/24 on-rounds keep identical annotations | 50 / 50 / 68 | 156 / 12 | 477 | 1,554 |
| H4: same-source mention-seed dedupe | same chunks; 14/16, 14/16 and 20/24 identical | 54 / 54 / 70 | 166 / 12 | 476 | 1,546 |
| H5: skip matching without specific terms | identical; inert, every query has specific terms | 56 / 56 / 78 | — | 471 | 1,570 |
| L1: drop stop-word-only n-grams | identical | 56 / 56 / 78 | — | 519 | 1,634 |
| H6a: drop n-grams in more than 10 notes | same chunks; 4/16, 4/16 and 16/24 identical; annotating falls from 8 to 2 | 64 / 64 / 88 | 208 / 8 | 53 | 1,143 |
| H6b: drop n-grams in more than 50 notes | same chunks; 14/16, 14/16 and 22/24 identical | 56 / 56 / 80 | 180 / 12 | 78 | 1,151 |
| **L3/L4: staged exact evaluation (this change)** | **identical, 112/112 rounds** | **56 / 56 / 78** | **178 / 12** | **110–233** | **1,229–1,327** |

- **Results common to every variant:**
  - All variants keep the 5 probe gains on the same permitted paths.
  - None displaces a support chunk or adds an unexpected chunk.
  - Graph-off is byte-identical, 56/56 off rounds each.
- **H3 and H4:** they trim rank-limited mention seeds but return the same
  chunks at no latency gain. At this budget, the seeds they remove never
  competed for a slot.
- **H6:** it is fast, but it changes which entities match. At the Gardener's
  threshold it removes 6 of 8 suite annotations. Its cutoff would be tuned to
  this corpus, so it is not pursued here.
- **L3 and L4:** L3 stages mention reads and note eligibility. L4, which is
  this change, reads every candidate's mention rows first and stages only
  endpoint eligibility and ranking. The safety fallback is then decided
  exactly as before.

## This change

- **Ordering:** contained-phrase candidates are evaluated in descending order
  of an upper bound on their match specificity. The bound is the longest
  indexed query phrase that produced them.
- **Stages:** 64 candidates first, then four times as many per stage.
- **Stop rule:** evaluation stops once the capped page is full and its last
  row is strictly more specific than any unevaluated bound. Per-entity
  evidence does not depend on the other candidates, so the page equals the
  full evaluation.
- **Fallback:** an unsafe mention, entity or source shape anywhere among the
  candidates keeps the original path, including its observable errors.
- **Scope:** exact and prefix tiers are unchanged.
- **Typical cost:** 46 of 56 calls finished after one stage and 10 after two.

## Results for this change

- **Control vs the clean build:**
  - 112/112 rounds are byte-identical to the switches-off build.
  - Tokens are identical, and so are the counters: added / annotating /
    rank_limited are 2/8/56, 2/8/56 and 2/34/78.
- **Graph entity matching:**
  - Within the experiment build, p95 fell from 473 to 233 ms (−51%) and p50
    from 139 to 101 ms.
  - The clean release build measured p95 227 ms and p50 81 ms.
- **Graph-on total:** p95 fell from 1,550 to 1,284–1,327 ms, never worse.
  Graph-off totals are unchanged within noise.
- **Suite v2:** see the next section.

## Held-out suite v2

Live `5619a92` baseline, the deployed binary on the restore, and this change
all ran the 22-item approved suite with the cap off.

- **Byte identity:**
  - The restore control matched the live baseline in 44/44 off and 44/44 on
    rounds.
  - This change matched both in 44/44 off and 44/44 on rounds.
- **Statuses:** every per-item status is unchanged. This change neither adds
  nor fixes a guard violation.
- **Latency:** graph entity matching p95 fell from 489 to 185 ms, and
  graph-on total p95 from 1,543 to 1,254 ms.

| Item | Category | Graph-off support | Graph-on gain | Status (control = this change) |
| --- | --- | --- | --- | --- |
| E02, E04, E06, E07, E09, E11 | edge | miss | no | fail |
| Q4c | edge (corrected) | miss | **yes** (`derived_from` outbound, 1 hop) | needs_judgement |
| E05 | base search | present | — | fail |
| E08 | base search | present | swap (support kept) | fail |
| N1 | derived-duplicate guard | present | **violated**: swap | fail |
| N2 | seed-only guard | present | **violated**: swap | fail |
| N5, N6 | derived-duplicate guard | present | nothing added | fail |
| N3, N4, B5 | base search, known miss | miss | — | fail |
| B1–B4, Q5r, Q7r | base search | present | — | fail |

### The N1, N2 and E08 swaps

In every graph-on round of all three items, one chunk changes. A direct hit
that graph-off ranks below the second slot reaches it over a reviewed
`derived_from` outbound first hop from the support seed, which stays in
context. #221 lifts that neighbor to its anchor's strength, and it displaces
a chunk that is not a support chunk. `graph_candidates_added` is 0 because
the neighbor is also a direct hit, so it is counted as annotating.

| Item | vs seed: 5-shingle Jaccard / containment | Token Jaccard | Longest shared span (tokens) | Same source as seed |
| --- | --- | --- | --- | --- |
| N1 | 0.000 / 0.000 | 0.110 | 2 | no |
| N2 | 0.000 / 0.000 | 0.104 | 1 | no |
| E08 | 0.011 / 0.022 | 0.265 | 5 | no |

These measures cover the returned snippets (about 40–70 tokens). By them,
**none of the three neighbors is a near-duplicate of its seed**. They are
derived notes that restate the same content in different words, so a
lexical duplicate filter would not catch them. Suppressing these swaps means
deciding whether a `derived_from` step from a derived note to its origin
should lift at all. That is a ranking change, separate from this
latency-only change.

### Counter proposal

`graph_candidates_added` counts only chunks that are not direct hits, so a
lift-induced swap shows as `annotating`. Proposal: add
`graph_candidates_promoted`, which counts packed chunks whose graph-on score
exceeds their own direct score and that would not be packed graph-off. It
would read 1 per on-round for N1, N2 and E08 and 0 elsewhere on v2. The
existing counters stay as they are.

## Tests

- **Differential test against the oracle:** `graph_phrase_staging_matches_full_evaluation_for_hubs_duplicates_and_ties`
  compares the staged path with the unchanged original-order SQL. The fixture
  covers:
  - hub entities, 150 same-spelling duplicates, and ties across stage
    boundaries;
  - a hidden mention inflating one candidate's bound;
  - every scope and limits from 1 to 200;
  - repeated runs;
  - an unsafe alias shape, where the result, including its error, must match.
- **Sabotage:** relaxing the stop rule to `>=` makes the test fail at
  `limit=63`.
- **Agent negatives:**
  - `hub_entity_mention_seeds_stay_out_of_the_fixed_budget` checks
    graph-off equivalence, determinism and budget.
  - `duplicate_entity_records_do_not_crowd_out_reviewed_support` checks
    that the reviewed support still enters with its permitted path.
  - The existing hub, copy and uncorroborated-bridge negatives are unchanged
    and pass.

## Limitations

- **Corpus and sample:** one corpus, 2 graph-on rounds per item, and about
  56 graph-on calls per variant for latency percentiles.
- **Latency only:** this change makes matching faster. It does not discount
  hub or duplicate mention seeds. On this data, none of H1–H4 changed a
  returned chunk at the fixed budget.
- **Guard violations:** the N1/N2 swaps remain. They are not lexical
  near-duplicates, so they need a separate ranking decision.
- **Open hub path:** a query entity with many mentions can still fill the
  `max_seed_notes` frontier before hybrid anchors are added. The hub negative
  documents this rather than changing it.
