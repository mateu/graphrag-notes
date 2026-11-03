# Hot retrieval statement inventory (#179)

**Status: incomplete / AC-1 blocked on existing scan paths.** A green subset
check must not be read as proof that every hot statement is index-backed.
No production query, index, or migration definition is changed by #179.

The provider-free unit fixture creates 256 notes, entities, mentions, messages,
conversations and rows of each provenance/edge table. One entity matches the
needle among the distractors. Vectors have 1024 dimensions. SQL and native
bindings are captured from the **actual Repository calls** by a `cfg(test)`-only
connection observer, then replayed with EXPLAIN FULL. This avoids a hand-copied
query staying green after production SQL changes. All operator assertions reuse
`graph_retrieval_batch_tests::mention_plan_operators`; there is no second plan
parser. Plan output includes actual execution metrics.

| Entry / actual statement | Guard or evidence | Status |
| --- | --- | --- |
| Notes lexical all-term outer hydration SELECT | `indexed_hot_retrieval_statements_keep_their_access_paths` | All three `idx_note_search_*` FullTextScans; no TableScan |
| Notes lexical all-term inner candidate SELECT | Same fixture, extracted actual inner SQL, original bindings | Independently explained; same three indexes |
| Notes lexical any-term outer hydration SELECT | Same fixture; deliberately missing query term forces backfill | Same three indexes; no TableScan |
| Notes lexical any-term inner candidate SELECT | Same fixture, independently explained | Same three indexes; no TableScan |
| Notes exact vector SELECT | `exact_vector_plan_policy_tests` | **EXPLICIT EXCEPTION:** exact KnnTopK over note TableScan; [policy/qualification](exact-vector-plan-policy.md) |
| Messages lexical SELECT | Indexed hot fixture | `idx_message_content` FullTextScan; no TableScan |
| Messages exact vector SELECT | `exact_vector_plan_policy_tests` | **EXPLICIT EXCEPTION:** exact KnnTopK over message TableScan; [policy](exact-vector-plan-policy.md) |
| Conversations lexical SELECT | Indexed hot fixture | `idx_conversation_summary` and `idx_conversation_title` FullTextScans; no TableScan |
| Conversations exact vector SELECT | `exact_vector_plan_policy_tests` | **EXPLICIT EXCEPTION:** exact KnnTopK over conversation TableScan; [policy](exact-vector-plan-policy.md) |
| `graph_notes_by_ids` bound record-array SELECT | Indexed hot fixture | SourceExpr with one returned record, no TableScan; primary-key dereference has no named secondary index |
| Entity exact lexical candidates: mention_lexeme equality SELECT | `entity_lexeme_and_bound_mention_statements_reject_historical_scans` | Child of RETURN explained independently; `idx_mention_lexeme_lookup` |
| Entity exact lexical candidates: entity_lexeme equality SELECT | Same fixture | Child of RETURN explained independently; `idx_entity_lexeme_lookup` |
| Entity contained-phrase candidates: mention_lexeme IN SELECT | Same fixture | Independently explained with actual phrase n-grams; `idx_mention_lexeme_lookup` |
| Entity contained-phrase candidates: entity_lexeme IN SELECT | Same fixture | Independently explained; `idx_entity_lexeme_lookup` |
| Entity prefix candidate range SELECT | Same fixture | Actual prefix lower/upper bindings; `idx_entity_lexeme_lookup` |
| Entity candidate evidence: one mentions equality SELECT per bound entity | Same fixture, each statement independently explained | `idx_mentions_entity_note`; no mentions TableScan |
| Entity fast-path ranking outer SELECT (all three tiers) | Same fixture | Bound record set / SourceExpr; no TableScan |
| Entity fast-path ranking inner entity hydration SELECT (all three tiers) | Same fixture, actual inner SQL independently explained | Bound record set / SourceExpr; no TableScan |
| Entity eligibility LET note hydration, source hydration, in-memory filtering | Existing behavior fixtures; plan coverage pending | **INCOMPLETE:** multi-statement LET/RETURN context needs independent bound replay; outer ranking cannot prove it |
| Entity malformed/compound evidence fallback, >64-word authoritative path | Existing behavior fixtures; plan coverage pending | **INCOMPLETE:** deliberate fallback may scan; cannot infer indexed coverage from normal fast path |
| Ranked graph seeds: query-owning alias mentions SELECT | Entity/mentions fixture | `idx_mentions_entity_note`; no TableScan |
| Ranked graph seeds: direct-ranked mentions SELECT | Same fixture | `idx_mentions_entity_note`; no TableScan |
| Ranked graph seeds: non-ranked fallback mentions SELECT | Same fixture | `idx_mentions_entity_note`; no TableScan |
| Unranked/empty-query graph seed batch SELECT | Existing behavior fixtures | **INCOMPLETE:** not yet added to plan gate |
| Accepted edge expansion: supports/contradicts/related_to/derived_from, inbound/outbound/both | `edge_frontier_plan_tests::accepted_edge_frontiers_use_indexes_in_every_direction`, actual `graph_edge_batch_sql` statements | Guarded by v27 endpoint indexes in all twelve variants; [separate migration, behavior and hydration qualification](193-edge-frontier-indexes.md) |
| Provenance: note_from_conversation membership SELECT | Gap diagnostic | **BLOCKED:** TableScan, no `in` index |
| Provenance: note_from_message membership SELECT | Gap diagnostic | **BLOCKED:** TableScan, no `in` index |

## Negative controls (AC-2)

The entity/mentions fixture applies the **same indexed-path predicate** to a
positive canonical entity equality plan and a `WITH NOINDEX` variant, and to a
positive client-bound mention plan and an actual nested `$parent.id` plan inside
an entity SELECT. The de-indexed/correlated plans must fail the predicate; merely
checking a plan string for SQL text is not a negative control. The positive
controls prevent an always-false guard from passing this demonstration.

## Engine visibility limits

RETURN array concatenation does not expose its two lexical SELECT children;
they are explained separately with their captured parameters. Direct record
hydration uses SourceExpr rather than a named IndexScan. Blocks can retain only
DynamicScan, not a runtime-created IndexScan child; a green direct-bound probe
cannot establish a nested block's access plan. Existing graph alias tests retain
that distinction; uncovered fallback paths remain explicitly incomplete here.

## Reproduce existing gaps (not a CI no-scan gate)

```sh
cargo test --locked -p graphrag-db --lib \
  repository::hot_retrieval_plan_tests::existing_scan_gap_diagnostic \
  -- --exact --ignored --nocapture
```

This deliberately ignored diagnostic emits plans; it does **not** satisfy AC-1.
The three exact-vector statements now have a narrow [explicit policy exception](exact-vector-plan-policy.md)
and a non-ignored drift/exactness guard. Other existing scan paths remain tracked
separately. Do not relabel scans as an engine visibility exception: EXPLAIN
exposes these TableScans directly.

## Recorded evidence and follow-ups

[Full generated plans](evidence/179-hot-plans.json) include 51 plans from the
indexed-path guards, historical negative controls, and existing-scan diagnostic.
The confirmed scan paths and policy decisions are tracked separately:

- [#191: exact-vector policy/access paths](https://github.com/mateu/graphrag-notes/issues/191);
- [#192: provenance endpoint membership](https://github.com/mateu/graphrag-notes/issues/192);
- [#193: accepted-edge frontier expansion](https://github.com/mateu/graphrag-notes/issues/193).

AC-4 non-matching proof: temporary [PR #189](https://github.com/mateu/graphrag-notes/pull/189),
[successful 5-second job](https://github.com/mateu/graphrag-notes/actions/runs/41567747346/job/125693256221).
The `No migration changes` step succeeded, while toolchain/cache/test steps were
skipped. The PR was closed and its temporary branch deleted after qualification.
The matching-path job is exercised by #188's documentation-only edit to the
registry-derived helper; final run results belong in the PR qualification record.
