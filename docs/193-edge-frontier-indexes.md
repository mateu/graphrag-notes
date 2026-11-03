# Accepted-edge frontier indexes (#193)

Refs #179. Schema v27 adds non-unique `in` and `out` indexes to each of
`supports`, `contradicts`, `related_to`, and `derived_from`. The existing
`graph_edge_batch_sql` is unchanged: outbound uses `in`, inbound uses `out`,
and both directions use the planner's two-index union, followed by the original
eligibility, visited, confidence, ordering and per-source/per-table limit.
There is no endpoint or relation ID conversion or data rewrite.

## Migration ordering

The migration registry requires contiguous versions. Sibling #192 reserves v26
`provenance_endpoint_indexes`; this branch includes that exact predecessor DDL
so v27 can be tested and installed safely. Reconcile/land #192 before #193 and
preserve v26's SQL checksum. #193's own migration is
`v027_accepted_edge_frontier` / version **27** / `accepted_edge_frontier`.
All existing migration SQL remains immutable. Ordinary index creation backfills
previously stored edges; the fixture populates schema25 before upgrading twice.

## Plan and behavior guards

The retrieval PR lane runs the separate `repository::edge_frontier_plan_tests`
module's named tests and the existing independent legacy-membership differential:

- `accepted_edge_frontiers_use_indexes_in_every_direction`: 256 pre-migration
  edges per table; alternating inbound/outbound edges; actual emitted SQL;
  twelve `EXPLAIN FULL` plans. Each asserts its direction's endpoint indexes
  and rejects every `TableScan`. Both-direction plans contain `UnionIndexScan`.
- `indexed_edge_frontiers_match_scans_for_native_ids_and_self_loops`: all four
  tables and three direction modes, page caps 1/5, integer/string endpoint IDs,
  numeric/string/array/object relation IDs, self loops, null confidence and
  tied confidence, visited exclusion; compares indexed results with `WITH NOINDEX`.
- `edge_batches_preserve_pre_limit_filters_direction_confidence_cycles_and_sources`:
  independent legacy SQL guards filtering before limits, multiple frontiers,
  source visibility/generation, date/URI scopes, dangling/compound/range IDs,
  confidence, direction and visited behavior.

[Generated plans](evidence/193-edge-frontier-plans.json) include all twelve edge
plans and a separately explained bound endpoint hydration using `RecordIdScan`.
The latter is not inferred from the outer edge indexes. As elsewhere in the
inventory, EXPLAIN does not expose every internal `endpoint.*` dereference;
its eligibility behavior remains covered by the independent legacy-membership
fixture. No full edge-table scan or edge-plan visibility exception is accepted.

Reproduce using `scripts/run-exact-rust-test.py -p graphrag-db --lib --test-name`
and each fully qualified test name above (`--nocapture` emits evidence).
Touching the migration files also selects the complete conditional migration
and schema fixture job (`python3 scripts/run-migration-gate.py --run`).
