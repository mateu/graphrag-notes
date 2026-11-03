# Provenance membership indexes (#192)

Migration **26 / `provenance_endpoint_indexes`** adds non-unique `in` indexes
`idx_note_from_conversation_in` and `idx_note_from_message_in`. Historical
migration SQL/checksums and the baseline schema are unchanged. Initialization
builds the indexes over existing rows; neither provenance records nor their
native IDs are rewritten. There is no Repository SQL or result-format change.

This resolves the two provenance scan exceptions in the original
[#179 inventory](hot-retrieval-plans.md); its committed #188 plans remain
historical evidence, not plans for the post-v26 schema. Other #179 gaps are
outside this change.

## Regression guard

`repository::provenance_index_tests::provenance_membership_indexes_preserve_upgrade_and_fresh_results`
runs in the required `retrieval regression baseline` job on PRs/main pushes:

```sh
python3 scripts/run-exact-rust-test.py -p graphrag-db --lib \
  --test-name repository::provenance_index_tests::provenance_membership_indexes_preserve_upgrade_and_fresh_results \
  --nocapture
```

The provider-free fixture populates 512 rows in each provenance table plus
one-to-many, duplicate and dangling-endpoint rows. It captures actual
`Repository::graph_note_provenance_ids` SQL and native bindings, independently
explains both statements with `EXPLAIN FULL`, and reuses the existing operator
walker. Multiple IDs, duplicate inputs, nonexistent notes, nonexistent target
records, and distinct numeric/string note IDs are covered. Empty input stays
query-free. Existing textual output serialization/deduplication is preserved;
separate numeric-only/string-only requests assert native membership identity.

The real v25 fixture is a negative control: both actual statements table-scan
and fail the same indexed/no-TableScan predicate. After migration and repeat
initialization, every logical source/provenance record (including native IDs)
is unchanged and both plans use their expected indexes with no TableScan.
Post-upgrade writes and fresh initialization are checked too.

[Eight generated plans](evidence/192-provenance-plans.json) record v25,
upgraded, post-upgrade-write and fresh paths. This migration also selects the
existing required `conditional migration and schema fixtures` job; no skip,
exception or branch-protection change is introduced.
