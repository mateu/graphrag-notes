# Exact-vector access-path exception (#191)

## Decision and scope

The no-full-table-iteration policy in [the hot statement inventory](hot-retrieval-plans.md)
has **one explicit class of exception**: the exact cosine-vector SELECTs issued
by `Repository::vector_search_notes`, `vector_search_messages`, and
`vector_search_conversation_summaries`. Their current access path is
`KnnTopK(Cosine, dimension=1024, k=limit)` over a `TableScan` of respectively
`note`, `message`, and `conversation`. These are visible, intentional scans,
not EXPLAIN visibility limitations or indexed-plan passes. This policy chooses
exactness over the universal no-scan requirement for these three statements.
It does not authorize a scan in any lexical, graph, hydration, or other path.

No production SQL, index, schema, migration, portable format, service, database,
or approximate-search default changes as part of this decision. No migration
ID is needed. #179 remains a partial delivery; this exception resolves its
exact-vector policy conflict, not its other coverage gaps.

## Why not replace the scan with an index?

The locked engine is `surrealdb-core 3.3.5` (Cargo.lock). Inspection of that
release's `src/idx/planner/tree.rs` (`lookup_index_option`, `eval_hnsw_knn`,
`eval_diskann_knn`) shows vector index selection maps to `IndexOperator::Ann`.
HNSW and DiskANN are approximate candidate searches, not exhaustive cosine
ranking. The companion `surrealdb-catalog 3.3.5`
`src/schema/index.rs::Index` enum contains only Idx, Uniq, Hnsw, FullText,
Count and DiskAnn (no M-tree). In particular even a literal `<|k,COSINE|>` can be mapped by that
planner to ANN with `ef=max(k,ef_construction)` when index matching succeeds;
the spelling of the operator alone does **not** prove exactness. The observed
native-bound Repository path instead executes `KnnTopK` over a table scan.
This release has no M-tree exact-vector index path to migrate to. Older
SurrealDB M-tree documentation is not evidence of a supported index here.

Increasing ANN effort, overfetching and reranking cannot guarantee the exact
top-k over arbitrary eligible records. A filter index may reduce some scoped
searches, but cannot solve unscoped exact cosine ranking; source dereferences,
optional time restrictions and note-generation eligibility must also retain
their existing semantics. No safe, general index-backed replacement preserving
this contract was found. An eventual replacement requires a separate exactness
qualification, including migration/reopen/restore coverage if schema changes.

## PR-time guard and exactness evidence

`repository::exact_vector_plan_policy_tests::exact_vector_scan_exception_preserves_plans_and_filtered_top_k`
is a non-ignored named test in the existing retrieval-regression lane (every
PR and push to main). It uses the existing cfg(test) SQL/native-binding capture
and existing operator walker, not a copied serving query or new plan parser.

The provider-free 256-row-per-table, 1024-dimensional fixture covers all three
Repository methods, k=1/5/17, and four filter combinations (none, time, source,
both): **36 actual statements**. An independent Rust eligibility calculation
and analytic cosine formula verify returned IDs/order/count and distances
(absolute f32 tolerance 1e-7). Closer ineligible records exercise pre-top-k
filtering, stale note generations, legacy missing generations, missing source
links and missing message timestamps. Distances are deliberately unique so the
expected IDs do not depend on an unspecified tied-boundary admission order.
The existing native-float qualification remains the broader tied-boundary,
historical-number and full-payload comparison; this guard does not replace it.

For every call, the guard requires exactly one matching table scan beneath
exactly one Cosine KnnTopK, with the expected field, dimension and k, and no
IndexScan/FullTextScan/AnnTopK. The production operator must remain explicit
`<|k,COSINE|>` as well. Mutated plan negative controls reject an indexed-scan
substitution and a changed distance metric. A planner upgrade that removes the
scan intentionally fails: review/qualify the replacement and revise this policy
instead of silently accepting a new approximate or otherwise unqualified plan.
This is a **known-scan drift guard**, not an assertion that no scans occur.

Reproduce (build concurrency bounded):

```sh
CARGO_BUILD_JOBS=2 python3 scripts/run-exact-rust-test.py -p graphrag-db --lib \
  --test-name repository::exact_vector_plan_policy_tests::exact_vector_scan_exception_preserves_plans_and_filtered_top_k \
  --nocapture
```

[Generated evidence](evidence/191-exact-vector-plans.json) records the actual
plans and expected-output qualification. Execution timings are observations,
not latency limits or a bounded-retrieval guarantee. Scans remain linear in the
table population and are not a performance fix.
