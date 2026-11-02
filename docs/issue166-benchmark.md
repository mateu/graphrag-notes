# Issue 166 semantics and warm matcher benchmark

Baseline: `25bf7c94c3ffbf6f7ecb537ab600ca2f75e2db96`.
No canonicalization, ID merging, cap changes, or changed ordering is permitted.

## Safety and reproduction

Use only a copied fam backup beneath `~/.graphrag/shared/tmp/issue166-*`.
The benchmark example rejects paths outside that subtree after resolving
symlinks. Never pass a live database or the shared database directory.

1. Record a sorted file-name/size/SHA-256 manifest of the cold backup before
   copying. Keep the original read-only.
2. Build the example on the baseline and optimized revisions:
   `CARGO_BUILD_JOBS=2 RUSTC_WRAPPER= cargo build --release -p graphrag-db --example issue166_benchmark`.
3. Run each binary sequentially against its own backup copy:
   `issue166_benchmark BACKUP_COPY > RESULTS.jsonl 2> migration.log`.
   Opening applies migrations; the stderr timer includes database opening and
   schema initialization/backfill. Keep this separate from warm query latency.
4. Each query runs three discarded warmups and twenty measured invocations
   inside one process. p50/p95 use nearest ranks 10/19 of the sorted twenty.
   No compiler or other heavy work may run on fam during measurements.
5. Compare baseline and optimized JSONL:
   `python3 scripts/issue166-compare.py BEFORE.jsonl AFTER.jsonl`.
   Results include IDs, metadata, order, and the 20-result cap. A mismatch
   fails; identical results with any p95 >=100ms exits 2 rather than passing.

The public scoped matcher is called with an empty ranked-note list and no
time/source constraint; this includes authoritative endpoint eligibility.
Repository regression tests separately exercise ranked notes, hidden/current
generations, detached notes, source/time scope, malformed aliases and caps.

Candidate counts are lexical supersets from the v24 entity index, not returned
eligible hits. Exact/contained counts use full-query equality/all phrase
n-grams. The third count uses the entire query as a prefix: it is the actual
prefix candidate count only for the single-token prefix fixture. Multiword
queries normally stop at exact/contained; do not mislabel that diagnostic
count as a measured prefix execution.

The prefix fixture is `openc`: both `opencl` and `opencla` have exact
entities in this backup. A direct index probe found no `openc` lexeme. Confirm exact and contained candidate eligibility/results before
accepting the prefix fixture on any newer backup.

## SQL plans

The same example supports `BACKUP_COPY --sql FILE.sql` for bounded diagnostic
SQL. Record real `EXPLAIN FULL` output with bound record IDs, including the
lexical lookup and mention association lookup. SQL text that appears indexable
is not evidence of an indexed execution; reject an association TableScan.
Never compare an `out IN $ids` scan with an indexed per-record lookup without
checking the executed plan.

## Retention and cleanup

Keep aggregate timing/result/plan evidence with the delivery record. Remove
only the owned issue166 temporary directory after copying evidence out.
No service restart, deploy, merge, or changes to other temporary files belong
to this benchmark.

## Measured outcome (not accepted)

All eight query payloads match baseline exactly. Warm p95 gate fails in four cases. Three warmups, twenty measured calls sequentially on the pinned fam backup copy.

|Query|Exact/contained/prefix candidate counts|p50 ms|p95 ms|
|---|---|---:|---:|
|OpenClaw|[906, 906, 2752]|176.40|178.37|
|GraphRAG|[20, 20, 60]|2.83|2.90|
|Proxmox|[20, 20, 44]|3.30|3.37|
|how does openclaw work|[0, 934, 0]|189.47|190.99|
|how does memory work|[0, 340, 0]|58.89|59.60|
|main|[591, 591, 825]|96.57|100.70|
|index|[81, 81, 427]|11.65|11.76|
|openc|[0, 0, 2755]|620.49|650.25|

Backup manifest SHA256: `bb9b2c37a93c3100c16e7b1c078fb2e6925d2e4944c4bc0a14d59ce7bf09df41`. Original schema23 →25 open/migrate:126.249s; baseline23→24:65.280s. EXPLAIN FULL used idx_entity_lexeme_lookup, idx_mentions_entity_note, idx_mention_lexeme_lookup, idx_mention_lexeme_entity; zero TableScan operators. Remaining cost scales with equality mention probes and global SQL candidate evaluation. No canonicalization.
