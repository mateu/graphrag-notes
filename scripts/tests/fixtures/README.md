`graph-eval-20261102.json` is the sanitized Q1–Q8 baseline from the read-only clawd graph evaluation on 2026-11-02 (service build 0dbb186); it contains paths, scores, graph diagnostics, timing and hashes, not note bodies or credentials.


`graph-eval-20261103.json` is the read-only live baseline for the 2026-11-03
Q2/Q7 seed revision (#180), service build eabe6e7. It uses the same fixed
protocol without a per-source cap. The previous fixture is retained unchanged.
The private suite on clawd is
`project/graphrag-memory-sync/data/graph-eval-20261103/suite-seed-revision-20261103.json`;
its SHA-256 and the runner SHA-256 are recorded in report metadata. Accepted
proposal-backed edges were **0** immediately before the run (bounded
`list_proposals(status=accepted, limit=200)` returned no rows); no hops ≥ 1
were observed. This is a live snapshot, not a frozen-corpus ranking experiment.

## Suite correction and lineage

Both seeds now use `memory/2026-10-07-pr142-review.md`, read directly on clawd.
Q2 asks why an unchanged re-upload must preserve source identity and
enrichment/rollback proof. Q7 asks why rollback must reject reuse of the
conversion's request ID. Under `## Conversation Summary`, the assistant's
“Both regression tests now reproduce Codex’s reports” turn states that an
unchanged refresh loses reviewed proof and rollback wrongly accepts that ID;
the preceding finding explicitly says rollback must reject reuse. The suite
note records these passage locations and keeps the old rows.

The retired `memory/2026-10-06-graphrag-ingestion-canary.md` is synthetic
“NOT REAL MEMORY” and supplies no answer evidence. Broad Q7 lost-admission,
terminal-processing and changed-generation claims remain out of scope in the
`project/` canary receipt, not silently admitted to the memory corpus.

Both revised seeds are found graph-off in both repeats; they are **not seed
misses**. Seed and support now coincide, so these cases establish baseline
answerability only: they cannot prove graph lift or pass while that support is
already returned off. No alternative support/edge is invented.

## Old → new per-question path comparison

Each row applies to both off and on; timings changed but are not graph gains.

| Q | Gained paths (`memory/` prefix) | Lost paths (`memory/` prefix) | Reason |
|---|---|---|---|
| Q1 | none | none | Genuine seed miss unchanged. |
| Q2 | 2026-08-01-0701.md; 2026-10-07-pr142-review.md | 2026-07-25-1234.md; 2026-07-31-2051.md | Narrowed unchanged-refresh question and valid seed. |
| Q3 | none | none | Retrieval unchanged. |
| Q4 | none | none | Retrieval unchanged; no accepted-edge traversal yet. |
| Q5 | none | none | Retrieval unchanged; no accepted-edge traversal yet. |
| Q6 | none | none | Retrieval unchanged; no accepted-edge traversal yet. |
| Q7 | 2026-10-07-pr142-review.md; 2026-10-08-codex-cli-version.md | 2026-10-15-graphrag-ingestion-failure.md; 2026-10-27-write-conflict-retry.md | Narrowed rollback request-ID question and valid seed. |
| Q8 | none | none | Genuine seed miss unchanged. |

Only Q2/Q7 seed/support found flags flip false → true. All questions remain
fail, on/off paths match, and repeated responses are stable. The current suite
also incorporates the already-documented 2026-11-02 dated-support revision:
Q1/Q3/Q8 replace `memory/error-log.md`; Q2/Q7 replace the out-of-scope
`project/` support. Those are expected coverage-target changes, not new
retrieval changes in this issue. Q1/Q8 questions and seeds are untouched.
The service upgrade since the old fixture explains lower call timings; those
single-run deltas do not qualify performance.
