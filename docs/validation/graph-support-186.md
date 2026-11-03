# Accepted-neighbor support at a fixed budget (#186)

The [sanitized aggregate](graph-support-186.json) holds the before/after
fixed-suite comparison, the Q4 window and context diagnostics, and the
reproducibility checks. It records only ranks, scores, counters, hashes and
source paths that the committed
[baseline fixture](../../scripts/tests/fixtures/graph-eval-20261103.json)
already publishes. It contains no record IDs, query text, titles, snippets or
credentials.

## Protocol

The unchanged eight-question suite (`suite_sha256` `0c591bba…0140`) ran
through the unchanged repository runner (`runner_sha256` `dfb42b21…ccb5`)
with `--compare` against the committed fixture: notes scope, policies
off, on, on, off, `max_chunks=2`, `max_total_tokens=1000`,
`max_chunk_tokens=400`, both uncapped and `max_chunks_per_source=1`. All runs
were read-only `build_context`/`search_notes` calls over loopback MCP.

- **Before, live:** deployed `9b867f6` (binary SHA-256 `8ac529fb…17cc`), one
  accepted edge.
- **Before, isolated:** the same binary serving a verified restore of the
  post-migration schema-27 portable backup (218,860 records, embeddings
  included, the same single accepted edge) under a separate read-only
  credential. All 8/8 questions were identical to live in both cap modes:
  returned paths, scores, chunk graph paths, tokens and every diagnostic.
- **After, isolated:** the candidate binary (`4ecb8504…15bf`) on a fresh copy of
  the same restore. Without deploying, this is the closest live-equivalent
  after state.

## Where Q4's support lost

The accepted edge is `derived_from` from the Aug17 summary
(`memory/2026-08-17.md`, the top hybrid hit) to the Aug15 detail
(`memory/2026-08-15.md`, the support). The two-chunk budget over-fetches an
eight-hit window.

| | Graph-off rank / score | Graph-on rank / score | Graph evidence for the detail |
| --- | --- | --- | --- |
| Before | 106 / 0.004615 | 95 / 0.004918 | hop-0 entity-mention seed only |
| After | 106 / 0.004615 | **3** / 0.012842 | hops=1 outbound `derived_from` from the summary |

1. **Selection and calibration, fixed here.** Graph path strength was always
   the seed's lexical query coverage times decay and confidence. That held even
   when the seed was the top retrieved hit. With defaults, the accepted path
   could reach at most `0.8 × 0.011475 ≈ 0.0092`. The eight-hit window ended at
   0.010606, so the outbound candidate could never enter it. The detail's best
   evidence was its own weak mention seed (coverage 6/14). The only selected
   graph evidence was therefore the inbound path from that seed, which
   annotated the already-selected summary.
2. **Packing, by design after the fix.** The detail now ranks third. Slot two
   still goes to `memory/2026-08-13-1111.md`, a direct hit at 0.014924 in both
   modes. Under min-max relevance with novelty weight 0.25, that hit's lowest
   possible selection score is 0.5898. The detail's highest possible score is
   0.5442. No novelty or near-duplicate outcome can reverse the order without
   reserving a graph slot or demoting a stronger direct match, and this change
   does neither. With a third chunk, used only as a diagnostic and not as the
   suite protocol, graph-on adds the detail through the outbound accepted path:
   `graph_candidates_added=1`. Graph-off fills that slot with another source.
3. **The per-source cap is not the constraint.** With `max_chunks_per_source=1`
   the detail still ranks below the stronger direct hit, so the context is
   unchanged.

## Fixed-suite comparison

For the fixed suite, after equals before in 8/8 questions under both cap
modes. Returned paths, scores, chunk graph paths, tokens and all pre-existing
diagnostics are unchanged; the only differences are the new counters. Those
counters separate *path exists* from *support added* for every question. At
this budget `graph_candidates_added` is 0 everywhere. The selected graph
evidence (Q1, Q2, Q4 and Q7) is always an annotation on a direct hit. Every
other graph candidate was cut by the window (`graph_candidates_rank_limited`)
or lost packing. All six reports (live before, isolated before, isolated
after; each uncapped and capped) completed 32 rounds with zero errors, and
every question's repeated on and off rounds were stable. Q4 is the only
question with an accepted-edge path, before and after. No suite case passes
before or after, and no pass or answer-materiality claim is made.

## Fixture semantics

Q4's frozen `allowed_relations` are `supersedes` and `related_to`. The
reviewed edge is `derived_from`, which is the accurate relation here: the
summary restates the same canary head from the detail. Neither replaces the
other, and `related_to` would discard the provenance direction. Under the
frozen fixture, Q4 cannot pass through this edge even if the detail were
packed. This change leaves the baseline and runner untouched. Admitting
`derived_from` for Q4 would require a separate, explicitly versioned suite
revision. Even then, Q4 still fails at the frozen two-chunk budget for the
packing reason above.
