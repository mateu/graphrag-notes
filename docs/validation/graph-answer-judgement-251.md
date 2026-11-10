# Graph on/off owner judgement at `98fb72f` (#251)

This note records counts, verdicts and classifications only. It contains no
questions, expected answers, note text, snippets, source paths, record IDs or
credentials. Item names (`V3-E01` and so on) are positions in private suites.

## What was judged

The graph-context evaluation stops at `needs_judgement` when graph-off misses
the support chunk and every graph-on round delivers it over an allowed accepted
path. The [fourth pass condition](../retrieval-evaluation.md#graph-context-evaluation)
is a human decision that the answer differs materially. Seven items were
waiting for that decision in the most recent retained live runs.

No evaluation call was made for this record. It judges the retained runs.

| | Suite v3 (held out for #244) | Suite v2 |
| --- | --- | --- |
| Items | 7 (5 edge, 2 guard) | 22 (7 edge, 4 guard, 8 base, 3 known base misses) |
| Run (UTC) | 2026-11-05 18:39:33–18:40:03 | 2026-11-05 18:40:11–18:41:39 |
| Suite SHA-256 | `35c2f9e9…155a71ab` | `90e1ac16…b3bce86a` |
| Source map entries | 848 | 848 |

Common to both runs:

- **Build:** source revision `98fb72f`, binary SHA-256 `5c4e3faa…14dde950`,
  37 accepted reviewed edges.
- **Runner:** `scripts/evaluate-graph-context.py`, report schema 2,
  `runner_sha256` `b96bcc10…0dd40e7e`.
- **Protocol:** policies off, on, on, off; scope `notes`; `max_chunks=2`,
  `max_total_tokens=1000`, `max_chunk_tokens=400`; no per-source cap.
- **Calls:** 0 errors. Every item returned the same chunk IDs in both off rounds
  and in both on rounds.

**Freshness check, 2026-11-10 (read-only):** the running service binary has the
same SHA-256 and the service still reports 37 accepted edges (`derived_from`
27, `related_to` 5, `supports` 4, `contradicts` 1), so the retained runs
describe the deployed build. The corpus is a live snapshot and has grown
(source map 863), so a rerun today need not reproduce these rounds.

## How each item was judged

A private sheet was built from the retained raw responses. For each item it
shows the question, the expected answer, the graph-off context and the
graph-on context exactly as rendered to a client. Each item got four
classifications and a delegated yes/no/unsure recommendation, and the
repository owner then decided each item on 2026-11-10. The sheet, the per-item
facts and the decisions are kept with the private run evidence, next to the
earlier decision log.

- **Correctness:** is the expected answer in the rendered context: `full`,
  `partial` or `absent`? This is about the delivered text, not the chunk ID.
- **Citation:** is the answer-bearing text from the expected support chunk,
  reached over which accepted edge type and direction?
- **Displacement:** at two chunks, graph-on never only adds. Which graph-off
  chunk did it replace: the seed chunk, another chunk of the seed's file, or an
  unrelated chunk?
- **Cost:** mean client latency and total tokens, on minus off, per item.

The fallback token counter charges one token per UTF-8 byte, so at
`max_chunk_tokens=400` each delivered chunk is a span of at most about 400
bytes of a longer chunk. All seven support chunks were delivered truncated.

## Result per item

| Item | Path to support (1 hop) | Answer in graph-off | Answer in graph-on | Graph-on replaced | Owner: differs materially | Δ latency | Δ tokens |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| V3-E01 | `related_to` outbound | absent | full | seed chunk | **yes** | +91 ms | +145 |
| V3-E02 | `derived_from` outbound | absent | full | seed chunk | **yes** | +108 ms | +54 |
| V3-E03 | `derived_from` outbound | absent | full | unrelated chunk | **yes** | +194 ms | −10 |
| V3-E05 | `supports` outbound | absent | partial | other chunk of the seed file | **yes** | +170 ms | −74 |
| V2-E09 | `derived_from` inbound | absent | full | seed chunk | **yes** | +142 ms | +23 |
| V2-E07 | `derived_from` outbound | partial | partial | seed chunk | **no** | +134 ms | +11 |
| V2-Q4c | `derived_from` outbound | full | full | unrelated chunk | **no** | +73 ms | +9 |

Notes on the decisions:

- **Five yes.** In each, graph-off returned no answer-bearing text and graph-on
  returned it from the expected support chunk. In four of the five the
  graph-off context also held a plausible wrong answer: a similar figure or
  list from another date, host or entity (V3-E01, V3-E03, V2-E09), or a later
  statement that contradicts the one asked about (V3-E05). V3-E05 is `partial`: the delivered
  span answers both things the question asks, but not a detail that the
  expected answer also lists.
- **V2-E07, no: delivered chunk, undelivered answer.** The expected support
  chunk is in the graph-on context by ID, so every mechanical condition holds.
  But the rendered span is a different passage of that chunk; the answer
  passage sits earlier in the same chunk and is cut. Graph-off already had half
  of the answer from another note, and graph-on adds nothing to it. The
  evaluator checks chunk IDs, not rendered spans, so it cannot see this.
- **V2-Q4c, no: redundant.** The delivered chunk does answer the question, but
  the seed chunk that graph-off already returns states the same result. The
  graph adds corroborating detail, not a different answer. This item was
  recorded as an owner pass at the `5619a92` baseline on the narrower question
  of whether the delivered chunk answers; for this run the owner decided there
  is no material difference.

By what graph-on replaced, as #251 asked:

| Graph-on replaced | Items | Owner yes | Owner no |
| --- | --- | ---: | ---: |
| The seed chunk (seed file no longer in context) | V3-E01, V3-E02, V2-E07, V2-E09 | 3 | 1 |
| Another chunk; seed chunk kept | V3-E03, V3-E05, V2-Q4c | 2 | 1 |

Where the seed was dropped and the owner said yes, the seed carried only the
premise that the question itself restates.

## Guards in the same runs

Guards are graded mechanically: graph-on must keep the answer-bearing chunk
that graph-off delivers. They need no judgement.

- **Suite v3: 0 of 2 held.** In V3-N1 and V3-N2 both graph-off rounds deliver
  the answer-bearing seed chunk, and both graph-on rounds replace it with the
  seed's reviewed `derived_from` neighbour (outbound, 1 hop). The neighbour's
  delivered span does not carry the answer. These two items were written for
  exactly this risk in #244: a seed in the last slot being swapped for its
  neighbour when the two hold different facts.
- **Suite v2: 4 of 4 kept the support chunk** in every off and on round. In two
  of them graph-on still changes the other slot.

So in suite v3 the same mechanism produced four judged gains and two answer
losses, with one edge item unchanged.

## Totals

| | Suite v3 | Suite v2 |
| --- | ---: | ---: |
| Edge items | 5 | 7 |
| — pass (mechanical conditions and owner yes) | 4 | 1 |
| — owner rejected | 0 | 2 |
| — unresolved | 0 | 0 |
| — mechanical fail (support not delivered) | 1 | 4 |
| Guards that kept the answer-bearing chunk | 0 of 2 | 4 of 4 |
| Base items with support in off and on | n/a | 8 of 8 |
| Known base misses, missed off and on | n/a | 3 of 3 |
| Mean client latency per call, off → on | 1,022 → 1,163 ms (14 calls each) | 953 → 1,052 ms (44 calls each) |
| Mean total tokens per call, off → on | 711 → 744 | 758 → 758 |

Two of the four v2 mechanical fails (V2-E04, V2-E11) are diagnosed in #243.
V2-E07 and V2-E09, diagnosed there at an earlier build, now deliver their
support chunk. The v3 mechanical fail (V3-E04) returned identical chunks with
the graph off and on.

## What this does and does not support

It supports:

- At this budget, on these edge items, a reviewed edge from a retrieved seed
  can replace a wrong or empty answer with the right one: 5 of 12 edge items
  across the two suites, each from a support chunk that graph-off retrieval
  did not return within the budget.
- That gain is not free and not one-directional. Graph-on cost about 100–140 ms
  per call on average, and in suite v3 it removed the answer in both guards.
- Mechanical `needs_judgement` overstates benefit. Two of seven such items were
  rejected: one because the rendered span missed the answer, one because
  graph-off already had it.

It does not support:

- A general answer-quality claim. The questions are agent-written, seeded from
  the wording of one endpoint of an accepted edge, and chosen so that graph-off
  misses. They measure whether a known edge can deliver, not how often graph
  retrieval helps ordinary questions.
- Any claim about automatic recall or conversational use. No model answered
  these questions; a person judged the rendered contexts.
- A claim beyond two chunks and 400 estimated tokens per chunk, or beyond this
  live snapshot. A larger budget would change both the gains and the guard
  losses.
- A latency percentile. These are small client-side samples through a tunnel.

## Open observations

These are recorded for whoever picks up the ranking work. This note changes
nothing.

1. Both v3 guards are violated on the deployed build.
2. A support chunk can pass by ID while its rendered span omits the answer
   (V2-E07). Chunk-level grading cannot detect that.

## Replay with span evidence (#255)

The runner can now check a suite's `support_evidence` against the rendered
span. To test it, the retained raw responses of both runs were re-graded
offline: no service call was made and the approved suites were not edited.

- **Without evidence:** all 29 items reproduce their retained verdict,
  `fail_reasons`, chunk IDs and paths, and report `span_check: unchecked`.
- **With delegated evidence** (two to five required terms per item, taken from
  each expected answer, for the seven judged items):

| Item | Mechanical verdict before → after | Span check | Terms found / required, each on round | Owner decision |
| --- | --- | --- | --- | --- |
| V3-E01 | `needs_judgement` → `needs_judgement` | covered | 2 / 2 | yes |
| V3-E02 | `needs_judgement` → `needs_judgement` | covered | 3 / 3 | yes |
| V3-E03 | `needs_judgement` → `needs_judgement` | covered | 5 / 5 | yes |
| V3-E05 | `needs_judgement` → `needs_judgement` | covered | 2 / 2 | yes |
| V2-E09 | `needs_judgement` → `needs_judgement` | covered | 2 / 2 | yes |
| V2-E07 | `needs_judgement` → `fail` (`support_span_missing`) | missing | 0 / 3 | no |
| V2-Q4c | `needs_judgement` → `needs_judgement` | covered | 2 / 2 | no |

  The other 22 items are unchanged.
- **Truncation:** 214 of the 228 rendered chunks across both runs were
  truncated.
- **`off_answer_terms_found` is `false` for all seven, including V2-Q4c.** The
  graph-off context there states the same result in different words, so exact
  terms do not find it. The flag cannot replace reading the context for that
  pattern; it stays a flag.

The evidence terms are a delegated choice kept with the private run evidence.
They are not an owner-approved suite revision.
