# Finding a captured note

Current source gives indexed exact note titles priority in keyword and hybrid
retrieval, including hybrid retrieval with graph `auto` or `on`. The published
`v0.1.0-rc.2` assets predate this ranking correction.

For example, a note titled `Orion Notebook` with body `Available to the assistant
through a shared gateway` can be found by searching `Orion Notebook`, even when
many imported bodies repeat those words. The existing title index supplies this
signal; existing notes need no refresh or new embedding.

Title comparison ignores case and surrounding whitespace. Internal whitespace
and punctuation remain significant: `Orion Notebook`, `Orion  Notebook` and
`Orion-Notebook` are distinct titles. This preference applies to titles matched
by the full-text analyzer; a title that produces no analyzer tokens cannot be
retrieved through this index. Source, time, scope and successful source-generation
filters still apply. Setting the note hit-type weight to zero disables its title preference;
setting the hybrid full-text weight to zero disables hybrid title preference.
Keyword retrieval does not use hybrid channel weights.

Exact-title candidates are retained before component and final result limits.
Other candidates use the configured fusion score and deterministic tie-breaking.
Explain output includes `exact_title_match` so a title result's priority is
visible even when its numeric BM25 score is small or zero.

## Graph ranking

Graph retrieval supplements the hybrid candidates. It retains accepted-edge
confidence, hop decay, provenance and the existing traversal bounds.
`graph_seed_score` is the graph channel's strength relative to the stronger of
the configured vector and full-text channels, before hop and confidence decay.
Before final ranking, that relative strength is multiplied by the stronger
channel's weight. RRF then divides by `rrf_k + 1`; weighted fusion clamps to
`[0, 1]`. The note hit-type weight is applied last.
With defaults, a zero-hop graph result scores approximately `0.011475` under
RRF, or `0.7` under weighted fusion. Strong combined direct matches stay ahead,
while graph candidates can outrank weaker vector-only matches and add recall
in a full result list.

When a note appears in both channels, it receives the stronger calibrated
score and keeps its direct-retrieval and graph evidence. The graph score in
explain output is calibrated, including hop/confidence decay. These rules avoid
the previous fixed `0.03` graph score dominating the default RRF scale.
Higher configured graph strengths can intentionally favor graph evidence.
The default graph strength is now `1.0`. Existing configurations explicitly
setting the previous `0.03` retain that value and produce more conservative
graph contributions; set it to `1.0` to use the new default policy.

### Accepted neighbors of retrieved notes

A path's strength starts from its seed's query coverage. For a seed that the
direct channels did not retrieve, such as an entity-mention seed, that remains
the whole story. Coverage alone, however, put an accepted neighbor of the top
hybrid hit at no more than `0.8 × 0.011475 ≈ 0.0092` with defaults: below
every hit of a full `build_context` over-fetch window, which the vector channel
alone fills with scores of at least `0.7 / 68 ≈ 0.0103` at two chunks. The
traversal then only annotated hits that were already retrieved.

A seed that is itself in the ranked hybrid window now also carries its
calibrated retrieval strength: `graph_seed_score × fused score ÷ the zero-hop
graph score above`. A neighbor inherits that strength, times hop decay and edge
confidence, only when it has its own bounded evidence for the same query: it is
a hybrid or lexical-supplement candidate, or a query-entity mention seed. Its
path strength is then the larger of its coverage chain and the anchored chain,
so the accepted path, not a weaker zero-hop mention, becomes its evidence.
Neighbors without query evidence, such as a hub's unrelated neighbors, and any
path through such a bridge keep the coverage chain unchanged. With the default
`graph_seed_score`, an anchored neighbor scores at most its anchor's fused score
times decay and confidence, so it ranks below its anchor and competes with the
anchor's weaker direct competitors on the same final scale.

### Reviewed edges

The hop from a retrieved seed over a reviewed edge, one that a person created
or accepted, is not decayed (#216). Such a neighbor, still only with its own
query evidence, competes at its anchor's strength times edge confidence. With
the default `graph_seed_score` it scores at most its anchor's own score, never
a rounding step above it, and ties fall to the shared deterministic ranker,
which places a direct anchor before a graph-only neighbor. It can therefore
take a context slot from a direct match weaker than its anchor, but not from
one stronger than the anchor.

Hop decay still applies to unreviewed edges such as Gardener's automatic policy
acceptance, to seeds that the direct channels did not retrieve, to every later
hop, and to a reviewed `derived_from` edge followed from a note to a note
derived from it. Such a derivative restates its source, so it is a duplicate
path rather than independent support. Explain output reports the hop decay of
the chain that produced a path's score, so an undecayed reviewed hop shows
`decay` 1.0. See the [validation note](validation/graph-support-216.md).

No context slot is reserved for graph evidence. An anchored neighbor still
passes packing relevance, novelty, near-duplicate suppression, the optional
per-source cap and every token budget; a stronger direct match keeps its slot.
`auto` remains entity-seeded, so ordinary hybrid hits are not traversal anchors
there, and graph-off retrieval and packing are unchanged.

`build_context` diagnostics distinguish an existing path from added support.
`graph_candidates_added` counts selected chunks that only the graph channel
placed in the ranked window, `graph_candidates_annotating` counts selected
direct hits that merely carry a path (the two sum to
`graph_candidates_selected`), and `graph_candidates_rank_limited` counts graph
candidates cut by the over-fetch window before packing. Search pipelines report
the same window cut as `candidates_rank_limited`.

Graph retrieval fetches referenced endpoint documents and hydrates bounded
candidate IDs directly; its ranking and traversal bounds remain unchanged.
See [graph performance](graph-search-performance.md) for query plans, phase
timing, reproducible fixtures, and service build guidance. Keyword search
requires no inference providers; hybrid search still requires query embedding.
