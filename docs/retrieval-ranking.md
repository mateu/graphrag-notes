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

Graph retrieval fetches referenced endpoint documents and hydrates bounded
candidate IDs directly; its ranking and traversal bounds remain unchanged.
See [graph performance](graph-search-performance.md) for query plans, phase
timing, reproducible fixtures, and service build guidance. Keyword search
requires no inference providers; hybrid search still requires query embedding.
