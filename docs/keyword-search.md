# Keyword search without inference services

These commands are available in current source and the forthcoming Apple
Silicon `0.1.0-rc.2` candidate. Its assets are not published yet; use the
[source-build path](getting-started.md#build-from-source). The published
`v0.1.0-rc.1` predates keyword mode. See the [daily workflow](daily-workflow.md)
for a repeatable capture-to-reuse example.

Hybrid retrieval remains the default. Select keyword retrieval explicitly when
providers are stopped, when you need literal term matching, or after restoring
a backup that omitted embeddings:

```sh
graphrag search "Atlas launch" --mode keyword
graphrag search "Atlas" --mode keyword --scope messages
graphrag search "Atlas" --mode keyword --scope all --since-days 7
graphrag search "Atlas" --mode keyword --source-uri "file:///absolute/path/notes.md"
graphrag search "Atlas" --mode keyword --scope all --explain --format json
```

`--scope notes` is the default, `messages` searches original chat messages, and
`all` also includes conversation titles and summaries. Source and time filters
use the same fields as hybrid retrieval: note creation time, message creation
time, and conversation update time. Messages without a creation timestamp do
not match a time filter. Source URIs must exactly match the stored URI; `sources
list` or inspection can show it.

Keyword mode uses the existing full-text indexes and their BM25 scores. The
configured note/message/conversation weights apply before deterministic final
ordering; vector weights, RRF configuration, and graph expansion do not affect
keyword ranking. BM25 scores from different record indexes can have different
scales. `--scope all` therefore exposes each score and hit kind rather than
claiming a calibrated semantic similarity.

Keyword retrieval never embeds or extracts, checks provider health, reads the
inference cache, or initializes embedding compatibility metadata. It can search
legacy records with untracked vectors and records restored without vectors.
Imports and edits that change content still require the configured providers.

The default `--graph auto` resolves to `off` in keyword mode. Explicit
`--graph on` fails with exit code 2 and tells you to choose `--graph off` or
`--mode hybrid`. `--context` can add existing related-note data for notes scope;
it does not expand or change the keyword result ranking.

Hybrid failures keep their original failure status and suggest an exact
keyword command with the selected query, scope, filters, configuration,
database, output format, and result limit. The CLI never switches modes
implicitly. Hybrid retrieval and its existing rankings are unchanged.

## Machine output

Human output identifies the selected mode even for no matches. Ordinary JSON
retains `data.results` and adds `data.mode` and `data.channels`. Ordinary JSONL
retains one envelope per hit, with the original `data.id` location and additive
`mode`/`channels` fields. An empty ordinary JSONL response still emits zero rows.

`channels` identifies the base retrieval channels: `vector` and `full_text` in
hybrid mode, only `full_text` in keyword mode. Hybrid graph contributions retain
their separate graph summary and per-result evidence.

Explain JSON and JSONL preserve their existing result and pipeline shape.
`pipeline.filters.mode` and `pipeline.filters.channels` identify the selected
retrieval path, and keyword reports the resolved graph policy as `Off`. Keyword
results carry only full-text evidence: vector and graph are null, embedding
provider/model metadata is absent, and fused/final score kind is `bm25` with
the native full-text score and hit-type multiplier. Empty explain JSONL retains
one envelope with `data.result: null` and the pipeline metadata.

Navigation commands and complete stored content remain available for keyword
results using the same canonical IDs and revision guards as hybrid search.
