# GraphRAG Notes

A local-first GraphRAG notes system built around a Rust CLI, hybrid retrieval, and graph/provenance links.

## Get your first result

The published [0.1.0-rc.2 prerelease](https://github.com/mateu/graphrag-notes/releases/tag/v0.1.0-rc.2)
includes recoverable capture, explicit Markdown folder sync, offline keyword
search, source inspection/opening, connection review, and a terminal workspace.
Its binary targets **macOS Apple Silicon, macOS 15 or newer**. Download the
versioned installer, inspect it, and select this prerelease explicitly:

For an existing installation, [create and verify a portable backup before
switching binaries](docs/getting-started.md#upgrade-an-existing-installation).

```bash
curl -fsSL https://raw.githubusercontent.com/mateu/graphrag-notes/v0.1.0-rc.2/scripts/install.sh \
  -o /tmp/graphrag-install.sh
bash /tmp/graphrag-install.sh --help
bash /tmp/graphrag-install.sh --version 0.1.0-rc.2
export PATH="$HOME/.local/bin:$PATH"
graphrag --version
```

For a new configuration, choose the local Ollama preset:

```bash
graphrag init --backend ollama
graphrag init --backend ollama --write
```

For an existing configuration, run `graphrag init` to inspect it;
`--write` creates a new file and refuses to overwrite one.
The [source-build path](docs/getting-started.md#build-from-source) remains
available for development, Intel Macs, and Linux.

Use [Ollama](https://ollama.com/download) for the local inference backend.
Start Ollama in another terminal with `ollama serve` if it is not already
running, then download the two models and try the bundled fictional notes:

```bash
ollama pull bge-m3:latest
ollama pull phi4-mini:latest
graphrag init --check
graphrag doctor
graphrag import "$HOME/.local/share/graphrag-notes/samples/first-notes.md"
graphrag search "What is the Atlas project launch plan?" --limit 3
```

Expect a result containing the Atlas launch plan. Before the first import,
`doctor` can warn that the database does not exist yet; the import creates it.
`init` previews configuration without opening the database or contacting
providers. `--check` adds provider checks. Copy a result's **Inspect** or
**Open source** command to preserve its ID, revision, configuration, and database.
The [daily workflow](docs/daily-workflow.md) walks through capture, refresh,
find, inspect, open, reuse, review, and recovery with fictional Atlas notes.

For a terminal session, select a result once and reuse that selection:

```bash
graphrag capture --editor
graphrag workspace
# Inside: keyword Atlas → select 1 → inspect / open / copy
```

Browsing, keyword search, source status, and saved proposal review work offline.
The workspace holds the embedded database until you quit; other commands using
that database must wait for the workspace to exit.

The published [v0.1.0-rc.1](https://github.com/mateu/graphrag-notes/releases/tag/v0.1.0-rc.1)
remains an onboarding baseline and predates these daily-use commands. Its
[versioned install instructions](docs/getting-started.md#published-onboarding-baseline)
remain available. See the [rc.2 release guide](docs/releases/0.1.0-rc.2.md) for
candidate scope, installation, and publication verification.
Prereleases need an explicit `--version`; default installer lookup selects stable releases.

[Operating runbooks](docs/operations.md) cover existing database migration,
backups, and reindexing. [Daily workflow validation](docs/daily-workflow-validation.md)
describes automated measurements and live smoke testing; older feature
validation records retain their original evidence. Native Linux and Intel macOS
release acceptance follows in [#66](https://github.com/mateu/graphrag-notes/issues/66).
Current source adds an optional [shared MCP service](docs/shared-mcp.md) for
OpenClaw, Hermes and the remote CLI, with per-instance authorization and durable
capture retries, [uploaded Markdown and durable jobs](docs/remote-upload-jobs.md).
Published rc.2 predates these commands. The remaining remote editing and
two-computer deployment work is tracked under
[#56](https://github.com/mateu/graphrag-notes/issues/56).

## Architecture

For daily Markdown use, [register a folder and sync explicitly](docs/folder-sync.md).
Dry-run plans and unchanged sync work offline; missing files stay searchable
until a separate preview and confirmation prune their generated records.

```text
┌─────────────────────────────────────────────────────────────────┐
│                        CLI / Future Web UI                      │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                         Rust Service Layer                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Librarian  │  │   Search    │  │  Gardener   │             │
│  │   Agent     │  │   Agent     │  │   Agent     │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              SurrealDB (Embedded RocksDB)                │   │
│  │         Graph + Vector + Full-Text in one DB             │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                          HTTP/JSON
                                │
┌─────────────────────────────────────────────────────────────────┐
│                    Inference Backends                           │
│                                                                 │
│  Embeddings: TEI or Ollama                                      │
│  Extraction: TGI or Ollama                                      │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Hybrid Search**: combines semantic (vector) search with keyword (full-text) search
- **Knowledge Graph**: notes connect via typed relationships (`supports`, `contradicts`, `related_to`)
- **Entity Extraction**: local structured extraction via TGI or Ollama
- **Gardener Agent**: finds orphan notes and suggests connections
- **Local-First**: all data stored locally, inference runs locally
- **Chat Retrieval**: import chats, search messages, and build prompt-ready augmentation context with citations
- **Source lifecycle**: idempotent Markdown imports with inspect, dry-run deletion, and safe reimport operations
- **Markdown-aware ingestion**: structure-aware chunks retain heading provenance and stable identities across source refreshes

### Markdown chunking

Markdown imports use a deterministic, local chunker rather than blank-line
splitting. Sizes are **Unicode characters** (not bytes or model tokens):
`librarian.min_chunk_size`, `target_chunk_size`, and `max_chunk_size` define
the soft and hard bounds, while `chunk_overlap` copies a tail from the prior
chunk when it still fits below the hard maximum. Headings, paragraphs, lists,
block quotes, thematic boundaries, and fenced code are recognized. Short
adjacent blocks under the same heading are merged; long prose prefers sentence
boundaries, then UTF-8-safe character boundaries.

Fenced code is kept whole whenever it fits. A fenced block larger than the hard
maximum is exceptionally split at UTF-8-safe boundaries and marked as such in
its persisted chunk metadata. Displayed note content remains source text;
heading context is added only to the text embedded and indexed for search.

Each chunk records a deterministic key, ordinal, heading path, source line and
byte span, overlap predecessor, and content hash. Without configured overlap,
its source span reproduces the displayed source substring exactly; copied
overlap text is identified separately. Refreshes are copy-on-write: the previous successful
generation remains searchable until a complete pending generation promotes.
Unchanged chunks are aligned by content and heading context in document order,
so inserting or removing earlier chunks does not redirect their relationships.
They retain embeddings and original creation time, but receive a new note
record ID in the promoted generation. Changed/new chunks are re-embedded and
are eligible for fresh entity extraction; removed or ambiguous chunks are
deleted through the source lifecycle cascade. Tune these settings in `config.toml` or with
`GRAPHRAG_LIBRARIAN_{MIN,TARGET,MAX}_CHUNK_SIZE` and
`GRAPHRAG_LIBRARIAN_CHUNK_OVERLAP`.

### Augmentation packing

`augment` and `eval-augment` use a deterministic, local packing stage after
retrieval. It counts the whole rendered prompt block (including `<context>`,
citation labels, and headers), clips long hits around lexical query matches,
and suppresses near duplicates with token-set Jaccard similarity. The selection
score is `(1 - novelty_weight) * relevance + novelty_weight * novelty`; a
candidate below `min_relevance` is never chosen just because it is novel.

By default, token usage is a conservative **estimated** count that never
downloads a tokenizer or contacts a provider. Library callers with a locally
installed model tokenizer can inject a `TokenCounter` and receive **exact**
mode in `AugmentContext.diagnostics`. The human command prints the same stable
diagnostics (mode, header tokens, and drop reasons); `AugmentDiagnostics`
is paired with a versioned retrieval-evidence contract. Explain surfaces use
the same object for human and JSON output: fusion/vector/full-text channels,
accepted graph paths, provenance, selected source spans, token counts, and
typed inclusion or drop decisions. Evidence is observational—it never changes
ranking or context selection—and intentionally excludes prompts, credentials,
headers, and unrelated absolute local paths.
derives `Serialize` for JSON/API callers. A zero or too-small budget yields an
empty context rather than a prompt block that exceeds its cap.

## Runtime model

The current implementation is **Rust-first**.

The CLI talks directly to inference services via Rust clients:
- `TeiClient` for embeddings
- `TgiClient` for extraction

Supported backend modes:
- **Default:** TEI + TGI
- **Alternative:** Ollama for embeddings and extraction

### Default endpoints

- `TEI_URL=http://localhost:8081`
- `TGI_URL=http://localhost:8082`
- `TEI_PROVIDER=tei`
- `TGI_PROVIDER=tgi`

### Ollama mode

Set:

```bash
export TEI_PROVIDER=ollama
export TGI_PROVIDER=ollama
```

Defaults:
- Ollama URL: `http://localhost:11434`
- Embedding model: `bge-m3:latest` (matches the repo's 1024-dim schema)
- Extraction model: `phi4-mini:latest`

## Configuration and daily use

### Runtime configuration

GraphRAG Notes resolves runtime settings in this fixed order: compiled defaults,
an optional TOML file, compatible environment variables, and explicit CLI
flags. A configuration file is optional; a fresh install retains the historic
database location of `~/.graphrag/data-v3` and the existing local TEI/TGI
defaults.

| Layer | How it is selected |
| --- | --- |
| Defaults | Built into `graphrag` |
| TOML | `--config PATH`, otherwise `GRAPHRAG_CONFIG`, otherwise the OS configuration path if it exists |
| Environment | `GRAPHRAG_*`, plus the established `TEI_*`, `TGI_*`, and `OLLAMA_URL` variables |
| CLI | Explicit flags such as `--db-path`, `search --limit`, or `augment --max-tokens` |

The default configuration path is
`~/Library/Application Support/graphrag/config.toml` on macOS, or
`$XDG_CONFIG_HOME/graphrag/config.toml` on Linux (falling back to
`~/.config/graphrag/config.toml` when XDG_CONFIG_HOME is unset). `graphrag init`
prints the exact selected path. The database remains `~/.graphrag/data-v3` on
both platforms unless overridden.

The checked-in [`config.toml`](config.toml) is a complete template, but it is
not auto-loaded from the current directory. Use `init` to create a new
configuration at the OS location, or pass the template explicitly:

```bash
graphrag init --backend ollama
graphrag init --backend ollama --write
graphrag config validate
graphrag config show
graphrag --config ./config.toml search "configuration precedence"
```

`config validate` opens neither the database nor inference services, so invalid
values fail before application startup. `config show` emits the resolved TOML.
The current local-provider configuration has no secret-valued fields; avoid
putting credentials in a checked-in TOML file.

Hybrid retrieval defaults to reciprocal-rank fusion (RRF):
`vector_weight / (rrf_k + vector_rank) + fulltext_weight / (rrf_k + fulltext_rank)`.
This uses ranks because vector distance and BM25 are not calibrated to a common
scale. `[search]` also controls the bounded per-retriever candidate pool and
the relative weights used when `--scope all` merges notes, messages, and
conversation summaries. Ordering ties are stable: fused score, strongest
component rank, hit type, then canonical record ID. `weighted` is retained as
a configuration option only to compare with the pre-RRF behavior.

`search` and `augment` additionally accept `--graph=off|auto|on` (default:
`auto`). `off` preserves the hybrid-only path. `auto` and `on` use local,
deterministic canonical entity/alias matches to seed a bounded traversal over
accepted `supports`, `contradicts`, `derived_from`, and `related_to` edges;
they never call TGI and never read pending or rejected proposals. The resulting
notes are merged into the existing ranker and augmentation packer, rather than
using a second scoring or packing system. `auto` leaves the hybrid result
unchanged when it finds no useful graph candidates; `on` exposes the same safe
bounds explicitly for inspection.

Graph traversal defaults to one hop and is validation-capped at two. The
`[search].graph_*` settings bound entity/note seeds, per-node fanout, edge
types/directions, minimum confidence, per-hop decay, and the total candidate
set. Search output reports candidate counts; graph-derived citations carry the
canonical seed note, typed/directed edge IDs, confidence, hop/decay, source URI,
and original chat provenance IDs when available. This makes every graph result
reconstructable while preventing cycles, self-edges, and high-degree expansion.

Environment compatibility is preserved: `TEI_PROVIDER`, `TEI_URL`,
`TEI_MODEL`, `TGI_PROVIDER`, `TGI_URL`, `TGI_MODEL`, `OLLAMA_URL`, and
`TEI_MAX_BATCH` map to `[inference]`; `GRAPHRAG_DB_PATH` maps to
`[database].path`. The complete supported override list and defaults are in
[`.env.example`](.env.example).

The remaining established inference and Librarian environment names are also
typed and validated: `TEI_PROMPT_NAME_QUERY`, `TEI_PROMPT_NAME_PASSAGE`,
`STRICT_ENTITY_JSON`, `EXTRACT_MAX_ENTITIES`, `EXTRACT_MAX_RELATIONSHIPS`,
`TGI_OLLAMA_TIMEOUT_SECS`, `TGI_OLLAMA_OPTIONS`,
`SKIP_ENTITY_EXTRACTION`, `EXTRACT_LOG_EACH`, `EXTRACT_MAX_CHARS`,
`EXTRACT_PROGRESS_EVERY`, `EXTRACT_PROGRESS_EVERY_SECS`,
`IMPORT_PROGRESS_EVERY`, and `IMPORT_PROGRESS_EVERY_SECS`. Use
`[inference].ollama_options` as an inline TOML table (for example,
`{ temperature = 0, num_ctx = 1024 }`); the environment equivalent is a JSON
object. `EXTRACT_MAX_CHARS=0` intentionally preserves its legacy meaning of no
truncation. Invalid values are rejected by `config validate` rather than being
silently ignored.

### Resilient local inference processing

Long-running embedding and entity-extraction work is bounded by the resolved
`[inference]` processing controls. Every request has a timeout; only timeouts,
connection failures, HTTP 429, and retryable 5xx responses are retried with
bounded exponential backoff and deterministic jitter. Invalid requests,
dimension mismatches, and invalid structured output fail immediately.

Successful local results are cached in the database by operation, provider,
model, prompt/schema version, and normalized content hash. Use global flags to
adjust one invocation without changing configuration:

```bash
graphrag --concurrency 2 --retry-attempts 4 extract-entities --limit 100
graphrag --no-cache extract-entities --limit 100
graphrag jobs list --format json
graphrag jobs show processing_job:example --format json
graphrag jobs resume processing_job:example
graphrag jobs cancel processing_job:example
```

Jobs persist aggregate counts, checkpoint, timestamps, the selected item set,
and last error. For an actively running persistent job, press `Ctrl-C` in the
same `graphrag` process: RocksDB intentionally prevents a second CLI process
from opening that database concurrently. The in-process cancellation request
takes effect between atomic item updates. `jobs cancel` remains useful for a
stale/runnable job once the owning process has exited. The current item is
either committed before its checkpoint advances or left pending for the next
resume; source lifecycle generations remain searchable until their staged
import is promoted.

### Add notes

```bash
graphrag add "Machine learning models learn patterns from data"
graphrag add "Neural networks are inspired by biological brains" --title "Neural Networks Basics"
graphrag add "Rust is a solid fit for local tooling" --tags "rust,systems,tooling"
graphrag import notes.md
```

### Imported source lifecycle

Markdown files have one source identity: a normalized canonical `file://` URI.
The importer hashes UTF-8 content with SHA-256 after normalizing CRLF and CR
line endings to LF. Repeating an unchanged import is a no-op; `--force`
deliberately creates a fresh generation. A changed file stages its new notes
first, then removes only notes owned by the prior source generation. If an
embedding/import step fails, partial notes for the failed generation are
removed and the last successful generation remains searchable.

```bash
graphrag import notes.md                 # created, updated, or unchanged summary
graphrag import notes.md --force         # intentionally rebuild the generation
graphrag sources list --format json
graphrag sources show source:abc123
graphrag sources delete source:abc123 --dry-run
graphrag sources delete source:abc123 --yes
graphrag sources reimport source:abc123
```

`sources delete` removes generated notes, note edges, mentions, and note
provenance in that order. It never deletes notes without a source generation,
which protects manual and legacy records even when they reference an imported
source. Entity records are shared graph vocabulary, so unreferenced entities
are retained rather than risking deletion of a user-authored entity.

### Search your notes

```bash
graphrag search "how do neural networks work"
graphrag search "machine learning" --context
# Explicit full-text retrieval works with providers stopped or without vectors.
graphrag search "Atlas launch" --mode keyword --scope all
# Compare the hybrid baseline with bounded accepted-edge retrieval.
graphrag search "Atlas" --graph=off
graphrag search "Atlas" --graph=on
graphrag augment "Atlas" --graph=auto
# `--explain` is global; JSON and JSONL use the versioned output envelope.
graphrag --explain search "Atlas" --format json
graphrag augment "Atlas" --explain --format jsonl
```

`search --mode hybrid|keyword` defaults to `hybrid`. Keyword mode uses only
full-text indexes across the selected scope, keeps source/time filters, and
reports BM25 evidence without provider calls or embedding compatibility checks.
It disables graph expansion; `--graph on` requires hybrid mode. Hybrid failures
suggest a copyable keyword command and keep their failure status. See
[offline keyword search](docs/keyword-search.md) for ranking and output contracts.
See [retrieval ranking](docs/retrieval-ranking.md) for exact-title lookup and
calibrated graph contributions in current source.

`--explain` adds compact evidence lines to human output and emits the same
versioned evidence object for machine output. It reports final/fused rank,
vector distance or BM25 score when available, accepted graph paths, provenance,
and context token/span decisions; it does not change retrieval or packing.

### Review suggested connections

```bash
# Preview candidates without writing proposals or accepted edges.
graphrag garden scan --dry-run
# Persist reviewable related_to proposals; this never mutates accepted edges.
graphrag garden scan
graphrag garden proposals list --status pending
graphrag garden proposals accept proposed_edge:ID --reason "reviewed" --yes
# Batch acceptance is deliberately guarded and only accepts Gardener related_to proposals.
graphrag garden proposals accept --all --min-confidence 0.9 --yes
# Applies only the explicitly enabled auto-apply policy.
graphrag garden apply --yes
# Undo an accepted edge without losing the proposal audit trail.
graphrag edges undo related_to:ID --dry-run
graphrag edges undo related_to:ID --yes
```

`related_to` is symmetric and is stored in lexical note-ID order, so A↔B has
one canonical accepted edge. `supports` and `contradicts` are directional and
are never inferred from embedding similarity. Gardener auto-apply is disabled
by default: enable it only with both `[gardener].auto_apply = true` and an
appropriate `auto_apply_threshold` (or `GRAPHRAG_GARDENER_AUTO_APPLY=true`).

### Terminal workspace

```bash
graphrag workspace
```

`interactive` remains an alias. Use `help` inside for selection, navigation,
capture, source status, proposal review, history, and keyboard completion.

## CLI Commands

| Command | Description |
|---------|-------------|
| `init [--backend ollama\|tei-tgi]` | Preview setup; `--write` creates new config, `--check` checks providers |
| `config show/validate` | Inspect or validate the resolved runtime configuration |
| `doctor` | Diagnose configuration, database compatibility, and providers without changing data |
| `add <content>` | Add a new note |
| `import <file>` | Import notes from a markdown file (idempotent by normalized path and content hash) |
| `sources list/show/delete/reimport` | Inspect and safely manage imported file sources |
| `import-chats <file>` | Import chat export data |
| `migrate-chats <file>` | Migrate chats into conversation/message tables |
| `search <query>` | Search notes, messages, or all |
| `augment <query>` | Build prompt-ready retrieval context with citations |
| `eval-augment <file>` | Evaluate augmentation retrieval quality |
| `list` | List recent notes |
| `garden scan` / `garden proposals` | Persist, inspect, and review auditable Gardener proposals |
| `edges delete` / `edges undo` | Safely delete an accepted edge with `--dry-run` or `--yes` |
| `stats` | Show database statistics |
| `workspace` / `interactive` | Persistent terminal browsing, capture, and connection review |
| `embedding-dim` | Show embedding dimension for the active provider |
| `extract-entities` | Extract entities for notes missing entity links |
| `jobs list/show/resume/cancel` | Inspect and control durable local inference work |
| `backup create/verify/restore` | Create, validate, and safely restore a portable logical backup |
| `export <path> --format jsonl` | Stream portable logical records with a checksum manifest sidecar |
| `import-data <path>` | Validate and import JSONL only into an explicit fresh database |
| `reindex --notes|--messages|--summaries|--all` | Durably rebuild vectors with all-or-nothing model cutover |

### Retrieval evaluation

`eval-augment` accepts a JSON array or JSONL file. Legacy cases using `expected_ids` and
`expected_contains` remain valid. Version 2 cases may add `k`, graded `relevance` records,
expected source or conversation provenance, and forbidden IDs/text. Exact IDs are normalized
case-insensitively; substring expectations are reported separately from rank metrics.

```bash
# Create a stable JSON report to review or commit as a baseline.
graphrag eval-augment tests/fixtures/eval/cases-v2.jsonl --format json > /tmp/eval-baseline.json

# Fail only if a selected quality metric falls more than the permitted amount.
graphrag eval-augment tests/fixtures/eval/cases-v2.jsonl \
  --baseline /tmp/eval-baseline.json \
  --max-regression recall_at_k=0.02 \
  --max-regression mrr=0.02
```

Reports include Recall@k, Precision@k, MRR, nDCG@k for graded cases, provenance accuracy,
context budget use, and per-query/aggregate latency. Cases with no expectation are explicitly
`UNSCORED`: they contribute to latency and budget statistics but not relevance aggregates. The
fixture identifiers under `tests/fixtures/eval/` are synthetic and contain no private notes or
chat data.

## Development

### Run Tests

Basic test suite:

```bash
cargo --config 'build.rustc-wrapper = ""' test
```

Inference-backed integration test with local Ollama:

```bash
TEI_PROVIDER=ollama \
TGI_PROVIDER=ollama \
TEI_URL=http://localhost:11434 \
TGI_URL=http://localhost:11434 \
TEI_MODEL=bge-m3:latest \
TGI_MODEL=phi4-mini:latest \
cargo --config 'build.rustc-wrapper = ""' test -p graphrag-agents --test integration_test -- --ignored
```

### Project Structure

```text
graphrag-notes/
├── crates/
│   ├── core/      # Domain types (notes, entities, edges, chat export)
│   ├── db/        # SurrealDB layer and schema
│   ├── agents/    # Librarian, Search, Gardener, inference clients
│   └── cli/       # Command-line interface
├── docker/
└── tests/
```

## How It Works

### Data Model

**Notes** are the atomic units of knowledge:
- content
- embedding (currently 1024-dim in the Rust path)
- type (claim, definition, observation, etc.)
- tags

**Entities** are extracted concepts:
- people, organizations, technologies, concepts
- canonical names for deduplication

**Edges** are typed relationships:
- `supports`
- `contradicts`
- `related_to`
- `mentions`
- provenance links from notes to imported conversations/messages

### Search Pipeline

1. convert query to embedding
2. retrieve from vector and full-text search
3. merge and rerank
4. optionally enrich with graph context

### Agent Roles

| Agent | Trigger | Purpose |
|-------|---------|---------|
| Librarian | New content | Ingest, embed, extract entities |
| Search | User query | Fast hybrid retrieval |
| Gardener | Scheduled/manual | Find orphans, suggest links |

## Future Plans

- [ ] Alchemist Agent (synthesis / state-of-knowledge docs)
- [ ] Critic Agent (find contradictions, gaps)
- [ ] PDF/voice ingestion
- [ ] Web UI
- [ ] Multi-user support

## License

MIT
