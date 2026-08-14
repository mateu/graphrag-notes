# GraphRAG Notes

A local-first GraphRAG notes system built around a Rust CLI, hybrid retrieval, and graph/provenance links.

## Architecture

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

## Quick Start

### Prerequisites

- Rust 1.75+ (install via [rustup](https://rustup.rs/))
- Running local inference backends:
  - either **TEI + TGI**
  - or **Ollama**
- [sccache](https://github.com/mozilla/sccache) optional but recommended for fast builds

### SurrealDB 2.x → 3.x migration (embedded RocksDB)

If you already have a persistent v2 database, do **not** point the v3 app at it directly. The safe path is:

1. stop anything using the live DB
2. make a full copy of the v2 RocksDB directory
3. export that copy with a v2 Surreal binary using `--v3`
4. import into a fresh v3 RocksDB directory
5. validate with `stats`, `list`, and `search`

Example dry-run commands:

```bash
# 1) copy the old DB
cp -a ~/.graphrag/data ~/.graphrag-migration-backups/data-v2-copy-$(date +%Y%m%d-%H%M%S)

# 2) start SurrealDB 2.6.5 against the copied DB
/tmp/surreal2-binary/surreal2.6.5 start \
  rocksdb:~/.graphrag-migration-backups/data-v2-copy-YYYYMMDD-HHMMSS \
  --bind 127.0.0.1:8102 --unauthenticated

# 3) export in v3-compatible format
/tmp/surreal2-binary/surreal2.6.5 export \
  --endpoint http://127.0.0.1:8102 \
  --namespace graphrag \
  --database notes \
  /tmp/graphrag-v3-export.surql \
  --v3

# 4) start a fresh v3 target
~/.local/bin/surreal3.0.5 start \
  rocksdb:/tmp/graphrag-v3-restore \
  --bind 127.0.0.1:8103 --unauthenticated

# 5) import into v3
~/.local/bin/surreal3.0.5 import \
  --endpoint http://127.0.0.1:8103 \
  --namespace graphrag \
  --database notes \
  /tmp/graphrag-v3-export.surql

# 6) validate with the app (run one command at a time; RocksDB locks)
cargo run -q -p graphrag-cli -- --db-path /tmp/graphrag-v3-restore stats
cargo run -q -p graphrag-cli -- --db-path /tmp/graphrag-v3-restore list --limit 3
TEI_PROVIDER=ollama TGI_PROVIDER=ollama TEI_URL=http://127.0.0.1:11434 TGI_URL=http://127.0.0.1:11434 \
  cargo run -q -p graphrag-cli -- --db-path /tmp/graphrag-v3-restore search "migration" --limit 3
```

Notes:
- Use `rocksdb:/path/to/db`, not a plain filesystem path, with the Surreal CLI.
- Avoid concurrent access to the same DB path; overlapping processes will fail on the RocksDB `LOCK` file.
- Validate on a copied DB before doing a real cutover.

### Application schema migrations

On startup, GraphRAG Notes applies its own numbered schema migrations and records
them in the `schema_migration` table. This history is for application schema
changes only: it does not upgrade a SurrealDB 2.x data directory to 3.x. Use the
preceding export/import runbook for that engine upgrade.

To inspect the version that the running binary supports, use `graphrag
schema-version`. A database with a schema version newer than the binary is
rejected with a clear error; do not manually edit migration records. New
application migrations must be additive, immutable, and committed as a new
numbered migration rather than editing one that may already have run.

### Doctor and embedding compatibility

Run the read-only local-stack diagnostic before changing providers or
troubleshooting a database:

```bash
graphrag doctor
graphrag doctor --format json
```

`doctor` never applies migrations, repairs data, deletes records, or rebuilds
indexes. Its JSON contract has a stable `schema_version`, overall `status`,
`exit_code`, and a list of named checks. Exit code `0` is healthy, `1` is
warning-only, and `2` is a failed diagnostic.

Every vector write and vector query records or verifies the active embedding
provider, model, and dimension against the database metadata. A different
dimension or a different model with the same dimension is rejected before
vector work begins. Reindexing is deliberately not automatic; the diagnostic
prints the future command `graphrag reindex --all` rather than silently
changing an existing index.

| Doctor diagnostic | Corrective action |
| --- | --- |
| `database_open`: RocksDB lock | Stop the other GraphRAG process using the reported database path, then retry. |
| `application_schema` or `schema_objects` failed | Use a binary that supports the database or run one normal GraphRAG command to apply pending migrations; never edit migration rows manually. |
| `embedding_metadata` missing | Start a healthy embeddings provider, then run an ingestion or vector-search command to initialize the empty corpus metadata. |
| `embedding compatibility check failed` | Keep the prior embedding provider/model, or rebuild explicitly with `graphrag reindex --all` when that command is available. |
| `embedding_provider` or `extraction_provider` unavailable | Start the configured local provider. Database-only commands such as `list`, `stats`, and `schema-version` remain available while it is down. |

### 1. Start inference backends

#### Option A: TEI + TGI via Docker Compose

```bash
docker compose up -d
```

This starts:
- TEI embeddings on `http://localhost:8081`
- TGI extraction on `http://localhost:8082`

#### Option B: Ollama

Make sure Ollama is running, then set:

```bash
export TEI_PROVIDER=ollama
export TGI_PROVIDER=ollama
```

Recommended local models:

```bash
export TEI_URL=http://localhost:11434
export TGI_URL=http://localhost:11434
export TEI_MODEL=bge-m3:latest
export TGI_MODEL=phi4-mini:latest
```

You can verify the embedding model matches the schema with:

```bash
cargo run -q -p graphrag-cli -- embedding-dim
```

Expected output:

```text
Embedding dimension: 1024
```

### 2. Build and Run the CLI

```bash
cargo build --release
cargo run --release -p graphrag-cli -- --help
```

### Runtime configuration

GraphRAG Notes resolves runtime settings in this fixed order: compiled defaults,
an optional TOML file, compatible environment variables, and explicit CLI
flags. A configuration file is optional; a fresh install retains the historic
database location of `~/.graphrag/data-v3` and the existing local TEI/TGI
defaults.

| Layer | How it is selected |
| --- | --- |
| Defaults | Built into `graphrag` |
| TOML | `--config PATH`, otherwise `GRAPHRAG_CONFIG`, otherwise `~/.config/graphrag/config.toml` if it exists |
| Environment | `GRAPHRAG_*`, plus the established `TEI_*`, `TGI_*`, and `OLLAMA_URL` variables |
| CLI | Explicit flags such as `--db-path`, `search --limit`, or `augment --max-tokens` |

The checked-in [`config.toml`](config.toml) is a complete template, but it is
not auto-loaded from the current directory. Copy it to the XDG location or pass
it explicitly:

```bash
mkdir -p ~/.config/graphrag
cp config.toml ~/.config/graphrag/config.toml
graphrag config validate
graphrag config show
graphrag --config ./config.toml search "configuration precedence"
```

`config validate` opens neither the database nor inference services, so invalid
values fail before application startup. `config show` emits the resolved TOML.
The current local-provider configuration has no secret-valued fields; avoid
putting credentials in a checked-in TOML file.

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

### 3. Add Some Notes

```bash
graphrag add "Machine learning models learn patterns from data"
graphrag add "Neural networks are inspired by biological brains" --title "Neural Networks Basics"
graphrag add "Rust is a solid fit for local tooling" --tags "rust,systems,tooling"
graphrag import notes.md
```

### 4. Search Your Notes

```bash
graphrag search "how do neural networks work"
graphrag search "machine learning" --context
```

### 5. Run the Gardener

```bash
graphrag garden --dry-run
graphrag garden
```

### 6. Interactive Mode

```bash
graphrag interactive
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `add <content>` | Add a new note |
| `import <file>` | Import notes from a markdown file |
| `import-chats <file>` | Import chat export data |
| `migrate-chats <file>` | Migrate chats into conversation/message tables |
| `search <query>` | Search notes, messages, or all |
| `augment <query>` | Build prompt-ready retrieval context with citations |
| `eval-augment <file>` | Evaluate augmentation retrieval quality |
| `list` | List recent notes |
| `garden` | Run maintenance (find orphans, suggest connections) |
| `stats` | Show database statistics |
| `interactive` | Interactive REPL mode |
| `embedding-dim` | Show embedding dimension for the active provider |
| `extract-entities` | Extract entities for notes missing entity links |

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
