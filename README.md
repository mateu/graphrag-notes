# GraphRAG Notes

An evolving knowledge graph for your notes. Combines vector search, full-text search, and graph relationships to help you connect ideas.

## Architecture

```
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
│                      Python ML Worker                           │
│  ┌─────────────────────┐  ┌─────────────────────┐              │
│  │  Embeddings         │  │  Entity Extraction  │              │
│  │  (MiniLM-L6-v2)     │  │  (Pattern-based)    │              │
│  └─────────────────────┘  └─────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Hybrid Search**: Combines semantic (vector) search with keyword (full-text) search
- **Knowledge Graph**: Notes connect via typed relationships (supports, contradicts, related_to)
- **Entity Extraction**: Automatically identifies concepts, technologies, dates
- **Gardener Agent**: Finds orphan notes and suggests connections
- **Local-First**: All data stored locally, ML runs locally (no API keys needed)

## Quick Start

### Prerequisites

- Rust 1.75+ (install via [rustup](https://rustup.rs/))
- Python 3.10+ with [uv](https://github.com/astral-sh/uv)

### 1. Start the ML Worker

```bash
cd python
uv sync
uv run python -m ml_worker.server
```

The ML worker runs on `http://localhost:8100` and provides:
- `/embed` - Generate embeddings for text
- `/extract-entities` - Extract entities from text
- `/health` - Health check

### 2. Build and Run the CLI

```bash
# Build
cargo build --release

# Or run directly
cargo run --release -p graphrag-cli -- --help
```

### 3. Add Some Notes

```bash
# Add a note directly
graphrag add "Machine learning models learn patterns from data"

# Add with a title
graphrag add "Neural networks are inspired by biological brains" --title "Neural Networks Basics"

# Add with tags
graphrag add "Python is great for ML" --tags "python,ml,programming"

# Import from a file
graphrag import notes.md
```

### 4. Search Your Notes

```bash
# Hybrid search (vector + full-text)
graphrag search "how do neural networks work"

# With graph context (shows related notes)
graphrag search "machine learning" --context
```

### 5. Run the Gardener

```bash
# See what the gardener would do
graphrag garden --dry-run

# Apply high-confidence connections automatically
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
| `search <query>` | Search notes |
| `list` | List recent notes |
| `garden` | Run maintenance (find orphans, suggest connections) |
| `stats` | Show database statistics |
| `interactive` | Interactive REPL mode |

## Development

### Run Tests

```bash
# Rust tests
cargo test

# Python tests
cd python
uv run pytest
```

### Project Structure

```
graphrag-notes/
├── crates/
│   ├── core/      # Domain types (Note, Entity, Edge)
│   ├── db/        # SurrealDB layer
│   ├── agents/    # Librarian, Search, Gardener
│   └── cli/       # Command-line interface
└── python/
    └── src/ml_worker/  # Embedding & entity extraction
```

## How It Works

### Data Model

**Notes** are the atomic units of knowledge:
- Content (the actual text)
- Embedding (384-dim vector from MiniLM)
- Type (claim, definition, observation, etc.)
- Tags (user-defined)

**Entities** are extracted concepts:
- People, organizations, technologies, concepts
- Canonical names for deduplication

**Edges** are typed relationships:
- `supports` - one note supports another
- `contradicts` - notes are in conflict
- `related_to` - semantic similarity
- `mentions` - note mentions an entity

### Search Pipeline

1. **Query Understanding**: Convert query to embedding
2. **Parallel Retrieval**:
   - Vector search (semantic similarity)
   - Full-text search (keyword matching)
3. **Merge & Rerank**: Combine results
4. **Graph Enrichment**: Add related notes

### Agent Roles

| Agent | Trigger | Purpose |
|-------|---------|---------|
| Librarian | New content | Ingest, embed, extract entities |
| Search | User query | Fast hybrid search |
| Gardener | Scheduled/manual | Find orphans, suggest links |

## Future Plans

- [ ] Alchemist Agent (synthesis, "state of knowledge" docs)
- [ ] Critic Agent (find contradictions, gaps)
- [ ] PDF/voice ingestion
- [ ] Web UI
- [ ] Multi-user support

## License

MIT
