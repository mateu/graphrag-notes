# Operating an existing installation

Start with [getting started](getting-started.md) for a new installation. These
runbooks cover existing data, backup/recovery, model changes, and diagnostics.

## SurrealDB 2.x → 3.x migration (embedded RocksDB)

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

## Portable logical backups

`graphrag backup` is the application-level recovery format. It writes a
versioned `manifest.json` and streaming `records.jsonl` payload, then validates
the completed archive before publishing it. It preserves logical record IDs and
the graph/provenance references that use them; it does not copy a live RocksDB
directory.

Backup creation uses the resolved configured database path, which defaults to
`~/.graphrag/data-v3`. Inspect that path with `graphrag init` or
`graphrag config show` before creating a backup. If you select a configuration
with `--config PATH`, use the same flag for inspection and backup. Use
`--db-path PATH` only when deliberately backing up a different database.

```bash
# The destination must not already exist.
graphrag backup create /safe/backups/notes-2026-08-14
graphrag backup verify /safe/backups/notes-2026-08-14 --format json

# Verify first, then restore only into a fresh, nonexistent target.
graphrag backup restore /safe/backups/notes-2026-08-14 \
  --db-path /tmp/graphrag-restore --dry-run
graphrag backup restore /safe/backups/notes-2026-08-14 \
  --db-path /tmp/graphrag-restore

# JSONL transport has the same checksum-manifest contract. Import always
# names a fresh destination and can be validated without creating it.
graphrag export /safe/notes.jsonl --format jsonl --output json
graphrag import-data /safe/notes.jsonl --db-path /tmp/graphrag-import --dry-run --format json
graphrag import-data /safe/notes.jsonl --db-path /tmp/graphrag-import
```

Restore stages a sibling database, applies the current application migrations,
loads and validates the logical records, and only then renames the staged DB
into the requested target. It never overwrites an existing directory. A locked
or otherwise inaccessible directory is reported by the underlying RocksDB
open, leaving the requested target untouched.

Embeddings and inference caches are excluded by default. `--include-embeddings`
is accepted only when the archive can record the active provider, model, and
dimension. Configuration files, runtime caches, common secret fields, and local
absolute file URIs are not exported. Restoring a default archive preserves
searchable source text and graph structure, but vectors must be rebuilt with a
subsequent reindex operation before vector search is used.

This is distinct from the SurrealDB 2.x → 3.x engine migration above: use the
engine-specific Surreal export/import procedure to cross that engine boundary.
It is also distinct from reindexing/model migration, which rebuilds derived
vectors for an existing logical corpus rather than recovering its source and
graph data.

## Reindexing after an embedding-model change

Use `reindex` when intentionally changing embedding provider or model. It
probes the active provider first and rejects dimensions other than the current
1024-dimension index. `--dry-run` reports the immutable item count, target
identity, and provider-neutral input-character cost estimate without creating
a job or writing vectors.

```bash
graphrag reindex --all --dry-run --format json
graphrag reindex --all
# Resume the exact persisted job after an interruption or provider failure.
graphrag reindex --resume processing_job:... --format json
```

Reindex uses the normal durable inference cache in bounded batches, but writes
new vectors into inactive staging fields. Search continues with the prior model
until every selected item validates; one final transaction publishes all
vectors and advances embedding metadata. Failed or cancelled jobs therefore
leave the prior indexed generation intact.

## Application schema migrations

On startup, GraphRAG Notes applies its own numbered schema migrations and records
them in the `schema_migration` table. This history is for application schema
changes only: it does not upgrade a SurrealDB 2.x data directory to 3.x. Use the
preceding export/import runbook for that engine upgrade.

To inspect the version that the running binary supports, use `graphrag
schema-version`. A database with a schema version newer than the binary is
rejected with a clear error; do not manually edit migration records. New
application migrations must be additive, immutable, and committed as a new
numbered migration rather than editing one that may already have run.

## Doctor and embedding compatibility

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
prints the explicit command `graphrag reindex --all` rather than silently
changing an existing index.

| Doctor diagnostic | Corrective action |
| --- | --- |
| `database_open`: RocksDB lock | Stop the other GraphRAG process using the reported database path, then retry. |
| `application_schema` or `schema_objects` failed | Use a binary that supports the database or run one normal GraphRAG command to apply pending migrations; never edit migration rows manually. |
| `embedding_metadata` missing | Start a healthy embeddings provider, then run an ingestion or vector-search command to initialize the empty corpus metadata. |
| `embedding compatibility check failed` | Keep the prior embedding provider/model, or rebuild explicitly with `graphrag reindex --all` using the compatible 1024-dimension model. |
| `embedding_provider` or `extraction_provider` unavailable | Start the configured local provider. Database-only commands such as `list`, `stats`, and `schema-version` remain available while it is down. |
## Registered Markdown folders

Use `graphrag folders add NAME PATH`, `graphrag sync NAME --dry-run`, then
`graphrag sync NAME` for explicit folder ingestion. See [folder sync](folder-sync.md)
for include/exclude rules, durable resume, missing-file previews, and confirmed
pruning that preserves manual notes.
