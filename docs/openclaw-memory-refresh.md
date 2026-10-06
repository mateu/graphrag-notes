# Incremental indexed OpenClaw memory refresh

Run the repository-owned standard-library adapter on the computer that owns the
OpenClaw index and workspace. It sends supplied Markdown through the existing
authenticated MCP upload/jobs API; it never opens a GraphRAG database. A private
SSH tunnel carries requests to the corpus owner. No scheduler is required.

**Service requirement:** upgrade the host service to a version with
`service_status` from #94 before applying a refresh or cleanup. The adapter checks
that the authenticated caller matches the registered importer principal before
any writes. Older services fail closed with `service_status_required`; dry-run
and local status remain available without a service connection.

```sh
python3 scripts/refresh-openclaw-memory.py \
  --database "$HOME/.openclaw/agents/main/agent/openclaw-agent.sqlite" \
  --workspace "$HOME/.openclaw/workspace" \
  --state-dir "$HOME/.graphrag/openclaw-memory-refresh" \
  --collection-id clawd-main-memory --source-host clawd --source-agent main \
  --instance-id shiva-importer --server http://127.0.0.1:31057/mcp \
  --dry-run --format json
```

Use the actual workspace configured for that agent. The index must expose
`memory_index_sources` with `id,path,source,hash,mtime,size`. Only rows whose
`source` equals `memory`, with `MEMORY.md`, `USER.md`, or `memory/**/*.md` paths,
are admitted. Session indexes, conversation summaries, and other tables are
excluded. An unexpected memory path fails closed rather than silently widening
scope or interpreting an incomplete inventory as deletion.

The dry run takes one read-only SQLite transaction and validates all original
Markdown against its indexed SHA256 and byte size. The index contains document
metadata, not original Markdown bodies. SQLite and the filesystem do not share
an atomic snapshot: an original edited since indexing, a missing original, a
symlink, or an original changing while read therefore causes a failed snapshot
with zero uploads. Let OpenClaw finish indexing, then retry. Concurrent SQLite
writes do not mix old and new inventories in one export. `mode=ro` and
`query_only` prevent writes to the source database; ordinary SQLite WAL reader
coordination is used, without `immutable=1` ignoring live WAL contents.

The dry run creates no state, reads no credentials, and makes no network or model
calls. To apply, supply the importer token through `GRAPHRAG_TOKEN` (or a named
`--credential-env`) and repeat the command without `--dry-run`. Load credentials
through the existing private credential mechanism; never put a token in command
arguments. The adapter accepts only loopback HTTP `/mcp`, disables HTTP proxies,
and refuses redirects. Use a tunnel for remote access. No daily OpenClaw agent
configuration or tool permissions change.

## Identity, incremental updates, and recovery

Register one collection state directory per importer principal/host/agent and
endpoint. Protect its ancestors and backups; the directory must be owned by the
current user with mode 0700, and retained state files are mode 0600. The private
`collection.json` contains exact pinned Markdown, source paths, durable admission
IDs, and verification history. Never publish it. `freshness.json` contains only
bounded identities, timestamps, counts, and static failure/retry categories.

Source keys retain the original importer convention:
`openclaw-memory/HOST/AGENT/PATH/part-NNNN`, with a stable hashed long-path form.
Keep the **same authenticated importer principal** to update an existing seed;
using a different principal intentionally creates independent sources. The first
run with new adapter state registers previously imported sources by uploading
the same document keys with collection metadata. Server unchanged-content
processing avoids fresh embeddings while confirming ownership and provenance.
Legacy bespoke progress files are retained separately and are not silently
adopted as verified adapter state.

Before any initial upload, the adapter inspects existing sources by the
authenticated document key. A different extraction policy fails with
`existing_extraction_policy_requires_adoption`. If an earlier entity pilot
enriched an otherwise matching source, explicitly add `--adopt-existing-policy`
to retain that part's existing extraction policy. Adoption requires exact owner,
document key, supplied bytes, title and original provenance, a ready generation,
and compatibility with the service's current processing snapshot. It persists
the per-source policy for future changed uploads and recovery; new parts still
use the collection default. A changed processing snapshot fails closed and
requires a separately reviewed migration rather than silently re-embedding.

Metadata registration sets the server's `preserve_unchanged` upload guard. Under
the generation lock, exact input, title, extraction and processing policy must
still match; otherwise the job fails without replacing existing chunks or graph
mentions. A compatible registration retains the successful generation, chunk
IDs and previously extracted mentions without embedding or extraction calls.
This is a per-part guard, not an all-parts transaction. Resume a policy-review
failure with `--resume --adopt-existing-policy` only after reviewing its cause.

UTF-8 bytes, BOM, whitespace, and line endings are preserved exactly. Originals
over 64 KiB split at UTF-8/newline boundaries into approximately 49 KiB parts.
Part identities stay stable as originals grow or shrink. Original path, hash,
part index/count, and collection identity remain client-supplied provenance;
trusted `instance_id` and audit actor come from server authentication. The client
cannot overwrite those identities through metadata.

Every attempted update has a persisted request identity. Identical retries
recover its exact durable receipt; a later A→B→A content change receives a new
request so it cannot replay the historical first A. An unchanged refresh sends
no upload requests and performs no embedding or extraction, but verifies the
registered sources still match their last successful generations.

Each part uses the existing staged source generation. Its previous successful
chunks remain searchable until the replacement commits. Multi-part documents
are **not promoted as one transaction**: a partial refresh can contain successful
new parts alongside still-searchable old parts. Status reports this explicitly;
obsolete tails remain until reviewed cleanup. The first extraction policy is
fixed at registration; extraction defaults off. Use an explicit
`--extract-entities` registered collection only when extraction is intended.

An interruption retains exact pinned input and already accepted jobs continue
server-side. Resume without reading the mutable index again:

```sh
python3 scripts/refresh-openclaw-memory.py \
  --state-dir "$HOME/.graphrag/openclaw-memory-refresh" --resume --format json
```

After repairing a provider failure, add `--resume-jobs`. Resume attempts are
persisted and bounded (one by default, up to five with `--max-resumes`). Validation,
ownership, compatibility, and cancelled-job failures are not automatically
resumed. Preserve state and inspect the owned job; do not discard uncertain
admissions or start a newer snapshot over a pending attempt. The run deadline is
four hours; each observed active job is bounded to 20 minutes. A process lock
prevents two refreshes from using the same state directory concurrently.

```sh
python3 scripts/refresh-openclaw-memory.py \
  --state-dir "$HOME/.graphrag/openclaw-memory-refresh" --status --format json
```

Status needs no credentials, SQLite, MCP connection, or inference. It reports
last attempt, last fully verified success, `running/partial/failed/paused`,
created/changed/unchanged/failed/missing **part counts**, and pending parts.
Snapshot failures count unreadable/diverged originals under `failed`, since no
parts can safely be prepared from those originals. Plan counts describe proposed
parts; a partial status does not claim all proposed changes were committed.
Remote `doctor --refresh-status-file .../freshness.json` consumes this local
evidence with endpoint/principal matching; it is client evidence rather than a
live server freshness guarantee.

## Explicit reconciliation

A successful refresh previews original documents removed from the index and
tail parts made obsolete by resizing. Missing content remains searchable until
this separate action is confirmed. The importer needs an explicit `delete`
grant for cleanup, in addition to `read,upload,jobs`; ordinary refresh does not
require deletion permissions.

```sh
python3 scripts/refresh-openclaw-memory.py \
  --state-dir "$HOME/.graphrag/openclaw-memory-refresh" --reconcile --format json
```

Inspect the private `cleanup-preview.json` for exact paths, reasons, IDs, and
revisions. The public-safe command output shows removed-original/obsolete-part
counts and a plan SHA256. It makes no server changes. Apply that exact plan:

```sh
python3 scripts/refresh-openclaw-memory.py \
  --state-dir "$HOME/.graphrag/openclaw-memory-refresh" \
  --reconcile --yes --plan-sha256 REVIEWED_SHA256 --format json
```

`delete_uploaded_source` verifies authenticated source ownership, stored
collection provenance, a reviewed source revision, ready generation, and no
queued/running upload against that source. Upload admission, resume, and claim
transitions are serialized with retirement so a concurrent job cannot bypass the
active-job check. Generated chunks and their dependent
links retire atomically with the durable receipt. Detached/manual notes survive,
retaining original source provenance. Unrelated collections and manual captures
cannot be selected. A changed plan or source revision stops cleanup. After a
lost acknowledgement or interruption, rerun the identical reconciliation command
and hash to recover its receipts. Resume pending cleanup before a newer refresh.

Retirement marks earlier failed/cancelled upload executions for that source with
the explicit `retired` phase and `source_retired` error category. They retain
their original terminal status, cancellation flag, input, and admission receipt,
but cannot be resumed or claimed to recreate removed content. This fence also
survives portable backup/restore. A genuinely new upload request may deliberately
create the logical source again.

If cleanup was interrupted, freshness evidence reports `retry.action=reconcile`
and the original public-safe `retry.plan_sha256`; the human retry gives that
exact plan command. Normal evidence uses `resume` or `refresh` and a null plan.
An unchanged part is pending until it has been reverified during this attempt;
its copied last-good source snapshot is retained separately and does not claim
that a failed revalidation succeeded.

## Validation and optional scheduling

`python3 -m unittest discover -s scripts/tests -p test_openclaw_memory_refresh.py`
uses fictional SQLite originals and MCP/provider doubles. Rust uploaded-job
integration tests exercise real in-memory staged generations, owner/revision
guards, queued-job races, retained manual notes, and transactional receipt failure.
These tests run in the ordinary offline CI gates.

Before scheduling, privately verify two actual source-host-to-corpus-owner runs:
the first reaches complete, the second is unchanged with no new jobs/inference,
and search/inspection preserve expected provenance. Retain only sanitized counts
and timings in public evidence. Optional cron/launchd/systemd scheduling should
reuse the same private state, environment credential, and tunnel; detect a pending
attempt and alert rather than enabling automatic failed-job resume or deletion.

Collection counts describe registered parts. Operator output separately reports
server source actions (`created`, `updated`, `unchanged`); registering an existing
part may update its collection metadata while retaining its successful source
generation, chunks and graph mentions. Freshness counts retain their part-based
contract.

Every existing source must match its original URI, label, host, agent, path,
part ordinal and other provenance. For an edited legacy part using the same
extraction policy, only `original_sha256` and `parts` are content-version facts;
these may change with the newly indexed original. Registered edits validate the
last verified provenance first. Graph-policy adoption still requires exact
content/title and full original provenance apart from collection metadata.
A negative source lookup pins a `create_only` admission: the service rechecks
absence under its generation lock before any inference or source mutation.
A source that appeared after the lookup produces a failed conflict job requiring
an operator to inspect and start a reviewed new attempt.

Existing-source admissions also pin `expected_source_revision`. The service
checks this provider-free source snapshot under its generation lock, including
original content, provenance and processing policy. A queued source update that
wins after client preflight causes a conflict before mutation or inference.
The adapter never silently replaces that newly published original or its graph.

Reviewed cleanup records retired part identities in private collection state.
A retained source stub with detached/manual notes is explicitly marked retired
by the service. If the same indexed path reappears, the adapter requires that
collection's verified cleanup evidence and the exact retired source revision,
then creates a new generation using its retained per-source policy. Detached
notes remain intact. A different collection cannot silently claim an unknown
retired stub, and an old inspected draft cannot bypass the retirement fence.
