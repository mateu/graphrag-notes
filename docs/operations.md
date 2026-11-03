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

## SurrealDB 3.2 → 3.3 upgrade

SurrealDB 3.3 upgrades on-disk data in a one-way format change: after the
upgrade, a 3.2 binary can misread the migrated data. Before replacing a 3.2
GraphRAG binary or opening its database with 3.3, create and verify a portable
backup with `graphrag backup create` and `graphrag backup verify`. Keep the
previous binary available only for restoring that verified backup into a fresh
location; do not point it at a database that 3.3 has already opened.

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

## Shared-service usage records

The shared service (`graphrag serve`) logs one INFO line per MCP tool call,
named `graphrag_usage` (for example
`2026-11-02T01:00:00Z  INFO graphrag_usage principal="…" tool="search_notes" …`). Each line names the authenticated principal
and tool, the outcome (`ok` or the stable error code) and latency, and records
allowlisted options (`mode`, `graph`, `scope`, `limit`), the query length and
result statistics: hit count, top score and, for `build_context`, graph
candidates considered/selected, relevance drops and total tokens. Lines never
contain query text, record IDs, content, titles or source URIs.

For `search_notes` and `build_context`, usage lines also expose executed
retrieval phases as flat millisecond fields, rounded to 0.1 ms. The closed set is:

| Field | Work measured |
| --- | --- |
| `embedding_ms` | Query embedding |
| `lexical_ms` | Lexical retrieval |
| `vector_fusion_ms` | Vector retrieval and baseline fusion |
| `graph_query_candidates_ms` | Preparing ranked graph seed candidates |
| `graph_entity_matching_ms` | Query-to-entity matching |
| `graph_mentions_ms` | Mention lookup |
| `graph_seed_evidence_ms` | Seed evidence lookup |
| `graph_edge_expansion_ms` | Graph edge expansion |
| `graph_note_hydration_ms` | Loading graph candidate notes |
| `graph_provenance_ms` | Graph provenance lookup |
| `context_packing_ms` | Context packing (`build_context`) |

Phases that did not run are absent, not zero. Repeated phases (including
multiple graph hops) are summed per request. Timings cover measured retrieval
work, not the entire request wall time. These fields add no query text,
fingerprints, record IDs or content.
`graph_status` is `off` (explicitly disabled), `ok` (completed), `timed_out`
(budget expired; unchanged graph-off results returned), or `skipped` (graph work
not applicable). This is separate from `outcome`: a timeout fallback can have
`outcome=ok`, while errors retain their existing outcome codes. Search responses
include `data.graph_status` in the MCP result; `build_context` includes it in
its diagnostics. The CLI JSON envelope wraps that MCP result in its own `data`.

### Graph-phase budget and rollback

The host's `[search]` configuration accepts `graph_phase_budget_ms` (milliseconds;
default **2000**, **0** disables the bound). It covers graph candidate work,
including hydration and provenance, not baseline lexical/vector retrieval,
embedding, or context packing. Expiry abandons graph work and returns exactly
the graph-off baseline (same records, order and scores) with
`graph_status=timed_out`; it is not an overall request deadline. Keep a separate
client timeout for deployment checks.

Before running an older binary, **remove `graph_phase_budget_ms` from the
configuration**: older `SearchConfig` uses `deny_unknown_fields` and rejects the
new key. This configuration rollback does not override database-version rollback
restrictions described above.

### Post-deploy read probe

Run [`scripts/post-deploy-probe.py`](../scripts/post-deploy-probe.py) after an
owner-authorized deployment using an existing note's known keyword and a
read-capable credential inherited from the environment. It checks remote
`doctor`, keyword graph-off retrieval, inspection of the top note, and hybrid
graph-on retrieval with `graph_status=ok`. It fails closed at the first failing
step, with a per-command client timeout (15 seconds by default), and prints no
note text or IDs. See [remote diagnostics](remote-diagnostics.md#post-deploy-read-probe)
for invocation, doctor warning semantics, and isolated acceptance evidence.

Summarize usage records per principal and tool:

```bash
scripts/usage_report.py ~/.graphrag/shared/logs/service.stderr.log --since 2026-11-02
scripts/usage_report.py service.stderr.log --principal openclaw.clawd --json
```

The report shows call counts by day, error codes, latency p50/p95, the
empty-result rate, median hits and top score, the option mix, and how often
the graph contributed a selected `build_context` candidate. An empty result or
a low top score is a retrieval-quality signal; it does not show whether the
client used a nonempty result.

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

## Human-initiated fam deployments

[`scripts/deploy-fam.sh`](../scripts/deploy-fam.sh) versions the fam recipe. It
**does not deploy on merge**, provision a runner, or change the service plist or
configuration. Production use remains a separately authorized operator action.
It needs macOS launchd, Python 3.9+, Git, the pinned Rust toolchain under
`~/.cargo/bin`, and an already running, quiesced service (all durable jobs must
be completed). Never enable shell tracing or put a token on the command line.

```bash
# Inherit GRAPHRAG_TOKEN from the operator's environment/secret manager.
# Supply a full commit already reachable from origin/main, never a tag/short SHA.
scripts/deploy-fam.sh FULL_40_CHARACTER_SHA --dry-run
scripts/deploy-fam.sh FULL_40_CHARACTER_SHA
```

Defaults: source repository `~/dev/graphrag-notes`, state `~/.graphrag/shared`,
label `com.hunter.graphrag.shared`, plist in `~/Library/LaunchAgents`, service
port **31057**, rehearsal port **31201**. `--help` lists scratch overrides
(`--state`, `--repo`, `--label`, `--plist`, `--port`, `--rehearsal-port`, `--backup`,
`--query`). The plist must match the exact binary/config/database/loopback port
selected; it is read, never rewritten. Credentials come from the existing
plist's credential file for the service and inherited `GRAPHRAG_TOKEN` for read
probes. They are not copied into backups, command arguments, reports or logs.
Child output (which can include private data) is captured but never rendered or
persisted by the deploy tool. Command names, exit codes and timings are recorded.

The dry run is read-only, checks **cached** `origin/main`, current service,
verified vector-inclusive backup and disk capacity; it neither fetches nor
builds nor stops anything. Real deployment fetches fail-closed and verifies both
requested and deployed commits' ancestry. A private detached build worktree
leaves the source checkout unchanged; its HEAD must equal the requested SHA.
Builds use `RUSTC_WRAPPER= CARGO_BUILD_JOBS=2 nice -n 10 cargo build --release
--locked -p graphrag-cli`. Check `uptime` before running on fam. The script records
the actual toolchain and binary SHA256. Repeating a successful exact SHA with
its matching installed binary is a no-op.

A verified portable seed with embeddings is required **before stopping** the
service. The default is the prior deployment's `pre_deploy_backups.portable`;
`--backup` selects another old-binary-compatible archive. If migration files
differ, the candidate must restore this seed into a temporary DB, reach the
expected schema, preserve every per-table count, and pass authenticated keyword,
hybrid graph-on search and `build_context` on a separate loopback port. The
rehearsal process, restored DB and temporary archive are removed on success or
failure. It refuses unfinished jobs, to prevent validation from starting work.
The current live DB is never opened by rehearsal.

After preflight/rehearsal pass, cutover joins the old process after `bootout`,
verifies a byte-for-byte cold copy, and creates/verifies a **fresh** portable
backup with embeddings from the stopped database. A fresh backup failure
restarts the old binary without installing the candidate. This offline capture
necessarily occurs after stop; the already verified seed is the pre-stop gate.
The candidate is installed as `bin/graphrag.<sha7>` with an atomic active-binary
replacement. The previous binary and JSON remain versioned. Offline schema and
post-migration archive counts must match, then launchd readiness and all three
authenticated read probes must pass. The log gate's byte offset is captured
**after the old PID exits, before candidate bootstrap**: old shutdown cancellation
ERRORs are excluded; any candidate-window ERROR fails. Missing/truncated logs fail
closed. Log files must not rotate during the cutover.

### Recovery and retention

The atomic `deploy.lock` is held from preflight through recovery. Never remove a
lock because it looks old: identify the PID/owner and reconcile service/database
ownership first. A stopped or otherwise non-running service is refused as a
possible interrupted deployment. No automatic retry is attempted.

* Before any candidate DB-opening attempt: restore the old binary and restart;
  the database has not crossed the candidate boundary.
* After any candidate DB-opening attempt (including a failed command): stop and
  join the candidate; use the **previous binary** to restore the verified fresh
  portable backup into a nonexistent location. Re-export/verify restored counts,
  move the failed database aside, then swap the fresh restore into `database`.
  Only then install/start the old binary and restore its deployment JSON.
* If the candidate cannot be proven stopped, restoration fails, or counts differ,
  recovery fails closed. Reconcile launchd/PID ownership and the backup evidence
  manually before taking any further action. Do not launch the old binary against
  the existing database.

This conservative boundary applies even with no application migration diff:
SurrealDB **3.2 → 3.3** can change the physical format independently (see the
upgrade note above). A cold copy is evidence/last-resort physical recovery, not
the automatic cross-version rollback path.

Every cutover keeps `backups/deploy-TIMESTAMP-PID/REPORT.md`, `deployment.json`,
`previous-binary`, `previous-deployment.json`, the verified `cold-database`, and
pre/post portable archives. After migrated rollback, `failed-database` is retained
there too. The active success record is `bin/graphrag.deployment.json`; a failed
attempt's report does not replace the restored previous success record. Do not
remove cold copies, portable archives, old binaries or failed-migration DBs until
an operator has verified recovery and approved retention cleanup. No automatic
pruning is performed.

### Isolated acceptance rehearsal

Never test the script by stopping production. Restore a verified archive with an
old-compatible binary into a **new scratch state**, use a **scratch launchd
label**, and choose an unused port such as **31200** (internal migration rehearsal
on **31201**). Provide a scratch plist whose binary, DB, config and log paths all
point inside that state; reference the existing read-only service credential file
without copying it. The config must preserve provider/embedding identity but use
the scratch DB. Set the scratch deployment JSON to the actual restored old binary
and schema, and invoke with all scratch overrides and the verified seed. After
checking REPORT.md, record counts, hashes and probe results, boot out only the
scratch label and remove only the owned scratch state/plist. Leave production's
label, port, database, binary and checkout untouched.

The fake launchctl/service fault-injection suite is automatically discovered in
the existing required **format** CI lane (no extra pending required job):

```bash
python3 -m unittest discover -s scripts/tests -p 'test_*.py'
```

## Owner-approved fam deployment workflow

`.github/workflows/deploy-fam.yml` is a manual entry point for the versioned
`scripts/deploy-fam.sh <full-sha> [--dry-run]` interface delivered by #200.
**Merge after #200.** Adding this workflow does not authorize a deployment or
provision production access. There is no push, pull-request, or merge trigger.
The existing fam service is launchd `com.hunter.graphrag.shared`, with state
`~/.graphrag/shared` under its owning account.

### Administrative activation — explicit repository-owner approval required

Do not register a runner, create/change the Environment, alter repository or
branch protection settings, or modify the live service as part of this PR.
The owner must separately approve and perform these steps **before use**:

1. Create/protect the `fam-production` GitHub Environment. Require a trusted
   human reviewer, prevent self-review, disallow protection-rule bypass, and
   allow deployments from **selected branch `main` only**, not tags. Verify
   these settings before registering the runner: merely referencing an
   Environment in YAML does not establish protection. Approval is required
   for dry runs as well as migrations. Review the SHA, workflow revision,
   migration/rehearsal plan and rollback readiness before approving.
2. Create a dedicated runner group `fam-production`, restricted to this
   repository and the selected workflow
   `mateu/graphrag-notes/.github/workflows/deploy-fam.yml@refs/heads/main`.
   No PR, fork, branch, or other workflow may use it. Labels are routing, **not
   authorization**: repository scoping alone does not block another workflow
   in this repository. Workflow-level group restrictions require appropriate
   GitHub organization/enterprise support. If this personal repository or
   plan cannot enforce them, **leave the runner unregistered and the workflow
   inactive**; obtain owner approval for an enforceable hosting/isolation
   design first. Do not substitute a broadly accessible runner or relax this
   requirement. Audit all existing runner groups/workflows for overlapping
   access before activation.
3. On fam, install a checksum-verified GitHub Actions runner from the official
   runner download instructions into a dedicated directory, never the service
   checkout/state directory. Use a dedicated, non-admin deployment account,
   no sudo rights or unrelated SSH/cloud credentials, and only the filesystem
   and launchd access needed by the deploy script. macOS launchd user domains
   matter: confirm its approved identity can manage the existing service's
   domain without changing the service configuration. If that cannot be
   achieved with least privilege, stop and seek a separately approved design.
   Provision Rust/build prerequisites and read-only repository fetch access;
   no repository-write token is needed. Create Environment secret `FAM_GRAPHRAG_TOKEN` for an existing read-capable
   probe principal; it is inherited as `GRAPHRAG_TOKEN` only by the approved
   script step, never by builds or the hosted validator. Never inline it in
   YAML/logs. Rotate it by replacing the Environment secret and revoking the
   old service credential through the separately approved credential runbook.
4. With a short-lived registration token supplied interactively (never saved
   in docs/logs), configure `./config.sh` for this repository, runner group
   `fam-production`, and custom label `fam-graphrag-deploy`. Retain the default
   `self-hosted` and `macOS` labels. Install/start its service using the
   platform runner instructions under the approved account. Do not enable
   runner access for public forks or PR workflows. Keep the runner's workspace
   separate from the deploy script's managed checkout. Pin/verify runner
   upgrades and keep macOS/build tools patched.
5. Inspect Environment and group policy in GitHub, then approve one explicit
   dry-run dispatch. Confirm an invalid/non-main SHA fails in the hosted
   validator without reaching fam, branch dispatches skip both jobs, and a
   valid main SHA waits for Environment review before the runner starts.
   Confirm a different workflow cannot select the group. Do not weaken
   protection to run these checks. A live cutover needs separate owner
   approval after reviewing the dry-run evidence.

### Dispatch and evidence

After activation and the dependent script have landed on main:

```bash
# Use a full commit SHA, not a branch/tag/abbreviation. Default is dry-run.
gh workflow run deploy-fam.yml --repo mateu/graphrag-notes --ref main \
  -f sha=FULL_40_CHARACTER_MAIN_SHA -F dry_run=true
# Only for an explicitly owner-approved production cutover:
gh workflow run deploy-fam.yml --repo mateu/graphrag-notes --ref main \
  -f sha=FULL_40_CHARACTER_MAIN_SHA -F dry_run=false
```

The hosted validator and approved runner independently fetch main and require
commit ancestry. Inputs are shell environment values, not interpolated shell
code. The runner executes the deploy script from the approved main workflow
revision, not arbitrary input-commit code. The workflow passes
`--repo "$GITHUB_WORKSPACE"` (the #200 script option) so
even read-only dry-run ancestry uses this freshly fetched repository; the
script controls its own detached build checkout and repeats preflight.
Both dry runs and real deploys share the
`fam-production-deploy` concurrency lock, with cancellation disabled. GitHub
concurrency may replace an older pending run, so inspect the run being approved;
it is not a FIFO deployment queue. The script's host lock remains authoritative
for deployments initiated outside GitHub. Do not cancel a running cutover or
impose a timeout shorter than its build/rehearsal/rollback budget; a runner loss
or the 180-minute job limit requires incident inspection, not an automatic retry.

The final artifact step runs even after failure and collects only reports and
the deployment record from `~/.graphrag/shared/bin/graphrag.deployment.json`,
never databases, portable backups, config, or service logs. Only
`~/.graphrag/shared/backups/deploy-*/REPORT.md` files newer than this run
are staged; historical reports are excluded. The JSON is explicitly labelled
as a current-state snapshot: dry runs, safe no-ops and early preflight failures
do not write a fresh script report or update that JSON. Use the step conclusion
as the run outcome, never an older success record. Retention is 14 days;
reports can contain operational paths and should
remain accessible only to trusted repository readers. Missing reports warn
rather than hide the original preflight failure. Retain host backups according
to the script runbook, independently of GitHub artifact expiration. Dry-run is
the script's `--dry-run` path: it must not stop, build, migrate, or replace the
live service. This delivery has not registered a runner, changed Environment
policy, or executed either path on fam; activation/end-to-end proof is deferred
to those separately approved administrative steps.

### Runner rotation, removal, and incidents

For planned rotation, disable new workflow runs/runner scheduling and allow an
active deployment to finish; inspect the script lock and service health first.
Use GitHub's short-lived removal token with `./config.sh remove`, uninstall the
old runner service per the macOS instructions, remove its GitHub registration,
and securely remove its credentials/work directory. Re-register the replacement
with a fresh token and exactly the same restricted group/labels; recheck policy
and repeat the approved dry run. Rotate any host probe or read-only Git
credential it could access. For permanent removal, also disable the deployment
workflow and retain Environment/group protection until access is revoked.

For suspected compromise, immediately revoke runner access and exposed tokens,
block new approvals/dispatches, and preserve runner audit logs and run IDs.
Isolation may interrupt a cutover: inspect launchd, deployment JSON, backups,
failed databases and lock state using the script's recovery runbook. Never
blindly delete a lock, rerun a failed deployment, point the old binary at a
migrated database, or restore from an unverified backup. Preserve evidence and
obtain owner authorization before recovery changes. Rebuild the runner from
trusted media, rotate all reachable credentials, audit group/Environment rules
and workflow revisions, and repeat dry-run qualification before reactivation.

### Workflow lint (no production access)

Run `actionlint` and `zizmor` against the new workflow. Since the custom runner
label is intentionally not registered, supply actionlint an external config:

```bash
printf 'self-hosted-runner:\n  labels: [fam-graphrag-deploy]\n' > /tmp/fam-actionlint.yaml
actionlint -config-file /tmp/fam-actionlint.yaml .github/workflows/deploy-fam.yml
zizmor --no-progress .github/workflows/deploy-fam.yml
```
