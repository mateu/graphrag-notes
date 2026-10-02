# Explicit Markdown folder sync

Register a local folder once, preview its changes, then run a one-shot sync:

```bash
graphrag folders add work ~/notes/work
graphrag folders list
graphrag sync work --dry-run
graphrag sync work
graphrag sync --all --format json
```

Use the same global `--config PATH` and `--db-path PATH` for these commands
when you keep several installations. Registration requires an existing config;
create it with `graphrag init --write` first. `folders add` appends a declaration
to the selected config (`--config`, then `GRAPHRAG_CONFIG`, then the native
platform default). It preserves existing comments, settings, credentials, and
file permissions. It refuses a symlinked config target and duplicate names.
Registrations coordinate through a lock. Editor saves detected during commit
leave registration unapplied; inspect the reported recovery file and retry
with the updated config. Before installing a declaration, registration moves
the previous config to a sibling `CONFIG.folders-backup-*` file, retains its
permissions, and reports that path. Keep it until you have checked the new
config; remove it when you no longer need it. Arbitrary editors do not share
the registration lock, so do not edit the config during registration. The
config name is briefly absent during capture; an interrupted registration
can require copying the reported sibling backup back to the config path.
An interrupted registration can leave a `.toml.folders.lock` sidecar; remove
that lock only after verifying that another registration is not running.

You can also edit definitions directly:

```toml
[folders.work]
path = "/absolute/path/to/notes/work"
recursive = true
include = ["**/*.md", "**/*.markdown"]
exclude = ["**/.git/**", "**/.obsidian/**", "**/.graphrag/**"]
```

Names contain 1–64 ASCII letters, digits, hyphens, or underscores and start
with a letter or digit. Roots must be canonical absolute UTF-8 directory
paths. Registration resolves a root alias to its
canonical directory; duplicate and overlapping roots are rejected. Rules use
case-sensitive globs over slash-separated paths relative to the root. Include
rules select files, then exclude rules take precedence. `--include GLOB`
replaces the default includes; repeat it for several patterns. `--exclude GLOB`
adds to the default excludes. `--no-recursive` selects only files directly
inside the root. Registration supports both `[folders.NAME]` declarations
and inline TOML `folders = {...}` maps.

Sync reads each selected file to compare its normalized content hash. Newline
changes between LF and CRLF do not cause a reimport. An unchanged sync makes
no provider health checks or inference requests, creates no new processing
job, and keeps existing note IDs and source generations. A dry run reports
planned actions without inference, source promotion, or job writes. Normal
database startup still applies pending schema migrations if needed.

Changed files use the existing Markdown importer and embedding configuration.
Run `extract-entities` explicitly to extract entities from imported notes.
Each file stages a replacement source generation,
and its old successful generation stays searchable until replacement succeeds.
A failed file is reported independently and does not stop successful peers.

## Results and recovery

Each file reports `created`, `updated`, `unchanged`, `failed`, `missing`, or
`ignored`. Cancelled work that has not been attempted reports `pending`.
Dry-run `created`/`updated` rows describe the proposed actions. Applied rows
include the source ID and successful generation. Errors include a retry command
with the selected config and database paths.

Changed-file work creates a durable `folder_sync` processing job. Its scope
pins the selected roots and rules, and its items pin the canonical file URIs,
including matched files whose contents could not be read. Root discovery
failures without a file identity receive a fresh named-folder retry command.
Check progress or resume it explicitly:

```bash
graphrag jobs show processing_job:ID --format json
graphrag sync --resume processing_job:ID --dry-run --format json
graphrag sync --resume processing_job:ID
```

Resume uses the stored folder definitions even if the config has since changed.
It retries the current contents of the pinned files, skips already completed
unchanged peers without inference, and leaves newly discovered files for a
fresh sync. For pinned unchanged files it also completes any deferred
old-generation cleanup after a process stopped during promotion. A missing
pinned file remains an unresolved job failure until restored. `jobs resume`
directs folder jobs to `sync --resume`. Ctrl-C requests a safe stop between
files; resume uses the same durable file set. A process killed during a file
can leave a running job; explicit `sync --resume` recovers it after acquiring
the database. RocksDB allows one process to own a database at a time.

## Missing files and explicit pruning

Sync retains missing source records and their searchable notes. Preview
removal separately, then confirm the exact preview token:

```bash
graphrag folders prune work --format json
# Copy the report's confirm_command, or use its revision explicitly:
graphrag folders prune work --yes --revision TOKEN --format json
```

Prune previews the generated-note and dependent-record counts. Confirmation
rejects a changed token, an incomplete scan, an unavailable root, or a file
that has returned. It removes only source-generated notes and their dependents;
manual, detached, and legacy notes without a source generation remain, as do
their own relationships and provenance. Shared entity records remain.
When retained notes still reference a source, prune keeps its source metadata
as a provenance anchor. The anchor has no active generated generation or
content hash, so repeated missing/prune previews omit it. Restoring even the
same file contents reimports under its existing source ID. Portable backups
retain valid references for those manual notes and their source metadata.
Portable backup's existing privacy policy redacts host-local source URIs;
restored notes retain source IDs and content, without assuming another host's
filesystem paths exist.
A restored anchor cannot open its old local file and is not automatically
mapped to a registered folder.
Pruning also preserves terminal graph proposal history. Rejected and
superseded proposals may retain the IDs of removed endpoints; backups keep
those audit records without recreating deleted notes. Active proposals still
require existing endpoints.
Each source is rechecked immediately before deletion. If a later source
returns or becomes unverifiable after earlier deletions completed, the report
retains the actual per-file outcomes, reports failure, and provides a new
preview command; untouched missing sources remain retained.

Ownership follows canonical host-local `file://` Markdown source identities
under the selected root and its current rules. A file previously imported with
`graphrag import` is reused rather than duplicated. Such a matching file is also
included in a missing-source preview. Sources excluded by the current rules
are retained and excluded from pruning. Remote upload identities must use a
distinct namespace; folder roots and file URIs describe the server's host,
not another computer's local filesystem.

Symlink files and directories are never followed, including cycles and
symlinks pointing outside the root. They appear as ignored entries. A root
that becomes a symlink, including through an ancestor, is unavailable until the actual root is restored or
explicitly registered through its canonical target. Distinct hard-linked file
paths retain distinct source identities. Renames are a new path plus a missing
old path; sync does not guess a rename or transfer graph/provenance ownership.
Use the normal missing-source preview after verifying the new import.
An unreadable directory, invalid UTF-8 path/content, or another discovery error
prevents missing-file conclusions for that root and blocks its pruning.

## Automation contract

`--format json` produces one versioned envelope on stdout. `--format jsonl`
produces the same envelope on one line. Logging and provider progress stay on
stderr. Commands are `folders.add`, `folders.list`, `sync`, and `folders.prune`.
The sync envelope contains `dry_run`, `job_id`, `cancelled`, pinned `folders`,
per-file `files`, and `warnings`. Each file has its path, source URI/ID where
available, status, generation, hash, error, and retry command. A prune report
contains `revision`, its file/deletion previews, and `confirm_command`.

Successful sync, preview, and ordinary retained missing files exit **0**.
Validation errors exit **2**, missing named records exit **3**, and failed
files or cancellation exit **5** with `success=false` and useful per-file
results. Sync is one-shot: filesystem watching and cross-computer file
replication are deferred. No daemon keeps RocksDB open between runs.

See [folder sync validation](folder-sync-validation.md) for the offline checks
and an isolated manual failure/recovery walkthrough.
