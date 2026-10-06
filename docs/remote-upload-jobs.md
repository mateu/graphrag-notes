# Uploaded sources and durable service jobs

The shared service accepts supplied Markdown through `upload_source`. It never opens a caller path or fetches a caller URL. The server owns processing after admission; closing a client connection does not cancel a job. Remote source URIs identify server records and do not authorize file actions on another computer.

Provision `upload` to submit documents and `jobs` to inspect, cancel, or resume that authenticated instance's jobs. For example, add `--grant openclaw-a=upload,jobs` when running `scripts/provision-mcp-credentials.py`; default clients retain only read/capture access. `read` allows corpus retrieval and `get_source` content inspection. These grants are independent; jobs from another instance return unavailable even when their IDs are known. Caller IDs and worker leases are never accepted as tool arguments.

## Client commands

Use the existing encrypted remote endpoint and bearer environment variable:

```sh
graphrag --server https://notes.example/mcp --request-id daily-upload-001 \
  upload --document-key daily-planning --content-file ./daily.md --format json

graphrag --server https://notes.example/mcp jobs list --format json
graphrag --server https://notes.example/mcp jobs show processing_job:ID --format json
graphrag --server https://notes.example/mcp jobs cancel processing_job:ID
graphrag --server https://notes.example/mcp jobs resume processing_job:ID
graphrag --server https://notes.example/mcp sources show source:ID --format json
```

`--content-file` is read by the client and sent as text. A private draft is saved before sending; `--draft-dir` selects its persistent directory. Admission uncertainty retains a copyable recovery command with the same request ID, document key, title, extraction option, and resolved absolute draft directory, so it works from another working directory. That command includes `--recover-draft`, which explicitly adopts the retained private file for cleanup after an authoritative admission. Ordinary supplied files remain caller-owned. Recovery refuses symlinks, nonregular files, exposed permissions, or changed file identity/content. An authoritative admission returns the durable job ID; the server now owns its input. Inspect or resume that job instead of submitting another request merely because processing failed.

A request ID identifies one exact admission payload within an authenticated instance. Retry the identical request after a lost response. Reusing it for changed content, metadata, or extraction choice fails with a conflict. A document key identifies successive versions of one source within that instance. Submit a **new request ID** with the **same document key** for an intentional update. Other instances using the same key receive different sources. Uploads never silently change their request ID during a retry.

Add `--extract-entities` for a checkpointed extraction pass after notes become visible. This is opt-in; omission imports embedded Markdown chunks without running extraction.

## Tools and bounds

`upload_source` takes `request_id`, `document_key`, `content`, optional `title`/`provenance`, `extract_entities`, and optional `preserve_unchanged` (default false). The preserve guard permits only exact unchanged-input metadata registration under the generation lock: differing title, processing/extraction policy, missing/changed source or raw supplied bytes fail without replacing existing chunks/mentions. Unguarded legacy request fingerprints and receipt replay remain unchanged. Request IDs allow 128 characters/256 UTF-8 bytes; authenticated instance IDs contain 1–64 ASCII letters, digits, dots, dashes, or underscores. Other limits are 256 characters/512 bytes for document keys, 65,536 UTF-8 bytes for Markdown, 512 characters for titles, 16 KiB each for provenance and processing snapshots, and 200 prepared chunks. Byte bounds are checked separately from JSON Schema character bounds. Provenance URIs are descriptive user content; credential-bearing URIs are refused, and neither file URIs nor HTTP URIs are dereferenced.

`get_job`, `list_jobs` (1–100), `cancel_job`, and `resume_job` expose only owned uploaded jobs. Responses identify status, phase, attempted generation, progress, canonical checkpoint, safe error category, timestamps, and committed results. They contain no vectors, provider endpoints, bearer tokens, or worker authority. `get_source` accepts exactly one canonical `id` or `document_key`; key lookup is scoped to the authenticated instance. It reports extraction policy, an opaque processing-policy fingerprint, and compatibility with current service configuration without probing providers or exposing their endpoints. It returns exact Markdown bytes from the latest accepted origin job with both attempted and successful generation labels. Its revision covers those returned bytes, including line endings. During a failed refresh that content can describe the attempted version while retrieval continues serving the previous successful notes; inspect the labels before using it as committed context. A line-ending-only unchanged upload updates inspection without changing the backing text used by existing chunk spans.

`serve --max-concurrent-requests 1..64` (default 8) bounds ordinary HTTP work and separately bounds detached upload/resume admissions. Those admission slots remain occupied until their durable action finishes even if the HTTP client disconnects. Two reserved cancellation slots let an authenticated instance with `jobs` reconnect, discover tools, and cancel its job while ordinary requests are busy. Cancellation tasks retain their own bounded slots until settlement. Other calls receive a typed busy response when their capacity is full. Authentication and bounded request-body buffering also have a short ingress limit; body reads time out after five seconds.

The supported durable kind is `remote_upload` with uploaded Markdown input. Local embedding, reindex, extraction, and folder-sync jobs remain owned by their existing workers. Remote resume refuses unsupported kinds and IDs before changing their status.

## Persistence and recovery

Admission commits immutable input and stable job/source identities together. Workers validate saved input before claiming queued jobs with a service epoch and unique worker token, so malformed stored input cannot strand an untracked running job. Deterministic saved-input validation failures are marked failed before claiming, and a bounded scan continues to healthy queued jobs. Repairing that saved input requires an explicit resume before it can run again; storage failures remain retryable errors. Each short mutation phase validates that authority and the pinned source generation. Provider preparation runs outside those gates. Restart marks abandoned running jobs interrupted; explicit `resume_job` validates the saved provider/model/chunk configuration before queuing them. A stale worker cannot publish or finish a reclaimed job.

Admission order also fences an older unprepared request once a newer request for the same document has begun its generation. Cancelling an old queued upload and later resuming it cannot overwrite that newer version, including after source deletion and recreation. Restored histories without the optional admission sequence use their original admission timestamps.

The phases are preparation, atomic chunk staging, source promotion, and optional per-note extraction. Staging is hidden until promotion. Promotion reuses the existing source reconciliation rules for dependent links, proposals, and cleanup. A retry after visibility promotion avoids copying old dependents again. Extraction replaces mentions/entities and saves the exact canonical checkpoint in one transaction; resume continues after its last completed item.

Identical content is considered unchanged only when title, preparation configuration, extraction choice, and prior completed processing also match. It preserves the generation and note IDs, avoids provider work, and records the latest accepted provenance. Changed generation preparation retains the successful generation's provenance until promotion. A failed attempt does not relabel visible old notes.

Cancellation writes a durable flag independently of the worker transition gate and provider/capture semaphore. It stops provider preparation and takes effect between atomic mutations. If a mutation already entered its safe boundary, that write completes and its checkpoint remains accurate before cancellation is acknowledged. Service shutdown requests preparation cancellation, drains atomic writes and workers, and retains interrupted jobs for restart. Configure `serve --max-job-workers 1..4` (default 2) to bound concurrent worker preparation.

Terminalization checks the persisted cancellation flag atomically, so a concurrent accepted cancellation cannot become a completed or failed result. If status reads or terminal writes fail, the service retains the same fenced execution in a bounded worker slot and retries settlement with backoff. Recovery reads only status, ownership, and cancellation authority; damaged saved input or result data cannot block settlement. It does not rerun providers and stops after observing a terminal state or a changed owner. Graceful shutdown waits for a recovery write already in progress, then stops retrying unresolved storage faults. Their durable running leases remain available for startup reconciliation after repair.

Portable backups include only remote-owned processing jobs, preserving admission input and outcomes as opaque user content, like note bodies. Local runtime jobs and caches remain excluded. Restore clears worker epochs/tokens so no old worker authority survives, while completed/cancelled results and replay identities remain. Uploaded provenance and active/pending processing snapshots are preserved, including their embedding configuration objects in exports with or without vectors; ordinary host filesystem metadata and secret fields retain the existing sanitizer. Logical source and note IDs in completed history may outlive deleted records without recreating them.

Uploads and historical outcomes persist in the owner's corpus; this phase does not implement a retention policy or client replication. Use the owner's existing backup and source lifecycle controls.

`upload_source` also accepts two mutually exclusive, default-false guards:
`preserve_unchanged` requires an exact existing compatible original input and
retains its successful generation; `create_only` requires the caller's source
identity to remain absent when the worker begins. Both checks run under the
source generation lock before inference. Unguarded legacy admission fingerprints
remain byte-identical for durable retries.

`expected_source_revision` optionally pins an inspected existing source for an
edited upload. It is incompatible with `create_only`, may accompany
`preserve_unchanged`, and is checked before preparing a new generation. Omitting
it retains legacy upload/replay behavior; revisions and guard flags are frozen in
the admission payload.
