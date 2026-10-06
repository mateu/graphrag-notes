# Remote diagnostics

`graphrag --server http://127.0.0.1:3000/mcp --credential-env GRAPHRAG_TOKEN doctor`
reports readiness through the process that already owns the shared corpus. Use
`--format json` for a stable version-one machine report. It dispatches before
local configuration or database initialization, including when local config is
invalid. Use an HTTPS endpoint or the loopback endpoint of an encrypted tunnel.

Routine status is bounded (eight-second client timeout, at most a three-second
owner-side metadata read and a one-second owned-job read). It performs no
provider health request, embedding, extraction, capture, upload, backup,
migration, or second database open. It adds the read-only `service_status` MCP
tool under the existing `read` capability. Existing OpenClaw daily tool filters
are not changed: an operator can opt into this tool, or use the remote CLI.

## Reading the evidence

The report separates HTTP transport, authentication/read permission, readable
storage, embeddings, extraction, source refresh, owned jobs, and backup evidence.
A responding HTTP endpoint is not proof that storage, providers, or backups work.

Provider configuration is reported without exposing model endpoint, paths, or
credentials. Before ordinary operations observe provider health, availability
is `unknown`. The application caches health observations made by existing
ordinary search/capture preparation; observations older than sixty seconds are
`stale`. A recent `ready` health observation establishes that the health check
passed at that time; it does not establish current inference/model compatibility.
A provider stopped after an observation can therefore remain cached until the
next operation or expiry. Status itself never refreshes that observation.

Jobs require the existing `jobs` capability. Status returns counts for a bounded
recent sample of at most ten jobs belonging to the authenticated principal;
it does not return other principals' jobs, source paths, job inputs, or results.
Failed/interrupted work in that sample is `partial` with explicit inspect/repair/resume
guidance; an unrecognized state is `unknown`.
Missing permission is `forbidden`, not a storage failure. Backup evidence stays
`unknown` until there is a trusted recorded backup/restore evidence contract.
Ask the service owner to inspect their latest backup and restore verification.

Exit codes follow doctor conventions: `0` healthy, `1` incomplete/unknown evidence
or a warning, `2` transport/auth/storage or recorded refresh failure. Current
routine remote status normally exits `1`, because backup/freshness evidence is
not implied by connectivity. JSON reports remain available on failure.

## Optional collection freshness evidence

The incremental OpenClaw memory adapter writes a private `freshness.json` in
its collection state directory. On the client that owns that file:

```sh
graphrag --server http://127.0.0.1:3000/mcp --credential-env GRAPHRAG_TOKEN \
  doctor --format json --refresh-status-file /absolute/private/collection/freshness.json
```

Only a bounded regular file owned by the invoking user with permissions `0600`
is accepted; symlinks are refused. Evidence must match the canonical endpoint
SHA-256 and the service-authenticated `instance_id`. Wrong binding or invalid
evidence reports `unknown` and does not echo its content or filesystem path.
The machine report labels it `client_refresh_evidence`; this evidence is local
to one registered collection and cannot establish that the source has not
changed since the snapshot. `partial`, `running`, and `paused` remain partial;
`failed` is unavailable; `complete` preserves the recorded success time while
current freshness remains unknown. Counts describe upload parts, with failed
unreadable source documents also counted by the adapter.

Its version-one contract contains `collection_id`, `endpoint_sha256`,
`instance_id`, `status`, nullable RFC3339 `last_attempt_at`/`last_success_at`,
`counts` (`created`, `changed`, `unchanged`, `failed`, `missing`), `pending_parts`,
nullable static `error_code`, and `retry.action` (`resume`, `refresh`, or `reconcile`).
The optional `retained_extraction_policy_parts` count describes older graph
snapshots retained by explicit client adoption. A positive count makes source
readiness partial, including after a complete indexed original/vector refresh,
with explicit owner reprocessing guidance. Older status files omit the count;
that absence establishes no graph-policy evidence. Failed/partial collection
status still controls the indexed original/vector evidence separately.
Pending reconciliation also carries an optional `retry.plan_sha256` containing
the reviewed plan's 64-character lowercase SHA-256; cleanup remains explicit. It
contains no corpus text, database paths, request IDs, or bearer values.

## Explicit keyword recovery

A remote hybrid search that fails because a provider is unavailable or its
embedding configuration is incompatible prints an exact, shell-quoted retry
command. It preserves the query, scope, limit, lookback, source URI, output
format, endpoint and credential **environment variable name**. The retry uses
`--mode keyword --graph off` and requires an explicit user invocation. It never
prints the credential value or silently changes the requested search mode.
Protocol/schema incompatibilities require matching client/service versions and
do not produce a keyword retry, since changing retrieval mode cannot repair them.

An explicit deep check differs from routine status: the service owner runs
local `graphrag init --check` for active provider checks without
opening storage. Full local `doctor` also checks local corpus compatibility and
can run a dimension embedding probe; it must obey single-owner storage rules
and must not run against the active service database. Deep checks are not a
remote status option and do not expand client permissions.

## Isolated walkthrough

`cargo test -p graphrag-cli --test remote_doctor_http --locked` runs a fictional
in-memory service with a stopped provider and a private read-only credential.
It verifies reachable storage with unknown provider availability, observes a
hybrid failure and exact keyword recovery, reports cached provider unavailable,
accepts partial collection evidence, rejects another principal's evidence, and
distinguishes denied credentials from an unreachable endpoint. It asserts no
client database was created and no routine status health/inference call occurred.
