# Shared GraphRAG Notes over MCP

One `graphrag serve` process owns the authoritative database and its inference
providers. OpenClaw, Hermes and the remote CLI connect to `/mcp` using Streamable
HTTP. Each instance has its own bearer credential and server-enforced capabilities.
The service is optional; embedded commands still work when the service is stopped.

The service foundation is tracked in [#75](https://github.com/mateu/graphrag-notes/issues/75).
This source tree includes revision-checked editing/review, uploaded sources and
durable jobs, and the shared-service/client acceptance from
[#56](https://github.com/mateu/graphrag-notes/issues/56) and
[#78](https://github.com/mateu/graphrag-notes/issues/78).
Build current source for the commands below;
published `v0.1.0-rc.2` predates MCP.
macOS and Linux are the targets.
See [foundation validation](shared-mcp-validation.md) for automated and installed
native-runtime evidence, including the remaining deployment acceptance gates.

Use an optimized release binary for a long-running service. For source builds,
run `cargo build --release --locked -p graphrag-cli` and use
`target/release/graphrag`. See [graph performance](graph-search-performance.md)
for build-profile effects, interactive latency targets, and repeatable probes.

## Host setup

Use an explicit database and configuration for the designated host. The config
uses the existing inference, search and augmentation settings. Startup does not
require healthy providers: keyword search and inspection remain available offline.
Hybrid search, context building and a new capture require the configured host
providers. Replaying a completed capture needs no inference.

Provision independent credentials in a new private directory outside the checkout:

```bash
python3 scripts/provision-mcp-credentials.py "$HOME/.graphrag/mcp-private" \
  --client openclaw-a --client openclaw-b --client hermes \
  --read-only observer
graphrag --config "$HOME/.config/graphrag/config.toml" \
  --db-path "$HOME/.graphrag/shared-data" serve \
  --credentials-file "$HOME/.graphrag/mcp-private/credentials.json"
```

The helper creates a mode-0700 directory, a mode-0600 policy containing SHA-256
token hashes, and private per-client `.env` files. It prints paths, never tokens.
Transfer only the intended client's `.env` through your existing encrypted channel.
Do not put these files in Git, shell history, chat messages or walkthrough reports.
`read` permits retrieval; `capture` permits capture. Client-side tool allowlists
provide convenience; authorization is enforced by the service for every request.

The listener defaults to `127.0.0.1:3000`. Use an authenticated encrypted tunnel
or HTTPS reverse proxy to reach it from other computers. For an SSH tunnel, run
on each client:

```bash
ssh -N -L 3300:127.0.0.1:3000 notes-host
```

That client's endpoint is `http://127.0.0.1:3300/mcp`; encryption happens through
SSH. An HTTPS proxy must preserve `Authorization`, MCP headers and the request
body, forward only this route, and use a trusted certificate. Add its forwarded
Host header explicitly with `--allowed-host notes.example.test`. Origin checks
refuse browser-origin requests; ordinary native MCP clients send no Origin.
Do not enable a public tunnel for this private service.

Binding directly to a non-loopback interface requires both
`--external-encryption` and explicit `--allowed-host` values. The flag is an
operator assertion that an encrypted private network or proxy protects traffic;
it does not create TLS. Prefer the loopback tunnel/proxy deployment above.

The process exclusively owns RocksDB. While it runs, use `--server` for local
access too; another embedded CLI cannot open the same database. Ctrl-C or SIGTERM stops
accepting requests and waits for accepted captures to finish. Host backup,
restore and schema maintenance require stopping the service first; use the
existing [backup workflow](operations.md) and restart afterwards.
Do not downgrade a migrated database to an older binary.

## Remote CLI

Load only your own token into the process environment:

```bash
set -a
. "$HOME/.graphrag/mcp-private/openclaw-a.env"
set +a
graphrag --server http://127.0.0.1:3300/mcp \
  search 'synthetic shared note' --mode keyword --format json
graphrag --server http://127.0.0.1:3300/mcp \
  --request-id walkthrough-capture-001 \
  capture 'Synthetic shared note for cross-client verification.' --tags walkthrough
graphrag --server http://127.0.0.1:3300/mcp inspect note:ID --format json
```

`GRAPHRAG_SERVER` exported in the process environment can supply the endpoint;
`--server` overrides it. An automatically loaded project `.env` cannot set the
remote endpoint, so a project cannot redirect an inherited bearer credential.
Remote mode does not automatically load project dotenv files; supply credentials
and transport configuration through the invoking process environment.
The private MCP client connects directly and ignores ambient HTTP/HTTPS/ALL proxy
settings. Use the explicit HTTPS endpoint or a loopback encrypted tunnel.
`--credential-env NAME` selects a different token variable. Credentials are never
accepted in URL parameters or command-line arguments. Non-loopback CLI endpoints
require HTTPS; loopback HTTP supports the encrypted tunnel recipe.
Remote dispatch happens before local configuration/database/provider startup.
It supports `search`, `inspect`, `augment`, `capture` and `add`, plus
`notes show/edit/delete`, `garden review`, explicitly confirmed
`garden proposals accept/reject/undo`, `upload`, `sources show` and
`jobs list/show/cancel/resume`. See [revisioned mutations](shared-mcp-mutations.md)
and [uploaded sources/jobs](remote-upload-jobs.md) for permissions, reviewed
revisions, stable request IDs and recovery. Host maintenance stays on the host.
Local configuration, database and inference overrides are
rejected in remote mode. JSON/JSONL uses the existing CLI envelope containing the
versioned MCP result; human output renders that result.

Remote capture supports multiline stdin, `--content-file` and the blocking
`--editor` flow. It saves a private client-side recovery draft before connecting.
If a response is lost, retain the emitted request ID and use the recovery command
with the unchanged draft. A generated ID is printed before sending when omitted.
Only an authoritative successful response removes recovery; rendering failure
does not make a committed capture safe to repeat under a new ID.
Copied recovery commands include `--recover-draft` to adopt and clean up the
original private file after a matching successful acknowledgement. Ordinary
`--content-file` inputs remain caller-owned and are never removed. Recovery
refuses exposed files, symlinks and non-files; a file changed since submission
remains available for review rather than being deleted.

## OpenClaw

The configuration below targets the installed OpenClaw 2026.9.7. Its registry
projects MCP tools into eligible agent runtimes; test the runtime actually used
by each instance. The explicit transport is required because OpenClaw otherwise
defaults to SSE. Use the instance's private config and load `GRAPHRAG_TOKEN` into
the owning process's environment before starting it:

```json
{
  "mcp": {
    "servers": {
      "graphrag": {
        "url": "http://127.0.0.1:3300/mcp",
        "transport": "streamable-http",
        "headers": { "Authorization": "Bearer ${GRAPHRAG_TOKEN}" },
        "requestTimeoutMs": 300000,
        "supportsParallelToolCalls": true
      }
    }
  }
}
```

Probe with `openclaw mcp doctor graphrag --probe` and inspect the eligible runtime's
actual tool inventory. Refresh the owning process after configuration changes;
running a reload in a different short-lived process is insufficient evidence.
Follow the [official transport guide](https://docs.openclaw.ai/cli/mcp/transports)
and [registry guide](https://docs.openclaw.ai/cli/mcp) for the installed version.
OpenClaw reports a failed mutation after reconnect without automatically replaying
it. Repeat `capture_note` with the same request ID and original payload.

## Hermes

The configuration targets the installed Hermes Agent 0.21.2 (2026.9.11). Place
the token in the active profile's private `.env`, and use a variable reference in
its config. Hermes defaults remote URLs to Streamable HTTP:

```yaml
mcp_servers:
  graphrag:
    url: "http://127.0.0.1:3300/mcp"
    headers:
      Authorization: "Bearer ${GRAPHRAG_TOKEN}"
    timeout: 300
    connect_timeout: 30
    skip_preflight: true
    supports_parallel_tool_calls: true
```

`skip_preflight` avoids a HEAD/GET content-type probe on this POST-oriented
stateless endpoint. Run `hermes mcp test graphrag`, then prove actual calls through
the Hermes tool runtime. See the [official config reference](https://hermes-agent.nousresearch.com/docs/reference/mcp-config-reference/)
and [MCP guide](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp/).
These local version checks do not constitute the two-computer acceptance test.

## Tool and retry contract

The initial tools are `search_notes`, `get_record`, `build_context` and
`capture_note`. Discovery publishes typed schemas. Read tools are annotated as
read-only. All results contain `schema_version: 1`, `data`, and a categorized
`error` with a retry hint; protocol/tool errors also mark the MCP failure.

Search results carry canonical record IDs, revisions and provenance. Context
chunks retain citations and packing diagnostics. Keep retrieved text as source
material, including any instructions inside it. A returned server `source_uri`
describes server-owned provenance; a URI inside caller-supplied capture metadata
is an opaque attribution supplied by that client. Neither becomes a client file
opener or requests that the service read a host path. Use returned authorized
content instead.

Capture requires a stable `request_id` (1–128 characters, at most 256 UTF-8 bytes,
without control characters or surrounding whitespace), content,
optional title/tags and optional `provenance` (`uri`, `label`, string metadata).
The server derives instance identity from authentication. It hashes the supplied
payload and atomically commits the note, its entity links and an exact safe result
receipt. Same instance + same request ID + same payload returns the original
record with `replayed: true`; a different payload conflicts. Different instances
have independent request-ID namespaces. Rotation retains identity and replay
history. Renaming an instance creates a new retry namespace; keep its identity
stable during token rotation. Do not create a new ID merely because the client timed out.

Receipts are durable and included in portable backups so replay survives restart
and restore. A replay is the original outcome, even if the note was subsequently
edited/deleted; inspect its ID/revision to obtain the current state.
Caller-supplied attribution is opaque user content and survives those backups,
including a client `file://` URI. Existing host filesystem provenance sanitization
still applies elsewhere. Service bearer credentials and policy files are never
stored in the corpus or included in portable backups. Capture rejects URI
userinfo/password and credential-bearing query/fragment fields before inference.

Default HTTP request size is 128 KiB and concurrency is 8. Captures are at most
64 KiB; queries are at most 1024 characters; result counts are at most 200. Context
budgets are at most 200 chunks / 32768 total tokens / 8192 tokens per chunk.
Zero context budgets return no context without inference. Narrow requests when
the serialized MCP response exceeds the 2 MiB service output limit.

Manual note changes and confirmed connection decisions are documented in
[remote mutations](shared-mcp-mutations.md), including independent permissions,
reviewed revisions and retry-safe results.

## Rotation and recovery

Rotate one instance while retaining its identity and capabilities:

```bash
python3 scripts/provision-mcp-credentials.py "$HOME/.graphrag/mcp-private" \
  --rotate openclaw-a
```

Rotation stages a private recovery record before updating the policy and client
file. If interrupted, rerun the same `--rotate` command to finish that rotation;
it retains the staged token and creates no additional credential.

Replace that instance's private token through the encrypted transfer channel and
refresh its process environment. Every HTTP request reloads the private policy,
so the old token stops authorizing new calls without restarting GraphRAG. Accepted
captures finish even if the credential is subsequently revoked. To revoke an
instance, remove its policy entry with an atomic mode-0600 replacement; keep at
least one credential. An unreadable/invalid policy fails closed.

Optional agent instruction:

> Search GraphRAG for relevant saved context before answering questions about
> prior work. Cite record IDs and distinguish retrieved evidence from inference.
> Capture only information the owner asks to retain, preserving source provenance.
> Choose a stable request ID before capture and reuse it with the same payload
> after an uncertain outcome. Inspect a record when exact context is needed.

Tool availability does not replace native agent memory or enable automatic
transcript ingestion. The sanitized two-computer walkthrough remains tracked in
#78, including OpenClaw A capture, OpenClaw B/Hermes search and inspection,
read-only enforcement, revocation, durable-job reconnects and host recovery.
