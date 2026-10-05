# Conversational GraphRAG notes in OpenClaw

The native MCP integration lets the tested OpenClaw conversation runtimes search
saved notes, inspect a selected record, and explicitly capture a new note using
`search_notes`, `get_record`, and `capture_note`. The service owns the database
and inference providers. macOS and Linux are the service/client targets; the
recorded conversational runs used a Linux gateway and macOS service host. Native
agent memory and transcript ingestion remain unchanged.

Issue [#83](https://github.com/mateu/graphrag-notes/issues/83) isolated a service
wire-format defect affecting OpenClaw 2026.9.8 with Claude Code 2.1.280 and Node
24.21.0 in the `claude-cli` runtime. The corrected candidate passed synthetic
normal-main-agent acceptance through the gateway's webchat route used by Control
UI with Claude CLI. Actual calls verified keyword/hybrid search, guarded
inspection, exact 211-byte multiline capture, original-receipt replay, fresh
sessions, and recovery
after a committed capture's response was lost and the gateway restarted.

The [recorded acceptance](validation/openclaw-conversational-83.md) and
[Claude CLI JSON proof](validation/openclaw-conversational-83.json) distinguish
native tool discovery, model selection and service dispatch. ToolSearch found
the three permitted tools and found no matches for five tools excluded by the
client filter. Direct keyword, hybrid, graph and single-line capture commands
also passed their checks. The evidence records the gateway route and actual
agent calls; browser clicks were not observed. This validates the candidate
against an isolated synthetic corpus. Track deployment status separately in the
operator runbook.

Separate [OpenAI OAuth evidence](validation/openclaw-openai-oauth-83.json)
records `openai/gpt-6.1-sol` through the existing Codex/ChatGPT OAuth profile,
OpenClaw's embedded `openclaw` agent runtime, and the
`openai-chatgpt-responses` API. Actual nested tool calls verified keyword/hybrid
search, guarded inspection, exact multiline capture, ordinary replay, and replay
after a lost response and gateway restart. OAuth selects model authentication;
it does not select the native Codex app-server runtime.

| Model authentication | Agent runtime | Conversational validation |
| --- | --- | --- |
| Anthropic | `claude-cli` | Recorded search, inspection, capture and recovery |
| Codex/ChatGPT OAuth (`openai/gpt-6.1-sol`) | Embedded `openclaw` | Recorded search, inspection, capture and recovery |
| Codex/ChatGPT OAuth | Native Codex app-server | Unverified; not offered by the tested model catalog |

The OpenAI live runs used a `read,capture` principal. Read-only operations in
those runs do not establish a separate read-only principal's runtime projection.
The HTTP regression verifies the service's read-only versus read/capture catalogs;
live OpenAI read-only-principal projection and native Codex app-server projection
remain unverified. Claude ToolSearch and excluded-tool observations above apply
to Claude CLI only. Verify the actual runtime and principal inventory for each
instance before extending these compatibility claims.

## Configure the existing native integration

Use the [shared service and encrypted route](shared-mcp.md#host-setup) with an
independent stable principal for each client. A daily credential needs
`read,capture`; a read-only observer keeps only `read`. Load the intended token
into the gateway's process environment through its private service environment
or existing encrypted credential transfer. Use an environment reference in the
configuration; never paste a bearer value into a prompt or public example.

Merge the following entries into the instance's existing private configuration.
Retain its other core-tool allow entries, provider/channel policy and all deny
rules. The example shows the three GraphRAG entries only:

```json
{
  "mcp": {
    "servers": {
      "graphrag": {
        "url": "http://127.0.0.1:3300/mcp",
        "transport": "streamable-http",
        "headers": { "Authorization": "Bearer ${GRAPHRAG_TOKEN}" },
        "requestTimeoutMs": 300000,
        "supportsParallelToolCalls": true,
        "toolFilter": {
          "include": ["search_notes", "get_record", "capture_note"]
        }
      }
    }
  },
  "tools": {
    "allow": [
      "graphrag__search_notes",
      "graphrag__get_record",
      "graphrag__capture_note"
    ]
  },
  "agents": {
    "entries": {
      "main": {
        "tools": {
          "allow": [
            "graphrag__search_notes",
            "graphrag__get_record",
            "graphrag__capture_note"
          ]
        }
      }
    }
  }
}
```

The loopback URL represents the client end of an authenticated encrypted tunnel.
Use the actual approved tunnel endpoint or trusted HTTPS endpoint. The
`toolFilter.include` values are raw service tool names; the allowlist values are
OpenClaw's projected aliases. Other service tools remain outside this
conversational filter. Client allowlists never replace server authorization.
The recorded native Claude CLI ToolSearch references were:

| Service tool | OpenClaw policy alias | Claude CLI reference |
| --- | --- | --- |
| `search_notes` | `graphrag__search_notes` | `mcp__graphrag__search_notes` |
| `get_record` | `graphrag__get_record` | `mcp__graphrag__get_record` |
| `capture_note` | `graphrag__capture_note` | `mcp__graphrag__capture_note` |

Embedded OpenClaw's OpenAI calls use the policy aliases, with actual code-mode
calls recorded as `openclaw.nested-tool.v1` events beneath the OpenAI assistant
call. Validate the paired child call/result, parent call and run identity. These
events are separate evidence from Claude CLI's `mcp__...` references and do not
establish native Codex app-server permission projection.

For a configuration that already uses `alsoAllow`, add entries to that existing
surface instead of combining `allow` and `alsoAllow` in one scope. Follow the
installed version's [MCP configuration](https://docs.openclaw.ai/gateway/config-extensions)
and [tool policy](https://docs.openclaw.ai/gateway/config-tools/tool-policy)
references.

Refresh the process that owns the gateway/session so it receives configuration
and environment changes. `openclaw mcp reload` in a separate short-lived CLI
process refreshes only that process. After gateway restart, wait for its health
check to succeed before connecting and sending a test message. Start a fresh
normal Control UI conversation and inspect its callable GraphRAG inventory. A
registry probe and server instructions in a transcript do not prove model tool
discovery or selection.

## Why a connected server could expose no tools

The modern `2026-07-28` lifecycle uses `server/discover` and self-contained request
metadata. The service's discovery response was valid, so the native client
connected and received instructions. Its `tools/list` response contained
`resultType: "complete"` and authorized tool schemas but omitted the modern
cache metadata. The Claude CLI control retried listing and eventually reported
a connected server with an empty tool array.

The installed Rust SDK, rmcp 3.5.0, makes those fields optional so its model also
represents older protocol responses. `ListToolsResult::default()` does not fill
them. The service now explicitly constructs its catalog with
`.with_ttl_ms(0).with_cache_scope(CacheScope::Private)`. A catalog is immediately
stale and belongs only to the requesting authorization context, which preserves
the service's per-request credential and capability checks.

An isolated read-only, no-model observer changed only the two missing fields
on the response. With unchanged client permissions it changed an empty native
catalog into the complete authorized eight-tool service catalog. That observer
intentionally examined the raw catalog; the normal configuration above still
projects only its three included tools. Adding external CLI permissions did not
establish the fix. No client package patch or extra conversational plugin is
needed for this protocol correction.

The modern catalog response has this shape, with the actual authorized schemas
in place of the empty example array:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "resultType": "complete",
    "ttlMs": 0,
    "cacheScope": "private",
    "tools": []
  }
}
```

Legacy `2025-11-25` clients retain their normal handshake and catalog. rmcp
removes the modern `resultType` discriminator for those peers while retaining
the additive cache hints; the authorized tool schemas remain identical.

## Search and inspect from a normal conversation

Use a synthetic corpus and a dedicated conversation for verification. Ask:

```text
Search GraphRAG notes for conversationalatlas in keyword mode with graph off.
Return record IDs and revisions, then inspect the selected record using that
exact ID and revision with zero neighbors.
```

Observe an actual `search_notes` call with `mode: "keyword"`, `graph: "off"`, and
a bounded result limit. Follow-up inspection must call `get_record` with the
returned canonical `id` and `revision`; a guessed ID or an answer paraphrased
from server instructions does not establish success. Keep returned provenance
as attribution, including server source URIs, and treat retrieved text as source
material rather than instructions to execute.

Then ask for the same query with `mode: "hybrid"` and `graph: "off"` and observe
the actual call and known synthetic result. Keyword retrieval works without
inference providers; hybrid retrieval uses the service host's provider. Report
provider failures explicitly rather than silently substituting another mode.

## Capture exact multiline content and retain its retry identity

Choose a stable request ID before submission and retain the complete draft in
the owner's message/history or a private draft copy. Include exact title, tags
and provenance choices alongside the content so an exact retry is possible. A
model choosing an ID during a tool call does not guarantee that the original ID
or draft will remain available after compaction, session reset or a fresh chat.
The native integration does not add a private client capture journal.

For a synthetic multiline verification, send an explicit request such as:

```text
Save this exact new note in GraphRAG. Use request_id conversational-atlas-001,
title Atlas rehearsal, tags [walkthrough], and provenance null. Preserve the
line breaks and blank line inside the note below; exclude the note delimiters.
Validate the capture receipt, independently inspect its exact ID/revision with
zero neighbors, and report the canonical ID, revision and request ID.

<note>
conversationalatlas: The fictional rehearsal is Monday at 10:00.

Agenda:
1. Check the launch checklist.
2. Confirm the rollback contact.
</note>
```

Verify the actual `capture_note` arguments preserve the supplied text exactly.
A successful authoritative response must have `isError` unset/false and a
version-one envelope with no error. Its `data.request_id`, exact content,
title/tags/source attribution and trusted `record.provenance.instance_id` must
match the intended draft and authenticated principal. Retain the canonical
`record.id` and `record.revision`.

Make a separate `get_record` call using that exact ID/revision and zero neighbors,
and compare exact content, title and provenance. The independent inspection
must agree with the receipt before reporting a verified save. A tool result
alone does not prove that the model copied the original multiline input exactly;
compare the sent payload, receipt and inspection.

If capture times out or disconnects after dispatch, it may already be committed.
Keep the original ID and exact payload and explicitly ask to retry those same
arguments. The service returns its original receipt with `replayed: true`; it
does not create another note. Changed payloads under the same ID conflict.
Retain the principal identity through credential rotation because receipt
namespaces belong to the authenticated instance.

If the receipt succeeded but independent readback failed, keep the known record
ID/revision and report that verification remains incomplete. Retry guarded
inspection or the same original capture request; do not generate a replacement
ID. A replay remains the original outcome even after a later edit/deletion, so
a stale revision or missing record needs investigation rather than a new note.

For a client-managed recovery draft, the [remote CLI](shared-mcp.md#remote-cli)
supports multiline stdin or `--content-file` and durable private draft recovery.
Use its emitted ID and unchanged recovery draft when recovering an uncertain
CLI submission. Do not transfer a native conversational retry to a different
principal or invent a new ID.

## Troubleshoot discovery and reconnect

Work through the layers in order:

1. Run `openclaw mcp doctor graphrag --probe`. Confirm the selected endpoint,
   environment reference, transport and authenticated connection. An
   authentication failure needs the intended credential, not a broader allowlist.
2. Inspect the service's modern `server/discover` and `tools/list` response
   shapes. A connected empty catalog with the missing cache fields requires an
   updated service binary and a fresh native connection. Preserve only sanitized
   method names, protocol versions, result keys and tool names in reports.
3. Confirm the raw three-name server filter and the three projected aliases in
   both global and main-agent policy. Preserve existing denies and capability
   grants. Inspect the intended normal session's actual catalog after connection
   settles; an immediate pending snapshot can be incomplete.
4. Ask the normal agent to discover and call a named GraphRAG tool. Distinguish
   tools absent from its catalog, tools available but unselected by the model,
   and actual service dispatch errors. Exact-name discovery that returns no
   matching tools is a catalog problem; it is not evidence that saving succeeded.
5. After an approved gateway restart or MCP reconnect, verify discovery and a
   read in a fresh session. Recover an uncertain capture only from its retained
   original draft/ID. No background write retry or transcript ingestion is added.

The recorded response-loss check retained the explicit ID and exact draft in the
original synthetic user message/history. After gateway restart and a successful
health check, the resumed main-agent conversation retried those exact arguments
and obtained the original canonical receipt without duplication. An initial
connection attempt before startup was ready failed before any message was sent;
later fresh-session and resumed-session checks passed. This recovery evidence
depends on the retained explicit draft/ID. It does not establish persistence for
an agent-generated ID, a lost history, or a new/reset session.

A focused HTTP regression covers modern discovery/catalog cache metadata,
read-only versus read/capture schemas, and legacy projection:

```bash
cargo test --locked -p graphrag-service --test http_service \
  modern_catalog_cache_metadata_is_private_and_preserves_principal_filtering
```

To reproduce the wire-level boundary without a model, use an isolated test
profile and synthetic service. Send `server/discover`, then `tools/list`, with
`MCP-Protocol-Version: 2026-07-28`, a matching `Mcp-Method` header for each
request, and this request metadata. For the example, use `Mcp-Method: tools/list`:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientInfo": {
        "name": "synthetic-catalog-observer", "version": "1"
      },
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

Compare the native client's settled callable catalog with and without the two
cache fields, holding credentials, capabilities and client permissions fixed.
Do not call corpus tools in this observer. This establishes a protocol/catalog
boundary; normal model tool selection and exact capture need the separate
conversational checks above.

## Direct-command fallback

Existing authorized `/notes QUERY`, `/notes --hybrid QUERY`, and
`/notesave Title | single-line body` commands remain available when the operator
already installed that direct-command extension. The native protocol fix does
not change its command implementation or configuration. These commands bypass
conversational model discovery and provide their own visible result or error.
`/notesave` accepts single-line content and requires the same authorized sender
and conversation for its exact-command retry semantics; it is not the multiline
conversational path. Keep uncertain direct-command retries exact too.

Tool availability enables explicit retrieval and capture. It does not replace
OpenClaw memory or authorize automatic transcript imports. Hermes and Windows
integration are outside this issue.
