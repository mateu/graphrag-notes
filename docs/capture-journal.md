# Durable conversational capture recovery

The supported client integration is the repo-owned
[`scripts/mcp-capture-journal.py`](../scripts/mcp-capture-journal.py) loopback
proxy. Point a GraphRAG MCP connection at this proxy to persist every native
`capture_note` payload before it crosses the network. The draft and request ID
survive client restarts and chat resets. Recovery is an explicit operator action;
starting the proxy or a conversation never submits a retained draft.

The proxy needs Python 3.11+ on macOS or Linux, the current GraphRAG service with
`service_status`, and a stable principal with **Read + Capture**. It forwards the
client's existing bearer credential only in memory. It uses the same credential
to confirm the authenticated principal before each capture or recovery. Read
allows that check and independent revision-pinned `get_record` verification.
Credential rotation remains recoverable when the replacement authenticates as
the same stable principal. A different principal or endpoint halts recovery.

## Install on the MCP client host

Use the existing encrypted tunnel or trusted HTTPS upstream endpoint. Run the
proxy under the same OS account as the OpenClaw gateway, with its own private
journal directory. The upstream is the service, rather than the proxy's own URL:

```bash
python3 scripts/mcp-capture-journal.py \
  --journal-dir "$HOME/.graphrag/capture-journal" \
  proxy --server http://127.0.0.1:3300/mcp \
  --instance-id openclaw-clawd-daily --port 3301
```

Keep the proxy running using the client's existing process supervisor. Change
only this MCP connection's URL from its upstream URL to
`http://127.0.0.1:3301/mcp`. Retain its environment-based `Authorization` header,
transport, timeout, raw tool filter, projected allowlists, and deny rules. See
the [normal OpenClaw configuration](openclaw-conversational.md#configure-the-existing-native-integration).
The proxy adds no catalog entries and forwards discovery, reads and other MCP
RPCs with their original response bodies. Server authorization remains decisive.
Keep the daily filter limited to `search_notes`, `get_record`, and `capture_note`.
It does not require adding `service_status` to the model's tool filter: the proxy
uses that read capability itself to validate its connection.

This integration targets GraphRAG's bounded, stateless HTTP MCP endpoint. It
requires a single JSON RPC object per request and never redirects bearer calls or
uses ambient HTTP proxy settings. The listener binds to `127.0.0.1`, refuses
browser-origin requests, and bounds active connections and response size. It
preserves ordinary and modern MCP request headers and request metadata. A generic
SSE notification relay for other MCP servers is outside this integration.

After configuring the connection, start a fresh normal OpenClaw conversation and
explicitly ask it to save a fictional note using `capture_note`. Native MCP
captures through that configured connection are covered. Direct `/notesave`
commands or clients still pointing at the upstream bypass this proxy; route them
through it separately before treating their saves as journaled. Prompting alone
cannot persist a model-chosen payload at this boundary.

## States and explicit recovery

List retained entries without exposing note text:

```bash
python3 scripts/mcp-capture-journal.py \
  --journal-dir "$HOME/.graphrag/capture-journal" list
```

The versioned JSON output includes journal ID, stable principal, original request
ID, receipt record ID/revision when known, timestamps, and verification state.
Keep this metadata private too.

| State | Meaning | Next action |
| --- | --- | --- |
| `pending` | Exact payload is durable; transmission has not begun. | Explicitly recover when ready. |
| `uncertain` | Transmission may have started or its result was lost/invalid. | Replay the original payload and request ID. |
| `receipt_known` | Exact authoritative receipt matches the payload; separate pinned readback failed. | Repair the connection/read failure and explicitly recover. |
| `verified` | A separate `get_record` matched the original receipt's ID, revision, title, content and provenance. | Retain or explicitly clean up the local entry. |

The proxy fsyncs `uncertain` before sending a capture. A crash between this state
and the actual send is conservatively recoverable. A successful MCP capture
response remains the service's original authoritative response; it is not a
claim that the proxy's independent readback succeeded. Inspect the journal's
`independently_verified` field for that additional evidence. `verified_at`
records historical verification and does not assert that a subsequently edited
note still has its original revision.

After a reset, choose the original journal ID from `list`. Load the intended
credential through the existing private process environment, then run:

```bash
python3 scripts/mcp-capture-journal.py \
  --journal-dir "$HOME/.graphrag/capture-journal" \
  recover JOURNAL_ID --server http://127.0.0.1:3300/mcp \
  --instance-id openclaw-clawd-daily --credential-env GRAPHRAG_TOKEN
```

Recovery replays the exact stored capture arguments, including title, tag order,
provenance and original request ID, under the same endpoint/principal. The server
returns its original idempotent receipt. This preserves a single note after a
committed response was dropped. Recovery exits nonzero if the receipt is still
uncertain or pinned readback remains unavailable. Receipt or payload mismatches
halt without replacing the saved draft or inventing a new identity. If the note
was legitimately edited/deleted after its original capture, pinned readback can
fail while the original receipt still replays; inspect that record explicitly
before deciding how to resolve the retained evidence.

## Privacy, concurrent saves and retention

The journal directory must be owned by the current account with mode `0700`;
entry and lock files must be private regular files with mode `0600`. Symlinks,
devices, FIFOs, hard links and exposed files are refused. File replacements are
atomic and fsync both the file and directory. A storage failure prevents initial
submission; preserve the caller's original draft and repair storage before
retrying. A failure recording an already returned receipt leaves the durable
original payload available for exact recovery.

The key binds canonical endpoint, principal and request ID. Per-key OS locks
serialize concurrent duplicate calls across threads and processes; distinct IDs
remain independent. Reusing an ID with changed arguments fails before capture.
Do not manually edit/delete active journal files or `.lock` files. The journal
stores private note text and authoritative receipt content, but never bearer
headers, credentials, OAuth material, or chat transcripts. Routine proxy logs
contain no note text or credentials. Provisioning the directory does not import
existing chat history or replace OpenClaw's native memory.

Verified entries are retained until explicitly removed; there is no automatic
expiry. Choose a retention window appropriate for your recovery needs and protect
any private backups of the directory. The service retains its own idempotent
receipt after local cleanup:

```bash
python3 scripts/mcp-capture-journal.py \
  --journal-dir "$HOME/.graphrag/capture-journal" \
  cleanup JOURNAL_ID --yes
```

Cleanup only removes an independently verified local entry. Pending, uncertain
and receipt-known entries stay available for investigation/recovery. It never
deletes the service note or its receipt, and keeps the small lock file so active
and future retries continue to share the same synchronization identity.

## Verification

Run the deterministic fictional HTTP/storage fault suite:

```bash
python3 -m unittest discover -s scripts/tests -p test_capture_journal.py -v
```

It checks journal-before-send, client crash/restart, loss before/after commit,
exact replay without duplicate records, credential rotation and rejected auth,
principal/endpoint/request/payload mismatch, receipt-known readback failure,
concurrent duplicate/distinct saves, permissions and unsafe-file rejection,
catalog passthrough, origin rejection, and nonpersistence of credentials. Normal
OpenAI OAuth OpenClaw acceptance uses an isolated gateway profile, synthetic
corpus and this proxy URL; record that separate evidence before claiming live
conversational reset/recovery acceptance.
