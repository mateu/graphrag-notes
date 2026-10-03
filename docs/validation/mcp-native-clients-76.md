# Installed native-client smoke after revision-aware reads

On 2026-10-03, the packaged harness passed on one Apple Silicon macOS host.
The [sanitized report](mcp-native-clients-76.json) is separate from the preserved
[foundation report](mcp-native-clients.json).

The tested source candidate reported `graphrag 0.1.0-rc.2`; its binary SHA-256 was
`25820c3487acfac91d26b12f14cf36897bed07e2a35e8998509c2e5a6c8f1aee`.
This is a built candidate, not the published rc.2 asset, which predates MCP.
Wire schema was 1. A host-created portable-backup manifest confirmed database
schema 17; that archive was created for the schema check, not restored.

| Installed runtime | Observed version |
| --- | --- |
| OpenClaw / Node | 2026.9.7 / v26.7.0 |
| Hermes / Python / MCP SDK | 0.21.2 / 3.11.15 / 2.0.0 |

Both actual runtimes discovered `search_notes`, `get_record`, `build_context`,
`capture_note`, `get_note`, `get_proposal` and `list_proposals`. The first four
were called through OpenClaw's session runtime and Hermes' tool registry. The
three new read tools were discovered only in this run.

Two OpenClaw contexts and Hermes used distinct authenticated instance IDs to
share three synthetic notes. Identical capture retries produced three durable
receipts and three notes; the original receipt replayed unchanged after service
restart. Both clients returned cited context within the 1000-token budget
(114 reported tokens). Host export preserved the opaque client `file://` source
URI and included none of the generated credentials.

All ordinary writers retained only `read,capture` grants. A read-only runtime
could read and denied capture; its runtime may refuse an unadvertised tool
before transport. A revoked credential could no longer connect/call while a
valid native control read the same endpoint. OpenClaw redacts HTTP status/body,
so this native result proves denial against a healthy control; the exact 401
status is covered separately by service HTTP tests.

The harness preserved the invoked Hermes virtualenv Python symlink, used fresh
private HOME/profile/workspace directories, and stopped every private service,
client and inference process cleanly. Seven harness failure/isolation
regressions and all 20 repository Python tests passed; Node syntax and diff
checks passed. No personal configuration, real notes, remote host or agent LLM
session was used.

Mutation/upload/job scenarios require the combined candidate and separate
explicit grants. Two-computer and conversational-session evidence, plus the
remaining deployment/maintenance checklist, remain pending in
[the #78 walkthrough](../shared-mcp-walkthrough.md). This smoke does not close
#78 or the parent #56.
