# Shared MCP validation checkpoints

The initial foundation checkpoint below was validated October 3, 2026 on an
Apple Silicon Mac, Rust 1.97.1. Its #75/#56 counts and original four-tool evidence
are historical observations. Current combined proof is recorded separately in
the [integrated source candidate report](validation/mcp-native-clients-final-candidate.md)
and [upload lifecycle/recovery validation](remote-upload-validation.md).
[Setup and contracts](shared-mcp.md) describe the implemented source workflows.

## Initial foundation automated verification

At that initial checkpoint, `cargo test --workspace --locked` passed 610 tests,
with five pre-existing live
inference tests ignored. `cargo clippy --workspace --all-targets --all-features
--locked -- -D warnings`, formatting, exact workspace MSRV, installer/release
fixtures and 11 Python tests passed.

New tests include principal-scoped atomic capture/replay, conflicting payloads,
concurrent callers, rollback fault injection, revision equality, cancellation,
offline replay and bounded context parity. Real HTTP tests verify legacy MCP
initialization/discovery, two clients, trusted identities, read-only denial,
credential rotation, Host/Origin checks, request/output limits and detached capture
draining after disconnect/shutdown. CLI subprocess tests verify that remote mode
bypasses malformed local configuration, creates no local database, and retains
private capture drafts after failed connections or cancelled editors. Capture
recovery rejects incompatible/mismatched authoritative responses.

Production backup tests create, verify and restore an archive into a fresh
persistent database, then replay the original request. They preserve exact
receipts and opaque client provenance, allow historical references after deletion,
keep existing host path/credential sanitization, and accept earlier schema-15
version-1 archives. Migration 16 preserves existing schema-15 data.

## Initial foundation installed native tool runtimes

The [sanitized evidence](validation/mcp-native-clients.json) records actual tool
discovery/calls through these installed runtimes:

| Client | Version | Runtime entry point |
| --- | --- | --- |
| OpenClaw A/B | 2026.9.7 | `createSessionMcpRuntime`, `getCatalog`, `callTool` |
| Hermes | 0.21.2, MCP SDK 2.0.0 | `register_mcp_servers`, `tools.registry.dispatch` |

The harness used fresh private HOME/profile/state/workspace directories, a
separate persistent GraphRAG corpus, independent generated credentials, and
deterministic host inference doubles. It supplied explicit temporary client
configuration and exercised the discovered runtime handlers directly. No personal
notes, existing client profiles or LLM sessions were used. All synthetic server
and provider processes stopped after validation.

All four original foundation tools were discovered at that checkpoint.
OpenClaw A captured a fictional note; OpenClaw B
and Hermes found and inspected that exact record. Both produced cited context
within a 1000-token cap (114 tokens measured). The same request ID under three
distinct principals produced exactly three notes and three durable receipts.
Retries within one principal returned the original result, including after
service restart, without duplicate captures. Opaque client `file://` attribution
survived host export; no authenticated bearer credential appeared in the corpus.

This proves compatibility with installed native tool runtimes on one computer.
The [opt-in repository harness and two-host walkthrough](shared-mcp-walkthrough.md)
make that distinction explicit and provide reproducible validation without
personal profiles or notes.
Later packaged checkpoints cover [revision-aware reads](validation/mcp-native-clients-76.md)
and the [combined mutation/upload-job candidate](validation/mcp-native-clients-77.md).
The [integrated review checkpoint](validation/mcp-native-clients-integrated-checkpoint.md)
adds the reviewed ancestor changes and hardened harness; it remains separate
from the [later integrated source candidate](validation/mcp-native-clients-final-candidate.md)
and two-host/session acceptance.
Each report identifies its tested binary, observed schema and exact native calls;
the original foundation evidence above remains a separate historical result.
The two-computer deployment and full conversational agent walkthrough remain
[#78](https://github.com/mateu/graphrag-notes/issues/78).
[Revision-checked editing/review](shared-mcp-mutations.md) and
[uploaded sources/jobs](remote-upload-jobs.md) are implemented in the stacked
#80/#81 source changes for #76/#77. Their same-host native proof appears in the
new candidate report; automated upload and CLI recovery proof appears in the
upload validation. #56 stays open until its child acceptance criteria, including
the actual two-computer and conversational-session walkthrough, are complete.
