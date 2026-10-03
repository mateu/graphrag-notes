# Shared MCP foundation validation

Validated October 3, 2026 on an Apple Silicon Mac, Rust 1.97.1. The foundation
implements #75 under #56. [Setup and contracts](shared-mcp.md) describe its scope.

## Automated verification

`cargo test --workspace --locked` passed 610 tests, with five pre-existing live
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

## Installed native tool runtimes

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

All four tools were discovered. OpenClaw A captured a fictional note; OpenClaw B
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
Each report identifies its tested binary, observed schema and exact native calls;
the original foundation evidence above remains a separate historical result.
The two-computer deployment and full conversational agent walkthrough remain
[#78](https://github.com/mateu/graphrag-notes/issues/78). Revision-checked remote
editing/review and uploaded sources/jobs are tracked in #76 and #77; #56 stays open.
