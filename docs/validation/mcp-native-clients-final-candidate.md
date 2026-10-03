# Source candidate native SDK validation after Codex fixes

The October 3, 2026 extended smoke passed on Apple Silicon macOS using an
immutable executable with SHA-256
`896e081973977236150240aa9de21ff8e4648ebd07647772a9ef8a974ee86d6d`.
The snapshot hash was independently checked against the recorded evidence.
The builder attributed the Rust build to `d146db2ba6b00f93cdd1196c790b4a8424263cbd`;
Git comparison confirms all production Rust sources, manifests and lockfile
match report base `d81c6ac6698590b480a7ab7d771084441ed56adf`. Differences are
native harness/adapters/tests and documentation. The executable reports
`graphrag 0.1.0-rc.2`; published rc.2 predates this MCP source candidate.
Wire schema was 1 and the host backup manifest reported database schema 18.

The [sanitized JSON proof](mcp-native-clients-final-candidate.json) preserves
the new outcomes and source attribution. The previous final-candidate
[report](mcp-native-clients-pre-codex-review.md) and
[JSON](mcp-native-clients-pre-codex-review.json) remain explicit historical
checkpoints before the Codex isolation and Linux cleanup fixes.

Installed OpenClaw 2026.9.7 (Node v26.7.0) and Hermes 0.21.2 (Python 3.11.15,
MCP SDK 2.0.0) exercised their native MCP APIs on one host. Two OpenClaw runtime
contexts and Hermes used independent principals. Thirty recorded extended
calls completed, with additional successful restart replay batches. Native
read-only and revoked-client probes recorded enforcement and its runtime
reporting limits.

The smoke verified exact shared snapshots, revision-checked edits, stale
revision rejection, confirmed deletion and capture replay without resurrection.
Uploaded work survived runtime disposal; cancellation during provider
preparation and resume completed the same job and generation. Duplicate
admission replayed, changed payload conflicted, and a foreign jobs principal
could not access the owner's job. Readers observed shared source content and
trusted provenance. Restart preserved capture/delete/upload receipts and
completed chunk IDs.

Export recorded four capture receipts, three mutation receipts and four live
notes, including one uploaded chunk. Inference used twenty-eight synthetic
fixture requests. All private runtimes, service and provider stopped with
`cleanup_errors: []`. This SDK smoke used no personal profiles or notes,
separate computers, or conversational LLM sessions; those last two coverage
flags remain false in the JSON. Separate conversational evidence belongs to
[the two-host walkthrough](../shared-mcp-walkthrough.md).

The tested harness revision `6217f07543cf7971bd87b66962786684afb42384` is unchanged
at the report base. Its observed macOS gate ran 44 Python checks: 43 passed and
the Linux-only regression skipped. All 26 native tests passed on Linux,
including orphan reaping under a controlled non-reaping supervisor. The gate
covers fail-closed private directory writes and bounded descendant cleanup.
The builder separately reported 705 Rust tests passing, five ignored, and
workspace Clippy, format and build passing. These are attributed observations.
GitHub Codex review of the new heads is in progress; no approval is claimed.

Proposal decisions, source refresh, entity extraction, restart during a running
job and host backup restore were unperformed in this SDK smoke. The backup
was created only to observe its schema. Separate [upload validation](../remote-upload-validation.md)
records automated lifecycle and CLI recovery proof. This smoke alone does not
close the two-host walkthrough or the broader #56 acceptance work.
