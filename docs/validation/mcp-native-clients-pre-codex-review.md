# Historical native validation before Codex review fixes

This preserved checkpoint predates the Codex isolation and Linux descendant
reaping fixes. Its binary, test counts and review status describe that earlier
run. The [current candidate report](mcp-native-clients-final-candidate.md) records
the replacement proof.

The October 3, 2026 installed-client extended smoke passed on Apple Silicon
macOS against code reported by the builder as
`a1bcef17b05994a1de0a951e5c686bb437b2ac80`. The immutable executable snapshot
used throughout the run had SHA-256
`ff64eeddbfaa47ba6c5181876a5e661bcb9d5d2689817352fc62a2cf24590362`.
Its version label is `graphrag 0.1.0-rc.2`; this is an MCP-enabled source
candidate, while published rc.2 predates MCP. Wire schema was 1 and the host
backup manifest reported database schema 18.
Published PR #81 head `58b8b7d` has byte-identical production sources,
manifests, lock and scripts to the tested builder tree; its remaining changes
are a standalone HTTP test and documentation. The file comparison was verified
before the final harness rebase.

The [sanitized JSON proof](mcp-native-clients-pre-codex-review.json) preserves
catalogs, exact outcomes, binary attribution and coverage limits. Earlier
foundation, #76/#77 and integrated review checkpoints remain separate evidence.
This run used harness revision `9b6bbf8`, including installed-runtime containment,
linked/special-log rejection, forced-descendant detection, bounded shutdown and
bounded macOS parent reaping after process-group signal denial. An unconfirmed
group remains a failed run even after parent reaping succeeds.

OpenClaw 2026.9.7 (Node v26.7.0) and Hermes 0.21.2 (Python 3.11.15, MCP SDK
2.0.0) completed thirty recorded extended tool calls plus five restart-batch
calls through their actual native APIs. Two OpenClaw runtime contexts and
Hermes shared the synthetic corpus with independent authenticated principals.
Read-only and revoked-client probes recorded their actual runtime enforcement.

The run verified exact manual-note snapshots, revision-checked edits, stale
revision rejection, explicit delete confirmation and immutable replay without
resurrection. Upload processing continued after its submitting runtime disposed;
cancellation during provider preparation reached a safe terminal boundary.
Resume completed the same job/generation, duplicate admission replayed, changed
payload conflicted and a separately granted jobs principal could not access the
owner's job. Readers inspected the shared source and trusted provenance.
Restart preserved capture/delete/upload outcomes and completed chunk IDs.

Export showed four capture receipts, three mutation receipts and four live notes,
including one uploaded chunk; inference used twenty-eight synthetic host fixture
requests. All private subprocess groups, service and provider stopped cleanly,
with no forced descendant cleanup or cleanup error. No personal profiles,
personal notes, external host or conversational LLM session were used.

After the final ancestor rebase and exit-race fix, forty stacked Python tests
passed, including twenty-two native tests; those native tests also passed under
the installed Hermes Python. Node syntax, Python compilation and diff checks
passed. Real-process fault injection covers denied TERM/KILL exit races and
ensures a successful parent cannot hide an unconfirmed residual group.
The builder separately reported 697 Rust tests passing, five ignored, workspace
Clippy/format/build passing and eighteen pre-harness Python tests. These are
attributed observations, not a live test inventory.

The latest requested Copilot review was unavailable because the requesting
account exhausted its review quota. Prior agreed findings were fixed;
independent cleanup audits found no actionable issue. That unavailable review
provides no approval.

Two-computer deployment and real agent sessions remain pending in
[the #78 walkthrough](../shared-mcp-walkthrough.md). Source refresh, entity
extraction, running-job restart, proposal decisions and backup restore were
unperformed here. A backup was created solely to observe its schema; separate
[upload validation](../remote-upload-validation.md) records automated lifecycle
and CLI recovery proof. This candidate smoke does not close #78 or #56.
