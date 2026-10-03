# Integrated native-client review checkpoint

The 2026-10-03 same-host extended smoke passed against integrated production
code reported by the builder as
`d4f541e8590ce0de6d0d0540a18ac3838c176978`. The harness itself copied and hashed
the supplied executable before any client/server call and used that snapshot
throughout. Its SHA-256 was
`322de8704df22f7a701feb671de9960d74ad0686edc614968646c62100c28159`;
wire schema was 1 and the host archive manifest reported database schema 18.
The [sanitized JSON evidence](mcp-native-clients-integrated-checkpoint.json)
preserves exact outcomes. Earlier foundation/#76/#77 reports remain unchanged.

This is a review checkpoint, **not the eventual final foundation**. Further
ancestor fixes and their sequential propagation require a separate candidate
run; test/documentation-only commits do not change this recorded binary.

Installed OpenClaw 2026.9.7 (Node v26.7.0) and Hermes 0.21.2 (Python 3.11.15,
MCP SDK 2.0.0) completed thirty recorded extended native calls and five calls in
restart-replay batches. Both edited/read the same exact snapshots. Stale
revision and unconfirmed deletion errors were categorized; confirmed deletion
and original capture replay preserved immutable outcomes without resurrecting
the note.

An admitted uploaded job continued after its OpenClaw runtime disposed. Hermes
requested cancellation while the synthetic provider was paused. Resume
completed the same job and source generation with one published chunk.
Identical upload retry retained its admission; changed payload was rejected.
A separately granted jobs principal could not get/cancel/resume the owner's
job or see it in its list. Ordinary clients shared exact source content and
trusted provenance. Restart preserved completed chunk IDs and immutable
capture/delete/upload receipts. Host export contained four capture receipts,
three mutation receipts and four live notes; all private processes shut down
cleanly.

The finished harness gate passed 31 Python tests, including sixteen native
failure/isolation regressions, plus Node syntax, Python compilation and diff
checks. Log cleanup rejects symbolic/hard links and special files without
following targets, and supplied runspace roots require exact 0700 mode with
write/search access. Ordinary identities retain `read,capture`; writer/uploader
and foreign-job grants are separate opt-in identities.

No personal profile, real note, remote host or conversational LLM session was
used. Source refresh, entity extraction, running-job restart, proposal
decisions and backup restore were not exercised. A backup was created only to
record its schema. Two-computer/session and remaining recovery/maintenance
acceptance stay pending in [the #78 walkthrough](../shared-mcp-walkthrough.md).
