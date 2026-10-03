# Installed native clients with revision mutations and uploaded jobs

On 2026-10-03, the packaged `--extended` harness passed on one Apple Silicon
macOS host. The [sanitized machine-readable report](mcp-native-clients-77.json)
records the actual installed native APIs and synthetic outcomes. The
[foundation](mcp-native-clients.json) and [earlier read/capture](mcp-native-clients-76.json)
reports remain separate and unchanged.

The pinned source candidate reported `graphrag 0.1.0-rc.2`; binary SHA-256 was
`39eb5a43a2dfe09f61decb677f8bbba7b5b0d702886748f6a5d8defa24e55df9`.
The builder reported source revision `bc802aa25bc096b6213734b3f063290d5239a1c0`;
the hash identifies the tested binary. A host-created portable-backup manifest
reported database schema 18. Wire envelopes used schema 1. This built candidate
is separate from the published rc.2 asset, which predates MCP.

| Actual installed runtime | Observed version/API |
| --- | --- |
| OpenClaw / Node | 2026.9.7 / v26.7.0; session runtime catalog and tool calls |
| Hermes / Python / MCP SDK | 0.21.2 / 3.11.15 / 2.0.0; native registration and registry dispatch |

Ordinary OpenClaw A/B and Hermes retained `read,capture`. The opt-in run
explicitly provisioned a separate writer with revision-mutation grants, an
uploader with `upload,jobs`, and a foreign principal with `read,jobs`. No
capabilities were inferred from tool names or granted to the ordinary clients.

Thirty individually recorded extended native calls completed, followed by
five calls in restart-replay batches. Both actual clients read editable
snapshots and performed edits; each observed the other's exact committed
content/revision. An old opening revision returned `revision_conflict`.
Unconfirmed deletion returned `invalid_input`; confirmed deletion replayed its
original outcome across clients. Replaying the original capture returned its
immutable receipt without resurrecting the deleted note. Three successful
mutation receipts were present after retries and failed preconditions.

An OpenClaw-uploaded synthetic Markdown document reached a controlled provider
pause after the admitting native runtime disposed. Hermes inspected the still
running host-owned job and requested explicit cancellation. The job reached
`cancelled`, resumed under the same job ID, and completed with one published
chunk. Identical upload retry preserved admission/source/job IDs; modified
content under that request ID returned `revision_conflict`.

The foreign principal had the jobs capability, yet its actual native
`get_job`, `cancel_job` and `resume_job` calls each returned `not_found`, and its
job list was empty. Ordinary OpenClaw/Hermes readers shared exact uploaded
content, generation and trusted uploader provenance. After clean service
restart, original capture/delete receipts, upload admission and completed
chunk IDs remained stable.

Host export contained four capture receipts, three mutation receipts and four
live notes (the three foundation captures plus one uploaded chunk). It
preserved opaque client source URIs and contained no generated bearer token.
The foundation checks also repeated cited context, independent capture
namespaces, read-only denial and revocation against a healthy native control.
Every private client, service and synthetic inference process stopped cleanly.

The original authored-worktree gate passed 24 Python tests, including eleven
native harness isolation/failure regressions. Rebasing onto the combined stack
added an ancestor credential test, bringing that gate to 25. Copilot review
then identified a retained-log symlink issue: cleanup now rejects symbolic and
hard links and special files before reading them, with two dedicated
regressions. The post-review stacked gate passed 27 tests, including thirteen
native harness regressions. Node syntax, Python compilation and diff checks
passed. The installed-client extended smoke passed again after this hardening
against the same pinned candidate, with clean process and log cleanup.

No real corpus, personal profile, remote host or conversational LLM session was
used. Proposal decisions, source refresh, entity extraction, restart during a
running job and backup restore were not exercised. A backup was created to
record the schema only. Those boundaries are explicit in the report; this
same-host runtime smoke does not close #78. Follow the
[two-host acceptance checklist](../shared-mcp-walkthrough.md) for the remaining
deployment/session/recovery evidence.
