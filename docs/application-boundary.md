# Shared application operations

This design defines the local operation boundary required by terminal workspace
issue #63 and the future remote service in #56. The terminal and the ordinary
CLI use the same retrieval, ingestion, inspection, and proposal policy. Issue
#63 implements an embedded backend; it does not introduce an HTTP server, MCP
transport, remote credentials, or remote CLI configuration.

## Ownership and adapters

`graphrag-application` exposes typed requests, results, errors, and an
`ApplicationOperations` interface. `EmbeddedApplication` receives the existing
`Repository`, configured search and librarian agents, and shared inference
providers. It delegates domain work to those agents and the repository. It
does not open a database, parse command-line arguments, print output, resolve
filesystem paths, launch programs, or run database queries of its own.

The CLI bootstrap opens one embedded connection. The terminal keeps that
connection for its entire session and clones the same repository when needed.
Those clones retain the existing proposal-lifecycle coordination. No terminal
action starts another `graphrag` process against the same corpus.

An embedded persistent database has one owning process. Another CLI process
must fail promptly with an actionable database-lock error identifying the
selected database and advising the user to exit the owning workspace before
retrying. It must not kill the owner or wait indefinitely. Memory-backed
sessions remain useful for tests and ephemeral browsing; recoverable captures
still require a persistent corpus.

The future service becomes the sole database owner. A remote adapter invokes
the same operations without opening a client-side RocksDB store or requiring
client-side inference. Service authentication, authorization, request identity,
and transport negotiation belong to #56 rather than this local implementation.

## Initial operations

The initial typed interface supports bounded keyword/hybrid search, recent
visible notes, record inspection, capture, source status, proposal inspection
and deliberate decisions, and corpus statistics. Search requests carry query,
scope, retrieval mode, graph mode, entity filter, and a bounded result limit.
Inspection carries the canonical record reference and a bounded neighboring
message count. Capture carries content, optional title, and tags, rather than
a file pathname or editor command.

The existing `SearchAgent` remains authoritative for retrieval order, graph
expansion, scopes, and compatibility checks. The existing librarian remains
authoritative for embeddings, extraction, and atomic note/entity persistence.
The repository remains authoritative for visibility, source generations,
record revisions, and proposal lifecycle. The boundary adds validation and
orchestration rather than a second implementation of those policies.

Requests and operation errors are versioned with application contract version
1. DTOs use canonical string IDs, content, provenance, bounded collections,
and primitive options. Existing domain results may be retained internally
without publishing database connection types, query access, filesystem paths,
or provider objects as remote capabilities. The CLI continues to adapt these
results into its existing schema-version-1 JSON/JSONL output; application
contract versioning does not change existing CLI envelopes or exit codes.

## Record identity and revision checks

Display numbers identify positions within one terminal result snapshot. They
never become record IDs. Every selectable result stores its full canonical
`note:`, `message:`, or `conversation:` ID and the revision obtained from
provider-free inspection. Search again replaces the result snapshot and clears
the old selection. Navigation within a snapshot preserves the original ID and
revision.

Inspect, open, and copy resolve that exact reference again before acting. A
missing record or a revision mismatch produces an actionable error directing
the user to repeat the search. They never follow a source URI, title, ordinal,
or generated successor to a different note. The repository's existing
inspection fingerprint is the revision contract, including source-generation
and linked conversation information where applicable.

Guarded content edits and detach retain the existing full opening-note
snapshot check inside the atomic write transaction. An inspection fingerprint
is an observational read guard; it is not a substitute for that mutation
snapshot. Editing through the future service must supply an expected revision
and use a repository-backed conflict check.

Proposal cards retain proposal metadata plus both endpoint revisions. The
shared decision operation rebuilds the card and refuses changed proposals or
endpoints before invoking the existing accept/reject/undo repository methods.
Acceptance requires explicit human confirmation and remains manual and
audited. This preserves the existing inbox's freshness check and lifecycle
coordination; it does not claim that its pre-mutation read is a new SQL
compare-and-swap. Stronger remote concurrency guarantees must be implemented
and tested in #56 before exposing competing-client mutation tools.

## Provider availability and cancellation

Starting a workspace performs no inference health checks. Keyword search,
recent notes, inspection, source status, proposal review/decisions, and stats
work with unavailable providers. Hybrid search checks embeddings when that
action is selected. Capture checks embeddings and, unless extraction is
disabled by existing runtime policy, extraction. A provider failure returns to
the workspace so the next offline operation remains usable. The boundary
does not silently change a hybrid request into keyword retrieval.

Shared resilient provider wrappers preserve existing request limits, retries,
cache behavior, and concurrency semaphores across the session. Availability is
evaluated per action rather than cached as a permanent session-wide result.

Each action gets a fresh cancellation flag. Cancelling one action must not
cancel subsequent commands. Read-only work and provider-health preparation
may be abandoned safely. A mutation future must not be dropped after a write
can have begun and then reported as though no data was saved. Capture observes
cancellation at safe preparation boundaries; once its atomic persistence
begins, it finishes and reports the committed note ID. A successfully committed
capture remains successful even if a cancellation arrives during persistence.
On cancellation or provider failure before commit, the client retains its
private draft and reports recovery guidance.

Existing checkpointed processing continues to use librarian/reindex
cancellation flags at item boundaries and existing durable job records. A
future service owns those jobs independently of a client connection. A
disconnect will not imply cancellation; explicit cancel/resume and reconnect
behavior, durable capture retry identities, and upload limits are #56 work.

## Local paths and interaction

Draft creation, editor execution, history/completion, clipboard access, and
source opening are client-adapter responsibilities. Native paths and executable
arguments must not become application operation requests. Local open validates
the selected revision and uses the existing literal-argv opener behavior.

Source URIs describe the corpus owner's source, not a file automatically
available on another client. A future remote client must use an explicit path
mapping or supported source-content workflow before opening such a URI.
Remote imports upload content with stable identity rather than asking the
service to read arbitrary client-supplied paths. Host-managed folders and
client uploads require distinct refresh contracts in #56.

The terminal must remain readable in a narrow window and usable with piped
commands for deterministic tests. An interactive line editor may add keyboard
history/completion, but transport operations do not depend on terminal state.
History must not persist captured private content by default.

## Errors and compatibility

Application errors distinguish validation, not found, revision conflict,
provider unavailable, cancellation, and internal failures. Database ownership
errors are produced during bootstrap before an operation backend exists. The
CLI adapter preserves established exit codes and machine output behavior.
Future unauthorized, forbidden, service-unreachable, and retry-identity
conflicts require #56 implementation rather than fictitious local behavior.

The boundary introduces no schema migration, no new source-generation policy,
and no change to existing archives or embedded installations. macOS and Linux
are supported targets. Windows is outside this project's requested scope.

## Validation

Focused tests must establish zero provider requests for offline operations,
identical existing retrieval order, stale-selection rejection after edits or
source refresh, shared proposal freshness behavior, and cancellation that
does not poison subsequent actions. Capture tests cover provider/cancellation
failure before commit, one atomic committed note, and retained recovery input.

Terminal walkthroughs cover an empty corpus, unavailable providers, result
selection without retyping IDs, open/copy, proposal review, narrow terminals,
competing database ownership, and recovery to the prompt after errors.
Non-interactive CLI/JSON regression checks remain required. Network and real
OpenClaw/Hermes interoperability validation belongs to #56.
