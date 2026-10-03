# Observed two-host conversational MCP checkpoint

The October 3, 2026 synthetic walkthrough completed actual conversational calls
from OpenClaw A and Hermes on **shiva (macOS)** and OpenClaw B on **clawd (Linux)**,
plus separately attributed lifecycle and portable-maintenance checks. The
[sanitized JSON proof](mcp-two-host-walkthrough.json) preserves exact service
envelopes, identities, failures and source attribution. This is an observed
checkpoint that demonstrates the required native capture/read walkthrough
acceptance. The dated follow-ups below record the later endpoint-compatibility,
public-job-status and queue scheduling fixes under their own source/runtime attribution. No issue
closure or GitHub approval is claimed.

The preceding vector-restored checkpoint had SHA-256
`3a99eb0eb6d324e3903e8f119f4f3656cad7544b1930f747df6742f6ebdb942b`, source
`02a49a2449153d9b1f3ec67bab2907ab4c610bd6`. Its frozen hash was independently
checked. Earlier conversation/job checks used `896e0819…`, built from `d146db2`;
default-vectorless maintenance used `8d69d630…`, built from `568481f`. Full hashes
and the distinct scope of every window remain in the JSON. The last source
comparison has only CLI editor/archive-exporter changes since the original run;
MCP/service/application/database/core sources are identical at those recorded
revisions. This comparison does not apply to later source fixes.

## Actual conversational observations

| Client | Observed model | Successful calls on the vector-restored checkpoint |
| --- | --- | --- |
| OpenClaw A, shiva, 2026.9.7 | Google `gemini-3.8-flash` | `capture_note`, original receipt replay |
| OpenClaw B, clawd, 2026.9.8 | Anthropic `claude-sonnet-5` | `search_notes`, `get_record`, `get_job`, `get_source` |
| Hermes, shiva, 0.21.2 | Ollama Cloud `deepseek-v4.1-flash` | `search_notes`, both writers' `get_record`, `build_context` |

A's exact fictional capture was:

> twohostatlasfinal: The fictional Atlas launch meeting is Monday at 10:00.

B and Hermes inspected the same canonical writer record:

- ID: `note:aba607cd7a8ac6d14f48601c1142d186512b428df2c26f1e7c19370738684c84`
- Revision: `6a08ec335f19fbba21bd2ed55b32a9b6a70ad5193f3984bbb71e6ea67247c720`
- Trusted authenticated writer: `openclaw-shiva`

The original Hermes context cited that record as `[C1]`, using 120 estimated
tokens under requested limits of 300 total, 120 per chunk and three chunks.
On the vector-restored checkpoint, Hermes returned three citations and 298
estimated tokens within the same budget. Its two later chunks were truncated;
the JSON retains every cited ID and selected source URI.

| Observed window | Successful MCP calls | Outer agent bridge turns | Recorded failed attempts |
| --- | ---: | ---: | --- |
| Original conversation | 6 | 11 | A guest bridge error before dispatch |
| Default-vectorless restore | 8 | 17 | Hermes rejected batch; B two schema errors |
| Vector-inclusive restore | 9 | 18 | Hermes rejected batch before separate reads |

All nine agent sessions completed; the three vector-restored processes exited 0.
The last three sessions had no MCP service errors. Hermes' outer bridge still returned one
explicit rejected multi-local batch; that failure is counted even without an
`is_error` flag. B's default-restored CLI summary reports fifteen invocations/four
failures, while its saved outer transcript has nine turns/two schema errors.
Those counts use distinct units. Content/details envelope duplicates likewise
do not count as extra MCP calls. Raw execution code and reasoning are excluded.

## Route, authentication and durable jobs

GraphRAG listened on shiva loopback port 31056. An encrypted reverse SSH forward
exposed clawd loopback port 33156 to B. MCP negotiation was `2025-11-25`, with
service envelope schema 1. Owning-credential service-host catalog probes returned
13 tools for clawd, 11 for shiva OpenClaw and 13 for Hermes.

Scripted RPC checks proved read-only reads, `forbidden` mutation, invalid/revoked
credential 401, continued reads for other principals and writer rotation. The old
token failed; the new token retained its identity and immutable capture replay.
Identical capture payload replayed; changed payload returned `revision_conflict`.
The same request ID under B created a distinct canonical note. These were RPC
checks, rather than additional conversational agent calls.

During a controlled SSH outage, B's installed native runtime could not connect.
After reconnection it discovered its catalog and read the same running job.
The discarded-acknowledgement case used a separate local controlled relay.

Eight ordered service-host job phases completed. An initial out-of-order
cancel/resume invocation was rejected before the corrected sequence completed.
Three jobs covered two source identities: the base survived disconnect, explicit
cancellation and resume at generation 1; restart interrupted extraction in the
second job, and explicit resume completed that same job/generation with extraction.
Each completed job had one chunk and zero failed chunks. Foreign Hermes received
three `not_found` job denials; shiva's client lacked jobs grants and received four
`forbidden` denials.

Refresh retained generation 1 retrieval until generation 2 promotion. Unpromoted
replacement content stayed absent; promotion retired the old record with
`not_found`, and duplicate admission replayed. Source/record provenance retained
trusted `openclaw-clawd`, generation IDs and accepted synthetic attribution.
Client-only URIs and supplied `/Projects/topic` titles remained opaque data.
B's later conversational calls read the completed **base job only**, and the
ready generation 2 source. The separate maintenance RPC proof establishes all
three completed jobs. No LLM is claimed to have submitted, cancelled, resumed,
refreshed or replayed these uploaded jobs.

The first 40 inference ledger rows form the original conversation/lifecycle
snapshot: 20 paired real forwarded requests, all HTTP 200 on shiva, comprising
six `bge-m3:latest` embedding calls, three `phi4-mini:latest` extraction calls and
eleven model-discovery requests. Later calls are outside that prefix's hash and
counts. Host inference is separate from client agent providers. Interrupting only
the private forwarder preserved keyword reads and inspection; an uncached hybrid
search returned `provider_unavailable`. The personal Ollama service was unchanged.

## Portable maintenance and limits

Default-vectorless create/verify/fresh restore passed under `8d69d630…`: schema 18,
51 records (14 entities, 21 mentions, five notes, three jobs, three capture receipts
and five sources). Original capture/upload/refresh receipts replayed without
provider work. Current restored capture revisions changed while IDs/content/
provenance remained intact; the immutable receipt retained its original revision.
All three agents subsequently read/replayed that default-restored service.

The initial vector-inclusive archive exposed an existing zero-vector entity
export issue. Its CLI fix produced the recorded `3a99eb0e…` checkpoint. Actual
vector-inclusive create/verify/fresh restore passed with 52 records: the same
counts plus one embedding metadata record. Original inspection preserved every
canonical field, including revision; three jobs, generation 2 and the opaque
supplied title survived. All three agents then succeeded on that fresh restore.

The builder's recorded gate reports 710 Rust passes, five existing ignored across
33 targets, 24 archive regressions and workspace Clippy/format/build passing.
The unchanged harness gate records 43 macOS Python passes plus one Linux-only
skip, and all 26 Linux native tests passing. These are attributed observations.
The [same-host SDK smoke](mcp-native-clients-final-candidate.md) remains separate,
with its two-computer and real-LLM flags false.

Proposal decisions and resume after a superseding generation were unperformed.
Both maintenance archives contained completed jobs; restoration of an in-flight
worker lease was unperformed, while live restart/interruption/resume was observed.
These supplemental limits do not mark the demonstrated native walkthrough
acceptance pending. The preceding checkpoint does not attest the later source
fixes; the follow-up below records their acceptance.

## October 3 follow-up: reviewed recovery and fresh restore

The later endpoint-compatibility and quarantined-status changes were reviewed at
source `f6af11e3d3ed541dfc9bfde62be6eab9a8c42148`. Their observed executable had
independently checked SHA-256
`a861e61249f8e2e2882c26cc64bb5b19765b4f8ee61bd92750c9802013af7d7e`.
The report worktree's production crates and Cargo files match that tested source.
The builder recorded 715 Rust passes, five existing ignored across 33 targets,
and passing workspace Clippy, format and build. Focused regressions passed:
21 database jobs tests and 11 application jobs tests. Independent Codex review
found no actionable issue in the two source-fix commits. Subsequent source
changes require their own acceptance attribution. The unchanged native
harness retains its separately recorded macOS and Linux gates above.

| Client session on the fresh restored service | Outer turns | Successful MCP calls | Retained diagnostics |
| --- | ---: | ---: | --- |
| A `2aadfa40-779d-4447-9a00-f0d6ac9a260f` | 4 | 1 capture replay | One unawaited-promise discovery warning, recovered before capture |
| Hermes `20261003_121525_68fce7` | 6 | 4 search/record/context calls | One rejected multi-local bridge batch |
| B `8d30b5e3-b6e6-4474-9170-0b98b489c0c6` | 9 | 4 search/record/job/source calls | None |

All three launcher processes exited 0. The nine successful service calls produced
11 saved envelope representations, with zero service errors. A replayed its
original immutable capture receipt; Hermes and B inspected the exact original
writer record, including revision `6a08ec33…` and trusted `openclaw-shiva`.
Hermes inspected both writers and built three cited chunks using 296 estimated
tokens within the requested 300-total/120-per-chunk limits; its final two chunks
were truncated. B read the completed **base job only** and the ready generation 2
source with its opaque `/Projects/topic` title. These are actual conversational
calls using the same three configured models listed above.

Separate authenticated service-host RPC submitted one new synthetic upload with
title `/Projects/reviewed-provider`, completed generation 1 through a real
`bge-m3:latest` embedding request, and replayed its admission. Inspection of the
actual portable archive found 64-character endpoint digests for both provider
roles in that new job's processing options, with no raw endpoint URL or service
credential. Three older completed jobs retain their original options without
endpoint identity. Focused fixtures proved changed endpoints or missing legacy
identity reject execution/resume before provider work, while malformed saved
input remains visible through owner-scoped status APIs without weakening strict
execution/resume. Those mismatch/quarantine cases were regression tests, rather
than additional live malformed-input or endpoint-change attempts.

Actual vector-inclusive create, verify and fresh restore succeeded at schema 18
with 55 records: 14 entities, 21 mentions, six notes, four jobs, three capture
receipts, six sources and one embedding metadata record. Restored RPC replayed
all three capture receipts and four upload admissions with no inference POSTs,
and returned all four completed statuses. The owning list contained four jobs;
foreign Hermes get/cancel probes returned `not_found` and its list was empty.
A lacked jobs permission and received `forbidden`. Hermes could read the new
shared source. Original canonical inspection and generation 2 provenance/title
survived. The three sessions above then used this freshly restored service.

This follow-up adds three sessions, bringing the distinct recorded windows to
12 sessions, 32 successful MCP calls and 65 outer transcript turns. The JSON
preserves the original `896e0819…`, `8d69d630…` and `3a99eb0e…` observations and
hashes separately. Job mutation, archive and four-job status checks remain RPC
orchestration, rather than LLM job submissions. Completed-job archive coverage
does not establish restoration of an in-flight worker lease. Observed prior
owners stopped before the fresh restore; the acceptance runtime remains privately
managed. Permanent installation, personal profile changes and final runtime
cleanup are separate operator follow-up.

## October 3 follow-up: serialized uploads and final queue restore

The next tested Rust checkpoint was
`24b2d0c095c7b5c3158e8ce954cdd8e7cb6eef19`, including the database scheduler fix
`a58f497` and two-worker HTTP regression. Its independently checked executable
SHA-256 was
`60d349c33df481e16fd747ccdf77341f5deb887c7657c585710eba17d483f35f`.
The report checkout's production crates and Cargo files match that source;
scripts match the earlier tested harness. The recorded full gate passed 721 Rust
tests, zero failed and five existing ignored across 33 targets (26 nonempty),
with workspace all-target/all-feature Clippy, format and build passing. Focused
checks passed all 26 database job tests and four service upload HTTP tests.
Independent Codex review found no actionable issue in the frozen fix and its
regressions after resolving mixed legacy/current ordering and legacy timestamp
ties. Later source changes still need separate runtime attribution.

The scheduler reserves a principal's logical source at claim, before a generation
exists, and retains that ownership through terminal settlement. Waiting versions
stay queued. Claims and stale-resume fencing share a total order: legacy entries
by timestamp/ID, followed by numbered admissions by order/timestamp/ID. Focused
regressions cover concurrent claims, cancellation/recovery, strict failed resume,
mixed histories after clock rollback, tied legacy timestamps, malformed-input
quarantine, principal isolation and over 100 blocked updates without starving an
unrelated document. The two-worker HTTP fixture proves unrelated work can finish
while the first source's synthetic provider remains held.

Actual host-side authenticated RPC used a controlled private forwarder and real
shiva inference. The successful **002** run persisted five responses showing the
second accepted version still `queued`, without generation or service error,
while the first version was preparing generation 1. A distinct unrelated document
occupied the other worker and reached its own held real embedding request.
Controlled release completed the first/second/unrelated jobs at generations
1/2/1, without manual resume. The current source contained the second version,
retained the opaque `/Projects/serialized` title, and all three original admission
receipts replayed. The inference window records six paired requests: three
`bge-m3:latest` embeddings and three discovery requests, all HTTP 200.

The hold proof combines distinct provider pause markers and five sample times
before controlled release. Maximum marker age was 0.964419 seconds against a
150-second automatic release threshold. Gate presence was checked when both
markers were obtained, with no other pause controller; it was **not recorded
individually at each sample**. Forwarded ledger timestamps represent response
completion. The JSON preserves this timing scope. A first instrumentation attempt
had already completed three jobs when its replay marker hook failed; its queued
response envelopes were not saved. That wrapper failure and its real completed
jobs are preserved separately, without reconstructing missing evidence.

Actual vector-inclusive create/verify/fresh restore succeeded with schema 18 and
69 records: 14 entities, 21 mentions, ten notes, ten jobs, three capture receipts,
ten sources and one embedding metadata record. All ten archived jobs were
completed, including the initial instrumentation attempt. Seven jobs retain both
64-character endpoint digests; three legacy completed jobs retain no endpoint
identity. The archive has no raw provider URL in processing options or service
credential. Restored RPC replayed all three captures and ten full upload receipts
and preserved ten completed status/results, with zero inference POSTs during
replay. The owner list contained ten jobs. Hermes get/cancel probes returned 20
`not_found` denials and an empty owner list; A received 20 `forbidden` get/cancel
denials and a forbidden list. Seven uploaded sources matched their shared-reader
views. Original canonical inspection survived unchanged.

| Actual session on the final queue-restored service | Outer turns | Successful MCP calls |
| --- | ---: | ---: |
| A `87946f63-b9ba-43f3-8adb-3938b8735636` | 3 | 1 original capture replay |
| Hermes `20261003_124158_436c14` | 6 | 4 search/record/context calls |
| B `96b25061-07fd-404b-9783-53b8bf093b0d` | 9 | 4 search/record/job/source calls |

All three processes exited 0 using the same configured client models above.
Nine successful MCP calls yielded 11 envelope representations and zero service
errors. Hermes retained one rejected multi-local bridge batch, then inspected
both writers and returned three citations using 280 estimated tokens within the
300-total/120-per-chunk budget. Its second citation names the successful queued
generation 2 chunk; only its final chunk was truncated. A replayed the original
capture, and Hermes/B inspected its exact `6a08ec33…` revision and trusted writer.
B directly read the **second 002 job at completed generation 2** and its ready
generation 2 source/title. It did not submit or resume that job.

Across the five separately attributed windows, the report now records 15 real
sessions, 41 successful MCP calls and 83 outer transcript turns. Earlier hashes,
sessions, failures and counts remain intact. Queue mutations and ten-job archive
checks remain RPC orchestration. In-flight portable worker-lease restore remains
unperformed. Prior owners exited before the fresh restore; the final acceptance
runtime remains privately managed. Permanent installation, personal profiles
and final cleanup remain separate operator follow-up. This report does not
claim issue closure or GitHub review approval.
