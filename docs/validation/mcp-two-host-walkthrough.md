# Observed two-host conversational MCP checkpoint

The October 3, 2026 synthetic walkthrough completed actual conversational calls
from OpenClaw A and Hermes on **shiva (macOS)** and OpenClaw B on **clawd (Linux)**,
plus separately attributed lifecycle and portable-maintenance checks. The
[sanitized JSON proof](mcp-two-host-walkthrough.json) preserves exact service
envelopes, identities, failures and source attribution. This is an observed
checkpoint that demonstrates the required native capture/read walkthrough
acceptance. Later endpoint-compatibility and public-job-status fixes need separate
source/runtime attribution. No issue closure or GitHub approval is claimed.

The last observed executable had SHA-256
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
acceptance pending. This checkpoint does not attest later source fixes.
