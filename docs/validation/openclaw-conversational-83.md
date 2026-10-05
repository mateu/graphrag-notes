# OpenClaw conversational validation — issue 83

The corrected service passed Claude CLI normal main-agent native search, guarded
inspection, exact multiline capture and retry against an isolated synthetic
corpus. Requests used the installed gateway webchat route used by Control UI; no
browser click was observed. Daily production rollout is outside this fixture
evidence.

The [Claude CLI JSON proof](openclaw-conversational-83.json) records actual paired
native tool calls and result checks. It omits private machine paths, hostnames,
configuration, credentials, session identifiers, query/body text and principal
identity. Canonical record IDs and revisions are represented by SHA-256 hashes.
The [supported setup](../openclaw-conversational.md) describes normal configuration,
reconnect behavior, exact retry and direct-command fallback.

The later [embedded OpenClaw OpenAI OAuth proof](openclaw-openai-oauth-83.json)
records a separate tested runtime and corpus; its results are detailed below.
Neither report validates the native Codex app-server runtime. Model OAuth
authentication alone does not establish that runtime's MCP permission projection.

## Client, source and discovery boundary

The client used OpenClaw 2026.9.8, Claude Code 2.1.280, Node 24.21.0 and Anthropic
`claude-sonnet-5` through `claude-cli`. Actual native call blocks were imported
from `claude-cli` and named `mcp__graphrag__search_notes`,
`mcp__graphrag__get_record` and `mcp__graphrag__capture_note`. MCP used modern
`2026-07-28`; separate raw readback used the supported legacy `2025-11-25` path.
The Rust MCP SDK was rmcp 3.5.0; the executable reported `graphrag 0.1.0-rc.2`.

The tested service executable SHA-256 was
`f2c43195869b04161ac8a33c0b95837705384af79498c8f25269614914437e89`.
Source attribution is base commit
`b4bb762f16ee605960415254f153f9f64a8e3834`, plus the fixed service and HTTP-test
file SHA-256 values recorded in the JSON. This identifies the tested working
source without embedding a self-referential final documentation commit.

An old-service normal-session baseline reproduced discovery failure: actual
paired `ToolSearch` results included a no-match response, and no GraphRag tool
was dispatched. An isolated no-model control held client policy unchanged and
changed only `ttlMs: 0` and `cacheScope: "private"` on `tools/list`. Without those
fields, four HTTP-success listing attempts still left the native catalog empty.
Adding the two fields populated all eight authorized raw service tools after one
listing request. The fixed service then populated that catalog without a proxy
response mutation. These controls made zero model calls, sent zero user
messages, and read/wrote zero notes.

A separate normal-session `ToolSearch` result contained structured definitions
for exactly the three permitted tools. An exact-name discovery request for
`build_context`, `get_note`, `list_proposals`, `get_proposal` and `get_source`
returned no matching definitions. Those phases dispatched no GraphRag tool.
The raw eight-tool service catalog and the observed three-tool native model
filter are separate facts. No vendor patch, extra conversational bridge or
permission expansion established the fix.

## Actual conversational and fallback observations

| Scenario | Passed observation |
| --- | --- |
| Keyword search and follow-up | Native `search_notes` used keyword mode, graph off, bounded limit and exact query hash; both known seed records matched. Native `get_record` used the selected returned ID/revision and zero neighbors. A separate raw MCP readback matched content/title/actor. |
| Hybrid search | Native `search_notes` used hybrid mode, graph off and the seed's actual server source URI; the known synthetic record and revision matched. Query and source URI are retained only as hashes. |
| Multiline capture | Native `capture_note` sent the exact 211-byte UTF-8 body, title, tags, provenance and fixed request identity. Its authoritative receipt, same-turn native guarded inspection and independent raw readback agreed. |
| Exact ordinary retry | The retained acceptance conversation reused the exact draft/request identity. The native receipt had `replayed: true` and the original canonical ID/revision; guarded native and independent readback still matched. |
| Controlled unknown outcome | Exactly one native capture reached the service; its response was withheld after authoritative success. The agent retained the exact draft in normal user history, reported uncertainty, made no automatic repeat and did not report verified success. |
| Retry after gateway restart | Independent readback established the loss capture's commit before retry. After restart, the same retained history still contained the original exact draft; explicit native retry returned the original receipt with `replayed: true`, then exact guarded native/independent readback. |
| Fresh sessions and reconnect | Fresh normal acceptance sessions discovered and called tools. A fresh session after gateway restart/native reconnection repeated keyword search and guarded inspection. No separate MCP-only disconnect scenario is claimed. |
| Direct fallback | Six command turns covered keyword, hybrid, explicit graph, multiline refusal, single-line save and exact save replay. They showed no positive new-turn model usage or native GraphRag call. Independent single-line readback and canonical replay matched. |

Capture and replay used the following exact body SHA-256:
`9030f4ed9d0393db9155b170bd23108749b439af3928722ee506981def62a1f9`.
The ordinary request hash was
`5420da2c0b2db54813ec6b892534080f7852c411e451c9120c3d425e135c8368`;
the controlled-loss request hash was
`0f4ced4ed616a8ab1a0b6ad3dfd6d2e5fa77858a69a18800d2fe19626a9924ee`.
Actual argument-key lists, modes, provider/model/import source, paired result
counts and canonical equality hashes appear in the JSON. Success depends on
structured call/result pairing and authoritative readback, not assistant prose.

The saved proxy observation predates explicit retry: it recorded one forwarded
capture, zero blocked retry attempts, an authoritative successful upstream
receipt, and a withheld downstream response. Independent service readback also
predated retry. The subsequent replay's ID/revision hashes equal both records.

## Embedded OpenClaw with OpenAI OAuth

On 2026-10-04, OpenClaw 2026.9.8 with Node 24.21.0 used
`openai/gpt-6.1-sol`, the existing Codex/ChatGPT OAuth profile, the embedded
`openclaw` agent runtime and `openai-chatgpt-responses` API. The gateway ran on
Linux and the service on macOS. These were actual normal main-agent gateway
webchat calls, with no browser interaction claimed. The native Codex runtime was
not offered by the tested model catalog and remains unverified.

Keyword/hybrid search and revision-guarded inspection used read-only operations
against an existing source-scoped daily note, with independent MCP readback and
zero production writes. They used a read/capture principal, not a separate
read-only credential. Both service catalog capability sets are covered by the
HTTP regression; live read-only-principal OpenAI projection is unverified.

Capture checks used a separate synthetic service corpus and the same existing
read/capture credential policy. A globally disabled MCP alias was enabled only
for two dedicated test sessions, with the daily endpoint unchanged. Actual
`openclaw.nested-tool.v1` child calls/results were paired with their OpenAI
assistant parent calls and run identities. Claude ToolSearch observations were
not used to establish OpenAI discovery or dispatch.

The exact 211-byte multiline draft, including whitespace, Unicode and final
newline, matched capture arguments, the authoritative receipt, native guarded
inspection and independent raw MCP readback. Ordinary retry returned the original
canonical record/revision with `replayed: true`.

The loss proxy withheld one selected response after authoritative commit. The
model reported uncertainty without claiming a saved receipt or automatically
repeating capture. Independent readback proved commitment before retry. After
gateway restart and a successful health check, read-only checks confirmed the
persisted OAuth/model/runtime/routing selection and retained exact draft before
any reselection. Explicit retry returned the original receipt and record, with
guarded native and independent readback.

An owner-stopped backup including embeddings was independently verified: exactly
13 records, comprising four notes, four sources, four capture receipts and one
metadata record. The notes were two seeds and two test captures; neither retry
created a duplicate. All 63 offline parser/capture/proxy cases passed, and
independent Codex review found no actionable evidence findings. Sessions were
archived, temporary overlays/alias/processes removed, and original configuration
restored byte-for-byte. The daily service and existing note were unchanged.

The separate JSON records the tested source commit `31b5695`, the same executable
SHA-256 as the Claude run, exact body and identity hashes, actual paired tool-call
metadata, and private-proof digests. It omits raw bodies, canonical IDs, OAuth
profile/account identifiers, credentials, hostnames and private paths. These
OpenAI results support the embedded OpenClaw path only; native Codex app-server,
live read-only-principal projection, and a macOS OpenAI gateway run remain open
validation limits.

## Claude CLI corpus audit, checks and limits

An independently verified portable backup, including embeddings, recorded schema
18, six notes, six sources and six capture receipts with six distinct receipt
note IDs. The ordinary retry identity and controlled-loss retry identity each
had exactly one receipt. The six notes comprise two seeds, the ordinary capture,
an extra acknowledged setup trial, the controlled-loss capture and the direct
fallback save. The report does not claim a four-note minimal corpus or conceal
the setup trial; the authoritative count found no duplicate request or note ID.

Recorded checks passed: 744 Rust workspace tests, zero failures and five ignored;
workspace Clippy; 44 Python script checks with 43 passing and one skipped; all 16
release checks; and the install smoke.

The synthetic corpus stayed on the service host, which retained database and
inference ownership. No test notes were written to production. The normal
configuration was restored exactly, retaining its environment credential
reference and three-tool filter, and the synthetic service, proxy and tunnel
were stopped. Daily production rollout and its direct-command checks remain
outside this report.

Recovery here uses an explicit stable request ID and exact draft in durable
normal user history. The native integration adds no private capture journal and
does not guarantee recovery after compaction, session reset or history loss.
No automatic memory replacement, transcript ingestion or synchronization was
enabled. Only passed proof scenarios are included; the baseline passed by
reproducing the expected failure.
