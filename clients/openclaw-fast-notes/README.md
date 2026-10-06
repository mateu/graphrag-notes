# OpenClaw direct notes client

This directory maintains the direct `/notes` and `/notesave` handler that was
previously installed only on an OpenClaw host. Version 0.5.0 adds bounded reuse
for **read searches**. The command still returns the owning service's results
without starting an agent model. Ordinary `/notes` and `--hybrid` retain hybrid
search with graph on; `--keyword` and `--no-graph` are explicit opt-outs.

`/notesave Title | body` keeps its separate fresh MCP client, exact request and
payload, durable receipt, and independently pinned `get_record` verification.
A missing acknowledgement remains unconfirmed; repeating the exact command is
an explicit operator action. The read pool cannot dispatch a write. Capture
resolves the current plugin configuration on every invocation, rechecks its
endpoint/credential/actor after preparation, and permits one capture POST plus
one exact revision-pinned readback POST. SDK retries cannot silently repeat a
capture; an expired session remains unconfirmed until an explicit retry.

## Ownership and dependencies

The inspected prior handler was `graphrag-fast-notes` 0.4.0, source SHA-256
`6621568fac07395d6e23401a25edc847eca236627a34eb1832b63cc291d3e1ca`.
OpenClaw reported 2026.9.8 and its installed public MCP SDK reported 1.30.0
(MIT). This package uses the SDK's public `client/index.js` and
`client/streamableHttp.js` exports. A configured absolute `sdkAnchor` can resolve
that installed dependency; otherwise resolution starts beside this package.
The lockfile pins the tested SDK to 1.30.0. Node 22 or later is required.

Install the package through the host's normal OpenClaw plugin workflow, retaining
its existing MCP server, credential source, command permissions and model
configuration. Replacing an existing installation requires stopping the owning
gateway, retaining the old plugin files, verifying the new files, and restarting
that owner. Review a concrete candidate and run isolated fixtures before a live
upgrade. This repository does not automatically change any host installation.

## Pool contract

A gateway plugin instance owns the pool. Keys bind the canonical URL and selected
credential to its server, token environment and SDK context. Digests and the raw
credential comparison remain in process memory; no credential or credential
fingerprint is written to diagnostics. Every HTTP request sends the selected
Bearer again, and the service authorizes every tool call. The credential is
rechecked against the caller's current configuration and authorization after
asynchronous initialization and before each search HTTP dispatch.

Defaults are four total connections (including closing or still initializing
connections), four active searches per connection, 30 seconds idle and five
minutes absolute lifetime. Admission fails as busy without a queue when a bound
is reached. Configurable limits are:

| Setting | Default | Allowed range |
| --- | ---: | ---: |
| `poolMaxEntries` | 4 | 1–4 |
| `poolMaxInFlight` | 4 | 1–4 |
| `poolIdleMs` | 30000 | 1000–300000 |
| `poolLifetimeMs` | 300000 | 5000–900000 |

`reuseConnections: false` selects fresh clients. Hosts without the public
`registerService` lifecycle also use fresh clients. The registered service closes
connections on stop or configuration reload. Fresh reads also share this lifecycle
and retain their resource reservations during pending initialization or cleanup.
Switching between fresh and reused modes cannot bypass the bound. Closing has a bounded caller wait;
a stuck SDK still occupies its reservation until initialization and cleanup
actually settle. Stop/start does not bypass that resource limit.

Endpoint or credential rotation retires prior connections in that context.
Timeout, revoked credentials, transport loss, expired sessions and malformed
responses retire the shared transport. A timeout on one active search also
invalidates concurrent searches on that connection: the timed-out caller reports
its deadline and other callers report unavailability. None is automatically
replayed. The next explicit command may reconnect. Graph failures retain their
requested policy and offer explicit keyword guidance.

The owning GraphRAG service uses stateless JSON MCP responses. Reuse refuses an
unexpected SSE tool response and closes it, so completed requests cannot leave
unbounded live stream readers. Initialization, search and HTTP error bodies share
a 2 MiB response limit, with Content-Length and truncation checks. SDK stream retries are disabled and a second
`tools/call` POST for one lease is independently blocked.

## Profiling

Set `GRAPHRAG_NOTES_PROFILE_TIMINGS=1` in the gateway's process environment to
emit one sanitized JSON timing entry per valid search. It contains a random
command ID, a hashed session key, policy, outcome, whether initialization was
reused, bounded HTTP request count, and handler/connect/search timestamps. It
contains no endpoint, query, corpus text, record ID, sender, credential or
credential digest. Leave profiling disabled for normal operation.

`GRAPHRAG_NOTES_DISABLE_REUSE=1` provides an explicit fresh-client baseline
without editing saved configuration. Compare the same candidate handler, same
read-only commands and same corpus in controlled process runs. Preserve the
normal configuration, default model, command permissions and plugin settings;
record hardware, runtime versions and concurrent load. First observations are
not guaranteed cold starts. Use at least 20 successful warm observations per
route and report every failed command separately.

[The profiling runner](profile-gateway.mjs) accepts an owner-provided connected
gateway adapter, private known-note cases and diagnostic entries. It creates a
fresh acceptance session, uses `deliver:false`, fences events by session and
run ID, and joins a diagnostic entry by the same session and nested command
interval. It checks the final formatted metadata line of each ranked result, decodes escaped
actor underscores, and ignores citation-looking strings in note content. It requires independently
pinned full readback before and after the run. Raw proof belongs outside the
repository. Its report includes only fixed route names, counts and timings.

Gateway event receipt and browser rendering are separate observations. A real
normal UI walkthrough must additionally establish that the matching reply
renders in the owned acceptance session. No driver event alone proves browser
rendering. Model-orchestrated natural-language calls require a separate sample
set and must not be counted as direct commands. HTTP connect/search intervals
are nested in handler/gateway intervals; do not subtract unrelated percentiles
or claim that SDK initialization measures an SSH handshake on a persistent
tunnel. The runner computes per-command dispatch residuals only from matched
intervals. Correctness fixtures run in CI; live latency thresholds are opt-in.

## Verification

Run verification from a separate writable checkout or copied plugin directory.
For a versioned release bundle, copy this directory out of
`DATA_DIR/releases/vVERSION` before running npm installation or tests. npm creates
`node_modules`; changing the installed version bundle prevents the release
installer from verifying its exact contents during same-version reinstallation.
Plugin installation in OpenClaw remains a separate operator action.

```sh
cd /path/to/writable/graphrag-fast-notes
npm ci --ignore-scripts --no-audit --no-fund
npm test
```

The tests cover the existing direct search/capture contracts, real SDK HTTP
initialization and concurrent calls, per-request authentication, revocation,
rotation, idle/absolute expiry, timeout invalidation, configuration reload,
shutdown, hanging cleanup bounds, JSON response policy, and explicit capture
receipt recovery. All data and credentials are fictional fixtures.
