# OpenClaw dispatch and connection reuse (#108)

The maintained OpenClaw 0.5.0 client passed the normal gateway read workflow with bounded MCP reuse. Warm connection preparation fell from median 11.55–13.59 ms with fresh clients to 0.05–0.07 ms with reuse. Overall reply improvements were modest and varied by route: the hybrid median increased slightly. The backend and saved hybrid/graph-on defaults remained unchanged.

The accepted comparison used two private known-note queries through the normal webchat WebSocket API, one first observation and 20 alternating warm observations per route, with fresh and reuse roles run sequentially. Both roles used identical client payloads from `dbd320bd2e264d833cc2e125e9e33d0551ffa4a6`, OpenClaw 2026.9.8, MCP SDK 1.30.0 and Node 24.21.0. The unchanged backend was published v0.1.0-rc.3. Heavy builds and provider experiments were held; the aggregate records the observed host load.

## Warm gateway replies

Each cell is milliseconds. Percentiles use the nearest-rank rule over 20 successful observations per route. Every requested measured command is counted.

| Route | Fresh median | Reuse median | Fresh p95 | Reuse p95 |
| --- | ---: | ---: | ---: | ---: |
| Keyword / graph off | 74.17 | 72.18 | 131.21 | 135.16 |
| Hybrid / graph off | 482.13 | 485.49 | 548.05 | 525.33 |
| Hybrid / graph on, title | 484.35 | 467.73 | 534.28 | 507.11 |
| Hybrid / graph on, body | 477.88 | 470.02 | 561.28 | 527.36 |

## Matched command phases

The handler emits bounded optional diagnostic entries. The runner joins each entry to the same owned session, policy and nested gateway command interval. Connect includes client preparation or reuse admission; search covers the MCP tool request; handler covers permission checks, preparation, request and formatting. Dispatch before/after is computed separately for each command from the matching handler and gateway timestamps on the client host. The aggregate retains median, p95, minimum and maximum for every phase.

| Route | Fresh connect median | Reuse connect median | Fresh handler median | Reuse handler median |
| --- | ---: | ---: | ---: | ---: |
| Keyword / graph off | 11.851 | 0.052 | 21.10 | 13.80 |
| Hybrid / graph off | 11.551 | 0.071 | 425.20 | 413.42 |
| Hybrid / graph on, title | 13.041 | 0.051 | 424.30 | 411.98 |
| Hybrid / graph on, body | 13.586 | 0.057 | 424.56 | 413.83 |

Search and gateway dispatch remain material. Search medians were 9.11/13.54 ms for keyword and approximately 408–414 ms for the other routes. Dispatch-before medians were approximately 38–46 ms and dispatch-after medians approximately 12–15 ms. These are individual phase distributions; subtracting their aggregate percentiles would not identify a causal contribution.

## First observations and correctness

First observations are separate from warm percentiles and are not guaranteed cold. The JSON retains all their measured phases. First gateway replies were:

| Route | Fresh first | Reuse first |
| --- | ---: | ---: |
| Keyword / graph off | 646.25 | 623.81 |
| Hybrid / graph off | 530.21 | 989.97 |
| Hybrid / graph on, title | 492.20 | 711.42 |
| Hybrid / graph on, body | 491.78 | 571.52 |

- Fresh and reuse each passed 86/86 gateway commands: two contract probes plus 84 searches, with 84 unique same-command diagnostic joins. All 80 warm reuse searches reused an initialized connection.
- Every returned ranked record ID and actor matched across the two roles. Independent full revision-pinned readback matched before and after, and saved configuration, default model, permissions and client sources remained unchanged. Raw cases, records, credentials and configuration fingerprints remain private.
- Four simultaneous gateway requests passed all four replies, diagnostic joins, pins and owned-session archives. Gateway request overlap was four; handler overlap was one. This is an acceptance observation, with no concurrent handler throughput or percentile claim.
- After removing the owned profiling flags and restarting the gateway, the installed 0.5.0 client passed ten normal commands, with full pinned readback, exact source/metadata checks and unchanged settings. The accepted workflow totals 186/186 successful commands, including 180 read searches and six contract probes. Every owned session was archived.

## Setup failures and startup readiness

Three earlier adapter preflights failed before sending any measured chat command, and one owner transition failed before candidate admission. Two additional successful fresh roles (172 commands) remain separately retained and are excluded from the final comparison. The original 0.4.0 client and normal read readiness were verified after owned rollbacks. These failures are not successful latency samples.

The actual gateway listener appeared 26.56–29.63 seconds after systemd reported the owner active. A 12-second authenticated-hello setup deadline expired before the listener existed. The private measurement adapter now allows a bounded 60-second native hello/reconnect wait, retains 12-second RPC and five-second cleanup bounds, and rejects terminal authentication, scope or version failures. It submits no chat/tool retry during startup. This setup wait is outside measured search intervals. Product search deadlines and saved preferences remain unchanged.

## Remaining acceptance

Normal gateway final-event delivery is verified. Visible browser rendering remains pending because the browser connection exposed no available browser. [Issue #108](https://github.com/mateu/graphrag-notes/issues/108) remains open for that explicit criterion. The observations contain no natural-language model turns and measure no SSH handshake time. Two queries and 20 warm samples per route provide a narrow operational check, rather than a broad relevance or tail guarantee.

See the [sanitized phase aggregate](openclaw-dispatch-108.json), [maintained client and profiling runner](../../clients/openclaw-fast-notes/README.md), and [PR #112](https://github.com/mateu/graphrag-notes/pull/112).
