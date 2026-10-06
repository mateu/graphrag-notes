# Measuring search latency (#95)

`scripts/benchmark-search.py` is an opt-in, read-only standard-library MCP client.
Use a release build and an isolated restored corpus with private cases prepared
using [retrieval evaluation](retrieval-evaluation.md). Record source/binary,
model digest, config and backup identities, corpus size, hardware and load in
private suite metadata. Choose narrow terms, broad queries, entities and filters;
keep judged relevance cases so faster incorrect results do not count as success.

```sh
RUST_LOG=graphrag_service=debug,graphrag_agents=debug,graphrag_db=debug \
  /path/to/release/graphrag --config /private/config.toml \
  --db-path /private/restored-corpus serve \
  --credentials-file /private/read-policy.json --listen 127.0.0.1:3000 \
  2>/private/backend.log

python3 scripts/benchmark-search.py \
  --suite /private/workload.json --endpoint http://127.0.0.1:3000/mcp \
  --credential-env GRAPHRAG_TOKEN --rounds 20 --concurrency 1,4 \
  --timeout 30 --deadline-seconds 900 --backend-log /private/backend.log \
  --output /private/latency-001.json --summary /private/aggregate-001.json
```

Use the existing private credential environment mechanism. The runner refuses
plaintext remote HTTP, bearer values in arguments, proxies and redirects; HTTP
loopback through an encrypted SSH tunnel is supported. It never opens RocksDB.
Protect raw queries, IDs, revisions and debug traces in owner-private directories.
New reports are mode0600, directories mode0700, and existing reports are refused.
The public summary uses only categories/policies, numeric counts/timings, static
failure labels and suite hash; it never exports arbitrary suite metadata.

## What is measured

Each lane initializes MCP once, then performs search and revision-pinned
inspection before its next search. The measured `search_rpc_ms` spans the HTTP
search request/response, excluding initialization, result inspection and any
language model deciding which tool to use. Clients use stateless HTTP requests;
this does not model a persistent SDK connection pool. One and four lanes are
explicit client loads, bounded to eight; outstanding provider work may consume
separate server/provider concurrency limits. Four lanes do not claim four
simultaneous search requests at every instant because readbacks occur between
searches. Compare against an OpenClaw direct gateway separately.

The first observation for each policy is separate from twenty warmed repetitions.
It is **not** evidence of a cold embedding model or an empty filesystem cache:
an earlier hybrid route may load the model. To measure model loading explicitly,
start a separate isolated Ollama listener using the existing model files,
confirm its running-model list is empty, and capture the first hybrid request
before warmup. Do not unload a production provider or stop another corpus owner.
Record first unloaded-model latency and model `load_duration` separately from
steady-state percentile samples; model loading and OS cache coldness differ.

A private service debug span hashes the exact JSON-RPC identity. Existing backend
phase events inherit that span through async polling, including concurrent
requests. The runner joins **only that same RPC hash**, and reports how many
application phases were observed. Without tracing there is no backend phase
claim. `application_search` includes application retrieval and enrichment but
excludes MCP request parsing, authentication, ingress admission, response
serialization and transport. Other phases can nest or repeat; their times are
not additive. Never subtract unrelated requests to infer transport or model time.
Debug logging has measurement overhead; retain both traced and untraced runs.

## Budgets, failures and limits

Public p50/p95 use nearest-rank p95 and require at least twenty successful
samples. Counts, failures and per-category timings stay visible. The warmed RPC
operating targets are keyword<=250ms and hybrid/graph<=1000ms; graph overhead is
the paired same-query/round graph-minus-hybrid RPC delta<=250ms. This delta is an
interleaved comparison, not an additive internal graph timer or a simultaneous
paired experiment. Failures prevent a successful budget claim. Keep hardware
limits, corpus/load size and sample denominators beside any published result.

Transport timeouts close the client request and count that comparison as failed;
there is no automatic retry or mode fallback. They do not guarantee upstream
inference immediately stops. The suite deadline prevents admitting new requests
once expired and bounds subsequent request socket timeouts; every scheduled
comparison remains accounted for. Socket timeouts bound blocking I/O, rather
than a total slow-response wall deadline. An unavailable initialization marks all work
assigned to that lane failed. Inspect server traces for work that completes after
a timed-out connection. Hard cancellation of upstream provider work is a
separate service capability, not inferred from closing an HTTP socket.

Failed guarded inspection is a failed sample even when a search timing exists.
Successful samples retain rank metrics, provenance, revision-pinned readback and
private IDs. Paired graph results also count judged-positive RR regressions;
latency improvement never establishes a relevance improvement. Wall-clock
thresholds belong in controlled runs; fictional protocol/metric/correlation
correctness remains in normal CI. Do not introduce ANN, cache or concurrency
changes without paired relevance and timing evidence.

Natural-language OpenClaw measurements additionally include discovery/model
planning/response generation. Report those separately, with actual tool calls
and direct gateway timings; do not label a multi-second model conversation as
the MCP backend's search latency.
